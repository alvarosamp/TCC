"""
MLOps Pipeline Completo:
  1. Edge Benchmark — converte melhor modelo pra TFLite + quantização int8
  2. Drift Detection — teste KS sobre erro de reconstrução
  3. Retreinamento Automático — quando drift é detectado
  4. OTA Simulado — substituição atômica do modelo

Roda DEPOIS dos 3 Optuna + comparação final.
Usa o MELHOR modelo encontrado (lê dos JSONs).
"""

import json
import os
import time
import warnings
import shutil
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ---------- Paths ----------
TCC_BASE = os.environ.get("TCC_BASE", "/mnt/c/TCC_data")
DATA_DIR = os.environ.get("TCC_PROCESSED_DIR", os.path.join(TCC_BASE, "processed"))
RESULTS_BASE = os.environ.get("TCC_RESULTS_BASE", os.path.join(DATA_DIR, "results"))
MLOPS_DIR = os.path.join(RESULTS_BASE, "mlops")
EDGE_DIR = os.path.join(MLOPS_DIR, "edge")
OTA_DIR = os.path.join(MLOPS_DIR, "ota_versions")
os.makedirs(EDGE_DIR, exist_ok=True)
os.makedirs(OTA_DIR, exist_ok=True)

JANELA_NPTS = 800
SEED = 42
np.random.seed(SEED)


# =====================================================================
# PARTE 1 — EDGE BENCHMARK (TFLite + Quantização int8)
# =====================================================================

def encontrar_melhor_modelo():
    """Lê os JSONs dos 3 Optuna e retorna path do melhor modelo (por AUC-PR estação)."""
    candidatos = [
        ("Dense AE", "optuna_dense_ae", "resultados_optuna_dense_ae.json", "dense_ae_opt_estacao.keras"),
        ("CNN 1D AE", "optuna_cnn1d_ae", "resultados_optuna_cnn1d_ae.json", "cnn1d_ae_opt_estacao.keras"),
        ("LSTM AE", "optuna_lstm_ae", "resultados_optuna_lstm_ae.json", "lstm_ae_opt_estacao.keras"),
    ]
    melhor_nome, melhor_path, melhor_auc = None, None, -1
    melhor_json = None

    for nome, pasta, json_file, keras_file in candidatos:
        json_path = os.path.join(RESULTS_BASE, pasta, json_file)
        model_path = os.path.join(RESULTS_BASE, pasta, keras_file)
        if os.path.exists(json_path) and os.path.exists(model_path):
            with open(json_path) as f:
                data = json.load(f)
            auc = data["estacao"]["test"]["auc_pr"]
            print(f"  {nome}: AUC-PR estação = {auc:.4f}")
            if auc > melhor_auc:
                melhor_nome, melhor_path, melhor_auc = nome, model_path, auc
                melhor_json = data

    if melhor_nome is None:
        print("  ERRO: nenhum modelo Optuna encontrado. Rode os scripts Optuna primeiro.")
        return None, None, None, None

    print(f"\n  MELHOR: {melhor_nome} (AUC-PR = {melhor_auc:.4f})")
    return melhor_nome, melhor_path, melhor_auc, melhor_json


def converter_tflite(model, nome, X_cal):
    """Converte modelo Keras pra TFLite (float32 e int8 quantizado)."""
    print(f"\n  Convertendo {nome} pra TFLite...")

    # float32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float = converter.convert()
    float_path = os.path.join(EDGE_DIR, f"{nome}_float32.tflite")
    with open(float_path, "wb") as f:
        f.write(tflite_float)
    float_size = os.path.getsize(float_path)
    print(f"    Float32: {float_size/1024:.1f} KB")

    # int8 quantizado
    def representative_dataset():
        for i in range(min(500, len(X_cal))):
            sample = X_cal[i:i+1].astype(np.float32)
            yield [sample]

    converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_q.representative_dataset = representative_dataset
    converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_q.inference_input_type = tf.int8
    converter_q.inference_output_type = tf.int8

    try:
        tflite_int8 = converter_q.convert()
        int8_path = os.path.join(EDGE_DIR, f"{nome}_int8.tflite")
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
        int8_size = os.path.getsize(int8_path)
        print(f"    Int8:    {int8_size/1024:.1f} KB ({100*int8_size/float_size:.0f}% do float32)")
        return float_path, int8_path, float_size, int8_size
    except Exception as e:
        print(f"    Int8 falhou (normal pra LSTM): {e}")
        return float_path, None, float_size, None


def benchmark_tflite(tflite_path, X_test, n_runs=100):
    """Mede tempo de inferência do TFLite."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    times = []
    for i in range(n_runs):
        sample = X_test[i % len(X_test):i % len(X_test) + 1]
        sample = sample.reshape(input_shape).astype(input_dtype)
        interpreter.set_tensor(input_details[0]["index"], sample)
        t0 = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "mean_ms": round(float(np.mean(times)), 3),
        "median_ms": round(float(np.median(times)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "min_ms": round(float(np.min(times)), 3),
        "max_ms": round(float(np.max(times)), 3),
    }


def parte1_edge_benchmark():
    print("=" * 70)
    print("PARTE 1/4 — EDGE BENCHMARK")
    print("=" * 70)

    nome, model_path, auc, model_json = encontrar_melhor_modelo()
    if nome is None:
        return {}

    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape

    # preparar dados de calibração e teste
    data = np.load(os.path.join(DATA_DIR, "dataset_v3_split_estacao.npz"))
    X_train_n = data["X_train"][data["y_train"] == 0]
    X_test = data["X_test"]

    # reshape se necessário (CNN/LSTM)
    if len(input_shape) == 3:
        ts, ft = input_shape[1], input_shape[2]
        X_cal = X_train_n[:500].reshape(-1, ts, ft)
        X_bench = X_test[:100].reshape(-1, ts, ft)
    else:
        X_cal = X_train_n[:500]
        X_bench = X_test[:100]

    float_path, int8_path, float_size, int8_size = converter_tflite(model, nome, X_cal)

    resultados = {
        "melhor_modelo": nome,
        "auc_pr_estacao": auc,
        "float32": {"path": float_path, "size_kb": round(float_size / 1024, 1)},
    }

    print(f"\n  Benchmarking float32...")
    bench_float = benchmark_tflite(float_path, X_bench)
    resultados["float32"]["inferencia"] = bench_float
    print(f"    Inferência: {bench_float['mean_ms']:.3f}ms (média) | {bench_float['p95_ms']:.3f}ms (p95)")

    if int8_path:
        print(f"  Benchmarking int8...")
        bench_int8 = benchmark_tflite(int8_path, X_bench)
        resultados["int8"] = {
            "path": int8_path,
            "size_kb": round(int8_size / 1024, 1),
            "inferencia": bench_int8,
            "compression_ratio": round(float_size / int8_size, 2) if int8_size else None,
        }
        print(f"    Inferência: {bench_int8['mean_ms']:.3f}ms (média) | {bench_int8['p95_ms']:.3f}ms (p95)")
        speedup = bench_float["mean_ms"] / bench_int8["mean_ms"] if bench_int8["mean_ms"] > 0 else 0
        print(f"    Speedup int8 vs float32: {speedup:.2f}x")
        resultados["speedup_int8"] = round(speedup, 2)

    tf.keras.backend.clear_session()
    return resultados


# =====================================================================
# PARTE 2 — DRIFT DETECTION (Kolmogorov-Smirnov)
# =====================================================================

def calcular_scores_modelo(model_path, X):
    """Calcula MSE de reconstrução pra um conjunto de dados."""
    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape
    if len(input_shape) == 3:
        ts, ft = input_shape[1], input_shape[2]
        X_in = X.reshape(-1, ts, ft)
    else:
        X_in = X
    X_rec = model.predict(X_in, batch_size=1024, verbose=0)
    if len(input_shape) == 3:
        X_rec = X_rec.reshape(-1, JANELA_NPTS)
    return np.mean((X - X_rec) ** 2, axis=1)


def parte2_drift_detection():
    print("\n" + "=" * 70)
    print("PARTE 2/4 — DRIFT DETECTION (Kolmogorov-Smirnov)")
    print("=" * 70)

    nome, model_path, _, _ = encontrar_melhor_modelo()
    if nome is None:
        return {}

    # usar split temporal: train é o "passado", test é o "futuro"
    data = np.load(os.path.join(DATA_DIR, "dataset_v3_split_temporal.npz"))
    X_train_n = data["X_train"][data["y_train"] == 0]
    X_test_n = data["X_test"][data["y_test"] == 0]

    print(f"\n  Calculando distribuição de referência (train)...")
    scores_ref = calcular_scores_modelo(model_path, X_train_n[:5000])
    print(f"    Ref: mean={np.mean(scores_ref):.6f} std={np.std(scores_ref):.6f}")

    print(f"  Calculando distribuição atual (test)...")
    scores_new = calcular_scores_modelo(model_path, X_test_n[:5000])
    print(f"    New: mean={np.mean(scores_new):.6f} std={np.std(scores_new):.6f}")

    # KS test
    ks_stat, p_value = ks_2samp(scores_ref, scores_new)
    drift_detected = p_value < 0.05

    print(f"\n  KS statistic: {ks_stat:.6f}")
    print(f"  p-value:      {p_value:.6f}")
    print(f"  Drift:        {'SIM — DETECTADO' if drift_detected else 'NÃO detectado'}")
    print(f"  (threshold:   p < 0.05)")

    # simular monitoramento em janelas deslizantes
    print(f"\n  Simulando monitoramento em janelas de 500 amostras...")
    window_size = 500
    n_windows = min(10, len(X_test_n) // window_size)
    drift_log = []
    for i in range(n_windows):
        window = X_test_n[i * window_size:(i + 1) * window_size]
        scores_w = calcular_scores_modelo(model_path, window)
        ks_w, p_w = ks_2samp(scores_ref, scores_w)
        drift_w = p_w < 0.05
        drift_log.append({
            "window": i, "ks_stat": round(float(ks_w), 6),
            "p_value": round(float(p_w), 6), "drift": drift_w,
        })
        status = "DRIFT" if drift_w else "ok"
        print(f"    Window {i}: KS={ks_w:.4f} p={p_w:.4f} → {status}")

    tf.keras.backend.clear_session()

    return {
        "modelo": nome,
        "ks_global": {"ks_stat": round(float(ks_stat), 6),
                      "p_value": round(float(p_value), 6),
                      "drift_detected": drift_detected},
        "monitoramento": drift_log,
        "ref_stats": {"mean": round(float(np.mean(scores_ref)), 6),
                      "std": round(float(np.std(scores_ref)), 6)},
    }


# =====================================================================
# PARTE 3 — RETREINAMENTO AUTOMÁTICO
# =====================================================================

def parte3_retreinamento(drift_result):
    print("\n" + "=" * 70)
    print("PARTE 3/4 — RETREINAMENTO AUTOMÁTICO")
    print("=" * 70)

    if not drift_result.get("ks_global", {}).get("drift_detected", False):
        print("  Drift NÃO detectado — retreinamento não é necessário.")
        print("  (Simulando retreinamento mesmo assim para demonstração)")

    nome, model_path, _, _ = encontrar_melhor_modelo()
    if nome is None:
        return {}

    # simular: retreinar com dados mais recentes (test vira parte do train)
    data = np.load(os.path.join(DATA_DIR, "dataset_v3_split_temporal.npz"))
    X_train_old = data["X_train"][data["y_train"] == 0]
    X_val_n = data["X_val"][data["y_val"] == 0]
    X_val_a = data["X_val"][data["y_val"] == 1]
    X_test_n = data["X_test"][data["y_test"] == 0]
    X_test_a = data["X_test"][data["y_test"] == 1]

    # "novo" train = 70% do train antigo + 30% do test normal (simula dados recentes)
    n_new = len(X_test_n) // 3
    X_train_new = np.concatenate([X_train_old, X_test_n[:n_new]], axis=0)
    np.random.shuffle(X_train_new)

    print(f"  Train original: {X_train_old.shape}")
    print(f"  Train com dados recentes: {X_train_new.shape} (+{n_new} janelas novas)")

    # carregar e retreinar o melhor modelo
    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape

    if len(input_shape) == 3:
        ts, ft = input_shape[1], input_shape[2]
        X_tr = X_train_new.reshape(-1, ts, ft)
        X_vn = X_val_n.reshape(-1, ts, ft)
    else:
        X_tr = X_train_new
        X_vn = X_val_n

    # compilar com LR menor (fine-tuning)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss="mse")

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                          restore_best_weights=True, verbose=1)

    print(f"\n  Retreinando (fine-tuning, LR=5e-5)...")
    t0 = time.time()
    h = model.fit(X_tr, X_tr, validation_data=(X_vn, X_vn),
                  epochs=30, batch_size=256, callbacks=[es], verbose=1)
    tempo = time.time() - t0
    print(f"  Retreino concluído em {tempo/60:.1f} min")

    # avaliar modelo retreinado
    if len(input_shape) == 3:
        r_n = model.predict(X_test_n[n_new:].reshape(-1, ts, ft), batch_size=1024, verbose=0)
        r_a = model.predict(X_test_a.reshape(-1, ts, ft), batch_size=1024, verbose=0)
        s_n = np.mean((X_test_n[n_new:] - r_n.reshape(-1, JANELA_NPTS)) ** 2, axis=1)
        s_a = np.mean((X_test_a - r_a.reshape(-1, JANELA_NPTS)) ** 2, axis=1)
    else:
        r_n = model.predict(X_test_n[n_new:], batch_size=1024, verbose=0)
        r_a = model.predict(X_test_a, batch_size=1024, verbose=0)
        s_n = np.mean((X_test_n[n_new:] - r_n) ** 2, axis=1)
        s_a = np.mean((X_test_a - r_a) ** 2, axis=1)

    y_true = np.concatenate([np.zeros(len(s_n)), np.ones(len(s_a))])
    y_scores = np.concatenate([s_n, s_a])
    auc_pr_new = average_precision_score(y_true, y_scores)
    auc_roc_new = roc_auc_score(y_true, y_scores)

    retrained_path = os.path.join(MLOPS_DIR, f"retrained_{nome}.keras")
    model.save(retrained_path)

    print(f"\n  Modelo retreinado: AUC-PR={auc_pr_new:.4f} AUC-ROC={auc_roc_new:.4f}")

    tf.keras.backend.clear_session()

    return {
        "modelo_original": nome,
        "train_size_original": len(X_train_old),
        "train_size_retrained": len(X_train_new),
        "dados_novos_adicionados": n_new,
        "auc_pr_retrained": round(float(auc_pr_new), 4),
        "auc_roc_retrained": round(float(auc_roc_new), 4),
        "tempo_retreino_min": round(tempo / 60, 2),
        "epochs": len(h.history["loss"]),
        "retrained_path": retrained_path,
    }


# =====================================================================
# PARTE 4 — OTA SIMULADO
# =====================================================================

def parte4_ota_simulado(edge_result, retrain_result):
    print("\n" + "=" * 70)
    print("PARTE 4/4 — OTA SIMULADO (Over-The-Air Update)")
    print("=" * 70)

    if not retrain_result.get("retrained_path"):
        print("  Sem modelo retreinado — pulando OTA.")
        return {}

    # simular: converter modelo retreinado pra TFLite e "enviar" pro device
    retrained_model = tf.keras.models.load_model(retrain_result["retrained_path"])
    input_shape = retrained_model.input_shape

    data = np.load(os.path.join(DATA_DIR, "dataset_v3_split_temporal.npz"))
    X_cal = data["X_train"][data["y_train"] == 0][:500]
    if len(input_shape) == 3:
        ts, ft = input_shape[1], input_shape[2]
        X_cal = X_cal.reshape(-1, ts, ft)

    print("  Convertendo modelo retreinado pra TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(retrained_model)
    tflite_bytes = converter.convert()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(OTA_DIR, f"v_{timestamp}")
    os.makedirs(version_dir, exist_ok=True)

    # salvar modelo + metadata
    model_ota_path = os.path.join(version_dir, "model.tflite")
    with open(model_ota_path, "wb") as f:
        f.write(tflite_bytes)

    metadata = {
        "version": timestamp,
        "modelo_base": retrain_result["modelo_original"],
        "auc_pr": retrain_result["auc_pr_retrained"],
        "auc_roc": retrain_result["auc_roc_retrained"],
        "dados_novos": retrain_result["dados_novos_adicionados"],
        "size_kb": round(len(tflite_bytes) / 1024, 1),
        "created_at": datetime.now().isoformat(),
    }
    with open(os.path.join(version_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # simular deploy atômico
    current_link = os.path.join(OTA_DIR, "current")
    if os.path.islink(current_link) or os.path.exists(current_link):
        old_target = os.path.realpath(current_link) if os.path.islink(current_link) else current_link
        print(f"  Modelo anterior: {old_target}")
        if os.path.islink(current_link):
            os.unlink(current_link)
        elif os.path.isdir(current_link):
            shutil.rmtree(current_link)

    # criar symlink (deploy atômico)
    try:
        os.symlink(version_dir, current_link)
        print(f"  Symlink criado: {current_link} → {version_dir}")
    except OSError:
        # Windows/WSL pode não suportar symlink — copia
        shutil.copytree(version_dir, current_link)
        print(f"  Cópia criada: {current_link}")

    print(f"  OTA concluído!")
    print(f"    Versão: {timestamp}")
    print(f"    Tamanho: {metadata['size_kb']:.1f} KB")
    print(f"    AUC-PR: {metadata['auc_pr']:.4f}")

    # listar todas as versões
    versions = sorted(os.listdir(OTA_DIR))
    versions = [v for v in versions if v.startswith("v_")]
    print(f"\n  Versões no dispositivo: {len(versions)}")
    for v in versions:
        meta_path = os.path.join(OTA_DIR, v, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                m = json.load(f)
            print(f"    {v}: AUC-PR={m['auc_pr']:.4f} | {m['size_kb']:.1f} KB")

    tf.keras.backend.clear_session()

    return metadata


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("MLOps PIPELINE COMPLETO")
    print(f"  TensorFlow: {tf.__version__}")
    print(f"  GPU: {gpus[0].name if gpus else 'CPU'}")
    print("=" * 70)

    resultados = {}

    # Parte 1
    resultados["edge"] = parte1_edge_benchmark()

    # Parte 2
    resultados["drift"] = parte2_drift_detection()

    # Parte 3
    resultados["retrain"] = parte3_retreinamento(resultados["drift"])

    # Parte 4
    resultados["ota"] = parte4_ota_simulado(resultados["edge"], resultados["retrain"])

    # salvar tudo
    out_path = os.path.join(MLOPS_DIR, "resultados_mlops.json")
    with open(out_path, "w") as f:
        json.dump(resultados, f, indent=2, default=str)
    print(f"\n  Resultados salvos: {out_path}")

    # resumo
    print("\n" + "=" * 70)
    print("RESUMO MLOps")
    print("=" * 70)
    if resultados.get("edge"):
        e = resultados["edge"]
        print(f"  Edge: {e.get('melhor_modelo', '?')} | Float32: {e.get('float32', {}).get('size_kb', '?')} KB")
        if "int8" in e:
            print(f"         Int8: {e['int8']['size_kb']} KB | Speedup: {e.get('speedup_int8', '?')}x")
    if resultados.get("drift"):
        d = resultados["drift"]["ks_global"]
        print(f"  Drift: KS={d['ks_stat']:.4f} p={d['p_value']:.6f} → {'DETECTADO' if d['drift_detected'] else 'não'}")
    if resultados.get("retrain"):
        r = resultados["retrain"]
        print(f"  Retrain: AUC-PR={r['auc_pr_retrained']:.4f} ({r['tempo_retreino_min']:.1f}min)")
    if resultados.get("ota"):
        o = resultados["ota"]
        print(f"  OTA: v{o.get('version', '?')} | {o.get('size_kb', '?')} KB")
    print("=" * 70)


if __name__ == "__main__":
    main()