"""
Passo 6 v3 — LSTM Autoencoder para detecção de anomalias.

Diferença vs CNN 1D: memória temporal explícita conecta início, meio
e fim da janela. LSTM "lembra" o que viu 20 segundos atrás.

Hipótese: AUC-PR no split por estação deve subir acima de 0.20.
"""

import json
import os
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ---------- Verificar GPU ----------
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs disponíveis: {len(gpus)}")
if gpus:
    print(f"  {gpus[0].name}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("  AVISO: rodando em CPU")

# ---------- Paths ----------
DATA_DIR = "/mnt/c/TCC_data/processed"
RESULTS_DIR = "/mnt/c/TCC_data/processed/results/passo_06"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- Hiperparâmetros ----------
JANELA_NPTS = 800
TIMESTEPS = 50        # 800 / 50 = 16 features por timestep
FEATURES = 16         # cada timestep vê 16 amostras (400ms @ 40Hz)
EPOCHS = 200
BATCH_SIZE = 256
PATIENCE = 25
LR = 1e-3
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ---------- Modelo ----------
def build_lstm_ae(timesteps, features):
    """
    Encoder:
      LSTM(128, return_sequences=True)  → (50, 128)
      LSTM(64)                          → (64,)      ← BOTTLENECK

    Decoder:
      RepeatVector(50)                  → (50, 64)
      LSTM(64, return_sequences=True)   → (50, 64)
      LSTM(128, return_sequences=True)  → (50, 128)
      TimeDistributed(Dense(16))        → (50, 16)   ← OUTPUT

    Total: ~330k parâmetros.

    Reshape strategy:
      (800,) → (50 timesteps, 16 features)
      Cada timestep = 16 amostras consecutivas = 400ms de sinal
      LSTM processa 50 passos → "vê" a evolução do sinal ao longo de 20s
    """
    inputs = tf.keras.layers.Input(shape=(timesteps, features))

    # --- Encoder ---
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    encoded = tf.keras.layers.LSTM(64, return_sequences=False)(x)

    # --- Decoder ---
    x = tf.keras.layers.RepeatVector(timesteps)(encoded)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    decoded = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(features, activation="tanh")
    )(x)

    model = tf.keras.Model(inputs, decoded, name="lstm_ae")
    return model


# ---------- Reshape ----------
def reshape_para_lstm(X):
    """(N, 800) → (N, 50, 16)"""
    return X.reshape(-1, TIMESTEPS, FEATURES)


def reshape_para_flat(X):
    """(N, 50, 16) → (N, 800)"""
    return X.reshape(-1, JANELA_NPTS)


# ---------- Métricas ----------
def calcular_scores(model, X):
    X_3d = reshape_para_lstm(X)
    X_rec = model.predict(X_3d, batch_size=1024, verbose=0)
    X_rec_flat = reshape_para_flat(X_rec)
    return np.mean((X - X_rec_flat) ** 2, axis=1)


def avaliar(scores_normal, scores_anomalo):
    y_true = np.concatenate([
        np.zeros(len(scores_normal)),
        np.ones(len(scores_anomalo)),
    ])
    y_scores = np.concatenate([scores_normal, scores_anomalo])

    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_vals = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1_vals)
    f1_best = float(f1_vals[best_idx])
    threshold_best = float(thresholds[min(best_idx, len(thresholds) - 1)])

    return {
        "auc_roc": round(float(auc_roc), 4),
        "auc_pr": round(float(auc_pr), 4),
        "f1_best": round(f1_best, 4),
        "threshold": round(threshold_best, 6),
        "n_normal": len(scores_normal),
        "n_anomalo": len(scores_anomalo),
    }


def salvar_figuras(history, scores_val_n, scores_val_a,
                   scores_test_n, scores_test_a, nome_split):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"LSTM AE — Split {nome_split}", fontsize=14)

        ax = axes[0, 0]
        ax.plot(history.history["loss"], label="Train loss")
        ax.plot(history.history["val_loss"], label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.set_title("Curva de Treino")

        ax = axes[0, 1]
        ax.hist(scores_val_n, bins=100, alpha=0.6, label="Normal", density=True)
        ax.hist(scores_val_a, bins=100, alpha=0.6, label="Anômalo", density=True)
        ax.set_xlabel("Erro de Reconstrução (MSE)")
        ax.set_ylabel("Densidade")
        ax.legend()
        ax.set_title("Distribuição de Erros (Val)")

        ax = axes[1, 0]
        y_true = np.concatenate([np.zeros(len(scores_test_n)),
                                  np.ones(len(scores_test_a))])
        y_scores = np.concatenate([scores_test_n, scores_test_a])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = roc_auc_score(y_true, y_scores)
        ax.plot(fpr, tpr, label=f"AUC-ROC = {auc_val:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        ax.set_title("Curva ROC (Test)")

        ax = axes[1, 1]
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        baseline = len(scores_test_a) / (len(scores_test_n) + len(scores_test_a))
        ax.plot(recall, precision, label=f"AUC-PR = {ap:.4f}")
        ax.axhline(y=baseline, color="r", linestyle="--",
                   alpha=0.3, label=f"Baseline = {baseline:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.set_title("Curva Precision-Recall (Test)")

        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"lstm_ae_{nome_split}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Figura salva: {path}")
    except Exception as e:
        print(f"  AVISO: não conseguiu salvar figuras: {e}")


# ---------- Treino de um split ----------
def treinar_split(nome_split, path_npz):
    print("=" * 70)
    print(f"TREINO — LSTM AE — Split {nome_split}")
    print("=" * 70)

    print("Carregando dados...")
    data = np.load(path_npz)
    X_train_all, y_train = data["X_train"], data["y_train"]
    X_val_all, y_val = data["X_val"], data["y_val"]
    X_test_all, y_test = data["X_test"], data["y_test"]

    X_train_normal = X_train_all[y_train == 0]
    X_val_normal = X_val_all[y_val == 0]
    X_val_anomalo = X_val_all[y_val == 1]
    X_test_normal = X_test_all[y_test == 0]
    X_test_anomalo = X_test_all[y_test == 1]

    print(f"  Train (normal only): {X_train_normal.shape}")
    print(f"  Val normal:          {X_val_normal.shape}")
    print(f"  Val anômalo:         {X_val_anomalo.shape}")
    print(f"  Test normal:         {X_test_normal.shape}")
    print(f"  Test anômalo:        {X_test_anomalo.shape}")

    # reshape pra LSTM: (N, 800) → (N, 50, 16)
    X_train_3d = reshape_para_lstm(X_train_normal)
    X_val_3d = reshape_para_lstm(X_val_normal)

    print(f"  Reshape: (N, 800) → (N, {TIMESTEPS}, {FEATURES})")
    print(f"  Cada timestep = {FEATURES} amostras = {FEATURES/40*1000:.0f}ms de sinal")
    print(f"  Total timesteps = {TIMESTEPS} passos = {TIMESTEPS*FEATURES/40:.1f}s")

    model = build_lstm_ae(TIMESTEPS, FEATURES)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss="mse")
    model.summary()
    print(f"\n  Parâmetros totais: {model.count_params():,}")
    print(f"  Razão amostras/params: {len(X_train_normal)/model.count_params():.1f}:1")

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE,
        restore_best_weights=True, verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1,
    )

    print("\nIniciando treino...")
    t0 = time.time()
    history = model.fit(
        X_train_3d, X_train_3d,
        validation_data=(X_val_3d, X_val_3d),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    tempo_treino = time.time() - t0
    print(f"\n  Treino concluído em {tempo_treino/60:.1f} min")
    print(f"  Épocas rodadas: {len(history.history['loss'])}")
    print(f"  Melhor val_loss: {min(history.history['val_loss']):.6f}")

    print("\nCalculando scores de anomalia...")
    scores_val_n = calcular_scores(model, X_val_normal)
    scores_val_a = calcular_scores(model, X_val_anomalo)
    scores_test_n = calcular_scores(model, X_test_normal)
    scores_test_a = calcular_scores(model, X_test_anomalo)

    print("\n--- Resultados VAL ---")
    res_val = avaliar(scores_val_n, scores_val_a)
    for k, v in res_val.items():
        print(f"  {k}: {v}")

    print("\n--- Resultados TEST ---")
    res_test = avaliar(scores_test_n, scores_test_a)
    for k, v in res_test.items():
        print(f"  {k}: {v}")

    model_path = os.path.join(RESULTS_DIR, f"lstm_ae_{nome_split}.keras")
    model.save(model_path)
    print(f"\n  Modelo salvo: {model_path}")

    salvar_figuras(history, scores_val_n, scores_val_a,
                   scores_test_n, scores_test_a, nome_split)

    return {
        "split": nome_split,
        "hiperparametros": {
            "lstm_units": [128, 64, 64, 128],
            "timesteps": TIMESTEPS,
            "features": FEATURES,
            "epochs_rodadas": len(history.history["loss"]),
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "patience": PATIENCE,
            "params_total": model.count_params(),
        },
        "tempo_treino_min": round(tempo_treino / 60, 2),
        "melhor_val_loss": round(float(min(history.history["val_loss"])), 6),
        "val": res_val,
        "test": res_test,
    }


# ---------- Main ----------
def main():
    print("=" * 70)
    print("PASSO 6 v3 — LSTM Autoencoder")
    print(f"  TensorFlow: {tf.__version__}")
    print(f"  GPU: {gpus[0].name if gpus else 'CPU'}")
    print(f"  Dados: {DATA_DIR}")
    print(f"  Resultados: {RESULTS_DIR}")
    print("=" * 70)

    resultados = {}

    resultados["estacao"] = treinar_split(
        "estacao",
        os.path.join(DATA_DIR, "dataset_v3_split_estacao.npz"),
    )

    resultados["temporal"] = treinar_split(
        "temporal",
        os.path.join(DATA_DIR, "dataset_v3_split_temporal.npz"),
    )

    out_path = os.path.join(RESULTS_DIR, "resultados_passo_06.json")
    with open(out_path, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"\n  Resultados salvos: {out_path}")

    # Resumo final com comparação completa
    print("\n" + "=" * 70)
    print("RESUMO FINAL — LSTM AE")
    print("=" * 70)
    print(f"{'Métrica':<20} {'Estação':>12} {'Temporal':>12}")
    print("-" * 44)
    for m in ("auc_pr", "auc_roc", "f1_best"):
        v_est = resultados["estacao"]["test"][m]
        v_tmp = resultados["temporal"]["test"][m]
        print(f"{m:<20} {v_est:>12.4f} {v_tmp:>12.4f}")

    # Comparação completa: Dense → CNN → LSTM
    print("\n" + "=" * 70)
    print("TABELA COMPARATIVA COMPLETA — Dense AE → CNN 1D → LSTM")
    print("=" * 70)

    modelos = {}
    paths = {
        "Dense AE": "/mnt/c/TCC_data/processed/results/passo_04/resultados_passo_04.json",
        "CNN 1D":   "/mnt/c/TCC_data/processed/results/passo_05/resultados_passo_05.json",
        "LSTM":     out_path,
    }
    for nome, path in paths.items():
        if os.path.exists(path):
            with open(path) as f:
                modelos[nome] = json.load(f)

    if len(modelos) == 3:
        for split in ("estacao", "temporal"):
            print(f"\n  Split: {split}")
            print(f"  {'Modelo':<12} {'AUC-PR':>10} {'AUC-ROC':>10} {'F1':>10}")
            print(f"  {'-'*42}")
            for nome in ("Dense AE", "CNN 1D", "LSTM"):
                m = modelos[nome][split]["test"]
                print(f"  {nome:<12} {m['auc_pr']:>10.4f} {m['auc_roc']:>10.4f} {m['f1_best']:>10.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()