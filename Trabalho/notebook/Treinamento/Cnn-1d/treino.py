"""
Passo 5 v3 — CNN 1D Autoencoder para detecção de anomalias.

Diferença vs Dense AE: filtros convolucionais capturam padrões LOCAIS
no sinal temporal (onset, forma de onda, envelope). Esses padrões são
mais universais entre estações do que a distribuição estatística global.

Hipótese: AUC-PR no split por estação deve subir significativamente.
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
RESULTS_DIR = "/mnt/c/TCC_data/processed/results/passo_05"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- Hiperparâmetros ----------
JANELA_NPTS = 800
EPOCHS = 200
BATCH_SIZE = 256
PATIENCE = 20
LR = 1e-3
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ---------- Modelo ----------
def build_cnn1d_ae(input_length):
    """
    Encoder:
      Conv1D(32, k=7) → MaxPool(2) → 400
      Conv1D(64, k=5) → MaxPool(2) → 200
      Conv1D(128,k=3) → MaxPool(2) → 100
      Conv1D(32, k=3)              → 100  ← BOTTLENECK

    Decoder (simétrico):
      Conv1D(128,k=3) → UpSample(2) → 200
      Conv1D(64, k=5) → UpSample(2) → 400
      Conv1D(32, k=7) → UpSample(2) → 800
      Conv1D(1,  k=7)               → 800  ← OUTPUT

    Total: ~120k parâmetros (4x menos que Dense AE).
    """
    # --- Encoder ---
    inputs = tf.keras.layers.Input(shape=(input_length, 1))

    # Bloco 1: 800 → 400
    x = tf.keras.layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Bloco 2: 400 → 200
    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Bloco 3: 200 → 100
    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Bottleneck: 100 × 32
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(x)
    encoded = tf.keras.layers.BatchNormalization()(x)

    # --- Decoder ---
    # Bloco 4: 100 → 200
    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(encoded)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling1D(size=2)(x)

    # Bloco 5: 200 → 400
    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling1D(size=2)(x)

    # Bloco 6: 400 → 800
    x = tf.keras.layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling1D(size=2)(x)

    # Saída: 800 × 1
    decoded = tf.keras.layers.Conv1D(1, kernel_size=7, padding="same", activation="tanh")(x)

    model = tf.keras.Model(inputs, decoded, name="cnn1d_ae")
    return model


# ---------- Métricas ----------
def calcular_scores(model, X):
    """Erro de reconstrução (MSE por janela) = score de anomalia."""
    X_3d = X.reshape(-1, JANELA_NPTS, 1)
    X_rec = model.predict(X_3d, batch_size=1024, verbose=0)
    X_rec = X_rec.reshape(-1, JANELA_NPTS)
    return np.mean((X - X_rec) ** 2, axis=1)


def avaliar(scores_normal, scores_anomalo):
    """Calcula AUC-PR, AUC-ROC, F1 ótimo."""
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
        fig.suptitle(f"CNN 1D AE — Split {nome_split}", fontsize=14)

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
        path = os.path.join(RESULTS_DIR, f"cnn1d_ae_{nome_split}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Figura salva: {path}")
    except Exception as e:
        print(f"  AVISO: não conseguiu salvar figuras: {e}")


# ---------- Treino de um split ----------
def treinar_split(nome_split, path_npz):
    print("=" * 70)
    print(f"TREINO — CNN 1D AE — Split {nome_split}")
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

    # reshape pra Conv1D: (N, 800) → (N, 800, 1)
    X_train_3d = X_train_normal.reshape(-1, JANELA_NPTS, 1)
    X_val_3d = X_val_normal.reshape(-1, JANELA_NPTS, 1)

    model = build_cnn1d_ae(JANELA_NPTS)
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

    model_path = os.path.join(RESULTS_DIR, f"cnn1d_ae_{nome_split}.keras")
    model.save(model_path)
    print(f"\n  Modelo salvo: {model_path}")

    salvar_figuras(history, scores_val_n, scores_val_a,
                   scores_test_n, scores_test_a, nome_split)

    return {
        "split": nome_split,
        "hiperparametros": {
            "filters": [32, 64, 128, 32],
            "kernels": [7, 5, 3, 3],
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
    print("PASSO 5 v3 — CNN 1D Autoencoder")
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

    out_path = os.path.join(RESULTS_DIR, "resultados_passo_05.json")
    with open(out_path, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"\n  Resultados salvos: {out_path}")

    print("\n" + "=" * 70)
    print("RESUMO FINAL — CNN 1D AE")
    print("=" * 70)
    print(f"{'Métrica':<20} {'Estação':>12} {'Temporal':>12}")
    print("-" * 44)
    for m in ("auc_pr", "auc_roc", "f1_best"):
        v_est = resultados["estacao"]["test"][m]
        v_tmp = resultados["temporal"]["test"][m]
        print(f"{m:<20} {v_est:>12.4f} {v_tmp:>12.4f}")

    # comparação com Dense AE
    dense_path = "/mnt/c/TCC_data/processed/results/passo_04/resultados_passo_04.json"
    if os.path.exists(dense_path):
        with open(dense_path) as f:
            dense = json.load(f)
        print("\n" + "-" * 44)
        print("COMPARAÇÃO: Dense AE → CNN 1D AE")
        print("-" * 44)
        for split in ("estacao", "temporal"):
            for m in ("auc_pr", "auc_roc"):
                d = dense[split]["test"][m]
                c = resultados[split]["test"][m]
                delta = c - d
                seta = "↑" if delta > 0 else "↓"
                print(f"  {split:10s} {m:10s}: {d:.4f} → {c:.4f}  ({seta} {abs(delta):.4f})")

    print("=" * 70)


if __name__ == "__main__":
    main()