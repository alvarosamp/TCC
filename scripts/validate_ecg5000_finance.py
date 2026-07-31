"""
Valida RF, Tiny CNN e Tiny TCN nos datasets ECG5000 (biomedical) e
Finance S&P 500 (financial stress).
Salva em results/experiment_ecg5000_finance_validation.json.
"""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.statistical_features import extract_statistical_features
from src.training.evaluate import evaluate_scores

WINDOW_STEP_S = {"ecg5000": 1.0/360.0, "finance": 5.0 * 86400.0}


def extract_multi(X, sample_rate=1.0):
    """28 features × C canais + C(C,2) correlações de Pearson."""
    N, W, C = X.shape
    feat_per_ch = [extract_statistical_features(X[:, :, c], sample_rate=sample_rate)
                   for c in range(C)]
    F_chan = np.concatenate(feat_per_ch, axis=1)
    corrs = []
    for i in range(C):
        for j in range(i+1, C):
            r = np.array([np.corrcoef(X[n, :, i], X[n, :, j])[0, 1]
                          for n in range(N)]).reshape(-1, 1)
            corrs.append(r)
    F_corr = np.concatenate(corrs, axis=1) if corrs else np.empty((N, 0))
    return np.concatenate([F_chan, F_corr], axis=1)


def fp_per_hour(y_true, y_pred, step_s):
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    hours = len(y_true) * step_s / 3600.0
    return round(fp / hours, 3) if hours > 0 else 0.0


def build_tiny_cnn(W, C, n_blocks=2, base_filters=8, kernel_first=5):
    inp = tf.keras.Input((W, C))
    x = inp
    for i in range(n_blocks):
        f = base_filters * (2 ** i)
        k = kernel_first if i == 0 else 3
        x = tf.keras.layers.Conv1D(f, k, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, x)


def build_tiny_tcn(W, C, filters=8, kernel_size=3, n_blocks=2):
    inp = tf.keras.Input((W, C))
    x = tf.keras.layers.Conv1D(filters, 1, padding="same")(inp)
    for b in range(n_blocks):
        dil = 2 ** b
        res = x
        x = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal",
                                   dilation_rate=dil, activation="relu")(x)
        x = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal",
                                   dilation_rate=dil, activation="relu")(x)
        if res.shape[-1] != filters:
            res = tf.keras.layers.Conv1D(filters, 1, padding="same")(res)
        x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, x)


def train_model(model, X_tr, y_tr, X_va, y_va, epochs=50, batch=64):
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(curve="PR", name="auc_pr")])
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_auc_pr", patience=7,
                                          mode="max", restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=epochs, batch_size=batch, callbacks=[cb], verbose=0)
    return model


def eval_model(model, X_te, y_te, step_s):
    proba = model.predict(X_te, verbose=0).ravel()
    ev = evaluate_scores(y_te, proba, threshold=0.5)
    fph = fp_per_hour(y_te, (proba >= 0.5).astype(int), step_s)
    return {
        "auc_pr":     round(float(ev["auc_pr"]), 4),
        "vus_pr":     round(float(ev["vus_pr"]), 4),
        "pa_f1":      round(float(ev["pa_f1"]), 4),
        "fp_per_hour": fph,
        "n_params":   int(model.count_params()),
    }


def run_domain(name, npz_path, step_s, sr):
    print(f"\n{'='*60}\nDomínio: {name}\n{'='*60}")
    data = np.load(npz_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    W, C = X_train.shape[1], X_train.shape[2]
    print(f"Train: {X_train.shape}  anom={y_train.mean():.4f}")
    print(f"Val:   {X_val.shape}    anom={y_val.mean():.4f}")
    print(f"Test:  {X_test.shape}   anom={y_test.mean():.4f}")

    # ── Random Forest ──────────────────────────────────────────────
    print("\n[RF] Extraindo features...")
    t0 = time.time()
    Ftr = extract_multi(X_train, sr); Fva = extract_multi(X_val, sr); Fte = extract_multi(X_test, sr)
    print(f"  Feature shape: {Ftr.shape}  ({time.time()-t0:.1f}s)")
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(Ftr, y_train)
    proba = rf.predict_proba(Fte)[:, 1]
    ev = evaluate_scores(y_test, proba, threshold=0.5)
    fph = fp_per_hour(y_test, (proba >= 0.5).astype(int), step_s)
    rf_res = {"auc_pr": round(float(ev["auc_pr"]),4), "vus_pr": round(float(ev["vus_pr"]),4),
              "pa_f1": round(float(ev["pa_f1"]),4), "fp_per_hour": fph, "n_params": int(rf.n_estimators)}
    print(f"  RF: AUC-PR={rf_res['auc_pr']}  VUS-PR={rf_res['vus_pr']}  PA-F1={rf_res['pa_f1']}  FP/h={fph}")

    # ── Tiny CNN ───────────────────────────────────────────────────
    print("\n[CNN] Treinando...")
    cnn = build_tiny_cnn(W, C)
    train_model(cnn, X_train, y_train, X_val, y_val)
    cnn_res = eval_model(cnn, X_test, y_test, step_s)
    print(f"  CNN: {cnn_res}")

    # ── Tiny TCN ───────────────────────────────────────────────────
    print("\n[TCN] Treinando...")
    tcn = build_tiny_tcn(W, C)
    train_model(tcn, X_train, y_train, X_val, y_val)
    tcn_res = eval_model(tcn, X_test, y_test, step_s)
    print(f"  TCN: {tcn_res}")

    dataset_stats = {
        "window_size": int(W), "n_channels": int(C), "sampling_rate": float(sr),
        "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        "anomaly_rate_train": round(float(y_train.mean()),4),
        "anomaly_rate_val":   round(float(y_val.mean()),4),
        "anomaly_rate_test":  round(float(y_test.mean()),4),
    }
    return {f"random_forest_{name}": rf_res,
            f"tiny_cnn_{name}": cnn_res,
            f"tiny_tcn_{name}": tcn_res,
            f"dataset_{name}": dataset_stats}


# ── ECG5000 ────────────────────────────────────────────────────────
# Prepara ECG5000
from src.data.adapters.ecg5000 import build_ecg5000_dataset
ecg_stats = build_ecg5000_dataset(
    raw_dir=Path("/tmp/ecg5000_raw"),
    out_dir=Path("/tmp/ecg5000_processed"),
)
print("ECG5000 preparado:", ecg_stats)

results = {}
results.update(run_domain(
    "ecg5000",
    "/tmp/ecg5000_processed/dataset.npz",
    step_s=1.0/360.0,
    sr=360.0,
))

# ── Finance ─────────────────────────────────────────────────────────
results.update(run_domain(
    "finance",
    "/tmp/finance_processed/dataset.npz",
    step_s=5.0 * 86400.0,  # 5 dias em segundos → FP/h sem sentido mas mantém contrato
    sr=1.0 / 86400.0,
))

out = Path("/home/vish8/tcc_atual/TCC/results/experiment_ecg5000_finance_validation.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\n✓ Resultados salvos em {out}")
print(json.dumps(results, indent=2))
