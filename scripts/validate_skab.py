"""
Valida RF, Tiny CNN e Tiny TCN no dataset SKAB.
Salva em results/experiment_skab_validation.json.
"""
from __future__ import annotations
import json, sys, warnings, time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
import tensorflow as tf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.statistical_features import extract_statistical_features
from src.training.evaluate import vus_pr_score, pa_f1_score, evaluate_scores

WINDOW_STEP_S = 30.0

# ─────────────────────────────────────────────
# Carrega dataset
# ─────────────────────────────────────────────
data = np.load("/tmp/skab_processed/dataset.npz")
X_train = data["X_train"]  # (N, 60, 8)
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]

print(f"Train: {X_train.shape}, anomaly_rate={y_train.mean():.4f}")
print(f"Val  : {X_val.shape},   anomaly_rate={y_val.mean():.4f}")
print(f"Test : {X_test.shape},  anomaly_rate={y_test.mean():.4f}")

n_channels = X_train.shape[2]

# ─────────────────────────────────────────────
# Features para RF
# ─────────────────────────────────────────────
def extract_multi(X):
    """28 features × C canais + C(C,2) correlações de Pearson."""
    N, W, C = X.shape
    feat_per_ch = []
    for c in range(C):
        feat_per_ch.append(extract_statistical_features(X[:, :, c], sample_rate=1.0))
    F_chan = np.concatenate(feat_per_ch, axis=1)  # (N, 28*C)

    # Correlações de Pearson entre pares de canais
    corrs = []
    for i in range(C):
        for j in range(i+1, C):
            r = np.array([
                np.corrcoef(X[n, :, i], X[n, :, j])[0, 1]
                for n in range(N)
            ]).reshape(-1, 1)
            corrs.append(r)
    F_corr = np.concatenate(corrs, axis=1) if corrs else np.empty((N, 0))
    return np.concatenate([F_chan, F_corr], axis=1)

print("\nExtractando features (RF)...")
t0 = time.time()
Ftr = extract_multi(X_train)
Fva = extract_multi(X_val)
Fte = extract_multi(X_test)
print(f"Features shape: {Ftr.shape}, {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────
print("\n=== Random Forest ===")
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(Ftr, y_train)
proba_test = rf.predict_proba(Fte)[:, 1]

def fp_per_hour(y_true, y_pred, step_s=WINDOW_STEP_S):
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    hours = len(y_true) * step_s / 3600.0
    return round(fp / hours, 3) if hours > 0 else 0.0

ev = evaluate_scores(y_test, proba_test, threshold=0.5)
auc_pr = float(ev["auc_pr"])
vus_pr = float(ev["vus_pr"])
pa_f1  = float(ev["pa_f1"])
pred_test = (proba_test >= 0.5).astype(int)
fph    = fp_per_hour(y_test, pred_test)
rf_res = {"auc_pr": round(auc_pr,4), "vus_pr": round(vus_pr,4),
          "pa_f1": round(pa_f1,4), "fp_per_hour": fph,
          "n_params": int(rf.n_estimators)}
print(f"  AUC-PR={auc_pr:.4f}  VUS-PR={vus_pr:.4f}  PA-F1={pa_f1:.4f}  FP/h={fph}")

# ─────────────────────────────────────────────
# Constroção de modelos neurais
# ─────────────────────────────────────────────
W, C = X_train.shape[1], X_train.shape[2]

def build_tiny_cnn(W, C, n_blocks=2, base_filters=8, kernel_first=5):
    inp = tf.keras.Input((W, C))
    x = inp
    for i in range(n_blocks):
        x = tf.keras.layers.Conv1D(base_filters*(2**i), kernel_first if i==0 else 3,
                                   padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, x)

def build_tiny_tcn(W, C, filters=8, kernel_size=3, n_blocks=2):
    inp = tf.keras.Input((W, C))
    x = tf.keras.layers.Conv1D(filters, 1, padding="same")(inp)
    for b in range(n_blocks):
        dil = 2**b
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

def train_model(model, X_tr, y_tr, X_va, y_va, epochs=30, batch=64):
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(curve="PR", name="auc_pr")])
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_auc_pr", patience=5,
                                          mode="max", restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=epochs, batch_size=batch, callbacks=[cb], verbose=0)
    return model

def eval_model(model, X_te, y_te):
    proba = model.predict(X_te, verbose=0).ravel()
    ev    = evaluate_scores(y_te, proba, threshold=0.5)
    auc   = float(ev["auc_pr"])
    vus   = float(ev["vus_pr"])
    paf1  = float(ev["pa_f1"])
    fph   = fp_per_hour(y_te, (proba >= 0.5).astype(int))
    return auc, vus, paf1, fph, int(model.count_params())

# ─────────────────────────────────────────────
# Tiny CNN
# ─────────────────────────────────────────────
print("\n=== Tiny CNN ===")
cnn = build_tiny_cnn(W, C, n_blocks=2, base_filters=8, kernel_first=5)
cnn.summary(print_fn=lambda x: None)
train_model(cnn, X_train, y_train, X_val, y_val)
auc, vus, paf1, fph, n = eval_model(cnn, X_test, y_test)
cnn_res = {"auc_pr": round(auc,4), "vus_pr": round(vus,4),
           "pa_f1": round(paf1,4), "fp_per_hour": round(fph,3), "n_params": n}
print(f"  AUC-PR={auc:.4f}  VUS-PR={vus:.4f}  PA-F1={paf1:.4f}  FP/h={fph:.3f}  Params={n}")

# ─────────────────────────────────────────────
# Tiny TCN
# ─────────────────────────────────────────────
print("\n=== Tiny TCN ===")
tcn = build_tiny_tcn(W, C, filters=8, kernel_size=3, n_blocks=2)
tcn.summary(print_fn=lambda x: None)
train_model(tcn, X_train, y_train, X_val, y_val)
auc, vus, paf1, fph, n = eval_model(tcn, X_test, y_test)
tcn_res = {"auc_pr": round(auc,4), "vus_pr": round(vus,4),
           "pa_f1": round(paf1,4), "fp_per_hour": round(fph,3), "n_params": n}
print(f"  AUC-PR={auc:.4f}  VUS-PR={vus:.4f}  PA-F1={paf1:.4f}  FP/h={fph:.3f}  Params={n}")

# ─────────────────────────────────────────────
# Salva resultados
# ─────────────────────────────────────────────
results = {
    "random_forest_skab": rf_res,
    "tiny_cnn_skab": cnn_res,
    "tiny_tcn_skab": tcn_res,
    "dataset_skab": {
        "window_size": int(W),
        "n_channels": int(C),
        "sampling_rate": 1.0,
        "n_train": int(len(X_train)),
        "n_val":   int(len(X_val)),
        "n_test":  int(len(X_test)),
        "anomaly_rate_train": round(float(y_train.mean()), 4),
        "anomaly_rate_val":   round(float(y_val.mean()), 4),
        "anomaly_rate_test":  round(float(y_test.mean()), 4),
    }
}

out = Path("/home/vish8/tcc_atual/TCC/results/experiment_skab_validation.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nResultados salvos em {out}")
print(json.dumps(results, indent=2))
