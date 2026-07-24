"""
Validação completa dos experimentos para os 3 artigos.
Computa: AUC-PR, PA-F1, VUS-PR, Event-F1, FP/h para todos os modelos.
Salva resultados em results/experiment_validation.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    roc_auc_score, f1_score
)

# ── helpers ────────────────────────────────────────────────────────────────────

def get_anomaly_segments(y):
    segs = []
    in_seg = False
    for i, v in enumerate(y):
        if v == 1 and not in_seg:
            start = i; in_seg = True
        elif v == 0 and in_seg:
            segs.append((start, i)); in_seg = False
    if in_seg:
        segs.append((start, len(y)))
    return segs

def point_adjust(y_true, y_pred_bin):
    y_adj = y_pred_bin.copy()
    for s, e in get_anomaly_segments(y_true):
        if y_pred_bin[s:e].any():
            y_adj[s:e] = 1
    return y_adj

def pa_f1(y_true, scores, thr):
    y_pred = (scores >= thr).astype(int)
    y_adj  = point_adjust(y_true, y_pred)
    if y_adj.sum() == 0:
        return {"pa_f1": 0.0, "pa_precision": 0.0, "pa_recall": 0.0}
    tp = int((y_adj & y_true).sum())
    fp = int((y_adj & ~y_true.astype(bool)).sum())
    fn = int((~y_adj.astype(bool) & y_true).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"pa_f1": round(f1, 4), "pa_precision": round(prec, 4), "pa_recall": round(rec, 4)}

def event_f1(y_true, scores, thr, min_overlap=0.1):
    y_pred = (scores >= thr).astype(int)
    true_segs = get_anomaly_segments(y_true)
    pred_segs = get_anomaly_segments(y_pred)
    tp = sum(
        1 for s, e in true_segs
        if y_pred[s:e].mean() >= min_overlap
    )
    fp = len(pred_segs) - sum(
        1 for s, e in pred_segs
        if y_true[s:e].any()
    )
    fp = max(fp, 0)
    fn = len(true_segs) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "event_f1": round(f1, 4),
        "event_precision": round(prec, 4),
        "event_recall": round(rec, 4),
        "n_true_segments": len(true_segs),
        "event_tp": tp, "event_fp": fp, "event_fn": fn
    }

def vus_pr(y_true, scores, b_max=None, n_buffers=10):
    if b_max is None:
        b_max = int(len(y_true) * 0.01)  # 1% comprimento
    buffers = np.linspace(0, b_max, n_buffers, dtype=int)
    aucs = []
    for b in buffers:
        y_buf = y_true.copy().astype(float)
        if b > 0:
            kernel = np.ones(2 * b + 1)
            y_buf = np.minimum(np.convolve(y_buf, kernel, mode='same'), 1.0)
        aucs.append(average_precision_score(y_buf, scores))
    return {
        "vus_pr": round(float(np.mean(aucs)), 4),
        "vus_pr_per_buffer": [round(a, 4) for a in aucs],
        "buffer_sizes": [int(b) for b in buffers]
    }

def fp_per_hour(y_true, y_pred_bin, window_size=800, sampling_rate=40.0):
    samples_per_window = window_size
    # cada janela = window_size amostras / sampling_rate segundos
    secs_per_window = samples_per_window / sampling_rate
    total_normal_windows = int((y_true == 0).sum())
    total_normal_hours = total_normal_windows * secs_per_window / 3600.0
    fp_count = int(((y_pred_bin == 1) & (y_true == 0)).sum())
    fph = fp_count / total_normal_hours if total_normal_hours > 0 else 0.0
    return round(fph, 3)

def best_threshold_f1(y_true, scores):
    prec, rec, thrs = precision_recall_curve(y_true, scores)
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    idx = np.argmax(f1s[:-1])
    return float(thrs[idx])

# ── carrega dataset ────────────────────────────────────────────────────────────

DATASET = "/mnt/d/PipelineGenerico/data/dataset_seismic_edge_v1_split_evento.npz"
WINDOW_SIZE = 800
SAMPLING_RATE = 40.0

from src.features.statistical_features import extract_statistical_features

print("Carregando dataset...", flush=True)
t0 = time.time()
data = np.load(DATASET)
X_val,  y_val  = data["X_val"].astype(np.float32),  data["y_val"].astype(int)
X_test, y_test = data["X_test"].astype(np.float32), data["y_test"].astype(int)
X_train, y_train = data["X_train"].astype(np.float32), data["y_train"].astype(int)
print(f"  Dataset carregado em {time.time()-t0:.1f}s", flush=True)
print(f"  Test: {X_test.shape}, anomaly={y_test.mean():.1%}", flush=True)

results = {}

# ── avalia modelo ────────────────────────────────────────────────────────────

def evaluate_model(name, scores_val, scores_test):
    print(f"\n  [{name}]", flush=True)
    thr = best_threshold_f1(y_val, scores_val)
    y_pred_test = (scores_test >= thr).astype(int)

    auc_pr = float(average_precision_score(y_test, scores_test))
    try:
        auc_roc = float(roc_auc_score(y_test, scores_test))
    except Exception:
        auc_roc = None

    fph = fp_per_hour(y_test, y_pred_test, WINDOW_SIZE, SAMPLING_RATE)
    _pa  = pa_f1(y_test, scores_test, thr)
    _ev  = event_f1(y_test, scores_test, thr)
    _vus = vus_pr(y_test, scores_test)

    r = {
        "threshold": round(thr, 4),
        "auc_pr":    round(auc_pr, 4),
        "auc_roc":   round(auc_roc, 4) if auc_roc else None,
        "fp_per_hour": fph,
        **_pa, **_ev, **_vus
    }
    for k, v in r.items():
        if k not in ("vus_pr_per_buffer", "buffer_sizes"):
            print(f"    {k}: {v}", flush=True)
    return r

# ── Features para modelos clássicos ───────────────────────────────────────────
print("\nExtraindo features estatísticas (val+test)...", flush=True)
F_val  = extract_statistical_features(X_val,  sample_rate=SAMPLING_RATE)
F_test = extract_statistical_features(X_test, sample_rate=SAMPLING_RATE)
print(f"  Features shape: val={F_val.shape}, test={F_test.shape}", flush=True)

# ── Logistic Regression ────────────────────────────────────────────────────────
print("\nLogistic Regression...", flush=True)
lr = joblib.load("artefacts/models/logistic_regression.joblib")
sc_val_lr  = lr.predict_proba(F_val)[:, 1]
sc_test_lr = lr.predict_proba(F_test)[:, 1]
results["logistic_regression"] = evaluate_model("Logistic Regression", sc_val_lr, sc_test_lr)
clf_lr = lr[-1] if hasattr(lr, '__getitem__') else lr
results["logistic_regression"]["n_params"] = int(clf_lr.coef_.size + clf_lr.intercept_.size)

# ── Random Forest ──────────────────────────────────────────────────────────────
print("\nRandom Forest...", flush=True)
rf = joblib.load("artefacts/models/random_forest.joblib")
sc_val_rf  = rf.predict_proba(F_val)[:, 1]
sc_test_rf = rf.predict_proba(F_test)[:, 1]
results["random_forest"] = evaluate_model("Random Forest", sc_val_rf, sc_test_rf)
results["random_forest"]["n_params"] = int(rf.n_estimators * rf.estimators_[0].tree_.node_count)

# ── Extra Trees ────────────────────────────────────────────────────────────────
print("\nExtra Trees...", flush=True)
et = joblib.load("artefacts/models/extra_trees.joblib")
sc_val_et  = et.predict_proba(F_val)[:, 1]
sc_test_et = et.predict_proba(F_test)[:, 1]
results["extra_trees"] = evaluate_model("Extra Trees", sc_val_et, sc_test_et)
results["extra_trees"]["n_params"] = int(et.n_estimators * et.estimators_[0].tree_.node_count)
del et  # libera 921 MB

# ── Neural models ──────────────────────────────────────────────────────────────
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

for model_name, model_file in [("tiny_cnn", "artefacts/models/tiny_cnn.keras"),
                                ("tiny_tcn", "artefacts/models/tiny_tcn.keras")]:
    print(f"\n{model_name}...", flush=True)
    m = tf.keras.models.load_model(model_file)
    n_params = int(m.count_params())
    # adiciona dimensão canal (N, 800) → (N, 800, 1)
    Xv3 = X_val[:, :, np.newaxis]
    Xt3 = X_test[:, :, np.newaxis]
    sc_val_n  = m.predict(Xv3,  batch_size=512, verbose=0).flatten()
    sc_test_n = m.predict(Xt3, batch_size=512, verbose=0).flatten()
    results[model_name] = evaluate_model(model_name.upper(), sc_val_n, sc_test_n)
    results[model_name]["n_params"] = n_params
    del m

# ── TFLite sizes ──────────────────────────────────────────────────────────────
tflite_sizes = {}
for f in Path("artefacts/edge").glob("*.tflite"):
    tflite_sizes[f.name] = os.path.getsize(f)
results["tflite_sizes_bytes"] = tflite_sizes
results["tflite_sizes_kb"] = {k: round(v/1024, 1) for k, v in tflite_sizes.items()}

# ── dataset stats ─────────────────────────────────────────────────────────────
results["dataset"] = {
    "name": "seismic_edge_v1_split_evento",
    "window_size": WINDOW_SIZE,
    "sampling_rate": SAMPLING_RATE,
    "n_train": int(len(y_train)),
    "n_val": int(len(y_val)),
    "n_test": int(len(y_test)),
    "anomaly_rate_train": round(float(y_train.mean()), 4),
    "anomaly_rate_val":   round(float(y_val.mean()),   4),
    "anomaly_rate_test":  round(float(y_test.mean()),  4),
    "n_anomaly_segments_test": len(get_anomaly_segments(y_test)),
}

# ── salva resultados ──────────────────────────────────────────────────────────
out = Path("results/experiment_validation.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n\nResultados salvos em {out}")
print(json.dumps({k: {
    "auc_pr": v.get("auc_pr"),
    "pa_f1":  v.get("pa_f1"),
    "vus_pr": v.get("vus_pr"),
    "event_f1": v.get("event_f1"),
    "fp_per_hour": v.get("fp_per_hour"),
    "n_params": v.get("n_params"),
} for k, v in results.items() if isinstance(v, dict) and "auc_pr" in v}, indent=2))
