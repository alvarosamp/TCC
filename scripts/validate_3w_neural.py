"""Trains Tiny CNN and Tiny TCN on 3W dataset (7 channels). RF result already saved."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json, time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from sklearn.metrics import average_precision_score, precision_recall_curve
from pathlib import Path

def get_anomaly_segments(y):
    segs, in_seg = [], False
    for i, v in enumerate(y):
        if v == 1 and not in_seg:
            start = i; in_seg = True
        elif v == 0 and in_seg:
            segs.append((start, i)); in_seg = False
    if in_seg: segs.append((start, len(y)))
    return segs

def pa_f1_score(y_true, scores, thr):
    y_pred = (scores >= thr).astype(int)
    y_adj  = y_pred.copy()
    for s, e in get_anomaly_segments(y_true):
        if y_pred[s:e].any(): y_adj[s:e] = 1
    tp = int((y_adj & y_true.astype(bool)).sum())
    fp = int((y_adj & ~y_true.astype(bool)).sum())
    fn = int((~y_adj.astype(bool) & y_true.astype(bool)).sum())
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    return round(2*prec*rec/(prec+rec) if (prec+rec) else 0.0, 4)

def vus_pr_score(y_true, scores, n_buffers=10):
    b_max = max(1, int(len(y_true) * 0.005))
    aucs = []
    for b in np.linspace(0, b_max, n_buffers, dtype=int):
        y_buf = y_true.copy().astype(float)
        if b > 0:
            y_buf = np.minimum(np.convolve(y_buf, np.ones(2*b+1), mode='same'), 1.0)
        aucs.append(average_precision_score(y_buf, scores))
    return round(float(np.mean(aucs)), 4)

def fp_per_hour(y_true, y_pred, window_size=60, sr=1.0):
    secs = window_size / sr
    normal_h = (y_true == 0).sum() * secs / 3600
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return round(fp / normal_h if normal_h > 0 else 0.0, 3)

def best_threshold_f1(y_val, scores):
    p, r, thr = precision_recall_curve(y_val, scores)
    f1s = np.where((p + r) > 0, 2*p*r/(p+r), 0)
    return float(thr[np.argmax(f1s[:-1])])

def evaluate(name, scores_val, scores_test, y_val_arr, y_test_arr, n_params=None):
    thr = best_threshold_f1(y_val_arr, scores_val)
    y_pred = (scores_test >= thr).astype(int)
    auc  = round(float(average_precision_score(y_test_arr, scores_test)), 4)
    pa   = pa_f1_score(y_test_arr, scores_test, thr)
    vus  = vus_pr_score(y_test_arr, scores_test)
    fph  = fp_per_hour(y_test_arr, y_pred)
    r = {"auc_pr": auc, "pa_f1": pa, "vus_pr": vus, "fp_per_hour": fph,
         "threshold": round(thr, 4)}
    if n_params is not None:
        r["n_params"] = n_params
    print(f"  [{name}] auc_pr={auc}  vus_pr={vus}  pa_f1={pa}  fp/h={fph}", flush=True)
    return r

print("Carregando dataset 3W...", flush=True)
data = np.load("/mnt/d/PipelineGenerico/data/dataset.npz")
X_train, y_train = data["X_train"].astype(np.float32), data["y_train"].astype(int)
X_val,   y_val   = data["X_val"].astype(np.float32),   data["y_val"].astype(int)
X_test,  y_test  = data["X_test"].astype(np.float32),  data["y_test"].astype(int)
print(f"  train={X_train.shape} val={X_val.shape} test={X_test.shape}", flush=True)

rng = np.random.RandomState(42)
def subsample(X, y, n):
    if len(X) <= n: return X, y
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]

from src.training.neural_models import build_tiny_cnn, build_tiny_tcn

out_path = Path("results/experiment_3w_validation.json")
with open(out_path) as f:
    results = json.load(f)

model_cfgs = [
    ("tiny_cnn_3w", build_tiny_cnn,
     {"n_blocks": 4, "base_filters": 16, "kernel_first": 7, "kernel_other": 5,
      "dense_units": 16, "dropout": 0.1, "spatial_dropout": 0.0}),
    ("tiny_tcn_3w", build_tiny_tcn,
     {"n_blocks": 4, "filters": 16, "kernel_size": 7,
      "dilation_base": 2, "dense_units": 16, "dropout": 0.1, "spatial_dropout": 0.0}),
]

for mname, builder, hparams in model_cfgs:
    print(f"\n{mname}  input=(N, 60, 7)...", flush=True)
    model = builder(60, hparams, n_channels=7)
    n_p = int(model.count_params())
    print(f"  params={n_p}", flush=True)

    Xtr2, ytr2 = subsample(X_train, y_train, 200_000)
    Xv2,  yv2  = X_val[:50_000], y_val[:50_000]

    cb = [tf.keras.callbacks.EarlyStopping(
        patience=4, restore_best_weights=True,
        monitor='val_auc_pr', mode='max', verbose=1
    )]
    model.fit(Xtr2, ytr2, validation_data=(Xv2, yv2),
              epochs=20, batch_size=512, callbacks=cb, verbose=0)

    sv = model.predict(X_val,  batch_size=512, verbose=0).flatten()
    st = model.predict(X_test, batch_size=512, verbose=0).flatten()
    results[mname] = evaluate(mname.upper(), sv, st, y_val, y_test, n_params=n_p)
    del model

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Salvo em {out_path}", flush=True)

results["dataset_3w"] = {
    "window_size": 60, "n_channels": 7, "sampling_rate": 1.0,
    "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
    "anomaly_rate_train": round(float(y_train.mean()), 4),
    "anomaly_rate_val":   round(float(y_val.mean()),   4),
    "anomaly_rate_test":  round(float(y_test.mean()),  4),
}
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("\nResultados finais 3W:")
summary = {k: {m: v.get(m) for m in ["auc_pr","vus_pr","pa_f1","fp_per_hour","n_params"]}
           for k, v in results.items() if isinstance(v, dict) and "auc_pr" in v}
print(json.dumps(summary, indent=2))
