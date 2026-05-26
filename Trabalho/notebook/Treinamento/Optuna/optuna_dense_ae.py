"""
Optuna — Dense Autoencoder.
Busca hiperparâmetros ótimos com pruning inteligente.
Otimiza no split ESTAÇÃO (o mais difícil), depois retreina nos dois splits.

Espaço de busca:
  - latent_dim: 16, 32, 64, 128
  - n_camadas encoder: 1–4
  - unidades por camada: 64–512
  - dropout: 0.0–0.3
  - learning_rate: 1e-4 – 1e-2
  - batch_size: 128, 256, 512
"""

import json
import os
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("GLOG_minloglevel", "2")

# XLA pode usar Triton para GEMM na GPU e emitir muitos avisos do tipo
# "ptxas warning: Registers are spilled...". Isso é aviso de performance
# (não é falha), mas polui o log e pode dar a impressão de erro.
#
# Por padrão, desligamos o Triton GEMM. Para reativar (se quiser performance), rode com:
#   TCC_DISABLE_TRITON_GEMM=0
if os.environ.get("TCC_DISABLE_TRITON_GEMM", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}:
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_enable_triton_gemm=false" not in xla_flags:
        os.environ["XLA_FLAGS"] = (xla_flags + " --xla_gpu_enable_triton_gemm=false").strip()
warnings.filterwarnings("ignore")

import numpy as np
import optuna
from optuna.pruners import MedianPruner
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _is_notebook_like() -> bool:
    """Heurística simples para detectar execução via Jupyter/VS Code Notebook.

    Nesses ambientes, a barra de progresso do Optuna/tqdm pode virar widget/webview
    e às vezes o VS Code pede "atualize a página" e a execução trava.
    """

    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _optuna_show_progress_default() -> bool:
    override = os.environ.get("OPTUNA_SHOW_PROGRESS")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "y", "on"}
    return not _is_notebook_like()

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"GPU: {gpus[0].name}")

# ---------- Paths ----------
TCC_BASE = os.environ.get("TCC_BASE", "/mnt/c/TCC_data")
DATA_DIR = os.environ.get("TCC_PROCESSED_DIR", os.path.join(TCC_BASE, "processed"))
RESULTS_BASE = os.environ.get("TCC_RESULTS_BASE", os.path.join(DATA_DIR, "results"))
RESULTS_DIR = os.path.join(RESULTS_BASE, "optuna_dense_ae")
os.makedirs(RESULTS_DIR, exist_ok=True)

STUDY_NAME = os.environ.get("OPTUNA_STUDY_NAME", "optuna_dense_ae")
STORAGE_URL = os.environ.get(
    "OPTUNA_STORAGE_URL",
    f"sqlite:///{os.path.join(RESULTS_DIR, 'study.db')}",
)

JANELA_NPTS = 800
N_TRIALS = 10
EPOCHS_OPT = 40
EPOCHS_FINAL = 100
PATIENCE = 12
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------- Dados ----------
print("Carregando dados...")
data_est = np.load(os.path.join(DATA_DIR, "dataset_v3_split_estacao.npz"))
X_train_est = data_est["X_train"][data_est["y_train"] == 0]
X_val_est_n = data_est["X_val"][data_est["y_val"] == 0]
X_val_est_a = data_est["X_val"][data_est["y_val"] == 1]
X_test_est_n = data_est["X_test"][data_est["y_test"] == 0]
X_test_est_a = data_est["X_test"][data_est["y_test"] == 1]

data_tmp = np.load(os.path.join(DATA_DIR, "dataset_v3_split_temporal.npz"))
X_train_tmp = data_tmp["X_train"][data_tmp["y_train"] == 0]
X_val_tmp_n = data_tmp["X_val"][data_tmp["y_val"] == 0]
X_val_tmp_a = data_tmp["X_val"][data_tmp["y_val"] == 1]
X_test_tmp_n = data_tmp["X_test"][data_tmp["y_test"] == 0]
X_test_tmp_a = data_tmp["X_test"][data_tmp["y_test"] == 1]
print(f"  Train estação: {X_train_est.shape} | Train temporal: {X_train_tmp.shape}")


def calcular_auc_pr(model, X_n, X_a):
    r_n = model.predict(X_n, batch_size=1024, verbose=0)
    r_a = model.predict(X_a, batch_size=1024, verbose=0)
    s_n = np.mean((X_n - r_n) ** 2, axis=1)
    s_a = np.mean((X_a - r_a) ** 2, axis=1)
    y = np.concatenate([np.zeros(len(s_n)), np.ones(len(s_a))])
    s = np.concatenate([s_n, s_a])
    return average_precision_score(y, s)


def avaliar_completo(model, X_n, X_a):
    r_n = model.predict(X_n, batch_size=1024, verbose=0)
    r_a = model.predict(X_a, batch_size=1024, verbose=0)
    s_n = np.mean((X_n - r_n) ** 2, axis=1)
    s_a = np.mean((X_a - r_a) ** 2, axis=1)
    y = np.concatenate([np.zeros(len(s_n)), np.ones(len(s_a))])
    s = np.concatenate([s_n, s_a])
    auc_pr = average_precision_score(y, s)
    auc_roc = roc_auc_score(y, s)
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best = np.argmax(f1)
    return {
        "auc_pr": round(float(auc_pr), 4),
        "auc_roc": round(float(auc_roc), 4),
        "f1_best": round(float(f1[best]), 4),
        "threshold": round(float(thr[min(best, len(thr)-1)]), 6),
    }


def build_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    latent = trial.suggest_categorical("latent_dim", [16, 32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    units = sorted(
        [trial.suggest_int(f"units_{i}", 64, 512, step=64) for i in range(n_layers)],
        reverse=True,
    )

    inp = tf.keras.layers.Input(shape=(JANELA_NPTS,))
    x = inp
    for u in units:
        x = tf.keras.layers.Dense(u, activation="relu")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(latent, activation="relu")(x)
    for u in reversed(units):
        x = tf.keras.layers.Dense(u, activation="relu")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(JANELA_NPTS, activation="tanh")(x)

    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


class PruneCallback(tf.keras.callbacks.Callback):
    def __init__(self, trial):
        self.trial = trial
    def on_epoch_end(self, epoch, logs=None):
        self.trial.report(logs.get("val_loss", 999), epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def objective(trial):
    tf.keras.backend.clear_session()
    bs = trial.suggest_categorical("batch_size", [128, 256, 512])
    model = build_model(trial)
    try:
        model.fit(X_train_est, X_train_est,
                  validation_data=(X_val_est_n, X_val_est_n),
                  epochs=EPOCHS_OPT, batch_size=bs,
                  callbacks=[PruneCallback(trial)], verbose=0)
    except optuna.TrialPruned:
        raise
    return calcular_auc_pr(model, X_val_est_n, X_val_est_a)


class FakeTrial:
    def __init__(self, p):
        self.p = p
    def suggest_int(self, name, *a, **k):
        return self.p[name]
    def suggest_float(self, name, *a, **k):
        return self.p[name]
    def suggest_categorical(self, name, *a, **k):
        return self.p[name]


def retreinar(params, X_tr, X_vn, X_va, X_tn, X_ta, nome):
    print(f"\n  Retreinando {nome}...")
    tf.keras.backend.clear_session()
    model = build_model(FakeTrial(params))
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                          restore_best_weights=True, verbose=1)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                patience=5, min_lr=1e-6, verbose=1)
    t0 = time.time()
    h = model.fit(X_tr, X_tr, validation_data=(X_vn, X_vn),
                  epochs=EPOCHS_FINAL, batch_size=params["batch_size"],
                  callbacks=[es, rlr], verbose=1)
    tempo = time.time() - t0
    r_val = avaliar_completo(model, X_vn, X_va)
    r_test = avaliar_completo(model, X_tn, X_ta)
    model.save(os.path.join(RESULTS_DIR, f"dense_ae_opt_{nome}.keras"))
    print(f"  {nome} TEST — AUC-PR={r_test['auc_pr']:.4f} AUC-ROC={r_test['auc_roc']:.4f} F1={r_test['f1_best']:.4f}")
    return {"val": r_val, "test": r_test, "tempo_min": round(tempo/60, 2),
            "epochs": len(h.history["loss"])}


def main():
    print("=" * 70)
    print(f"OPTUNA — Dense AE | {N_TRIALS} trials")
    print("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        load_if_exists=True,
    )
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=_optuna_show_progress_default())
    tempo_busca = (time.time() - t0) / 60

    print(f"\n  Busca concluída em {tempo_busca:.1f} min")
    print(f"  Melhor trial #{study.best_trial.number}: AUC-PR={study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    best = study.best_params
    res = {"best_params": best, "best_auc_pr_val": round(study.best_value, 4),
           "tempo_busca_min": round(tempo_busca, 1)}

    res["estacao"] = retreinar(best, X_train_est, X_val_est_n, X_val_est_a,
                               X_test_est_n, X_test_est_a, "estacao")
    res["temporal"] = retreinar(best, X_train_tmp, X_val_tmp_n, X_val_tmp_a,
                                X_test_tmp_n, X_test_tmp_a, "temporal")

    with open(os.path.join(RESULTS_DIR, "resultados_optuna_dense_ae.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTADO FINAL — Dense AE Optuna")
    print("=" * 70)
    for s in ("estacao", "temporal"):
        r = res[s]["test"]
        print(f"  {s:10s} AUC-PR={r['auc_pr']:.4f}  AUC-ROC={r['auc_roc']:.4f}  F1={r['f1_best']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()