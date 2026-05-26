"""
Optuna — CNN 1D Autoencoder.
Busca hiperparâmetros ótimos com pruning inteligente.

Espaço de busca:
  - n_blocos conv: 2, 3, 4
  - filtros por bloco: 16–256
  - kernel_size: 3, 5, 7, 9, 11
  - bottleneck_filters: 8, 16, 32, 64
  - com/sem BatchNorm
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

TCC_BASE = os.environ.get("TCC_BASE", "/mnt/c/TCC_data")
DATA_DIR = os.environ.get("TCC_PROCESSED_DIR", os.path.join(TCC_BASE, "processed"))
RESULTS_BASE = os.environ.get("TCC_RESULTS_BASE", os.path.join(DATA_DIR, "results"))
RESULTS_DIR = os.path.join(RESULTS_BASE, "optuna_cnn1d_ae")
os.makedirs(RESULTS_DIR, exist_ok=True)

STUDY_NAME = os.environ.get("OPTUNA_STUDY_NAME", "optuna_cnn1d_ae")
STORAGE_URL = os.environ.get(
    "OPTUNA_STORAGE_URL",
    f"sqlite:///{os.path.join(RESULTS_DIR, 'study.db')}",
)

JANELA_NPTS = 800
N_TRIALS = 10
EPOCHS_OPT = 40
EPOCHS_FINAL = 120
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

# reshape pra 3D
def to3d(X): return X.reshape(-1, JANELA_NPTS, 1)
def to2d(X): return X.reshape(-1, JANELA_NPTS)

print(f"  Train estação: {X_train_est.shape} | Train temporal: {X_train_tmp.shape}")


def calcular_auc_pr(model, X_n, X_a):
    r_n = model.predict(to3d(X_n), batch_size=1024, verbose=0)
    r_a = model.predict(to3d(X_a), batch_size=1024, verbose=0)
    s_n = np.mean((X_n - to2d(r_n)) ** 2, axis=1)
    s_a = np.mean((X_a - to2d(r_a)) ** 2, axis=1)
    y = np.concatenate([np.zeros(len(s_n)), np.ones(len(s_a))])
    s = np.concatenate([s_n, s_a])
    return average_precision_score(y, s)


def avaliar_completo(model, X_n, X_a):
    r_n = model.predict(to3d(X_n), batch_size=1024, verbose=0)
    r_a = model.predict(to3d(X_a), batch_size=1024, verbose=0)
    s_n = np.mean((X_n - to2d(r_n)) ** 2, axis=1)
    s_a = np.mean((X_a - to2d(r_a)) ** 2, axis=1)
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
    n_blocks = trial.suggest_int("n_blocks", 2, 4)
    use_bn = trial.suggest_categorical("use_batchnorm", [True, False])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    bottleneck_f = trial.suggest_categorical("bottleneck_filters", [8, 16, 32, 64])

    filters = []
    kernels = []
    for i in range(n_blocks):
        f = trial.suggest_categorical(f"filters_{i}", [16, 32, 64, 128, 256])
        k = trial.suggest_categorical(f"kernel_{i}", [3, 5, 7, 9, 11])
        filters.append(f)
        kernels.append(k)

    # encoder
    inp = tf.keras.layers.Input(shape=(JANELA_NPTS, 1))
    x = inp

    # verificar que MaxPool não reduz demais
    current_len = JANELA_NPTS
    actual_blocks = 0
    for i in range(n_blocks):
        if current_len < 4:
            break
        x = tf.keras.layers.Conv1D(filters[i], kernels[i], padding="same", activation="relu")(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        current_len //= 2
        actual_blocks += 1

    # bottleneck
    x = tf.keras.layers.Conv1D(bottleneck_f, 3, padding="same", activation="relu")(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)

    # decoder (simétrico)
    for i in range(actual_blocks - 1, -1, -1):
        x = tf.keras.layers.Conv1D(filters[i], kernels[i], padding="same", activation="relu")(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling1D(2)(x)

    out = tf.keras.layers.Conv1D(1, 7, padding="same", activation="tanh")(x)

    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

    # verificar output shape
    if model.output_shape != (None, JANELA_NPTS, 1):
        raise optuna.TrialPruned()

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
    try:
        model = build_model(trial)
    except (optuna.TrialPruned, Exception):
        raise optuna.TrialPruned()

    X_tr = to3d(X_train_est)
    X_vn = to3d(X_val_est_n)
    try:
        model.fit(X_tr, X_tr, validation_data=(X_vn, X_vn),
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
    h = model.fit(to3d(X_tr), to3d(X_tr), validation_data=(to3d(X_vn), to3d(X_vn)),
                  epochs=EPOCHS_FINAL, batch_size=params["batch_size"],
                  callbacks=[es, rlr], verbose=1)
    tempo = time.time() - t0
    r_val = avaliar_completo(model, X_vn, X_va)
    r_test = avaliar_completo(model, X_tn, X_ta)
    model.save(os.path.join(RESULTS_DIR, f"cnn1d_ae_opt_{nome}.keras"))
    print(f"  {nome} TEST — AUC-PR={r_test['auc_pr']:.4f} AUC-ROC={r_test['auc_roc']:.4f} F1={r_test['f1_best']:.4f}")
    return {"val": r_val, "test": r_test, "tempo_min": round(tempo/60, 2),
            "epochs": len(h.history["loss"])}


def main():
    print("=" * 70)
    print(f"OPTUNA — CNN 1D AE | {N_TRIALS} trials")
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

    print(f"\n  Busca: {tempo_busca:.1f} min | Trial #{study.best_trial.number}: AUC-PR={study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    best = study.best_params
    res = {"best_params": best, "best_auc_pr_val": round(study.best_value, 4),
           "tempo_busca_min": round(tempo_busca, 1)}

    res["estacao"] = retreinar(best, X_train_est, X_val_est_n, X_val_est_a,
                               X_test_est_n, X_test_est_a, "estacao")
    res["temporal"] = retreinar(best, X_train_tmp, X_val_tmp_n, X_val_tmp_a,
                                X_test_tmp_n, X_test_tmp_a, "temporal")

    with open(os.path.join(RESULTS_DIR, "resultados_optuna_cnn1d_ae.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTADO FINAL — CNN 1D AE Optuna")
    print("=" * 70)
    for s in ("estacao", "temporal"):
        r = res[s]["test"]
        print(f"  {s:10s} AUC-PR={r['auc_pr']:.4f}  AUC-ROC={r['auc_roc']:.4f}  F1={r['f1_best']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()