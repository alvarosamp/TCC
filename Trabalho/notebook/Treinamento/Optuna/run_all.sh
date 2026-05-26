#!/usr/bin/env bash
# =====================================================
# Runner — Roda tudo de uma noite pra outra.
# Uso: bash run_all.sh
#
# Ordem:
#   1. Optuna Dense AE  (~1h)
#   2. Optuna CNN 1D AE (~3-4h)
#   3. Optuna LSTM AE   (~6-7h)
#   4. Comparação Final  (~5min)
#   5. MLOps Pipeline    (~10min)
#
# Total estimado: ~10-12h na RTX 4060
# Roda às 22h, acorda com tudo pronto às 8h.
# =====================================================

set -euo pipefail  # falha se qualquer etapa falhar (inclui pipes com tee)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

find_venv_activate() {
    local dir="$1"
    while true; do
        if [ -f "$dir/.venv/bin/activate" ]; then
            echo "$dir/.venv/bin/activate"
            return 0
        fi
        if [ "$dir" = "/" ]; then
            return 1
        fi
        dir="$(dirname "$dir")"
    done
}

echo "======================================================"
echo "  PIPELINE COMPLETO — $(date)"
echo "  Scripts em: $SCRIPT_DIR"
echo "======================================================"

# ativar venv se não estiver ativo
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -n "${VENV_ACTIVATE:-}" ]; then
        ACTIVATE_PATH="$VENV_ACTIVATE"
    else
        ACTIVATE_PATH="$(find_venv_activate "$SCRIPT_DIR" || true)"
    fi
    if [ -z "${ACTIVATE_PATH:-}" ]; then
        echo "ERRO: não encontrei .venv/bin/activate subindo a partir de: $SCRIPT_DIR" >&2
        echo "Dica: crie a venv em /home/vish8/tcc/Trabalho/.venv ou rode com VENV_ACTIVATE=/caminho/para/.venv/bin/activate." >&2
        exit 1
    fi
    echo "Ativando venv: $ACTIVATE_PATH"
    # shellcheck disable=SC1090
    source "$ACTIVATE_PATH"
fi

# ---------- Paths (WSL/Linux) ----------
# Observação: os scripts Python atuais usam paths fixos em /mnt/c/TCC_data.
# Aqui só garantimos que os diretórios existem e que os datasets estão no lugar.
if [ -d "/mnt/c/TCC_data" ]; then
    TCC_BASE_DEFAULT="/mnt/c/TCC_data"
else
    TCC_BASE_DEFAULT="$HOME/TCC_data"
fi

TCC_BASE="${TCC_BASE:-$TCC_BASE_DEFAULT}"
TCC_PROCESSED_DIR="${TCC_PROCESSED_DIR:-$TCC_BASE/processed}"
TCC_LOG_DIR="${TCC_LOG_DIR:-$TCC_BASE/logs}"
TCC_RESULTS_BASE="${TCC_RESULTS_BASE:-$TCC_PROCESSED_DIR/results}"

export TCC_BASE
export TCC_PROCESSED_DIR
export TCC_RESULTS_BASE

# ---------- Runtime knobs (more stable logs for nohup) ----------
# Optuna progress bar uses carriage-returns (\r) which can look like the log "stopped".
# If stdout is not a terminal (e.g., nohup > file), disable it by default.
if [ ! -t 1 ] && [ -z "${OPTUNA_SHOW_PROGRESS:-}" ]; then
    export OPTUNA_SHOW_PROGRESS=0
fi

# Reduce noisy XLA/Triton ptxas warnings by default. Re-enable with:
#   TCC_DISABLE_TRITON_GEMM=0
export TCC_DISABLE_TRITON_GEMM="${TCC_DISABLE_TRITON_GEMM:-1}"

mkdir -p "$TCC_LOG_DIR"

DATASET_EST="$TCC_PROCESSED_DIR/dataset_v3_split_estacao.npz"
DATASET_TMP="$TCC_PROCESSED_DIR/dataset_v3_split_temporal.npz"
if [ ! -f "$DATASET_EST" ] || [ ! -f "$DATASET_TMP" ]; then
    echo "ERRO: datasets não encontrados em: $TCC_PROCESSED_DIR" >&2
    echo "Esperado: $DATASET_EST" >&2
    echo "          $DATASET_TMP" >&2
    echo "Se você usa outro caminho, ajuste TCC_BASE/TCC_PROCESSED_DIR ou edite os scripts Python." >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERRO: não achei o interpretador '$PYTHON_BIN' no PATH." >&2
    exit 1
fi

echo ""
echo "Config:"
echo "  Python:         $(command -v "$PYTHON_BIN")"
echo "  TCC_BASE:       $TCC_BASE"
echo "  Processed dir:  $TCC_PROCESSED_DIR"
echo "  Results base:   $TCC_RESULTS_BASE"
echo "  Logs dir:       $TCC_LOG_DIR"

echo ""
echo "======================================================"
echo "  ETAPA 1/5 — Optuna Dense AE"
echo "  Início: $(date)"
echo "======================================================"
$PYTHON_BIN optuna_dense_ae.py 2>&1 | tee "$TCC_LOG_DIR/optuna_dense_ae.log"

echo ""
echo "======================================================"
echo "  ETAPA 2/5 — Optuna CNN 1D AE"
echo "  Início: $(date)"
echo "======================================================"
$PYTHON_BIN optuna_cnn1d_ae.py 2>&1 | tee "$TCC_LOG_DIR/optuna_cnn1d_ae.log"

echo ""
echo "======================================================"
echo "  ETAPA 3/5 — Optuna LSTM AE"
echo "  Início: $(date)"
echo "======================================================"
$PYTHON_BIN optuna_lstm_ae.py 2>&1 | tee "$TCC_LOG_DIR/optuna_lstm_ae.log"

echo ""
echo "======================================================"
echo "  ETAPA 4/5 — Comparação Final"
echo "  Início: $(date)"
echo "======================================================"
$PYTHON_BIN comparacao_final.py 2>&1 | tee "$TCC_LOG_DIR/comparacao_final.log"

echo ""
echo "======================================================"
echo "  ETAPA 5/5 — MLOps Pipeline"
echo "  Início: $(date)"
echo "======================================================"
$PYTHON_BIN mlops_pipeline.py 2>&1 | tee "$TCC_LOG_DIR/mlops_pipeline.log"

echo ""
echo "======================================================"
echo "  PIPELINE COMPLETO — CONCLUÍDO"
echo "  Fim: $(date)"
echo "======================================================"
echo ""
echo "  Resultados em: $TCC_PROCESSED_DIR/results/"
echo "  Logs em:       $TCC_LOG_DIR"
echo ""
echo "  Próximo passo: abra os JSONs de resultado e as figuras."
echo "======================================================"