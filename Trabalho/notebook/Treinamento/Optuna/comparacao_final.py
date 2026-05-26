"""
Comparação Final — Gera tabela do paper + figuras unificadas.
Lê os JSONs de resultado dos 3 Optuna e gera:
  - Tabela comparativa (console + CSV)
  - Figura com 6 curvas ROC (3 modelos × 2 splits)
  - Figura com 6 curvas PR (3 modelos × 2 splits)
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np

TCC_BASE = os.environ.get("TCC_BASE", "/mnt/c/TCC_data")
DATA_DIR = os.environ.get("TCC_PROCESSED_DIR", os.path.join(TCC_BASE, "processed"))
RESULTS_BASE = os.environ.get("TCC_RESULTS_BASE", os.path.join(DATA_DIR, "results"))
OUT_DIR = os.path.join(RESULTS_BASE, "comparacao_final")
os.makedirs(OUT_DIR, exist_ok=True)

MODELOS = {
    "Dense AE":  os.path.join(RESULTS_BASE, "optuna_dense_ae", "resultados_optuna_dense_ae.json"),
    "CNN 1D AE": os.path.join(RESULTS_BASE, "optuna_cnn1d_ae", "resultados_optuna_cnn1d_ae.json"),
    "LSTM AE":   os.path.join(RESULTS_BASE, "optuna_lstm_ae", "resultados_optuna_lstm_ae.json"),
}


def main():
    print("=" * 80)
    print("COMPARAÇÃO FINAL — Todos os modelos otimizados com Optuna")
    print("=" * 80)

    dados = {}
    for nome, path in MODELOS.items():
        if os.path.exists(path):
            with open(path) as f:
                dados[nome] = json.load(f)
            print(f"  {nome}: OK")
        else:
            print(f"  {nome}: ARQUIVO NÃO ENCONTRADO ({path})")

    if len(dados) < 3:
        print("\n  AVISO: nem todos os modelos foram encontrados.")
        print("  Rode os 3 scripts Optuna antes deste.")
        if len(dados) == 0:
            return

    # ---------- Tabela ----------
    print("\n" + "=" * 80)
    print("TABELA COMPARATIVA — Split Estação (generalização entre sensores)")
    print("=" * 80)
    header = f"{'Modelo':<14} {'AUC-PR':>10} {'AUC-ROC':>10} {'F1':>10} {'Params':>10}"
    print(header)
    print("-" * len(header))
    for nome in ("Dense AE", "CNN 1D AE", "LSTM AE"):
        if nome not in dados:
            continue
        r = dados[nome]["estacao"]["test"]
        params = dados[nome].get("best_params", {}).get("params_total", "—")
        print(f"  {nome:<12} {r['auc_pr']:>10.4f} {r['auc_roc']:>10.4f} {r['f1_best']:>10.4f} {str(params):>10}")

    print("\n" + "=" * 80)
    print("TABELA COMPARATIVA — Split Temporal (robustez ao longo do tempo)")
    print("=" * 80)
    print(header)
    print("-" * len(header))
    for nome in ("Dense AE", "CNN 1D AE", "LSTM AE"):
        if nome not in dados:
            continue
        r = dados[nome]["temporal"]["test"]
        print(f"  {nome:<12} {r['auc_pr']:>10.4f} {r['auc_roc']:>10.4f} {r['f1_best']:>10.4f}")

    # ---------- Melhor modelo ----------
    print("\n" + "=" * 80)
    melhor_est = max(dados.items(), key=lambda x: x[1]["estacao"]["test"]["auc_pr"])
    melhor_tmp = max(dados.items(), key=lambda x: x[1]["temporal"]["test"]["auc_pr"])
    print(f"MELHOR modelo (estação):  {melhor_est[0]} — AUC-PR = {melhor_est[1]['estacao']['test']['auc_pr']:.4f}")
    print(f"MELHOR modelo (temporal): {melhor_tmp[0]} — AUC-PR = {melhor_tmp[1]['temporal']['test']['auc_pr']:.4f}")

    # ---------- Hiperparâmetros ótimos ----------
    print("\n" + "=" * 80)
    print("HIPERPARÂMETROS ÓTIMOS")
    print("=" * 80)
    for nome, d in dados.items():
        print(f"\n  {nome}:")
        bp = d.get("best_params", {})
        for k, v in bp.items():
            print(f"    {k}: {v}")
        if "best_reshape" in d:
            print(f"    reshape ótimo: {d['best_reshape'][0]} × {d['best_reshape'][1]} "
                  f"({d['best_reshape'][1]/40*1000:.0f}ms/step)")

    # ---------- CSV ----------
    csv_path = os.path.join(OUT_DIR, "tabela_comparativa.csv")
    with open(csv_path, "w") as f:
        f.write("modelo,split,auc_pr,auc_roc,f1_best,threshold\n")
        for nome in ("Dense AE", "CNN 1D AE", "LSTM AE"):
            if nome not in dados:
                continue
            for split in ("estacao", "temporal"):
                r = dados[nome][split]["test"]
                f.write(f"{nome},{split},{r['auc_pr']},{r['auc_roc']},{r['f1_best']},{r['threshold']}\n")
    print(f"\n  CSV salvo: {csv_path}")

    # ---------- JSON consolidado ----------
    json_path = os.path.join(OUT_DIR, "resultados_finais.json")
    with open(json_path, "w") as f:
        json.dump(dados, f, indent=2)
    print(f"  JSON salvo: {json_path}")

    # ---------- Figuras ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve

        colors = {"Dense AE": "#e74c3c", "CNN 1D AE": "#2ecc71", "LSTM AE": "#3498db"}

        for split_name in ("estacao", "temporal"):
            npz_path = os.path.join(DATA_DIR, f"dataset_v3_split_{split_name}.npz")
            data = np.load(npz_path)
            X_test = data["X_test"]
            y_test = data["y_test"]
            X_test_n = X_test[y_test == 0]
            X_test_a = X_test[y_test == 1]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"Comparação — Split {split_name.capitalize()}", fontsize=14)

            for nome in ("Dense AE", "CNN 1D AE", "LSTM AE"):
                if nome not in dados:
                    continue

                # encontrar modelo salvo
                if nome == "Dense AE":
                    model_dir = "optuna_dense_ae"
                    model_file = f"dense_ae_opt_{split_name}.keras"
                elif nome == "CNN 1D AE":
                    model_dir = "optuna_cnn1d_ae"
                    model_file = f"cnn1d_ae_opt_{split_name}.keras"
                else:
                    model_dir = "optuna_lstm_ae"
                    model_file = f"lstm_ae_opt_{split_name}.keras"

                model_path = os.path.join(RESULTS_BASE, model_dir, model_file)
                if not os.path.exists(model_path):
                    print(f"  AVISO: modelo não encontrado: {model_path}")
                    continue

                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)

                # calcular scores
                input_shape = model.input_shape
                if len(input_shape) == 3:
                    ts, ft = input_shape[1], input_shape[2]
                    r_n = model.predict(X_test_n.reshape(-1, ts, ft), batch_size=1024, verbose=0)
                    r_a = model.predict(X_test_a.reshape(-1, ts, ft), batch_size=1024, verbose=0)
                    s_n = np.mean((X_test_n - r_n.reshape(-1, 800)) ** 2, axis=1)
                    s_a = np.mean((X_test_a - r_a.reshape(-1, 800)) ** 2, axis=1)
                else:
                    r_n = model.predict(X_test_n, batch_size=1024, verbose=0)
                    r_a = model.predict(X_test_a, batch_size=1024, verbose=0)
                    s_n = np.mean((X_test_n - r_n) ** 2, axis=1)
                    s_a = np.mean((X_test_a - r_a) ** 2, axis=1)

                y_true = np.concatenate([np.zeros(len(s_n)), np.ones(len(s_a))])
                y_scores = np.concatenate([s_n, s_a])

                # ROC
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc_roc = dados[nome][split_name]["test"]["auc_roc"]
                axes[0].plot(fpr, tpr, label=f"{nome} ({auc_roc:.3f})",
                           color=colors[nome], linewidth=2)

                # PR
                prec, rec, _ = precision_recall_curve(y_true, y_scores)
                auc_pr = dados[nome][split_name]["test"]["auc_pr"]
                axes[1].plot(rec, prec, label=f"{nome} ({auc_pr:.3f})",
                           color=colors[nome], linewidth=2)

                tf.keras.backend.clear_session()

            axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
            axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
            axes[0].set_title("Curva ROC"); axes[0].legend()
            axes[0].grid(alpha=0.2)

            baseline = len(X_test_a) / (len(X_test_n) + len(X_test_a))
            axes[1].axhline(y=baseline, color="gray", linestyle="--", alpha=0.3, label=f"Baseline ({baseline:.3f})")
            axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
            axes[1].set_title("Curva Precision-Recall"); axes[1].legend()
            axes[1].grid(alpha=0.2)

            plt.tight_layout()
            fig_path = os.path.join(OUT_DIR, f"comparacao_{split_name}.png")
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"  Figura salva: {fig_path}")

    except Exception as e:
        print(f"  AVISO figuras: {e}")

    print("\n" + "=" * 80)
    print("COMPARAÇÃO FINAL CONCLUÍDA")
    print("=" * 80)


if __name__ == "__main__":
    main()