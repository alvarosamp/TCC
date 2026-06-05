# TCC — Detecção de Anomalias Sísmicas em Dispositivos de Borda

Pipeline de machine learning para classificação de eventos sísmicos, com foco em modelos leves exportáveis para microcontroladores (TinyML).

## Visão Geral

O projeto implementa um pipeline reprodutível (DVC + MLflow) que vai desde a geração de dados sintéticos até a exportação de modelos otimizados para edge (TFLite / C header). Múltiplas famílias de modelos são treinadas e comparadas automaticamente, e o melhor candidato edge é selecionado via `candidate_manifest.json`.

## Estrutura

```
├── src/
│   ├── data/          # Validação de dataset e adaptadores
│   ├── features/      # Extração de features estatísticas
│   ├── training/      # Treinamento, HPO (Optuna), avaliação e registro
│   ├── export/        # Exportação TFLite e C header
│   └── core/          # Schemas, perfis e configurações
├── config/            # Configs Hydra (modelos, perfis, pipeline)
├── scripts/           # Geração de dados sintéticos
├── artefacts/         # Modelos, métricas e relatórios (saída do pipeline)
├── dvc.yaml           # Definição do pipeline DVC
└── requirements.txt
```

## Pipeline

```
generate_data → validate_dataset → train_all → export
```

Executar pipeline completo:

```bash
dvc repro
```

Executar treino isolado (modo smoke):

```bash
TCC_MODELS_CFG=config/model/models_smoke.yaml python -m src.training.train_all
```

## Modelos Suportados

| Família | Modelos |
|---|---|
| `classical_supervised` | Logistic Regression, Random Forest, Extra Trees |
| `classical_unsupervised` | Isolation Forest |
| `neural_classifier` | Tiny CNN, Tiny TCN, LSTM Classifier |
| `autoencoder` | Dense Autoencoder, CNN Autoencoder |

## Resultados

| Modelo | AUC-PR | F1 | Edge | Optuna |
|---|---:|---:|:---:|:---:|
| Logistic Regression | 1.000 | 1.000 | — | sim |
| Random Forest | 1.000 | 1.000 | — | não |
| Tiny CNN | 1.000 | 1.000 | sim (593 params) | não |

## Rastreamento de Experimentos

```bash
mlflow ui --backend-store-uri sqlite:///artefacts/mlruns/mlflow.db
```

## Requisitos

```bash
pip install -r requirements.txt
```
