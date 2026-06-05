# TinyML/MLOps Pipeline for Complex Time-Series Anomaly Detection

Pipeline genérico de TinyML e MLOps para detecção de anomalias em séries temporais complexas, com validação inicial em sinais sísmicos e implantação embarcada em ESP32.

O projeto não é apenas um detector sísmico. A proposta é construir uma arquitetura reutilizável em que diferentes domínios de sensores possam ser convertidos para um contrato comum de dados e, a partir daí, usar o mesmo fluxo de validação, treinamento, avaliação, exportação TinyML, monitoramento e atualização.

```text
raw domain data
    ↓
domain adapter
    ↓
generic time-series dataset
    ↓
validation
    ↓
model training + HPO
    ↓
model comparison
    ↓
TinyML export
    ↓
quality gate
    ↓
edge deployment / OTA
```

## Objetivo

Redes de sensores contínuos geram grandes volumes de dados. Transmitir e armazenar tudo pode ser caro em energia, banda, armazenamento e infraestrutura.

Este projeto usa TinyML para transformar o dispositivo de borda em um filtro inteligente:

```text
sensor -> janela temporal -> preprocessing -> modelo TinyML -> score -> decisão local
```

A ideia é que o dispositivo envie ou armazene apenas janelas relevantes, como possíveis eventos anômalos, reduzindo custo operacional sem depender de processamento constante na nuvem.

## Estudo De Caso Atual

O primeiro domínio validado é sísmico.

O pipeline atual usa dados MiniSEED organizados como:

```text
raw/
  events/
  continuous/
  stationxml/
```

A variante principal do projeto é:

```text
seismic_edge_v1
```

Ela é edge-compatible:

- não usa `remove_response`;
- não exige `StationXML`;
- aproxima o preprocessing offline do preprocessing embarcado;
- reduz training-serving skew entre Colab e ESP32.

Pipeline de preprocessing:

```text
resample 40 Hz
detrend linear
demean
taper 5%
bandpass 0.5-15 Hz
z-score por janela
```

## Arquitetura

```text
configs/
  config.yaml
  models.yaml
  profiles/
    seismic_edge_v1.yaml

src/
  core/
    settings.py
    profile.py
    schemas.py

  data/
    adapters/
      seismic_edge.py
    validate_dataset.py

  features/
    statistical_features.py

  training/
    evaluate.py
    hpo.py
    model_registry.py
    neural_models.py
    train_all.py

  export/
    export_tflite.py
    export_header.py

  mlops/
    promote_model.py

  monitoring/
    drift.py

tests/
  test_profile.py
  test_schemas.py
  test_features.py
  test_model_registry.py
  test_neural_models.py
  test_pipeline_smoke.py

artifacts/
  reports/
  models/
  edge/
  registry/
  mlruns/
```

## Ideia De Generalização

A generalidade do projeto não está no adaptador de dados brutos. Cada domínio naturalmente tem seu próprio formato.

Exemplo sísmico:

```text
MiniSEED -> seismic_edge.py -> dataset NPZ genérico
```

Exemplo futuro com vibração:

```text
CSV/ADC vibration data -> vibration_edge.py -> dataset NPZ genérico
```

Depois do adaptador, todos os domínios seguem o mesmo contrato:

```text
X_train, y_train
X_val, y_val
X_test, y_test
```

Assim, os módulos de validação, treinamento, métricas, exportação, MLflow, DVC, quality gate e OTA continuam os mesmos.

## Tecnologias

### Machine Learning

- NumPy
- Pandas
- SciPy
- scikit-learn
- TensorFlow/Keras
- Optuna

### MLOps

- MLflow
- DVC
- YAML configs
- JSON manifests
- quality gate
- pytest

### Edge/TinyML

- TensorFlow Lite
- TensorFlow Lite Micro
- ESP32
- PlatformIO
- export `.tflite`
- export `.h`

### Futuro / Expansão

- Evidently para drift
- FastAPI para serving
- Docker
- Prometheus/Grafana para telemetria
- OTA manifest para atualização remota

## Modelos Suportados

O projeto usa um catálogo configurável em:

```text
configs/models.yaml
```

Exemplo:

```yaml
tiny_tcn:
  enabled: true
  family: neural_classifier
  edge_candidate: true
  export_tflite: true
```

Modelos atuais:

| Modelo | Família | Uso |
|---|---|---|
| Logistic Regression | classical_supervised | baseline |
| Random Forest | classical_supervised | baseline forte |
| ExtraTrees | classical_supervised | baseline forte |
| Isolation Forest | classical_unsupervised | anomaly baseline |
| Dense Autoencoder | autoencoder | reconstrução |
| CNN Autoencoder | autoencoder | reconstrução temporal |
| Tiny CNN | neural_classifier | candidato TinyML |
| Tiny TCN | neural_classifier | principal candidato TinyML |
| LSTM Classifier | neural_classifier | experimental |
| Tiny Transformer | neural_classifier | experimental |

## Fluxo Principal

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

No Colab:

```python
!pip install -r requirements.txt
```

### 2. Gerar Dataset Edge-Compatible

```bash
python -m src.data.adapters.seismic_edge
```

Saídas esperadas:

```text
processed_seismic_edge_v1/
  dataset_seismic_edge_v1_split_temporal.npz
  dataset_seismic_edge_v1_split_evento.npz
  inventario_seismic_edge_v1.csv
  dataset_seismic_edge_v1_info.json
```

### 3. Validar Dataset

```bash
python -m src.data.validate_dataset
```

Gera:

```text
artifacts/reports/dataset_validation_report.json
```

Essa etapa valida:

- chaves obrigatórias do NPZ;
- tamanho das janelas;
- labels;
- splits;
- estatísticas básicas.

### 4. Rodar Testes

```bash
pytest -q
```

Com cobertura:

```bash
pytest -q --cov=src
```

Os testes garantem que os módulos principais se comunicam:

```text
profile -> schemas -> features -> registry -> evaluate
```

### 5. Treinar Modelos

```bash
python -m src.training.train_all
```

O script:

- lê `configs/models.yaml`;
- treina apenas modelos com `enabled: true`;
- aplica Optuna quando configurado;
- calcula métricas de validação e teste;
- escolhe threshold pela validação;
- registra runs no MLflow;
- gera comparação final.

Saídas:

```text
artifacts/models/
artifacts/reports/model_comparison.csv
artifacts/reports/model_comparison.md
artifacts/reports/model_comparison.json
artifacts/reports/candidate_manifest.json
```

### 6. Exportar Para TinyML

```bash
python -m src.export.export_tflite
```

Gera:

```text
artifacts/edge/
  <modelo>_float32.tflite
  <modelo>_float16.tflite
  <modelo>_int8.tflite
  <modelo>_int8.h
  <modelo>_export_manifest.json
```

O `.h` é usado no firmware ESP32 com TensorFlow Lite Micro.

### 7. Promover Modelo

```bash
python -m src.mlops.promote_model
```

Aplica quality gate definido em:

```text
configs/config.yaml
```

Exemplo:

```yaml
quality_gate:
  primary_metric: auc_pr
  min_auc_pr: 0.80
  min_f1: 0.70
  max_model_size_kb: 300
```

Se aprovado, gera:

```text
artifacts/registry/production_manifest.json
```

Sempre gera:

```text
artifacts/reports/promotion_report.json
```

## DVC

O projeto pode ser executado de forma reproduzível com DVC.

Fluxo esperado:

```bash
dvc repro
```

Stages recomendados:

```text
test
prepare_seismic_edge
validate_dataset
train_all
export_tflite
promote_model
```

O DVC ajuda a garantir que mudanças em dados, configs ou código disparem novamente as etapas necessárias.

## MLflow

Os experimentos são registrados com MLflow.

Cada run registra:

- nome do modelo;
- família;
- parâmetros;
- profile;
- dataset;
- métricas de validação;
- métricas de teste;
- threshold escolhido na validação;
- artefatos do modelo;
- manifestos.

Para abrir a UI local:

```bash
mlflow ui --backend-store-uri artifacts/mlruns
```

## Métricas

A métrica principal é:

```text
AUC-PR
```

Ela foi escolhida porque detecção de anomalias costuma ser desbalanceada.

Métricas adicionais:

- AUC-ROC
- F1
- Precision
- Recall
- matriz de confusão
- tamanho do modelo
- tamanho TFLite
- compatibilidade com edge

## Rastreabilidade

Cada modelo candidato possui um manifesto:

```text
artifacts/reports/candidate_manifest.json
```

Cada modelo exportado possui:

```text
artifacts/edge/<modelo>_export_manifest.json
```

Cada modelo promovido possui:

```text
artifacts/registry/production_manifest.json
```

Esses arquivos conectam:

```text
dataset
profile
preprocessing
modelo
threshold
métricas
exportação
edge deployment
```

Essa rastreabilidade é essencial para MLOps e para futura estratégia OTA.

## TinyML E ESP32

O modelo `.tflite` é convertido para header C/C++:

```text
modelo_int8.tflite -> modelo_int8.h
```

No ESP32, o TensorFlow Lite Micro carrega o modelo como bytes no firmware:

```cpp
model = tflite::GetModel(MODEL_DATA);
```

A versão `int8` é a principal candidata para execução embarcada por reduzir tamanho, memória e custo computacional.

## Training-Serving Skew

Uma decisão importante do projeto foi criar o profile `seismic_edge_v1`, que remove a dependência de `remove_response`.

Antes:

```text
treino: remove_response + StationXML
ESP32: sem remove_response
```

Isso gera training-serving skew.

Agora:

```text
treino: edge-compatible
ESP32: edge-compatible
```

Essa mudança torna o pipeline mais coerente com a implantação real.

## Próximos Passos

- Finalizar benchmark no ESP32.
- Medir latência, RAM, tensor arena e energia estimada.
- Adicionar drift report com Evidently.
- Criar OTA manifest.
- Validar o pipeline em outro domínio, como vibração industrial.
- Adicionar FastAPI para demonstração.
- Adicionar Prometheus/Grafana para telemetria futura.

## Visão De Produto

O sistema proposto usa TinyML como um organizador de dados na borda.

Em vez de transmitir tudo:

```text
sensor -> cloud
```

o dispositivo decide localmente:

```text
normal -> descartar/resumir
anomalia -> salvar/transmitir
```

Isso pode reduzir:

- armazenamento;
- consumo de energia;
- uso de rede;
- custo de nuvem;
- tempo de análise.

## Frase Central

Este projeto demonstra uma arquitetura genérica de TinyML/MLOps para detecção de anomalias em séries temporais complexas, validada inicialmente em sinais sísmicos e preparada para adaptação a outros sensores e domínios industriais.
