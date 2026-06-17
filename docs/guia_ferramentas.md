# Guia de Ferramentas do Projeto TCC

Este guia descreve como usar cada ferramenta do pipeline, com comandos prontos para rodar.

---

## Sumario

1. [MLflow — Rastreamento de Experimentos](#mlflow)
2. [DVC — Versionamento de Dados e Pipeline](#dvc)
3. [Prometheus — Coleta de Metricas](#prometheus)
4. [Grafana — Visualizacao de Metricas](#grafana)
5. [OTA Simulado — Atualizacao de Modelo Edge](#ota-simulado)
6. [PlatformIO — Firmware ESP32](#platformio)
7. [Quality Gate — Promocao de Modelo](#quality-gate)
8. [Drift Detection — Monitoramento de Distribuicao](#drift-detection)

---

## MLflow

Rastreia parametros, metricas e artefatos de cada rodada de treinamento.

### Iniciar interface web

```bash
cd ~/tcc_atual/TCC
mlflow ui --backend-store-uri sqlite:///artefacts/mlruns/mlflow.db --port 5000
```

Acesse: http://localhost:5000

### O que voce encontra la

- **Experiments**: cada modelo treinado e um run registrado
- **Parameters**: hiperparametros usados (batch_size, lr, n_blocks...)
- **Metrics**: auc_pr, f1, precision, recall por epoch
- **Artifacts**: arquivo `.keras`, `.tflite`, relatorios JSON

### Registrar manualmente um run

O pipeline ja registra automaticamente ao rodar `train_all`. Para registrar um experimento externo:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///artefacts/mlruns/mlflow.db")
with mlflow.start_run(run_name="meu_experimento"):
    mlflow.log_param("modelo", "tiny_cnn")
    mlflow.log_metric("auc_pr", 0.877)
```

---

## DVC

Controla versoes de dados e reproduz o pipeline de forma rastreavel.

### Pipeline smoke (sintetico, rapido)

```bash
cd ~/tcc_atual/TCC
dvc repro
```

Etapas: `generate_data` → `validate_dataset` → `train_all` → `export_tflite`

### Pipeline real (dataset sismico)

```bash
dvc repro -f dvc_real.yaml
```

### Ver status do pipeline

```bash
dvc status
```

### Forcar re-execucao de uma etapa especifica

```bash
dvc repro -f --single-item train_all
```

### Ver grafo de dependencias

```bash
dvc dag
```

---

## Prometheus

Coleta metricas expostas pelo servidor FastAPI do projeto.

### Subir stack de monitoramento

```bash
cd ~/tcc_atual/TCC
docker compose -f docker-compose.monitoring.yml up -d
```

### Verificar se esta rodando

```bash
docker compose -f docker-compose.monitoring.yml ps
```

### Acessar Prometheus

Acesse: http://localhost:9090

### Metricas expostas pelo FastAPI

Com o servidor rodando (`uvicorn server.app.main:app`), o endpoint de metricas fica em:

```
http://localhost:8000/metrics
```

### Exemplo de query no Prometheus

```promql
# Numero de inferencias por modelo
inference_total{model="tiny_cnn"}

# Latencia media de inferencia
rate(inference_duration_seconds_sum[5m]) / rate(inference_duration_seconds_count[5m])
```

---

## Grafana

Visualiza as metricas coletadas pelo Prometheus em dashboards.

### Acessar Grafana

Acesse: http://localhost:3000

- Usuario padrao: `admin`
- Senha padrao: `admin`

### Configurar datasource Prometheus

1. Va em **Configuration > Data Sources**
2. Clique em **Add data source**
3. Selecione **Prometheus**
4. URL: `http://prometheus:9090`
5. Clique em **Save & Test**

### Importar dashboard do projeto

Os dashboards ficam em `monitoring/grafana/dashboards/`. Para importar:

1. Va em **Dashboards > Import**
2. Carregue o arquivo `.json` da pasta acima
3. Selecione o datasource Prometheus criado

### Parar a stack

```bash
docker compose -f docker-compose.monitoring.yml down
```

---

## OTA Simulado

Fluxo completo de atualizacao de modelo no dispositivo edge.

### Visao geral do fluxo

```
production_manifest.json
  -> ota_manifest.json
  -> pacote OTA (tflite + assinatura HMAC-SHA256)
  -> validation_report.json
  -> releases/latest.json
  -> device_update_check_report.json
  -> install_report.json
  -> rollback_report.json
```

### Rodar fluxo completo

```bash
cd ~/tcc_atual/TCC
source ../.venv/bin/activate

# 1. Gerar manifesto OTA a partir do modelo em producao
python -m src.ota.build_ota_manifest

# 2. Empacotar artefato + assinatura HMAC-SHA256
python -m src.ota.build_ota_package

# 3. Validar integridade do pacote (SHA-256 + assinatura)
python -m src.ota.validate_ota_package

# 4. Publicar release local (simula repositorio OTA)
python -m src.ota.publish_local_release

# 5. Simular dispositivo checando atualizacao
python -m src.ota.simulate_device_update_check

# 6. Simular instalacao do pacote
python -m src.ota.simulate_apply_update

# 7. Simular rollback (opcional, testa o caminho de falha)
python -m src.ota.simulate_rollback
```

### Artefatos gerados

```
artefacts/ota/ota_manifest.json
artefacts/ota/packages/<versao>/artifact.tflite
artefacts/ota/packages/<versao>/signature.json
artefacts/ota/packages/<versao>/validation_report.json
artefacts/ota/releases/latest.json
artefacts/ota/releases/<versao>/
```

### Verificar assinatura manualmente

```python
import json, hmac, hashlib

with open("artefacts/ota/packages/<versao>/signature.json") as f:
    sig = json.load(f)

with open("artefacts/ota/packages/<versao>/artifact.tflite", "rb") as f:
    data = f.read()

key = sig["key_hint"].encode()  # em producao real: chave secreta segura
mac = hmac.new(key, data, hashlib.sha256).hexdigest()
print("Valido:", hmac.compare_digest(mac, sig["hmac_sha256"]))
```

---

## PlatformIO

Compila e faz upload do firmware para o ESP32.

### Pre-requisito

```bash
pip install platformio
```

### Compilar firmware

```bash
cd ~/tcc_atual/TCC/PlatformIO/Projects/TCC
PLATFORMIO_RUN_JOBS=1 pio run -e esp32dev
```

> O `-j1` e obrigatorio: TFLite Micro estoura RAM do compilador com multiplos nucleos.

### Upload para ESP32

```bash
PLATFORMIO_RUN_JOBS=1 pio run -e esp32dev -t upload
```

### Monitor serial (ver saida CSV do ESP32)

```bash
pio device monitor --port /dev/ttyUSB0 --baud 115200
```

### Trocar modelo ativo

Edite `include/model_config.h`:

```cpp
// Opcoes disponiveis:
#define ACTIVE_MODEL MODEL_PIPELINE_TINY_CNN_FLOAT32  // recomendado agora
#define ACTIVE_MODEL MODEL_PIPELINE_TINY_CNN_INT8     // requer modelo sem REDUCE_MAX
#define ACTIVE_MODEL MODEL_TCN_FLOAT32
```

### Identificar chip conectado

```bash
python3 ~/.platformio/packages/tool-esptoolpy/esptool.py --port /dev/ttyUSB0 chip_id
```

### Interpretar saida serial

A saida vem em CSV com as colunas:

```
run, score, pred, expected_label, correct,
preprocess_ms, inference_ms, total_ms,
fps, cpu_percent_est, energy_mj_est,
tensor_arena_kb, heap_free_before_kb, heap_free_after_kb,
input_type, output_type
```

E ao final de cada rodada de 100 inferencias:

```
# summary,metric,avg,min,max
# summary,preprocess_ms,...
# summary,inference_ms,...
# summary,accuracy,...
```

---

## Quality Gate

Decide se um candidato pode ser promovido a producao.

### Rodar promocao

```bash
cd ~/tcc_atual/TCC
python -m src.mlops.promote_model
```

### Regras configuradas em `config/configs/config.yaml`

```yaml
quality_gate:
  primary_metric: auc_pr
  min_auc_pr: 0.80
  min_f1: 0.70
  max_fp_per_hour: 10
  max_val_test_auc_pr_gap: 0.08
  max_model_size_kb: 300
```

### Artefatos gerados

```
artefacts/reports/promotion_report.json   # resultado do gate
artefacts/registry/production_manifest.json  # manifesto de producao
```

---

## Drift Detection

Detecta mudancas na distribuicao dos dados em relacao a referencia de treinamento.

### Fluxo completo

```bash
cd ~/tcc_atual/TCC

# 1. Construir referencia estatistica a partir do treino
python -m src.monitoring.build_drift_reference

# 2. Checar drift no dataset atual
python -m src.monitoring.check_data_drift --dataset <caminho/dataset.npz> --split test

# 3. Avaliar politica de retreino
python -m src.monitoring.retrain_policy

# 4. Decidir se faz OTA baseado no drift
python -m src.monitoring.drift_to_ota_decision
```

### Metricas de drift usadas

| Metrica | O que mede |
|---|---|
| Z-shift | Deslocamento da media em desvios padrao |
| PSI | Population Stability Index — mudanca de distribuicao |
| KS p-value | Teste Kolmogorov-Smirnov — diferenca significativa? |

### Interpretacao

```
PSI < 0.1   → estavel
PSI 0.1-0.2 → atencao
PSI > 0.2   → drift significativo, retreino recomendado
```

### Artefatos gerados

```
artefacts/monitoring/drift_reference.json
artefacts/monitoring/drift_report.json
artefacts/monitoring/retrain_policy.json
artefacts/monitoring/ota_decision.json
```

---

## Fluxo Completo Integrado

Para rodar tudo do zero:

```bash
# 1. Dataset
python -m src.data.validate_dataset

# 2. Treino
python -m src.training.train_all --models-cfg config/model/models.yaml

# 3. Quality gate
python -m src.mlops.promote_model

# 4. Export edge
python -m src.export.export_tflite

# 5. Drift
python -m src.monitoring.build_drift_reference
python -m src.monitoring.check_data_drift --dataset data/dataset.npz --split test
python -m src.monitoring.retrain_policy
python -m src.monitoring.drift_to_ota_decision

# 6. OTA
python -m src.ota.build_ota_manifest
python -m src.ota.build_ota_package
python -m src.ota.validate_ota_package
python -m src.ota.publish_local_release
python -m src.ota.simulate_device_update_check
python -m src.ota.simulate_apply_update

# 7. Firmware ESP32
cd PlatformIO/Projects/TCC
PLATFORMIO_RUN_JOBS=1 pio run -e esp32dev -t upload
pio device monitor --port /dev/ttyUSB0 --baud 115200
```
