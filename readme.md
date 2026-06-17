# TCC — Pipeline Generico de TinyML/MLOps para Series Temporais

Pipeline completo de TinyML/MLOps para deteccao de anomalias em series temporais. O primeiro estudo de caso usa dados sismicos, mas a arquitetura e generica e foi projetada para ser aplicada em outros dominios: vibracao industrial, corrente eletrica, audio, telemetria e sensores distribuidos.

A ideia central e usar um modelo leve na borda como camada de triagem inteligente: o dispositivo processa janelas locais e so encaminha dados quando ha indicio confiavel de anomalia, reduzindo transmissao, armazenamento, consumo de energia e custo operacional.

---

## Status de Validacao

| Componente | Status |
|---|---|
| Pipeline MLOps completo | OK |
| Quality gate | OK |
| Drift detection (PSI, KS, z-shift) | OK |
| Exportacao TFLite (float32, float16, int8) | OK |
| Geracao de header C/C++ | OK |
| Build embarcada ESP32 | OK |
| Flash real no ESP32 (chip: ESP32-D0WD-V3) | OK |
| Inferencia embarcada float32 | OK |
| OTA simulado (manifesto, pacote, SHA-256, HMAC) | OK |
| Inferencia embarcada int8 | Diagnostico: kernel REDUCE_MAX incompativel com esta versao do TFLite Micro |

> **Nota sobre int8:** o modelo exportado usava `GlobalMaxPooling` (que vira `REDUCE_MAX` quantizado). O runtime TFLite Micro exige `input_zp == output_zp` nessa operacao, o que nao e garantido apos PTQ. A correcao e substituir por `GlobalAveragePooling` e reexportar. Isso nao invalida o pipeline — e um resultado tecnico documentado que orienta a proxima iteracao de arquitetura.

---

## Resultados do Modelo Atual (Tiny CNN — float32)

Modelo: `tiny_cnn` | Parametros: 15.377 | Dataset: sismico (split por evento)

| Metrica | Validacao | Teste |
|---|---:|---:|
| AUC-PR | 0.8598 | **0.8775** |
| AUC-ROC | 0.9546 | **0.9576** |
| F1 | 0.7908 | **0.8109** |
| Precision | 0.8343 | **0.8431** |
| Recall | 0.7517 | **0.7810** |
| Threshold | 0.7242 | 0.7242 |

**Quality gate:** aprovado em todos os criterios (min AUC-PR 0.80, min F1 0.70, max FP/h 10).

Tamanho dos artefatos exportados:

| Formato | Tamanho |
|---|---|
| float32.tflite | 66.2 KB |
| float16.tflite | 37.9 KB |
| int8.tflite | 25.7 KB |

Uso de recursos no ESP32 (firmware float32, compilado com huge_app.csv):

| Recurso | Usado | Total | % |
|---|---|---|---|
| RAM | 101.648 bytes | 327.680 bytes | 31.0% |
| Flash | 638.633 bytes | 3.145.728 bytes | 20.3% |

---

## Comparativo de Modelos

| Modelo | AUC-PR teste | F1 teste | Precision | Recall | FP/h |
|---|---:|---:|---:|---:|---:|
| Optuna Tiny CNN v4 (externo) | 0.9127 | 0.8526 | 0.8982 | 0.8114 | 4.90 |
| **Tiny CNN (atual)** | **0.8775** | **0.8109** | **0.8431** | **0.7810** | **6.75** |
| Tiny CNN (baseline) | 0.8982 | 0.7951 | 0.7310 | 0.8716 | 16.94 |
| Tiny TCN | 0.8964 | 0.7666 | 0.6790 | 0.8801 | 21.98 |
| Optuna Random Forest v4 | 0.8127 | 0.7367 | 0.7974 | 0.6846 | 9.26 |
| STA/LTA (baseline tradicional) | 0.1662 | 0.2760 | 0.1773 | 0.6230 | — |

---

## Arquitetura do Sistema

```
dados brutos
  -> adapter de dominio
  -> dataset generico (contrato NPZ)
  -> validacao de contrato
  -> treinamento e comparacao de modelos (MLflow)
  -> selecao de candidato
  -> quality gate
  -> exportacao edge (TFLite / header C)
  -> pacote OTA (manifesto + HMAC-SHA256)
  -> validacao e rollback
  -> firmware ESP32 (TFLite Micro)
```

### Estrutura de diretorios

```
src/
  core/          # settings, schemas e profiles
  data/          # validacao de dataset e adapters de dominio
  features/      # features estatisticas e espectrais
  training/      # treino, HPO, avaliacao e selecao de modelos
  export/        # exportacao TFLite e header C/C++
  mlops/         # quality gate e promocao de modelo
  monitoring/    # drift, politica de retreino e decisao drift -> OTA
  ota/           # fluxo OTA completo com assinatura HMAC
  reports/       # relatorio final automatico
  tests/         # smoke tests e testes de comunicacao

server/
  app/
    routes/      # endpoints FastAPI (devices, metrics)

config/
  configs/       # configuracao global e profiles
  model/         # catalogo de modelos (models.yaml)

docs/
  chapters/      # relatorio tecnico
  results/       # snapshots de resultados
  guia_ferramentas.md  # guia de uso de cada ferramenta

PlatformIO/
  Projects/TCC/
    src/main.cpp         # firmware ESP32
    include/             # model_config.h, preprocessing.h, modelos .h
    platformio.ini

monitoring/
  prometheus/
  grafana/dashboards/
```

---

## Contrato do Dataset

O pipeline espera arquivos `.npz` com:

```
X_train, y_train   # janelas de treino e labels
X_val,   y_val     # validacao
X_test,  y_test    # teste
```

- `X_*`: janelas de series temporais com tamanho fixo
- `y_*`: labels binarios — `0 = normal`, `1 = anomalo`
- Profile sismico atual: 800 amostras por janela (20 s a 40 Hz, overlap 50%)

---

## Preprocessamento Edge-Aware

O preprocessamento foi projetado para ser reproduzivel no microcontrolador:

```
resample 40 Hz
-> detrend linear
-> demean
-> taper 5%
-> bandpass 0.5-15 Hz (zerophase)
-> zscore por janela
```

`remove_response` foi excluido intencionalmente: depende de StationXML e resposta instrumental, inviavel em firmware.

---

## Modelos Suportados

| Familia | Exemplos | Uso |
|---|---|---|
| Classicos supervisionados | Random Forest, Extra Trees, Logistic Regression | Baselines |
| Classicos nao supervisionados | Isolation Forest | Cenarios com poucos labels |
| Redes leves | Tiny CNN, Tiny TCN, LSTM | Candidatos TinyML |
| Autoencoders | Dense AE, CNN AE | Deteccao por reconstrucao |

Configuracao em `config/model/models.yaml`. Candidato atual: `tiny_cnn` com `head_pooling: avg` (compativel com TFLite Micro int8).

---

## Features Tabulares (modelos classicos)

```
mean, std, min, max, median, abs_mean, abs_peak, rms,
crest_factor, peak_to_peak, energy, skewness, kurtosis,
percentis, iqr, zero_crossings, zero_crossing_rate,
dominant_freq, spectral_centroid, spectral_rolloff_85,
bandpower_0_3hz, bandpower_0p5_3hz, bandpower_3_8hz,
bandpower_8_15hz, spectral_entropy
```

---

## Drift Detection

Resultado validado no dataset sismico:

```
Drift level: high
Max z-shift: 0.0176
Max PSI: 0.3463
Min KS p-value: 0.000033
Politica: retrain_recommended
Decisao OTA: build_and_publish_ota
```

Interpretacao: medias globais pouco deslocadas, mas distribuicao interna mudou (PSI > 0.2, KS significativo). Sistema recomenda retreino e so libera OTA quando candidato passa o quality gate.

---

## OTA Simulado

Fluxo completo com assinatura HMAC-SHA256:

```
production_manifest.json
  -> ota_manifest.json
  -> pacote (artifact.tflite + signature.json)
  -> validation_report.json  (SHA-256 + assinatura verificados)
  -> releases/latest.json    (publicacao local)
  -> device_update_check_report.json
  -> install_report.json
  -> rollback_report.json
```

Versao atual publicada: `seismic_edge_v1_tiny_cnn_20260614`
SHA-256: `f71ac3b1ec6b9207e459176f1bb86623ae23ed9573b51a1c9c4d15d6987c6bc0`

---

## Monitoramento (Prometheus + Grafana)

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Metricas FastAPI: http://localhost:8000/metrics

---

## Como Rodar o Projeto

### 1. Ambiente

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Validar dataset

```bash
python -m src.data.validate_dataset
```

### 3. Treinar modelos

```bash
python -m src.training.train_all --models-cfg config/model/models.yaml
```

### 4. Promover candidato

```bash
python -m src.mlops.promote_model
```

### 5. Exportar TFLite

```bash
python -m src.export.export_tflite
```

### 6. Drift

```bash
python -m src.monitoring.build_drift_reference
python -m src.monitoring.check_data_drift --dataset data/dataset.npz --split test
python -m src.monitoring.retrain_policy
python -m src.monitoring.drift_to_ota_decision
```

### 7. OTA simulado

```bash
python -m src.ota.build_ota_manifest
python -m src.ota.build_ota_package
python -m src.ota.validate_ota_package
python -m src.ota.publish_local_release
python -m src.ota.simulate_device_update_check
python -m src.ota.simulate_apply_update
python -m src.ota.simulate_rollback
```

### 8. Firmware ESP32

```bash
cd PlatformIO/Projects/TCC
PLATFORMIO_RUN_JOBS=1 pio run -e esp32dev -t upload
pio device monitor --port /dev/ttyUSB0 --baud 115200
```

### 9. MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///artefacts/mlruns/mlflow.db
```

### 10. Testes

```bash
pytest
```

> Guia detalhado de cada ferramenta: [docs/guia_ferramentas.md](docs/guia_ferramentas.md)

---

## DVC

Pipeline smoke (validacao de comunicacao entre etapas):

```bash
dvc repro
```

Pipeline real (dataset sismico, separado para nao rodar acidentalmente):

```bash
dvc repro -f dvc_real.yaml
```

---

## Proximos Passos

- Servidor HTTP para OTA real (ESP32 baixa modelo via endpoint)
- Correcao do modelo int8: substituir `GlobalMaxPooling` por `GlobalAveragePooling` e reexportar
- Dataset multivariado (branch separada em desenvolvimento)
- Assinatura criptografica com chave assimetrica
- Device registry para multiplos dispositivos
- Artigo de sistema TinyML/MLOps

---

## Tese Tecnica

Um pipeline generico de TinyML/MLOps para series temporais pode reduzir custo de transmissao, armazenamento e energia ao executar uma primeira camada de decisao diretamente na borda. No estudo de caso sismico, o modelo `tiny_cnn` com 15.377 parametros atingiu AUC-PR 0.877 no conjunto de teste, passou pelo quality gate automatico, foi exportado para TFLite, gravado em um ESP32 real e validou inferencia embarcada. O fluxo OTA simulado com assinatura HMAC-SHA256 completa o ciclo MLOps de ponta a ponta.
