# TCC — Pipeline Generico de TinyML/MLOps para Series Temporais

> Pipeline completo de deteccao de anomalias em series temporais com modelo leve na borda, ciclo MLOps rastreavel e atualizacao OTA real via HTTP no ESP32.

O primeiro estudo de caso usa dados **sismicos** (MiniSEED), mas a arquitetura e generica e foi projetada para ser reaplicada em outros dominios: vibracao industrial, corrente eletrica, audio, telemetria e sensores distribuidos.

**Ideia central:** o dispositivo de borda processa janelas locais do sinal e so encaminha dados quando ha indicio confiavel de anomalia — reduzindo transmissao, armazenamento, consumo de energia e custo operacional.

---

## Indice

1. [Status de Validacao](#status-de-validacao)
2. [Arquitetura Geral](#arquitetura-geral)
3. [Fluxo do Pipeline](#fluxo-do-pipeline)
4. [Fluxo OTA Real](#fluxo-ota-real-esp32--http)
5. [Resultados do Modelo](#resultados-do-modelo)
6. [Comparativo de Modelos](#comparativo-de-modelos)
7. [Estrutura do Repositorio](#estrutura-do-repositorio)
8. [Dataset e Preprocessamento](#dataset-e-preprocessamento)
9. [Drift Detection](#drift-detection)
10. [Quality Gate](#quality-gate)
11. [Monitoramento](#monitoramento-prometheus--grafana)
12. [Como Rodar](#como-rodar-o-projeto)
13. [Proximos Passos](#proximos-passos)

---

## Status de Validacao

| Componente | Status | Detalhe |
|---|:---:|---|
| Pipeline MLOps completo | ✅ | Treino, avaliacao, selecao, promocao |
| Quality gate | ✅ | Todos os criterios aprovados |
| Drift detection | ✅ | PSI, KS e z-shift validados |
| Exportacao TFLite | ✅ | float32, float16, int8 gerados |
| Geracao de header C/C++ | ✅ | Pronto para compilacao embarcada |
| Build embarcada ESP32 | ✅ | RAM 31% / Flash 20% |
| Flash real no ESP32 | ✅ | Chip: ESP32-D0WD-V3 |
| Inferencia embarcada float32 | ✅ | Executando no hardware real |
| OTA simulado (HMAC-SHA256) | ✅ | Manifesto, pacote, assinatura, rollback |
| OTA real via HTTP | ✅ | Firmware baixa modelo do servidor FastAPI |
| Inferencia embarcada int8 | ⚠️ | REDUCE_MAX incompativel — correcao planejada |

> **Nota int8:** o modelo usava `GlobalMaxPooling` que vira `REDUCE_MAX` quantizado. O TFLite Micro exige `input_zp == output_zp` nessa op. Correcao: substituir por `GlobalAveragePooling` (`head_pooling: avg` ja esta no YAML) e retreinar. Nao invalida o pipeline — e resultado tecnico documentado.

---

## Arquitetura Geral

```
                        PIPELINE TINYML/MLOPS
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  DADOS          ENGENHARIA        MODELOS         PRODUCAO      │
  │                                                                 │
  │  MiniSEED  ──►  Adapter     ──►  Treino    ──►  Quality Gate   │
  │  (sismico)      Dominio          MLflow         (AUC-PR, F1,   │
  │                    │             Optuna          FP/h, gap)     │
  │                    ▼                │                │          │
  │              Dataset NPZ            ▼                ▼          │
  │              (contrato         Comparacao       Exportacao      │
  │               padrao)          de modelos       TFLite          │
  │                                     │           (float32/int8)  │
  │                                     ▼                │          │
  │                              Candidato Edge           ▼         │
  │                              (tiny_cnn)          OTA Package    │
  │                                                  HMAC-SHA256    │
  │                                                       │         │
  └───────────────────────────────────────────────────────┼─────────┘
                                                          │
                          HTTP                            │
  ┌──────────────────────────────────────┐                │
  │         SERVIDOR FASAPI              │◄───────────────┘
  │                                      │   publish_local_release
  │  GET /ota/latest  (latest.json)      │
  │  GET /ota/artifact (.tflite)         │
  │  POST /ota/report (resultado)        │
  │  GET /metrics     (Prometheus)       │
  │  GET /devices     (registry)         │
  └──────────────┬───────────────────────┘
                 │  WiFi / HTTP
                 ▼
  ┌──────────────────────────────────────┐
  │            ESP32-D0WD-V3             │
  │                                      │
  │  Boot ──► OTA check ──► Download     │
  │                │             │       │
  │           SHA-256 OK         │       │
  │                │             ▼       │
  │                └──► SPIFFS /ota_model│
  │                          │           │
  │                          ▼           │
  │               TFLite Micro           │
  │               Inferencia             │
  │               (800 amostras, 20s)    │
  │                          │           │
  │               score >= 0.724?        │
  │               anomalia detectada     │
  └──────────────────────────────────────┘
```

---

## Fluxo do Pipeline

```
  Dados Brutos (MiniSEED)
         │
         ▼
  ┌─────────────────┐
  │  Adapter de     │  raw/events/      → anomalo
  │  Dominio        │  raw/continuous/  → normal
  └────────┬────────┘
           │  resample 40Hz, detrend, demean,
           │  taper 5%, bandpass 0.5-15Hz, zscore
           ▼
  ┌─────────────────┐
  │  Dataset NPZ    │  X_train, X_val, X_test
  │  (contrato)     │  y_train, y_val, y_test
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Validacao de   │  valida shape, dtype, labels,
  │  Contrato       │  balanceamento e splits
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────────┐
  │  Treinamento (MLflow)                       │
  │                                             │
  │  tiny_cnn   tiny_tcn   random_forest  ...   │
  │      │                                      │
  │   Optuna HPO (60 trials)                    │
  │      │                                      │
  │  AUC-PR, F1, precision, recall, FP/h        │
  └──────────────────┬──────────────────────────┘
                     │ selecao do candidato edge
                     ▼
  ┌─────────────────┐
  │  Quality Gate   │  AUC-PR >= 0.80
  │                 │  F1 >= 0.70
  │  APROVADO ✅    │  FP/h <= 10
  │                 │  gap val-test <= 0.08
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Exportacao     │  tiny_cnn_float32.tflite  (66 KB)
  │  TFLite         │  tiny_cnn_float16.tflite  (38 KB)
  │                 │  tiny_cnn_int8.tflite     (26 KB)
  │                 │  tiny_cnn_int8.h          (header C)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────────┐
  │  Drift Detection                            │
  │                                             │
  │  z-shift: 0.0176   PSI: 0.3463             │
  │  KS p-value: 0.000033                       │
  │                                             │
  │  Nivel: HIGH → retrain_recommended          │
  │  Decisao: build_and_publish_ota             │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────────────────────────────────┐
  │  OTA Simulado                               │
  │                                             │
  │  production_manifest.json                   │
  │    → ota_manifest.json                      │
  │    → pacote (tflite + SHA-256)              │
  │    → assinatura HMAC-SHA256                 │
  │    → validation_report.json (todos OK)      │
  │    → releases/latest.json                   │
  └──────────────────┬──────────────────────────┘
                     │
                     ▼
  ┌─────────────────┐
  │  Servidor HTTP  │  FastAPI expoe /ota/latest
  │  FastAPI        │  e /ota/artifact
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  ESP32          │  OTA real: baixa, valida SHA-256,
  │  (hardware)     │  salva no SPIFFS, carrega em RAM,
  │                 │  executa inferencia com TFLite Micro
  └─────────────────┘
```

---

## Fluxo OTA Real (ESP32 + HTTP)

```
  ESP32 Boot
       │
       ▼
  SPIFFS.begin()          ← monta filesystem na flash interna
       │
       ▼
  wifi_connect()          ← conecta ao AP (timeout 15s)
       │
       ├── FALHA ──────────────────────────────────┐
       ▼                                           │
  GET /ota/latest         ← consulta servidor      │
  {                                                │
    "model": {                                     │
      "version": "seismic_edge_v1_...",            │
    },                                             │
    "artifact": {                                  │
      "sha256": "f71ac3..."                        │
    }                                              │
  }                                                │
       │                                           │
       ├── versao ja atual? ──► POST /ota/report   │
       │                        "skipped"          │
       │                                           │
       ▼                                           │
  GET /ota/artifact       ← streaming              │
  (512 bytes por vez)     ← para SPIFFS            │
       │                                           │
       ▼                                           │
  SHA-256 calculado                                │
  == SHA-256 do JSON?                              │
       │                                           │
       ├── NAO ──► SPIFFS.remove()                 │
       │           POST /ota/report "failed"       │
       ▼                                           │
  save_meta()             ← salva versao           │
  POST /ota/report        ← "success"              │
       │                                           │
       ▼                 ◄──────────────────────── ┘
  ota_load_model_into_ram()
       │
       ├── SPIFFS tem modelo? ──► s_model_buf = malloc(66KB)
       │                          model_data = s_model_buf
       │
       └── NAO ──────────────► model_data = MODEL_DATA (builtin)
                │
                ▼
  tflite::GetModel(model_data)
  AllocateTensors()
                │
                ▼
         loop() — inferencia continua
         (100 runs por rodada, saida CSV)
```

---

## Resultados do Modelo

**Modelo:** `tiny_cnn` | **Parametros:** 15.377 | **Dataset:** sismico (split por evento)

| Metrica | Validacao | Teste |
|---|---:|---:|
| AUC-PR | 0.8598 | **0.8775** |
| AUC-ROC | 0.9546 | **0.9576** |
| F1 | 0.7908 | **0.8109** |
| Precision | 0.8343 | **0.8431** |
| Recall | 0.7517 | **0.7810** |
| FP/h | — | **6.75** |
| Threshold | 0.7242 | 0.7242 |

**Quality gate:** aprovado — AUC-PR 0.877 ≥ 0.80 | F1 0.811 ≥ 0.70 | FP/h 6.75 ≤ 10 | gap val-test 0.018 ≤ 0.08

**Artefatos exportados:**

| Formato | Tamanho | Uso |
|---|---|---|
| float32.tflite | 66.2 KB | Inferencia embarcada atual |
| float16.tflite | 37.9 KB | Alternativa compacta |
| int8.tflite | 25.7 KB | Alvo final (correcao REDUCE_MAX pendente) |

**Uso de recursos no ESP32** (firmware compilado, `huge_app.csv`):

```
RAM   ████████████░░░░░░░░░░░░░░░░░░  31.0%  (101 KB / 320 KB)
Flash ████░░░░░░░░░░░░░░░░░░░░░░░░░░  20.3%  (638 KB / 3072 KB)
```

---

## Comparativo de Modelos

| Modelo | AUC-PR | F1 | Precision | Recall | FP/h | Params |
|---|---:|---:|---:|---:|---:|---:|
| Optuna Tiny CNN v4 (ref. externa) | 0.9127 | 0.8526 | 0.8982 | 0.8114 | 4.90 | — |
| **Tiny CNN (atual — aprovado)** | **0.8775** | **0.8109** | **0.8431** | **0.7810** | **6.75** | **15.377** |
| Tiny CNN (baseline sem HPO) | 0.8982 | 0.7951 | 0.7310 | 0.8716 | 16.94 | — |
| Tiny TCN | 0.8964 | 0.7666 | 0.6790 | 0.8801 | 21.98 | — |
| Optuna Random Forest v4 | 0.8127 | 0.7367 | 0.7974 | 0.6846 | 9.26 | — |
| Optuna Extra Trees v4 | 0.7901 | 0.7102 | 0.7589 | 0.6675 | 11.30 | — |
| STA/LTA (baseline tradicional) | 0.1662 | 0.2760 | 0.1773 | 0.6230 | — | — |

> Metrica primaria: AUC-PR (deteccao de anomalia e desbalanceada — normal >> anomalo).

---

## Estrutura do Repositorio

```
TCC/
│
├── src/                          # Codigo Python do pipeline
│   ├── core/                     # Settings, schemas, profiles
│   ├── data/                     # Validacao de dataset, adapters
│   ├── features/                 # Features estatisticas e espectrais
│   ├── training/                 # Treino, HPO (Optuna), avaliacao
│   ├── export/                   # Exportacao TFLite e header C
│   ├── mlops/                    # Quality gate e promocao
│   ├── monitoring/               # Drift, retrain policy, OTA decision
│   ├── ota/                      # Fluxo OTA simulado
│   └── reports/                  # Relatorio final automatico
│
├── server/                       # Servidor FastAPI
│   └── app/
│       ├── main.py               # App + endpoints raiz e /metrics
│       ├── metrics.py            # Prometheus exporter
│       ├── storage.py            # Persistencia em JSON
│       └── routes/
│           ├── ota.py            # GET /ota/latest, /artifact, POST /report
│           ├── devices.py        # Registry de dispositivos
│           ├── events.py         # Eventos de inferencia
│           └── feedback.py       # Feedback humano
│
├── PlatformIO/Projects/TCC/      # Firmware ESP32
│   ├── src/
│   │   ├── main.cpp              # Loop principal + OTA no setup()
│   │   └── ota_http.cpp          # WiFi, HTTP, SPIFFS, SHA-256
│   ├── include/
│   │   ├── ota_http.h            # Interface do modulo OTA
│   │   ├── model_config.h        # Selecao de modelo compilado
│   │   ├── preprocessing.h       # Pipeline de borda (detrend/bandpass/zscore)
│   │   ├── tiny_cnn_float32.h    # Modelo compilado (fallback OTA)
│   │   └── wifi_config.h.template# Template de credenciais WiFi
│   └── platformio.ini
│
├── config/
│   ├── configs/                  # config.yaml global e profiles
│   └── model/
│       └── models.yaml           # Catalogo de modelos (enable/disable/HPO)
│
├── artefacts/                    # Saidas do pipeline (nao versionadas)
│   ├── models/                   # .keras e .joblib treinados
│   ├── reports/                  # Metricas, comparativos, manifestos
│   ├── edge/                     # .tflite e .h exportados
│   ├── ota/                      # Pacotes e releases OTA
│   ├── monitoring/               # Drift e decisao OTA
│   └── registry/                 # Manifesto de producao
│
├── docs/
│   ├── guia_ferramentas.md       # Como usar MLflow, DVC, Prometheus...
│   ├── explicacao_codigo_ota.md  # Explicacao detalhada do firmware OTA
│   ├── slides_tcc.pptx           # Slides de apresentacao
│   └── results/                  # Snapshots de resultados
│
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/dashboards/
│
├── docker-compose.monitoring.yml
├── dvc.yaml                      # Pipeline smoke (sintetico)
├── dvc_real.yaml                 # Pipeline real (dataset sismico)
└── requirements.txt
```

---

## Dataset e Preprocessamento

### Origem dos Dados

```
raw/
  events/      → arquivos MiniSEED com eventos sismicos  (anomalo, y=1)
  continuous/  → registros continuos sem evento          (normal,  y=0)
```

### Janelamento

```
Sinal continuo (horas/dias)
│
├──────────────────────────────────────────────────────────
│   janela 0    │   janela 1    │   janela 2    │  ...
│  800 amostras │  800 amostras │  800 amostras │
│    (20 s)     │    (20 s)     │    (20 s)     │
│←── step 10s ──►
└──────────────────────────────────────────────────────────
                overlap = 50%
```

### Preprocessamento Edge-Aware

```
Sinal bruto (40 Hz)
      │
      ▼  resample para 40 Hz
      │
      ▼  detrend linear       ← remove tendencia de longo prazo
      │
      ▼  demean               ← remove media DC
      │
      ▼  taper 5%             ← suaviza bordas (evita efeito Gibbs)
      │
      ▼  bandpass 0.5-15 Hz   ← filtra ruido e frequencias nao relevantes
      │  (zerophase)
      │
      ▼  zscore por janela    ← normaliza amplitude para media=0, std=1
      │
      ▼
  norm_window[800] — entrada do modelo
```

> `remove_response` foi excluido: depende de StationXML + resposta instrumental, inviavel em microcontrolador.

### Contrato do Dataset

```python
# Arquivo .npz esperado pelo pipeline
X_train.shape = (n_janelas, 800)   # float32
y_train.shape = (n_janelas,)       # int {0, 1}
X_val, y_val   # idem
X_test, y_test # idem
```

---

## Drift Detection

```
  Dataset de Referencia (treino)         Dataset de Producao (teste)
         │                                         │
         └──────────────┬──────────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  Metricas de Drift  │
              │                     │
              │  z-shift:  0.0176   │  ← deslocamento de media
              │  PSI:      0.3463   │  ← mudanca de distribuicao
              │  KS p-val: 0.000033 │  ← significancia estatistica
              └─────────────────────┘
                        │
                        ▼
              PSI > 0.2 → drift significativo
                        │
                        ▼
              ┌─────────────────────┐
              │  Politica           │  retrain_recommended
              │  Decisao OTA        │  build_and_publish_ota
              └─────────────────────┘
                        │
                        ▼
              Candidato novo passa quality gate?
                  SIM → publica OTA
                  NAO → aguarda novo treino
```

| Metrica | Valor | Interpretacao |
|---|---|---|
| z-shift (max) | 0.0176 | Media global pouco deslocada |
| PSI (max) | 0.3463 | Distribuicao interna mudou (> 0.2 = alto) |
| KS p-value (min) | 0.000033 | Diferenca estatisticamente significativa |

---

## Quality Gate

Criterios configurados em `config/configs/config.yaml`:

```yaml
quality_gate:
  primary_metric: auc_pr
  min_auc_pr:            0.80   # AUC-PR minima
  min_f1:                0.70   # F1 minimo
  max_fp_per_hour:       10.0   # Falsos positivos por hora
  max_val_test_auc_pr_gap: 0.08 # Limite de overfitting val-test
  max_model_size_kb:    300.0   # Tamanho maximo do modelo
```

Resultado do candidato atual:

```
✅ AUC-PR:       0.877  (>= 0.80)
✅ F1:           0.811  (>= 0.70)
✅ FP/h:         6.75   (<= 10.0)
✅ gap val-test: 0.018  (<= 0.08)
✅ tamanho:      25.7KB (<= 300KB)
```

---

## Monitoramento (Prometheus + Grafana)

```
  FastAPI (:8000)
       │
       │  /metrics (formato Prometheus)
       ▼
  Prometheus (:9090)
       │
       │  scrape a cada 15s
       ▼
  Grafana (:3000)
  (dashboards de metricas do servidor e dispositivos)
```

```bash
# Subir stack completa
docker compose -f docker-compose.monitoring.yml up -d

# Acessar
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin)
# Metricas:   http://localhost:8000/metrics
```

---

## Como Rodar o Projeto

### Ambiente

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# ou .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Pipeline completo (passo a passo)

```bash
# 1. Validar dataset
python -m src.data.validate_dataset

# 2. Treinar todos os modelos habilitados
python -m src.training.train_all --models-cfg config/model/models.yaml

# 3. Promover candidato (quality gate)
python -m src.mlops.promote_model

# 4. Exportar para TFLite (float32, float16, int8)
python -m src.export.export_tflite

# 5. Drift detection
python -m src.monitoring.build_drift_reference
python -m src.monitoring.check_data_drift --dataset data/dataset.npz --split test
python -m src.monitoring.retrain_policy
python -m src.monitoring.drift_to_ota_decision

# 6. OTA simulado
python -m src.ota.build_ota_manifest
python -m src.ota.build_ota_package
python -m src.ota.validate_ota_package
python -m src.ota.publish_local_release
python -m src.ota.simulate_device_update_check
python -m src.ota.simulate_apply_update
python -m src.ota.simulate_rollback

# 7. Servidor HTTP (OTA real)
uvicorn server.app.main:app --host 0.0.0.0 --port 8000

# 8. Firmware ESP32
#    Copie e preencha as credenciais:
cp PlatformIO/Projects/TCC/include/wifi_config.h.template \
   PlatformIO/Projects/TCC/include/wifi_config.h
#    Edite: WIFI_SSID, WIFI_PASSWORD, OTA_SERVER_HOST

cd PlatformIO/Projects/TCC
PLATFORMIO_RUN_JOBS=1 pio run -e esp32dev -t upload
pio device monitor --port /dev/ttyUSB0 --baud 115200
```

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///artefacts/mlruns/mlflow.db
# http://localhost:5000
```

### DVC

```bash
dvc repro            # pipeline smoke (sintetico, rapido)
dvc repro -f dvc_real.yaml   # pipeline real (dataset sismico)
```

### Testes

```bash
pytest
```

> Guia completo de cada ferramenta: [`docs/guia_ferramentas.md`](docs/guia_ferramentas.md)
> Explicacao detalhada do codigo OTA: [`docs/explicacao_codigo_ota.md`](docs/explicacao_codigo_ota.md)

---

## Saida Serial Esperada (ESP32)

```
[OTA] Iniciando verificacao de atualizacao...
[WiFi] Conectando a 'MinhaRede'...
[WiFi] IP: 192.168.1.107
[OTA] GET http://192.168.1.100:8000/ota/latest
[OTA] Nova versao disponivel — iniciando download...
[OTA] Baixados: 67768 bytes
[OTA] SHA-256 OK.
[OTA] POST /ota/report -> HTTP 200
[OTA] Usando modelo do SPIFFS (OTA).
================================
Modelo ativo : PIPELINE_TINY_CNN_FLOAT32
Threshold    : 0.72419429
================================
run,score,pred,expected_label,correct,preprocess_ms,...
0,0.823451,1,1,1,2.1200,8.4300,...
1,0.102341,0,0,1,2.1100,8.4100,...
...
# summary,inference_ms,8.430000,8.100000,8.900000
# summary,accuracy,0.850000,17,20
```

---

## Proximos Passos

| Prioridade | Item |
|---|---|
| Alta | Corrigir int8: retreinar com `head_pooling: avg` ja configurado no YAML |
| Alta | Validar inferencia int8 real no ESP32 (tensor arena cabe em 100 KB) |
| Media | Series temporais multivariadas (branch em desenvolvimento) |
| Media | Reset automatico pos-OTA (`ESP.restart()`) |
| Media | Rollback automatico se `AllocateTensors()` falhar com modelo OTA |
| Baixa | Assinatura RSA/ECDSA no firmware (alem do HMAC-SHA256 atual) |
| Baixa | OTA de firmware completo (`esp_ota_ops.h`, particoes ota_0/ota_1) |
| Futura | Artigo de sistema TinyML/MLOps |

---

## Tese Tecnica

Um pipeline generico de TinyML/MLOps para series temporais pode reduzir custo de transmissao, armazenamento e energia ao executar uma camada de decisao diretamente na borda. No estudo de caso sismico:

- O modelo `tiny_cnn` com **15.377 parametros** atingiu **AUC-PR 0.877** no conjunto de teste
- Passou pelo **quality gate automatico** (5 criterios)
- Foi **exportado para TFLite** em 3 formatos de quantizacao
- Foi **gravado e executado em um ESP32 real** (chip ESP32-D0WD-V3)
- O **fluxo OTA via HTTP** foi implementado: ESP32 consulta servidor, baixa modelo, valida SHA-256, salva no SPIFFS e carrega em RAM
- O ciclo **drift → retreino → quality gate → OTA** fecha o loop MLOps de ponta a ponta
