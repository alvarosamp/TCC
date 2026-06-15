# TCC - Pipeline Generico de TinyML/MLOps para Series Temporais Complexas

Este repositorio organiza um projeto de TCC voltado ao desenvolvimento de um pipeline generico de TinyML/MLOps para deteccao de anomalias em series temporais complexas. O primeiro estudo de caso utiliza dados sismicos, mas a arquitetura foi pensada para ser reaplicada em outros sinais, como vibracao industrial, corrente eletrica, audio, telemetria e sensores distribuidos.

A ideia central e usar um modelo leve na borda como uma camada de triagem inteligente: o dispositivo processa janelas locais do sinal e so encaminha dados quando ha indicio confiavel de anomalia. Isso pode reduzir transmissao, armazenamento, consumo de energia e custo operacional.

## Visao Do Projeto

O projeto nao e apenas um treino de modelo. Ele busca representar um sistema de IA completo:

```text
dados brutos
  -> adapter de dominio
  -> dataset generico
  -> validacao de contrato
  -> treinamento e comparacao de modelos
  -> selecao de candidato
  -> quality gate
  -> exportacao edge
  -> pacote OTA simulado
  -> validacao e rollback
```

O objetivo tecnico e aproximar pesquisa, engenharia de software, MLOps e sistemas embarcados em um mesmo fluxo rastreavel.

## Principais Contribuicoes

- Pipeline generico para deteccao de anomalias em series temporais.
- Separacao entre adapter de dominio e nucleo generico de ML.
- Preprocessamento edge-aware para reduzir training-serving skew.
- Comparacao entre modelos classicos, redes neurais leves e autoencoders.
- Configuracao de modelos por YAML, com presets vindos de Optuna.
- Avaliacao com AUC-PR, AUC-ROC, F1, precision, recall e falsos positivos por hora.
- Manifestos JSON para rastreabilidade de candidato, producao e OTA.
- Quality gate para impedir promocao automatica de modelos ruins.
- Monitoramento de drift com referencia estatistica, relatorio de drift, politica de retreino e decisao drift -> OTA.
- Fluxo OTA simulado com manifesto, pacote, validacao SHA-256, publicacao local, check de dispositivo, instalacao simulada e rollback.
- Exportacao TFLite/TFLite Micro com header C/C++ e build embarcada validada para ESP32.

## Arquitetura

```text
src/
  core/          # settings, schemas e profiles
  data/          # validacao de dataset e adapters de dominio
  features/      # features estatisticas e espectrais
  training/      # treino, HPO, avaliacao e selecao de modelos
  export/        # exportacao TFLite e header C/C++
  mlops/         # quality gate e promocao de modelo
  monitoring/    # drift, politica de retreino e decisao drift -> OTA
  ota/           # fluxo OTA simulado
  tests/         # testes de comunicacao e smoke tests

config/
  configs/       # configuracao global e profiles
  model/         # catalogo de modelos e smoke config

docs/
  chapters/      # relatorio tecnico por capitulos
  results/       # snapshots de resultados

artefacts/       # saidas locais do pipeline; nao deve ser usado como fonte versionada pesada
data/            # datasets locais ou controlados por DVC
```

## Contrato De Dataset

O nucleo generico espera arquivos `.npz` com o contrato:

```text
X_train, y_train
X_val,   y_val
X_test,  y_test
```

Onde:

- `X_*`: janelas de series temporais com tamanho fixo.
- `y_*`: labels binarios, com `0 = normal` e `1 = anomalo`.
- No profile atual, cada janela possui `800` amostras, representando `20 s @ 40 Hz`.

## Caso Sismico Atual

O primeiro dominio usa dados MiniSEED:

```text
raw/events/      -> anomalo
raw/continuous/  -> normal
```

O pipeline edge-aware evita `remove_response`, pois essa etapa depende de StationXML e resposta instrumental, algo dificil de reproduzir em um microcontrolador. O preprocessamento usado no profile sismico e:

```text
resample 40 Hz
-> detrend linear
-> demean
-> taper 5%
-> bandpass 0.5-15 Hz
-> zscore por janela
```

Essa decisao torna o treinamento mais proximo do que pode ser implementado em firmware.

## Modelos

Os modelos sao configurados em `config/model/models.yaml`.

Familias suportadas:

| Familia | Exemplos | Uso |
|---|---|---|
| Classicos supervisionados | Random Forest, Extra Trees, Logistic Regression | Baselines e interpretabilidade |
| Classicos nao supervisionados | Isolation Forest | Cenarios com poucos labels |
| Redes leves | Tiny CNN, Tiny TCN, LSTM | Candidatos TinyML |
| Autoencoders | Dense AE, CNN AE | Deteccao por erro de reconstrucao |

No estado atual, o `tiny_tcn` esta habilitado como candidato principal, com hiperparametros vindos de uma rodada Optuna de 60 trials.

## Parametros Atuais Do Tiny TCN

```yaml
tiny_tcn:
  enabled: true
  family: neural_classifier
  edge_candidate: true
  export_tflite: true
  priority: candidate
  params:
    batch_size: 64
    pos_multiplier: 1.274375423547668
    filters: 24
    kernel_size: 11
    n_blocks: 3
    dilation_base: 2
    dropout: 0.03544702294960163
    spatial_dropout: 0.1142415531888814
    dense_units: 32
    learning_rate: 0.0019901844880576103
    l2_reg: 0.0000008596266772391992
    head_pooling: avg
    label_smoothing: 0.012938050702971032
    padding: same
    conv_type: separable
    use_batch_norm: false
    epochs: 80
    patience: 12
```

Resultado de validacao da busca Optuna:

```text
best_value     = 0.9671
val_auc_pr     = 0.9036
val_f1         = 0.8406
val_precision  = 0.8814
val_recall     = 0.8035
val_fp_per_hour = 49.81
params         = 5273
```

Observacao: o treino final precisa ser executado e comparado no teste real antes de conclusoes finais.

## Features Tabulares

Para modelos classicos, o projeto extrai features genericas por janela, incluindo:

```text
mean, std, min, max, median, abs_mean, abs_peak, rms,
crest_factor, peak_to_peak, energy, skewness, kurtosis,
percentis, iqr, zero_crossings, zero_crossing_rate,
dominant_freq, spectral_centroid, spectral_rolloff_85,
bandpower_0_3hz, bandpower_0p5_3hz, bandpower_3_8hz,
bandpower_8_15hz, spectral_entropy
```

As features espectrais apareceram como relevantes nos experimentos com arvores.

## Metricas

A metrica primaria e AUC-PR, pois deteccao de anomalias e normalmente desbalanceada.

Metricas adicionais:

- AUC-ROC
- F1
- precision
- recall
- falsos positivos por hora (`FP/h`)
- tamanho do modelo
- contagem de parametros
- threshold escolhido na validacao

O threshold e escolhido no conjunto de validacao e aplicado no teste para reduzir risco de otimizar diretamente no conjunto final.

## Fluxo De Treinamento

Comando principal:

```bash
python -m src.training.train_all --models-cfg config/model/models.yaml
```

Fluxo interno:

```text
carrega profile
-> carrega dataset NPZ
-> extrai features tabulares
-> treina modelos enabled=true
-> avalia validacao e teste
-> registra no MLflow
-> gera model_comparison
-> escolhe candidato edge
-> salva candidate_manifest.json
```

Saidas principais:

```text
artefacts/models/<modelo>.keras ou .joblib
artefacts/reports/<modelo>_metrics.json
artefacts/reports/model_comparison.csv
artefacts/reports/model_comparison.md
artefacts/reports/model_comparison.json
artefacts/reports/candidate_manifest.json
```

## DVC

O pipeline DVC atual funciona como smoke/reproducao local:

```text
generate_data
-> validate_dataset
-> train_all
-> export_tflite
```

Comando:

```bash
dvc repro
```

Atencao: o `dvc.yaml` atual ainda usa `scripts/generate_synthetic_data.py` e `models_smoke.yaml`. Portanto, ele valida a comunicacao entre etapas, mas nao representa o experimento cientifico final com o dataset real.

## MLflow

O pipeline registra parametros, metricas e artefatos no MLflow.

```bash
mlflow ui --backend-store-uri sqlite:///artefacts/mlruns/mlflow.db
```

## Quality Gate

O quality gate fica em:

```text
src/mlops/promote_model.py
```

Ele le:

```text
artefacts/reports/candidate_manifest.json
```

E gera:

```text
artefacts/reports/promotion_report.json
artefacts/registry/production_manifest.json
```

Regras configuradas em `config/configs/config.yaml`:

```yaml
quality_gate:
  primary_metric: auc_pr
  min_auc_pr: 0.80
  min_f1: 0.70
  max_fp_per_hour: 10
  max_val_test_auc_pr_gap: 0.08
  max_model_size_kb: 300
```

Comando:

```bash
python -m src.mlops.promote_model
```

## OTA Simulado

O projeto implementa um fluxo OTA simulado para conectar MLOps e dispositivo de borda.

Arquivos principais:

```text
src/ota/build_ota_manifest.py
src/ota/build_ota_package.py
src/ota/validate_ota_package.py
src/ota/publish_local_release.py
src/ota/simulate_device_update_check.py
src/ota/simulate_apply_update.py
src/ota/simulate_rollback.py
```

Fluxo:

```text
production_manifest.json
-> ota_manifest.json
-> pacote OTA local
-> validation_report.json
-> releases/latest.json
-> device_update_check_report.json
-> install_report.json
-> rollback_report.json
```

Comandos:

```bash
python -m src.ota.build_ota_manifest
python -m src.ota.build_ota_package
python -m src.ota.validate_ota_package
python -m src.ota.publish_local_release
python -m src.ota.simulate_device_update_check
python -m src.ota.simulate_apply_update
python -m src.ota.simulate_rollback
```

Estado atual: o pacote OTA usa o artefato `.tflite` int8 gerado pela etapa de exportacao edge. O header `.h` tambem e gerado para compilacao direta com TensorFlow Lite Micro.

## Exportacao Edge

Script:

```text
src/export/export_tflite.py
```

Comando:

```bash
python -m src.export.export_tflite
```

Saidas esperadas:

```text
artefacts/edge/<modelo>_float32.tflite
artefacts/edge/<modelo>_float16.tflite
artefacts/edge/<modelo>_int8.tflite
artefacts/edge/<modelo>_int8.h
artefacts/edge/<modelo>_export_manifest.json
```

A exportacao int8 usa representative dataset a partir do split de treino.

## Monitoramento De Drift

O projeto possui um ciclo de drift integrado ao fluxo MLOps:

```bash
python -m src.monitoring.build_drift_reference
python -m src.monitoring.check_data_drift --dataset <dataset.npz> --split test
python -m src.monitoring.retrain_policy
python -m src.monitoring.drift_to_ota_decision
```

Saidas principais:

```text
artefacts/monitoring/drift_reference.json
artefacts/monitoring/drift_report.json
artefacts/monitoring/retrain_policy.json
artefacts/monitoring/ota_decision.json
```

Resultado validado no dataset real:

```text
Drift level: high
Max z-shift: 0.0176
Max PSI: 0.3463
Min KS p-value: 0.000033
Politica: retrain_recommended
Decisao OTA apos quality gate: build_and_publish_ota
```

A interpretacao tecnica e que as medias globais das features mudaram pouco, mas a distribuicao interna mudou de forma relevante, detectada por PSI e KS. Por isso o sistema recomenda retreino e so libera OTA quando existe candidato aprovado no quality gate.

## Validacao Embarcada

O modelo aprovado foi exportado para TFLite int8 e convertido para header C/C++:

```text
artefacts/edge/tiny_cnn_int8.tflite
artefacts/edge/tiny_cnn_int8.h
artefacts/edge/tiny_cnn_export_manifest.json
```

O firmware PlatformIO foi configurado para usar:

```cpp
#define ACTIVE_MODEL MODEL_PIPELINE_TINY_CNN_INT8
#define MODEL_DATA tiny_cnn_int8_model_data
#define MODEL_THRESHOLD 0.7241942882537842f
```

Status de validacao:

```text
Build alvo ESP32: sucesso
RAM estatica: 33888 bytes, 10.3%
Flash: 597165 bytes, 19.0%
Upload real: pendente, pois a placa conectada foi identificada como ESP8266EX
```

Comando usado para confirmar a placa conectada:

```bash
python3 ~/.platformio/packages/tool-esptoolpy/esptool.py --port /dev/ttyUSB0 chip_id
```

Resultado obtido:

```text
Detecting chip type... ESP8266
Chip is ESP8266EX
```

Portanto, a validacao real de inferencia no hardware deve ser feita quando uma placa ESP32 estiver conectada. A adaptacao para ESP8266 nao e recomendada como caminho principal do TCC, porque o projeto usa TensorFlow Lite Micro e APIs/recursos planejados para ESP32.

## Como Rodar O Projeto

### 1. Criar ambiente

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Validar dataset

```bash
python -m src.data.validate_dataset
```

### 3. Treinar modelos configurados

```bash
python -m src.training.train_all --models-cfg config/model/models.yaml
```

### 4. Promover candidato

```bash
python -m src.mlops.promote_model
```

### 5. Exportar para TFLite

```bash
python -m src.export.export_tflite
```

### 6. Rodar fluxo OTA simulado

```bash
python -m src.ota.build_ota_manifest
python -m src.ota.build_ota_package
python -m src.ota.validate_ota_package
python -m src.ota.publish_local_release
python -m src.ota.simulate_device_update_check
python -m src.ota.simulate_apply_update
python -m src.ota.simulate_rollback
```

### 7. Rodar testes

```bash
pytest
```

## Resultados Externos Consolidados

Resultados de rodadas externas em `D:\PipelineGenerico\data`:

| Modelo | AUC-PR teste | F1 teste | Precision | Recall | FP/h | Observacao |
|---|---:|---:|---:|---:|---:|---|
| Optuna Tiny CNN classifier v4 | 0.9127 | 0.8526 | 0.8982 | 0.8114 | 4.896 | Melhor modelo completo anterior |
| Tiny CNN classifier | 0.8982 | 0.7951 | 0.7310 | 0.8716 | 16.944 | Boa rede leve |
| Tiny TCN classifier | 0.8964 | 0.7666 | 0.6790 | 0.8801 | 21.984 | Boa, mas com FP/h maior |
| Optuna Random Forest v4 | 0.8127 | 0.7367 | 0.7974 | 0.6846 | 9.264 | Melhor baseline classico completo |
| Optuna Extra Trees v4 | 0.7901 | 0.7102 | 0.7589 | 0.6675 | 11.296 | Forte, porem pesado |
| STA/LTA v4 | 0.1662 | 0.2760 | 0.1773 | 0.6230 | - | Baseline tradicional |

## O Que Ainda Falta Para Ficar 100%

Prioridade alta:

- Rodar a validacao real em uma placa ESP32 fisica: upload, monitor serial, latencia, memoria livre, score e acuracia nas janelas reais.
- Gerar um relatorio final consolidado juntando dataset, treino, drift, quality gate, exportacao edge, OTA e build embarcada.
- Atualizar os capitulos do TCC com metodologia, resultados, limitacoes e trabalhos futuros.

Prioridade media:

- Criar um stage DVC para treino real, separado do smoke test sintetico.
- Adicionar testes automatizados para os contratos de manifests: candidato, producao, drift, export e OTA.
- Padronizar encoding de documentos Markdown com acentos.
- Criar uma tabela comparativa final entre Tiny CNN, Tiny TCN, Random Forest e baselines tradicionais.

Prioridade futura:

- OTA real via servidor HTTP/local endpoint.
- Assinatura criptografica de manifestos ou pacotes.
- Device registry para multiplos dispositivos.
- Prediction drift e performance drift quando houver rotulos em producao.
- Artigo de sistema TinyML/MLOps.
- Artigo de comparacao de modelos edge.


## Melhorias Finais Implementadas

Alem do pipeline principal, o projeto agora inclui modulos finais de produto/MLOps:

```text
Assinatura OTA
  -> src/ota/sign_ota_package.py
  -> src/ota/crypto.py
  -> validation_report.json valida assinatura HMAC-SHA256

Device registry
  -> server/app/routes/devices.py
  -> registro, status, historico e checagem de compatibilidade do dispositivo

Prometheus/Grafana
  -> endpoint /metrics no FastAPI
  -> docker-compose.monitoring.yml
  -> monitoring/prometheus/prometheus.yml
  -> dashboard Grafana em monitoring/grafana/dashboards/

Prediction/performance drift
  -> src/monitoring/check_prediction_performance_drift.py
  -> usa eventos, scores, predicoes e feedback humano quando existir

DVC real
  -> dvc_real.yaml
  -> pipeline separado do smoke test para evitar rodadas longas acidentais

Relatorio final automatico
  -> src/reports/generate_final_report.py
  -> docs/results/final_report/relatorio_final.md
  -> docs/results/final_report/relatorio_final.html
```

O ESP32 real permanece como unica validacao fisica pendente, pois a placa conectada durante os testes foi identificada como ESP8266EX.

## Observacoes Sobre Versionamento

O Git deve versionar codigo, configs e documentacao. Artefatos pesados devem ficar fora do Git ou sob controle de DVC/storage externo.

Evitar versionar:

```text
*.npz
*.joblib
*.keras
*.tflite
*.h gerado de modelo, se muito grande
mlruns/
artefacts/models/
artefacts/edge/
artefacts/ota/packages/
artefacts/ota/releases/
__pycache__/
*.pyc
```

## Tese Tecnica

Um pipeline generico de TinyML/MLOps para series temporais pode reduzir custo de transmissao, armazenamento e energia ao executar uma primeira camada de decisao diretamente na borda. No estudo de caso sismico, modelos neurais leves e otimizados sao comparados com baselines classicos e regras tradicionais, enquanto o sistema completo fornece rastreabilidade, promocao controlada e simulacao de atualizacao OTA segura.
