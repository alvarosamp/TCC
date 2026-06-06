# Relatorio Tecnico Do Projeto

Este documento consolida os capitulos tecnicos implementados no projeto. A ideia e manter um registro claro do que foi construido, por que foi construido e quais arquivos foram gerados ou alterados.

## Capitulo 1 - Pipeline Generico De TinyML

### Objetivo

Construir uma estrutura generica para deteccao de anomalias em series temporais complexas, inicialmente aplicada a dados sismicos, mas preparada para outros dominios como vibracao, audio, corrente eletrica e telemetria industrial.

### O Que Foi Feito

- Definicao da ideia central do projeto: TinyML como organizador inteligente de dados na borda.
- Separacao entre adapter de dominio e nucleo generico de machine learning.
- Definicao do contrato generico de dataset:
  - `X_train`, `y_train`
  - `X_val`, `y_val`
  - `X_test`, `y_test`
- Uso de profile para conectar dataset, treino, metricas, exportacao e edge.

### Arquivos Principais

- `config/configs/config.yaml`
- `config/configs/profiles/seismic_edge_v1.yaml`
- `src/core/settings.py`
- `src/core/profile.py`
- `src/core/schemas.py`

## Capitulo 2 - Dataset E Preprocessamento Edge-Aware

### Objetivo

Remover dependencia de `remove_response` e StationXML no pipeline principal, reduzindo training-serving skew entre treinamento e ESP32.

### O Que Foi Feito

- Definicao do pipeline sismico edge-aware:
  - resample 40 Hz
  - detrend linear
  - demean
  - taper 5%
  - bandpass 0.5-15 Hz
  - zscore por janela
- Justificativa tecnica: `remove_response` depende da resposta instrumental da estacao e nao e trivial de embarcar no microcontrolador.
- Manutencao da proposta generica: o dominio sismico fica no adapter; o resto do pipeline recebe arrays genericos.

### Arquivos Principais

- `config/configs/profiles/seismic_edge_v1.yaml`
- `src/data/validate_dataset.py`
- `src/data/adapters/`

## Capitulo 3 - Treinamento E Comparacao De Modelos

### Objetivo

Permitir que o usuario escolha quais modelos treinar por configuracao, compare familias diferentes e selecione automaticamente o melhor candidato.

### O Que Foi Feito

- Configuracao de modelos por YAML.
- Suporte a modelos classicos e neurais:
  - Random Forest
  - Extra Trees
  - Tiny CNN
  - Tiny TCN
  - Autoencoders como experimentos
- Inclusao de presets baseados nas melhores rodadas externas:
  - `tiny_cnn` como candidato principal atual
  - `tiny_tcn` como candidato em avaliacao
- Selecao automatica por metrica principal, atualmente `auc_pr`.

### Arquivos Principais

- `config/model/models.yaml`
- `config/model/models_smoke.yaml`
- `src/training/train_all.py`
- `src/training/model_registry.py`
- `src/training/neural_models.py`
- `src/training/evaluate.py`

## Capitulo 4 - Features Estatisticas E Espectrais

### Objetivo

Fortalecer os baselines classicos e a interpretabilidade dos resultados.

### O Que Foi Feito

- Criacao/ajuste de features genericas por janela.
- Inclusao de features que apareceram como importantes nas rodadas externas:
  - `spectral_rolloff_85`
  - `bandpower_8_15hz`
  - `spectral_centroid`
  - `zero_crossing_rate`
  - `kurtosis`
  - `spectral_entropy`
  - `abs_peak`
  - `crest_factor`
- Validacao de que o extrator gera 28 features.

### Arquivos Principais

- `src/features/statistical_features.py`

## Capitulo 5 - Relatorios De Metricas

### Objetivo

Gerar relatorios mais proximos de uma avaliacao real de produto e operacao.

### O Que Foi Feito

- Inclusao de metricas de validacao e teste:
  - AUC-PR
  - AUC-ROC
  - F1
  - precision
  - recall
  - FP/h
- Calculo de falso positivo por hora a partir de:
  - matriz de confusao
  - `step_seconds` do profile
- Geracao de:
  - `model_comparison.csv`
  - `model_comparison.md`
  - `model_comparison.json`
  - `candidate_manifest.json`

### Arquivos Principais

- `src/training/train_all.py`
- `artefacts/reports/model_comparison.csv`
- `artefacts/reports/model_comparison.md`
- `artefacts/reports/model_comparison.json`
- `artefacts/reports/candidate_manifest.json`

## Capitulo 6 - Quality Gate

### Objetivo

Evitar que um modelo seja promovido apenas por ter sido treinado. O modelo precisa passar por regras minimas antes de virar candidato de producao.

### O Que Foi Feito

- Criacao do fluxo:
  - `candidate_manifest.json`
  - quality gate
  - `promotion_report.json`
  - `production_manifest.json`
- Regras previstas:
  - AUC-PR minima
  - F1 minimo
  - maximo de FP/h
  - gap maximo entre validacao e teste
  - tamanho maximo do modelo

### Arquivos Principais

- `src/mlops/promote_model.py`
- `artefacts/reports/promotion_report.json`
- `artefacts/registry/production_manifest.json`

## Capitulo 7 - Inicio Do OTA

### Objetivo

Comecar a ponte entre MLOps e firmware, sem ainda executar OTA real.

### O Que Sera Feito

- Criar um manifesto OTA a partir do modelo promovido para producao.
- Registrar:
  - modelo aprovado
  - threshold
  - profile
  - preprocessamento esperado
  - target ESP32
  - runtime TensorFlow Lite Micro
  - estrategia OTA
  - checksum SHA-256

### Arquivos Planejados

- `src/ota/__init__.py`
- `src/ota/build_ota_manifest.py`
- `artefacts/ota/ota_manifest.json`

## Observacoes Importantes

- Resultados com AUC-PR 1.0000 foram obtidos em dataset sintetico/facil e nao devem ser usados como resultado cientifico final.
- O pipeline esta funcionando como smoke test.
- A validacao cientifica deve ser feita com o dataset real processado.
- Artefatos pesados nao devem ser versionados no Git:
  - `.npz`
  - `.joblib`
  - `.keras`
  - `.tflite`
  - runs completos de Optuna

## Proximos Passos

1. Implementar `src/ota/build_ota_manifest.py`.
2. Gerar `artefacts/ota/ota_manifest.json`.
3. Exportar o modelo para TFLite/header.
4. Criar pacote de firmware/modelo para ESP32.
5. Planejar OTA com rollback e validacao de checksum.
6. Repetir o pipeline com dataset real.
