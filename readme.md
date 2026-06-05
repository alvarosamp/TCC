# TCC - Pipeline Generico de TinyML para Series Temporais Complexas

Este repositorio organiza o projeto de TCC para deteccao de anomalias em series temporais complexas, com foco em uma aplicacao TinyML executavel em microcontroladores. O primeiro dominio de validacao e sismico, mas a arquitetura foi pensada para ser reutilizada em outros sensores, como vibracao industrial, corrente eletrica, audio, telemetria ou sinais de manutencao preditiva.

A ideia central do projeto e usar um modelo leve na borda como um organizador inteligente: o dispositivo processa janelas locais do sinal e so encaminha dados quando existe indicio confiavel de anomalia. Isso reduz armazenamento, transmissao, consumo de energia e custo operacional.

## Objetivos

- Construir um pipeline generico para classificacao/anomalia em series temporais.
- Separar adaptadores de dominio do nucleo de machine learning.
- Comparar modelos classicos, redes neurais leves e baselines tradicionais.
- Otimizar modelos com Optuna usando uma metrica escolhida.
- Exportar o melhor candidato para TFLite e header C/C++ para TensorFlow Lite Micro.
- Preparar a base para MLOps: DVC, MLflow, manifestos, quality gate, drift e OTA.

## Arquitetura

```text
raw data
  -> adapter de dominio
  -> dataset generico NPZ
  -> validacao do contrato
  -> treinamento de modelos selecionados
  -> Optuna/HPO
  -> comparacao por metrica principal
  -> candidate_manifest.json
  -> exportacao TFLite / C header
  -> validacao no ESP32
```

O ponto mais importante e que o nucleo do pipeline nao depende de sismologia. O dado sismico entra por um adapter especifico; depois disso, o treinamento recebe apenas matrizes `X` e rotulos `y`.

## Contrato De Dataset

Todo dataset processado deve seguir o contrato abaixo:

```text
X_train, y_train
X_val,   y_val
X_test,  y_test
```

Formato esperado:

- `X_*`: array numerico com janelas de serie temporal.
- `y_*`: rotulos binarios, onde `0 = normal` e `1 = anomalo`.
- Cada janela deve ter tamanho fixo, por exemplo `800` amostras para `20 s @ 40 Hz`.

## Caso Sismico Atual

No dominio sismico, o dataset e construido a partir de arquivos MiniSEED:

```text
raw/events/      -> anomalo
raw/continuous/  -> normal
```

A versao edge-aware do preprocessamento remove a dependencia de `remove_response`, porque essa etapa exige StationXML e resposta instrumental, algo dificil de reproduzir no microcontrolador. A pipeline usada para aproximar treinamento e inferencia embarcada e:

```text
resample 40 Hz
-> detrend linear
-> demean
-> taper 5%
-> bandpass 0.5-15 Hz
-> zscore por janela
```

Essa decisao reduz o risco de training-serving skew: o modelo passa a ser treinado com transformacoes que tambem podem ser implementadas no ESP32.

## Estrutura Do Repositorio

```text
src/
  core/          # configs, schemas e perfis
  data/          # validacao de dataset e adapters de dominio
  features/      # features estatisticas e espectrais
  training/      # treinamento, Optuna, avaliacao e selecao
  export/        # exportacao TFLite e header C/C++
  mlops/         # quality gate, promocao e manifestos

config/          # configuracoes do pipeline e modelos
profiles/        # perfis de dominio/dataset
scripts/         # scripts utilitarios
docs/            # documentacao tecnica e resultados
artefacts/       # saidas locais do pipeline, nao deve receber datasets pesados
```

## Modelos Comparados

O pipeline permite selecionar modelos por configuracao. A ideia e evitar travar o projeto em uma unica arquitetura.

Modelos suportados ou planejados:

| Familia | Exemplos | Uso principal |
|---|---|---|
| Baseline de regra | STA/LTA | Comparacao com metodo tradicional |
| Classicos supervisionados | Random Forest, Extra Trees | Baselines fortes e interpretaveis |
| Classicos nao supervisionados | Isolation Forest | Cenario com poucos rotulos |
| Redes leves | Tiny CNN, Tiny TCN | Candidatos TinyML |
| Autoencoders | Dense AE, CNN AE, LSTM/GRU AE | Deteccao por erro de reconstrucao |

Fluxo desejado:

```text
modelos selecionados
  -> treino inicial
  -> Optuna quando habilitado
  -> avaliacao em validacao
  -> treino final
  -> teste final
  -> selecao do melhor candidato
```

## Resultados Atuais

Resultados consolidados a partir das rodadas externas em `D:\PipelineGenerico\data` no dia 2026-06-05.

| Modelo | AUC-PR teste | F1 teste | Precisao | Recall | FP/h | Observacao |
|---|---:|---:|---:|---:|---:|---|
| Optuna Tiny CNN classifier v4 | 0.9127 | 0.8526 | 0.8982 | 0.8114 | 4.896 | Melhor modelo completo ate agora |
| Tiny CNN classifier | 0.8982 | 0.7951 | 0.7310 | 0.8716 | 16.944 | Boa rede leve sem HPO final |
| Tiny TCN classifier | 0.8964 | 0.7666 | 0.6790 | 0.8801 | 21.984 | Boa, mas com mais falsos positivos |
| Optuna Random Forest v4 | 0.8127 | 0.7367 | 0.7974 | 0.6846 | 9.264 | Melhor baseline classico completo |
| Optuna Extra Trees v4 | 0.7901 | 0.7102 | 0.7589 | 0.6675 | 11.296 | Forte, porem pesado |
| STA/LTA v4 | 0.1662 | 0.2760 | 0.1773 | 0.6230 | - | Baseline tradicional |

O `optuna_tiny_tcn_classifier_v4-2` ainda estava em otimizacao. Resultado parcial de validacao:

```text
val_auc_pr = 0.8811
val_f1     = 0.8230
val_fp_h   = 6.976
params     = 6.017
```

Por enquanto, o melhor candidato completo para TinyML e o `optuna_tiny_cnn_classifier_v4`.

## Interpretabilidade

As arvores indicam que as features mais relevantes estao ligadas ao comportamento espectral do sinal:

- `spectral_rolloff_85`
- `bandpower_8_15hz`
- `spectral_centroid`
- `zero_crossing_rate`
- `kurtosis`
- `spectral_entropy`

Isso e importante porque mostra que os modelos nao estao usando apenas amplitude. Eles estao capturando forma e distribuicao de frequencia das janelas.

## MLOps

Ferramentas usadas ou previstas:

- DVC: reproducao do pipeline e versionamento de etapas.
- MLflow: rastreamento de experimentos, metricas e parametros.
- Optuna: otimizacao de hiperparametros.
- Pytest: testes de comunicacao entre modulos.
- Manifestos JSON: registro de datasets, modelos, metricas e candidatos.
- Quality gate: criterio minimo antes de promover modelo.
- Drift monitoring: etapa futura para decidir retreinamento.
- OTA: etapa futura para atualizar o firmware/modelo embarcado com seguranca.

## Como Executar

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Rodar o pipeline DVC:

```bash
dvc repro
```

Executar testes:

```bash
pytest
```

Abrir MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///artefacts/mlruns/mlflow.db
```

## Artefatos Pesados

Datasets, modelos grandes e runs completos nao devem ser versionados no Git. Exemplos que devem ficar fora do repositorio:

```text
*.npz
*.joblib
*.keras
*.tflite
mlruns/
artefacts/models/
D:/PipelineGenerico/data/
```

O repositorio deve guardar codigo, configuracoes, documentacao e relatorios leves. Os artefatos pesados devem ser controlados por DVC, storage externo ou disco local.

## Proximos Passos

1. Finalizar a otimizacao do Tiny TCN e comparar com o Tiny CNN.
2. Exportar o melhor modelo para TFLite float32, float16 e int8.
3. Converter o modelo escolhido para header C/C++.
4. Validar no ESP32 com `preprocessing.h` equivalente ao pipeline de treino.
5. Criar quality gate para promocao do modelo.
6. Implementar monitoramento de drift.
7. Planejar OTA com manifestos assinados, rollback e validacao de versao.

## Tese Tecnica Do Projeto

Um pipeline generico de TinyML para series temporais pode reduzir custo de transmissao, energia e armazenamento ao executar uma primeira camada de triagem diretamente no dispositivo de borda. No caso sismico, os resultados atuais indicam que redes neurais compactas otimizadas superam baselines tradicionais e modelos classicos em equilibrio entre AUC-PR, F1 e falsos positivos por hora.
