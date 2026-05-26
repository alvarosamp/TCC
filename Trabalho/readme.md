# TCC - TinyML para Deteccao de Anomalias em Series Temporais

Este repositorio consolida o trabalho de deteccao de eventos anomalos em sinais sismicos com foco em TinyML, validacao embarcada no ESP32 e evolucao futura para MLOps/OTA.

## Ideia central

O TinyML atua como um filtro inteligente na borda:

```text
sensor -> janela temporal -> preprocessing -> modelo TinyML -> score -> decisao local
```

Em vez de armazenar ou transmitir todos os dados, o dispositivo envia apenas eventos suspeitos, resumos e amostras de auditoria. A meta e reduzir armazenamento, consumo de energia, trafego de rede e custo operacional sem perder eventos relevantes.

## Estado atual

- Dataset sismico processado em janelas de 20 s.
- Split v4 por evento para reduzir vazamento entre treino, validacao e teste.
- Comparacao de modelos classicos, ML com features e deep learning compacto.
- Melhor candidato observado: Tiny TCN classifier.
- Firmware PlatformIO/ESP32 com TensorFlow Lite Micro.
- Preprocessing embarcado aproximando o pipeline offline.
- Estrutura inicial para pipeline generico versionado.

## Estrutura principal

```text
profiles/
  seismic_v1.json              # contrato versionado do pipeline sismico

src/tcc_pipeline/
  profile.py                   # leitura/validacao de profiles
  dataset.py                   # leitura e inspecao de datasets NPZ
  metrics.py                   # metricas padrao de classificacao/anomalia

scripts/
  create_split_evento_v4.py    # cria dataset v4 com split por evento
  inspect_dataset.py           # valida NPZ contra um profile

docs/
  metrics_summary.md           # resumo dos resultados do Colab
  project_inventory.md         # comparacao das pastas e organizacao

edge/platformio_snapshot/
  src/main.cpp                 # firmware de inferencia/benchmark
  include/preprocessing.h      # preprocessing embarcado
  include/model_config.h       # selecao do modelo e threshold
  scripts/                     # scripts de exportacao para ESP32

notebook/
  ...                          # historico original de notebooks/scripts

notebooks/
  Metricas_Alvaro.ipynb        # notebook consolidado de metricas do Colab

artefacts/
  ...                          # resultados, figuras, modelos e dados antigos
```

Os diretorios `notebook/` e `artefacts/` foram preservados como historico. A estrutura nova e o ponto recomendado para continuar o projeto.

## Pipeline sismico atual

O profile canonico esta em `profiles/seismic_v1.json`.

Parametros principais:

- Sampling rate: 40 Hz
- Janela: 800 amostras
- Duracao: 20 s
- Passo: 10 s
- Split principal: evento
- Metrica primaria: AUC-PR
- Normalizacao: z-score por janela

Pipeline offline usado no dataset:

```text
resample 40 Hz
detrend linear
demean
taper 5%
remove_response para VEL
bandpass 0.5-15 Hz zerophase
z-score por janela
```

Pipeline embarcado no ESP32:

```text
detrend linear
taper 5%
bandpass 0.5-15 Hz zerophase
z-score por janela
```

Limitacao conhecida: `remove_response` nao foi embarcado porque depende de StationXML e resposta instrumental. Para producao, o ideal e calibrar o sensor ou treinar uma variante sem essa etapa.

## Resultados principais

Resumo do notebook `notebooks/Metricas_Alvaro.ipynb`:

| Modelo | Tipo | AUC-PR | F1 |
|---|---|---:|---:|
| STA/LTA | classico | 0.1642 | 0.2721 |
| Random Forest | ML features | 0.7641 | 0.6899 |
| ExtraTrees | ML features | 0.7767 | 0.6964 |
| Tiny CNN classifier | DL compacto | 0.9020 | 0.8189 |
| Tiny TCN classifier | DL compacto | 0.9186 | 0.8247 |

Mais detalhes em `docs/metrics_summary.md`.

## Como validar um dataset

Instale as dependencias Python principais quando estiver em um ambiente novo:

```bash
pip install -r requirements.txt
```

Exemplo:

```bash
python scripts/inspect_dataset.py \
  --profile profiles/seismic_v1.json \
  --dataset /caminho/para/dataset_v4_split_evento.npz \
  --output artefacts/reports/dataset_v4_split_evento_profile_report.json
```

No Windows, ajuste o caminho do dataset conforme o local dos arquivos processados.

## Como recriar o split por evento

```bash
python scripts/create_split_evento_v4.py \
  --data-dir /caminho/para/processed \
  --split-base temporal \
  --nome evento
```

Arquivos esperados em `data-dir`:

- `inventario_v3.csv`
- `dataset_v3_split_temporal.npz`

Saidas:

- `dataset_v4_split_evento.npz`
- `inventario_v4_split_evento.csv`
- `dataset_v4_split_evento_info.json`

## Firmware e modelo embarcado

O firmware de referencia esta em `edge/platformio_snapshot`.

Arquivos principais:

- `src/main.cpp`: inicializa TFLite Micro, prepara janelas, executa inferencia e imprime metricas CSV.
- `include/model_config.h`: escolhe `TCN_INT8`, `TCN_FLOAT16` ou `TCN_FLOAT32`.
- `include/preprocessing.h`: implementa o preprocessing de borda.

O modelo `.tflite` e convertido para um array C/C++ em `.h` porque o TensorFlow Lite Micro no ESP32 recebe os bytes do modelo diretamente da memoria do firmware:

```cpp
model = tflite::GetModel(MODEL_DATA);
```

## Proximas etapas

1. Validar Python vs ESP32 para a mesma janela.
2. Transformar os scripts de treino em comandos usando `profiles/seismic_v1.json`.
3. Criar um segundo profile, por exemplo `cwru_v1.json`, para provar generalidade.
4. Implementar um `drift_report.py` simples.
5. Iniciar OTA primeiro como firmware completo.
6. Depois evoluir para OTA de modelo separado com manifest, hash, versao e rollback.

## Visao MLOps

O OTA nao deve ser disparado automaticamente apenas porque houve drift. O fluxo correto e:

```text
telemetria -> drift/queda de metrica -> analise -> retreinamento -> validacao -> aprovacao -> OTA -> health check -> rollback se falhar
```

O dispositivo deve reportar versoes:

- firmware_version
- model_version
- threshold
- preprocessing_version
- profile_name/profile_version

Esses campos evitam confundir mudanca real do ambiente com mudanca de modelo ou de preprocessing.
