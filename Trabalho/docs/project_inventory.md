# Inventario Organizado do Projeto

Este arquivo documenta como as pastas antigas foram comparadas e como a nova estrutura canonica deve ser usada.

## Fontes comparadas

### Pasta WSL do TCC

`/home/vish8/tcc/Trabalho/Trabalho`

Contem:

- `notebook/`: scripts e notebooks de coleta, preprocessamento, treinamento e Optuna.
- `artefacts/`: resultados, figuras, datasets antigos e modelos gerados.
- `config.py`: configuracao antiga de caminhos e parametros sismicos.
- `readme.md`: roadmap antigo do projeto.

### Projeto PlatformIO

`C:/Users/vish8/OneDrive/Documentos/PlatformIO/Projects/TCC`

Contem:

- `src/main.cpp`: firmware de inferencia e benchmark.
- `include/preprocessing.h`: preprocessing embarcado.
- `include/model_config.h`: selecao do modelo e threshold.
- `include/TCN/...`: modelos TFLite convertidos para headers.
- `scripts/export_real_dataset.py`: exporta janelas NPZ para `real_dataset.h`.
- `scripts/create_cru_dataset.py`: gera dataset cru simulado para testar preprocessing na borda.

### Notebook do Colab

`C:/Users/vish8/Downloads/Metricas_Alvaro.ipynb`

Copiado para:

`notebooks/Metricas_Alvaro.ipynb`

Contem a comparacao de STA/LTA, ensembles, autoencoders, CNN/TCN, Optuna e TFLite.

## Nova estrutura canonica

```text
profiles/
  seismic_v1.json

src/tcc_pipeline/
  profile.py
  dataset.py
  metrics.py

scripts/
  inspect_dataset.py
  create_split_evento_v4.py

docs/
  metrics_summary.md
  project_inventory.md

edge/platformio_snapshot/
  platformio.ini
  src/main.cpp
  include/model_config.h
  include/preprocessing.h
  scripts/*.py

notebooks/
  Metricas_Alvaro.ipynb
```

## Principio de organizacao

Os arquivos antigos foram preservados. A nova estrutura define quais arquivos sao a versao de referencia para continuar o TCC:

- `profiles/seismic_v1.json`: contrato do pipeline.
- `src/tcc_pipeline`: funcoes reutilizaveis.
- `scripts`: comandos reproduziveis.
- `docs`: leitura humana e narrativa tecnica.
- `edge/platformio_snapshot`: snapshot dos arquivos relevantes do firmware.

## Proximos passos recomendados

1. Rodar `scripts/inspect_dataset.py` no `dataset_v4_split_evento.npz`.
2. Adaptar scripts de treinamento para usar `profiles/seismic_v1.json`.
3. Criar `profiles/cwru_v1.json` para validar generalidade em outro dataset.
4. Criar um `drift_report.py` simples comparando baseline vs producao.
5. Depois iniciar OTA com versionamento de firmware, modelo, threshold e preprocessing.
