# Snapshot PlatformIO

Este diretorio guarda os arquivos relevantes do firmware ESP32 usado no TCC.

Ele nao substitui automaticamente o projeto PlatformIO original em:

`C:/Users/vish8/OneDrive/Documentos/PlatformIO/Projects/TCC`

Use este snapshot como referencia organizada dentro do repositorio do trabalho.

## Arquivos principais

- `src/main.cpp`: firmware de inferencia e benchmark.
- `include/preprocessing.h`: pipeline embarcado.
- `include/model_config.h`: selecao de modelo, nome e threshold.
- `include/TCN/`: modelos TFLite e headers C/C++ usados pelo firmware.
- `scripts/export_real_dataset.py`: exporta janelas para `real_dataset.h`.
- `scripts/export_balanced_dataset.py`: exporta janelas balanceadas para teste.
- `scripts/create_cru_dataset.py`: gera dataset cru simulado.

## Observacao

O projeto PlatformIO completo depende tambem da biblioteca TensorFlow Lite Micro instalada pelo PlatformIO. Essas dependencias nao foram copiadas para ca para evitar duplicar milhares de arquivos gerados.
