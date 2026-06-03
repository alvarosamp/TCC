# TCC - TinyML para Deteccao de Anomalias em Series Temporais

Este repositorio documenta e implementa uma arquitetura generica de deteccao de anomalias em series temporais com TinyML no ESP32, avaliacao de modelos e evolucao para MLOps/OTA. O dominio sismico e o primeiro estudo de caso validado.

A pasta principal do trabalho esta em:

[`Trabalho/`](Trabalho/)

## Ideia central

O TinyML atua como um filtro inteligente na borda:

```text
sensor -> janela temporal -> preprocessing -> modelo TinyML -> score -> decisao local
```

Em vez de armazenar ou transmitir todos os dados continuamente, o dispositivo decide localmente quando uma janela parece anomala. Isso pode reduzir armazenamento, consumo de energia, banda de rede e custo de nuvem.

## Estado atual

- Dataset sismico com janelas de 20 s a 40 Hz.
- Split v4 por evento para reduzir vazamento entre treino, validacao e teste.
- Comparacao de STA/LTA, modelos com features, CNN/TCN e autoencoders.
- Melhor modelo observado no notebook de metricas: Tiny TCN classifier.
- Firmware ESP32/PlatformIO com TensorFlow Lite Micro.
- Preprocessing embarcado aproximando o pipeline offline.
- Estrutura de pipeline generico versionado por profile.

## Entradas principais

- [README do trabalho](Trabalho/readme.md)
- [Resumo de metricas](Trabalho/docs/metrics_summary.md)
- [Mapa resultado -> codigo](Trabalho/docs/results_code_map.md)
- [Retreino generico](Trabalho/docs/retraining_generic.md)
- [Contrato generico](Trabalho/docs/generic_pipeline_contract.md)
- [Inventario do projeto](Trabalho/docs/project_inventory.md)
- [Profile canonico do pipeline](Trabalho/profiles/seismic_v1.json)
- [Snapshot do firmware ESP32](Trabalho/edge/platformio_snapshot/)

## Resultados resumidos

| Modelo | Tipo | AUC-PR | F1 |
|---|---|---:|---:|
| STA/LTA | classico | 0.1642 | 0.2721 |
| Random Forest | ML features | 0.7641 | 0.6899 |
| ExtraTrees | ML features | 0.7767 | 0.6964 |
| Tiny CNN classifier | DL compacto | 0.9020 | 0.8189 |
| Tiny TCN classifier | DL compacto | 0.9186 | 0.8247 |

Os resultados de quantizacao e teste embarcado ainda precisam ser consolidados no mapa de resultados quando os arquivos finais forem enviados.

## Dados e arquivos pesados

Datasets brutos, arquivos MiniSEED, `.npz`, `.npy`, modelos `.h5/.keras` e artefatos pesados nao devem ser versionados neste repositorio. Eles devem ficar em armazenamento externo, Google Drive ou disco local, por exemplo `D:\`.

O repositorio deve guardar codigo, configuracao, documentacao, notebooks principais e artefatos pequenos suficientes para reproduzir o pipeline.

## Proximas etapas

1. Consolidar resultados de quantizacao e testes do modelo.
2. Retreinar o Tiny TCN pelo pipeline generico em `Trabalho/scripts/train_generic_classifier.py`.
3. Criar `drift_report.py` para iniciar a camada MLOps.
4. Validar o pipeline em outro dataset para demonstrar generalidade.
5. Implementar OTA inicialmente como atualizacao completa de firmware.
