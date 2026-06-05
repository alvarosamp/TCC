# Snapshot De Resultados - Seismic Edge v1

Data: 2026-06-05
Fonte externa: `D:\PipelineGenerico\data`

## Ranking Dos Modelos Completos

| Rank | Modelo | AUC-PR teste | AUC-ROC teste | F1 teste | Precisao | Recall | FP/h | Params |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Optuna Tiny CNN classifier v4 | 0.9127 | 0.9680 | 0.8526 | 0.8982 | 0.8114 | 4.896 | 19.489 |
| 2 | Tiny CNN classifier | 0.8982 | 0.9624 | 0.7951 | 0.7310 | 0.8716 | 16.944 | 9.473 |
| 3 | Tiny TCN classifier | 0.8964 | 0.9627 | 0.7666 | 0.6790 | 0.8801 | 21.984 | 6.417 |
| 4 | Optuna Random Forest v4 | 0.8127 | 0.9340 | 0.7367 | 0.7974 | 0.6846 | 9.264 | - |
| 5 | Optuna Extra Trees v4 | 0.7901 | 0.9283 | 0.7102 | 0.7589 | 0.6675 | 11.296 | - |
| 6 | STA/LTA v4 | 0.1662 | 0.6187 | 0.2760 | 0.1773 | 0.6230 | - | - |

## Melhor Candidato Atual

O melhor candidato completo e o `optuna_tiny_cnn_classifier_v4`.

Metricas principais:

```text
AUC-PR teste = 0.9126674878
AUC-ROC teste = 0.9679960081
F1 teste = 0.8525883838
Precisao = 0.8982374460
Recall = 0.8113547612
FP/h = 4.896
Threshold = 0.3804919124
Parametros = 19.489
Tempo de treino = 1305.493 s
```

Intervalo bootstrap de AUC-PR:

```text
low  = 0.9040822240
high = 0.9206185859
```

## Tiny TCN Em Andamento

A rodada `optuna_tiny_tcn_classifier_v4-2` ainda nao tinha resultado final de teste no momento da coleta.

Melhor validacao parcial:

```text
val_auc_pr = 0.8810995644
val_f1 = 0.8230478589
val_precision = 0.8570491803
val_recall = 0.7916414294
val_fp_per_hour = 6.976
params = 6.017
```

Hiperparametros parciais:

```text
batch_size = 512
filters = 16
kernel_size = 11
n_blocks = 6
dilation_base = 3
dropout = 0.258533
spatial_dropout = 0.285220
dense_units = 32
lr = 0.002320
conv_type = separable
use_batch_norm = true
head_pooling = avgmax
label_smoothing = 0.027348
```

## Baselines Classicos

### Optuna Random Forest v4

```text
AUC-PR teste = 0.8126896220
AUC-ROC teste = 0.9339990120
F1 teste = 0.7367059964
Precisao = 0.7974107768
Recall = 0.6845899670
FP/h = 9.264
```

### Optuna Extra Trees v4

```text
AUC-PR teste = 0.7901481328
AUC-ROC teste = 0.9282872401
F1 teste = 0.7102445261
Precisao = 0.7588797814
Recall = 0.6674677080
FP/h = 11.296
```

## Autoencoders

Os autoencoders testados ficaram abaixo dos classificadores supervisionados nesta versao do dataset.

| Modelo | AUC-PR teste | F1 teste | Observacao |
|---|---:|---:|---|
| dense_ae | 0.1969 | 0.2330 | Muitos falsos positivos |
| cnn1d_ae | 0.1289 | 0.0000 | Nao encontrou ponto util |
| gru_ae | 0.2301 | 0.2352 | Muitos falsos positivos |
| lstm_ae | 0.1901 | 0.2320 | Muitos falsos positivos |
| tiny_cnn_ae | 0.0839 | 0.1439 | Fraco nesta configuracao |

## Leitura Tecnica

- A rede Tiny CNN otimizada e o melhor compromisso atual entre desempenho, tamanho e falsos positivos.
- Random Forest e Extra Trees continuam importantes como baselines interpretaveis, mas sao ruins para embarcar diretamente por tamanho e custo de memoria.
- STA/LTA e um baseline tradicional util para comparacao, mas ficou muito abaixo dos modelos treinados.
- Os autoencoders nao devem ser a linha principal nesta versao, embora possam ser uteis em cenarios futuros com poucos rotulos.
- Features espectrais aparecem de forma consistente como importantes: `spectral_rolloff_85`, `bandpower_8_15hz`, `spectral_centroid`, `zero_crossing_rate`, `kurtosis` e `spectral_entropy`.

## Cuidados

- Nao versionar arquivos `.joblib`, `.keras`, `.tflite`, `.npz` ou runs completos no Git.
- A pasta `reports-20260605T220510Z-3-002` contem arquivos `.joblib` muito pequenos, possivelmente incompletos; nao usar como evidencia sem verificacao.
- A comparacao final deve ser atualizada quando o Optuna do Tiny TCN terminar e gerar resultado de teste.
