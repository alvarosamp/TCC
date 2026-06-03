# Retreino Generico por Profile

Este fluxo e o caminho recomendado para novos treinos do TCC.

## Ideia

O treino nao deve depender de detalhes sismicos espalhados pelo codigo. O contrato passa a ser:

```text
profile JSON + dataset NPZ com splits -> treino -> metricas -> modelo -> exports
```

Qualquer dataset que respeite o contrato abaixo pode usar o mesmo script:

```text
X_train, y_train
X_val, y_val
X_test, y_test
```

O `profile` informa tamanho da janela, labels, metrica primaria, split principal e versoes de preprocessing.

## Comando recomendado

Exemplo para retreinar o Tiny TCN no dataset v4 por evento:

```bash
cd Trabalho

python scripts/train_generic_classifier.py \
  --profile profiles/seismic_v1.json \
  --dataset /caminho/para/dataset_v4_split_evento.npz \
  --model tiny_tcn \
  --output-dir artefacts/runs/generic_classifier \
  --epochs 40 \
  --batch-size 256 \
  --export-tflite all \
  --export-header
```

No Windows, um exemplo de dataset seria:

```powershell
python scripts/train_generic_classifier.py `
  --profile profiles/seismic_v1.json `
  --dataset "D:\TCC_data\processed\dataset_v4_split_evento.npz" `
  --model tiny_tcn `
  --epochs 40 `
  --batch-size 256 `
  --export-tflite all `
  --export-header
```

## Saidas geradas

O script cria uma pasta de run com:

- `tiny_tcn.keras`: modelo Keras treinado.
- `metrics.json`: metricas de validacao e teste.
- `manifest.json`: rastreabilidade completa do treino.
- `model_float32.tflite`, `model_float16.tflite`, `model_int8.tflite`: quando `--export-tflite all` for usado.
- `model_float32.h`, `model_float16.h`, `model_int8.h`: quando `--export-header` for usado.

Modelos `.keras` e `.tflite` sao pesados e nao devem ser versionados no GitHub. Guarde esses arquivos em disco local, Drive ou `D:\`.

## Por que isso e generico

O script nao sabe ler MiniSEED, nao conhece StationXML e nao possui regra especifica de sismologia. Ele apenas exige janelas numericas e labels binarios.

Para outro dominio, como rolamentos CWRU, voce cria outro profile, por exemplo:

```text
profiles/cwru_v1.json
```

e gera um NPZ com as mesmas chaves. O treino continua o mesmo.

## Cuidados tecnicos

1. O preprocessing usado para gerar o dataset precisa estar declarado no `profile`.
2. O threshold usado no teste vem da validacao, nao do teste.
3. A metrica primaria e AUC-PR porque o problema e desbalanceado.
4. Para ESP32, prefira comparar `float32`, `float16` e `int8`.
5. Para OTA futuro, o `manifest.json` e tao importante quanto o modelo, pois guarda profile, threshold, versao e metricas.

## Decisao para o projeto

O retreino generico deve virar o fluxo oficial. Notebooks continuam uteis para exploracao, mas resultado final de TCC precisa apontar para script reproduzivel.
