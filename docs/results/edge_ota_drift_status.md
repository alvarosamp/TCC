# Status Tecnico - Drift, OTA e Edge

## Resumo

O pipeline foi validado ate a etapa de build embarcada para ESP32. O upload real nao foi executado porque a placa conectada ao WSL foi identificada como ESP8266EX, enquanto o firmware e o alvo oficial do projeto sao ESP32.

## Drift

- Referencia gerada a partir do split de treino.
- Lote analisado: split de teste do dataset real.
- Resultado: drift alto.

Metricas observadas:

```text
Max z-shift: 0.0176
Max PSI: 0.3463
Min KS p-value: 0.000033
```

Decisao:

```text
retrain_recommended
```

## Modelo Candidato

O modelo retreinado foi aprovado no quality gate:

```text
Modelo: tiny_cnn
AUC-PR teste: 0.8775
F1 teste: 0.8109
Precision: 0.8431
Recall: 0.7810
FP/h: 6.75
Threshold: 0.7241942882537842
Parametros: 15377
```

## Decisao Drift -> OTA

Apos promocao do candidato:

```text
Acao OTA: build_and_publish_ota
Candidato: True
Aprovado: True
```

## Exportacao Edge

Artefatos gerados:

```text
artefacts/edge/tiny_cnn_float32.tflite
artefacts/edge/tiny_cnn_float16.tflite
artefacts/edge/tiny_cnn_int8.tflite
artefacts/edge/tiny_cnn_int8.h
artefacts/edge/tiny_cnn_export_manifest.json
```

Foi corrigido o gerador de header para escrever o array real de bytes do `.tflite`, nao o literal `{c_array}`.

## Firmware PlatformIO

Configuracao ativa:

```cpp
#define ACTIVE_MODEL MODEL_PIPELINE_TINY_CNN_INT8
#define MODEL_DATA tiny_cnn_int8_model_data
#define MODEL_THRESHOLD 0.7241942882537842f
```

Build ESP32:

```text
Status: SUCCESS
RAM: 33888 bytes, 10.3%
Flash: 597165 bytes, 19.0%
```

## Hardware Conectado

Comando:

```bash
python3 ~/.platformio/packages/tool-esptoolpy/esptool.py --port /dev/ttyUSB0 chip_id
```

Resultado:

```text
Detecting chip type... ESP8266
Chip is ESP8266EX
```

Conclusao: a placa conectada nao e ESP32. O firmware correto foi compilado, mas a validacao real de upload/inferencia precisa ser feita com uma placa ESP32.

## Proximas Etapas

1. Conectar ESP32 real.
2. Fazer upload do firmware compilado.
3. Abrir monitor serial e coletar metricas.
4. Gerar relatorio final consolidado.
5. Fechar escrita do TCC/artigo.
