# Resumo do experimento ESP8266 - TinyML OTA Feedback Server

## Visao geral

- Eventos exportados: 43
- Amostras historicas de status: 15
- Dispositivos registrados: 2
- Relatorios OTA exportados: 4

## Estado atual do dispositivo ESP8266

- device_id: esp8266_001
- device_type: esp8266
- location: teste_laboratorio
- firmware_version: fw_1.0.0
- model_version: unknown
- status: registered
- last_seen_at_utc: 2026-06-12T14:45:23.632763+00:00
- battery_level: 100.0
- free_memory_kb: 45.7109375
- signal_quality: -67.0
- wifi_ip: 192.168.66.70
- chip_model: ESP8266
- cpu_freq_mhz: 80
- flash_size_mb: 4
- sdk_version: 2.2.2-dev(38a443e)
- last_status_id: dst_37f0f149bf8e

## Distribuicao das predicoes

- normal: 23
- uncertain: 14
- anomaly: 6

## Distribuicao das prioridades

- low_priority: 17
- uncertain: 14
- high_normal: 9
- high_anomaly: 3

## Estatisticas dos scores

- media: 0.4359
- desvio_padrao: 0.2429
- minimo: 0.0130
- mediana: 0.4370
- maximo: 0.9770

## Estatisticas de memoria livre

- media: 45.4484
- desvio_padrao: 0.1381
- minimo: 45.3359
- mediana: 45.3359
- maximo: 45.7109

## Estatisticas de RSSI

- media: -65.4000
- desvio_padrao: 2.4437
- minimo: -72.0000
- mediana: -66.0000
- maximo: -62.0000

## Relatorios OTA

- Relatorios OTA do ESP8266: 2
- Ultimo status OTA: success
- Versao anterior: unknown
- Nova versao: seismic_edge_v1_tiny_tcn_20260610
- Mensagem: Manifesto OTA consultado com sucesso. Download real ainda nao aplicado neste MVP.

## Interpretacao tecnica

O experimento validou o fluxo fisico de comunicacao entre o ESP8266 e o servidor FastAPI. O dispositivo conseguiu registrar-se, enviar status periodico, transmitir eventos de inferencia simulada, consultar o manifesto OTA, detectar uma nova versao de modelo compativel com o target esp8266 e registrar o resultado da atualizacao no endpoint /ota/report. Os dados foram persistidos em arquivos JSON e exportados para CSV, permitindo analise posterior e geracao de graficos.

Nesta etapa, a inferencia TinyML ainda e simulada e o OTA foi validado no nivel de manifesto e versionamento. O download real e a substituicao do artefato do modelo no dispositivo permanecem como evolucao futura do MVP.