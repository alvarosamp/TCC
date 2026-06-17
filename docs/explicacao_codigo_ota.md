# Explicacao Completa: OTA HTTP no ESP32

Este documento explica todo o codigo de atualizacao de modelo via HTTP (OTA) implementado no firmware do ESP32, do conceito a cada linha de codigo.

---

## O que e OTA?

**OTA = Over-The-Air** — atualizar o software de um dispositivo sem precisar conectar um cabo.

No contexto deste TCC, OTA significa: **atualizar o modelo TFLite do ESP32 remotamente**, sem recompilar o firmware. Quando o pipeline MLOps gera um modelo melhor, aprovado pelo quality gate, o ESP32 pode baixar e usar o novo modelo automaticamente.

---

## Visao Geral do Sistema

```
  SERVIDOR (PC / nuvem)                    ESP32
  ┌─────────────────────────┐              ┌────────────────────────────┐
  │  FastAPI (uvicorn)      │              │  Arduino + TFLite Micro    │
  │  ─────────────────────  │  HTTP/JSON   │  ──────────────────────    │
  │  GET /ota/latest        │◄─────────────│  ota_check_and_download()  │
  │  GET /ota/artifact      │─────────────►│  stream_to_spiffs()        │
  │  POST /ota/report       │◄─────────────│  ota_post_report()         │
  │                         │              │                            │
  │  artefacts/ota/         │              │  SPIFFS                    │
  │    releases/latest.json │              │    /ota_model.tflite       │
  │    releases/<versao>/   │              │    /ota_meta.json          │
  │      artifact.tflite    │              │                            │
  └─────────────────────────┘              │  RAM                       │
                                           │    s_model_buf[]           │
                                           │    tensor_arena[]          │
                                           └────────────────────────────┘
```

---

## Fluxo Completo no Boot do ESP32

```
Power on / Reset
      │
      ▼
Serial.begin(115200)
      │
      ▼
SPIFFS.begin(true)    ← monta sistema de arquivos flash interno
      │               ← se nao existir, formata automaticamente (parametro true)
      ▼
wifi_connect()        ← tenta conectar ao AP em ate 15 segundos
      │
      ├──── FALHA ──────────────────────────────────────────┐
      │                                                     │
      ▼                                                     │
ota_get_installed_version()   ← le /ota_meta.json          │
      │                         do SPIFFS (versao local)    │
      ▼                                                     │
GET /ota/latest               ← consulta servidor          │
      │                                                     │
      ├── versao servidor == versao local?                  │
      │       SIM ──► POST /ota/report "skipped"           │
      │       NÃO ──►                                       │
      │               GET /ota/artifact                     │
      │               (streaming para SPIFFS)               │
      │                     │                               │
      │               SHA-256 valido?                       │
      │                 SIM ──► save_meta()                 │
      │                         POST /report "success"      │
      │                 NAO ──► remove arquivo              │
      │                         POST /report "failed"       │
      ▼                                                     │
ota_load_model_into_ram()  ◄──────────────────────────────┘
      │
      ├── SPIFFS tem modelo? ──► s_model_buf = malloc + leitura SPIFFS
      │                          model_data = s_model_buf
      │
      └── NAO ──────────────► model_data = MODEL_DATA (header compilado)
              │
              ▼
tflite::GetModel(model_data)
              │
              ▼
AllocateTensors()
              │
              ▼
loop() — inferencia continua
```

---

## Estrutura dos Arquivos

```
PlatformIO/Projects/TCC/
├── include/
│   ├── ota_http.h            ← interface publica do modulo OTA
│   ├── wifi_config.h         ← suas credenciais (NAO versionado)
│   ├── wifi_config.h.template← exemplo para voce copiar
│   └── model_config.h        ← selecao do modelo compilado (fallback)
│
└── src/
    ├── main.cpp              ← integra OTA no setup()
    └── ota_http.cpp          ← toda a logica OTA
```

---

## Explicacao por Arquivo

---

### `include/wifi_config.h`

Voce cria este arquivo (nao esta no git). Define:

```cpp
#define WIFI_SSID       "MinhaRede"
#define WIFI_PASSWORD   "senha123"
#define OTA_SERVER_HOST "192.168.1.100"   // IP do seu PC
#define OTA_SERVER_PORT  8000
#define DEVICE_ID       "esp32_001"
```

**Por que separado?** Credenciais nunca devem ir para o repositorio git. O `.gitignore` ja ignora `wifi_config.h`.

**Como descobrir o IP do servidor no WSL:**

```bash
ip addr show eth0 | grep "inet "
# ou
hostname -I
```

---

### `include/ota_http.h`

Define a **interface** do modulo OTA: quais funcoes existem e o que elas retornam.

#### Struct OtaResult

```cpp
struct OtaResult {
    bool update_available;   // o servidor tem versao mais nova?
    bool download_ok;        // o download foi concluido sem erro HTTP?
    bool sha256_ok;          // o arquivo baixado e integro?
    bool already_current;    // ja estava na versao mais recente?
    char version[64];        // versao disponivel no servidor
    char sha256[65];         // hash esperado (64 hex + terminador)
    char error[256];         // descricao do erro se algo falhou
};
```

`OtaResult` funciona como um "relatorio" da operacao OTA. Voce lê os campos para saber o que aconteceu:

```
update_available=true, sha256_ok=true  → atualizacao bem-sucedida
update_available=true, sha256_ok=false → download corrompido
already_current=true                   → nao precisava atualizar
update_available=false (todos false)   → erro HTTP ou JSON invalido
```

---

### `src/ota_http.cpp`

Implementacao completa. Vamos por partes.

---

#### Variaveis de estado interno

```cpp
static uint8_t* s_model_buf  = nullptr;
static size_t   s_model_size = 0;
```

`static` aqui significa **escopo de arquivo**: so o proprio `ota_http.cpp` acessa diretamente. O resto do programa acessa via `ota_get_model_data()`.

`s_model_buf` guarda o modelo carregado do SPIFFS em RAM. O `s_` e convencao para "static/modulo-level".

---

#### Funcao: `sha256_of_file()`

```
Arquivo no SPIFFS
      │
      ▼
mbedtls_md_starts()      ← inicializa contexto SHA-256
      │
      ▼
loop: mbedtls_md_update() ← processa 512 bytes por vez
      │                     (nao carrega o arquivo inteiro na RAM)
      ▼
mbedtls_md_finish()       ← gera hash final de 32 bytes
      │
      ▼
snprintf: converte 32 bytes para 64 chars hexadecimal
      │
      ▼
retorna string ex: "f71ac3b1ec6b9207..."
```

**Por que SHA-256?** Para garantir que o modelo baixado e identico ao publicado pelo servidor. Se um byte mudar (corrupcao de rede, truncamento), o hash muda completamente. O servidor publica o hash no `latest.json` e o ESP32 recalcula e compara.

**`mbedtls`** e a biblioteca de criptografia incluida no SDK do ESP32 — nenhuma dependencia extra necessaria.

---

#### Funcao: `stream_to_spiffs()`

```
GET /ota/artifact (HTTP)
        │
        ▼
HTTPClient.begin(url)
        │
        ▼
HTTPClient.GET() ──── code != 200? ──► retorna -1
        │
        ▼
SPIFFS.open("/ota_model.tflite", "w")
        │
        ▼
loop (enquanto stream disponivel):
    stream->readBytes(buf, 512)   ← le 512 bytes por vez
    file.write(buf, n)            ← escreve na flash
        │
        ▼ (ate content_length ou timeout 30s)
file.close()
        │
        ▼
retorna total de bytes escritos
```

**Por que streaming e nao carregar tudo na RAM?** O modelo tem 66 KB. O ESP32 tem 320 KB de RAM, mas boa parte ja esta usada (tensor_arena, buffers). Streaming evita pico de uso de memoria.

**Por que 512 bytes por vez?** Equilibrio entre quantidade de chamadas e uso de stack. Valores maiores alocam mais stack; menores sao mais lentos.

---

#### Funcao: `wifi_connect()`

```cpp
WiFi.mode(WIFI_STA);      // modo estacao (cliente), nao ponto de acesso
WiFi.begin(ssid, password);

while (WiFi.status() != WL_CONNECTED && elapsed < timeout_ms) {
    delay(500);
}
```

`WIFI_STA` = Station Mode. O ESP32 se comporta como um dispositivo conectando a um roteador, nao como um roteador.

O loop checa `WiFi.status()` a cada 500 ms. Os possiveis estados sao:
- `WL_CONNECTED` — conectado, IP atribuido
- `WL_DISCONNECTED` — tentando
- `WL_CONNECT_FAILED` — senha errada
- `WL_NO_SSID_AVAIL` — rede nao encontrada

O `timeout_ms` evita que o boot trave indefinidamente se a rede nao estiver disponivel.

---

#### Funcao: `ota_check_and_download()`

Esta e a funcao principal. Passo a passo:

**Passo 1: Consulta /ota/latest**

```
GET http://192.168.1.100:8000/ota/latest

Resposta JSON:
{
  "model": {
    "version": "seismic_edge_v1_tiny_cnn_20260614",
    ...
  },
  "artifact": {
    "sha256": "f71ac3b1...",
    "path": "..."
  }
}
```

**Passo 2: Parse com ArduinoJson**

```cpp
JsonDocument doc;
deserializeJson(doc, body);
const char* server_version = doc["model"]["version"] | "";
const char* sha256         = doc["artifact"]["sha256"] | "";
```

`| ""` e o operador de fallback do ArduinoJson: se o campo nao existir, usa string vazia.

**Passo 3: Comparacao de versoes**

```cpp
if (strcmp(server_version, current_version) == 0) {
    result.already_current = true;
    return result;  // nada a fazer
}
```

**Passo 4: Download**

Chama `stream_to_spiffs()` que salva em `/ota_model.tflite`.

**Passo 5: Validacao SHA-256**

```
SHA-256 calculado do arquivo baixado
           vs
SHA-256 informado pelo servidor em latest.json

SIM → ok, salva metadados
NAO → SPIFFS.remove() ← apaga arquivo corrompido
```

Se o hash divergir, o arquivo e deletado imediatamente para nao deixar lixo no SPIFFS.

---

#### Funcao: `ota_load_model_into_ram()`

```
SPIFFS.exists("/ota_model.tflite") ── NAO ──► return false
      │
      ▼
File.open()
size = file.size()   ← ex: 66.768 bytes (float32)
      │
      ▼
heap_caps_aligned_alloc(
    16,              ← alinhamento de 16 bytes (TFLite exige)
    size,
    MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL  ← RAM interna do ESP32
)
      │
      ▼
file.read(s_model_buf, size)
      │
      ▼
s_model_size = size
return true
```

**Por que alinhamento de 16 bytes?** O TensorFlow Lite Micro acessa dados em blocos SIMD (Single Instruction Multiple Data). Se o buffer nao estiver alinhado, pode gerar acessos nao alinhados que em alguns processadores causam crash ou reducao de performance.

**`MALLOC_CAP_INTERNAL`** garante que o buffer fica na RAM interna (mais rapida), nao em PSRAM externa (se existisse).

---

#### Funcao: `ota_post_report()`

```cpp
JsonDocument doc;
doc["device_id"]              = DEVICE_ID;
doc["previous_model_version"] = prev_version;
doc["new_model_version"]      = new_version;
doc["status"]                 = status;   // "success"|"failed"|"skipped"
doc["message"]                = message;

http.POST(body);  // POST /ota/report
```

O servidor salva isso em `server/data/ota_reports/ota_report_<id>.json`. Serve para auditoria: saber quais dispositivos atualizaram, quando, e se deu certo.

---

### `src/main.cpp` — Integracao no setup()

```
setup() {
  ...
  SPIFFS.begin(true)           // monta ou formata
  wifi_connect(SSID, PASS)     // conecta ao WiFi
  ota_get_installed_version()  // le versao atual do SPIFFS
  ota_check_and_download()     // baixa se ha versao nova
  ota_post_report()            // informa servidor
  ota_load_model_into_ram()    // carrega modelo do SPIFFS (se tiver)
  ...
  model_data = ota_get_model_data()  // ponteiro do SPIFFS
           ou
  model_data = MODEL_DATA            // header compilado (fallback)
  ...
  tflite::GetModel(model_data)   // TFLite usa o ponteiro
  AllocateTensors()
  ... inferencia normal ...
}
```

**O fallback e importante:** se nao houver WiFi, ou o download falhar, ou nao houver SPIFFS, o ESP32 usa o modelo compilado no header `.h`. O dispositivo nunca fica sem modelo.

---

## Mapa de Memoria do ESP32

```
Flash (4 MB total — huge_app.csv):
┌─────────────────────┬──────────┐
│ Bootloader          │  ~8 KB   │ offset 0x0000
├─────────────────────┼──────────┤
│ NVS (config WiFi)   │  20 KB   │ offset 0x9000
├─────────────────────┼──────────┤
│ OTA data            │   8 KB   │ offset 0xe000
├─────────────────────┼──────────┤
│ app0 (firmware)     │   3 MB   │ offset 0x10000
│   (compilado: 638KB)│          │
├─────────────────────┼──────────┤
│ SPIFFS (dados)      │ ~960 KB  │ offset 0x310000
│  /ota_model.tflite  │  66 KB   │
│  /ota_meta.json     │   1 KB   │
└─────────────────────┴──────────┘

RAM (320 KB total — uso atual ~31%):
┌─────────────────────┬──────────┐
│ Sistema + Arduino   │  ~50 KB  │
├─────────────────────┼──────────┤
│ Codigo compilado    │  ~50 KB  │
├─────────────────────┼──────────┤
│ tensor_arena        │ 100 KB   │ heap_caps_aligned_alloc
├─────────────────────┼──────────┤
│ s_model_buf (OTA)   │  66 KB   │ heap_caps_aligned_alloc
├─────────────────────┼──────────┤
│ Disponivel          │  ~54 KB  │
└─────────────────────┴──────────┘
```

**Observacao:** quando o modelo vem do SPIFFS (OTA), ha dois buffers em RAM: `tensor_arena` (100 KB) e `s_model_buf` (66 KB). Total: 166 KB de 320 KB (52%). Dentro do limite.

---

## Seguranca: SHA-256

```
Servidor publica:
  artifact.tflite  →  SHA-256: "f71ac3b1ec6b..."  (em latest.json)

ESP32 faz:
  1. Baixa artifact.tflite para SPIFFS
  2. Calcula SHA-256 do arquivo salvo
  3. Compara com o hash do JSON

Se divergirem:
  • Arquivo e deletado do SPIFFS
  • Status "failed" enviado ao servidor
  • ESP32 continua com modelo anterior (fallback)
```

**Por que isso importa?** Sem validacao, um arquivo corrompido na transmissao ou um arquivo trocado por um atacante poderia ser carregado como modelo. O SHA-256 garante que o arquivo e exatamente o que o servidor publicou.

**Limitacao atual:** o hash vem no mesmo JSON do servidor. Para seguranca maxima, o ideal e assinar o hash com chave privada (RSA/ECDSA) e validar com chave publica no dispositivo. O projeto ja tem HMAC-SHA256 na camada MLOps — essa seria a proxima evolucao para o firmware.

---

## Como Testar

### 1. Servidor (PC)

```bash
cd ~/tcc_atual/TCC
source .venv/bin/activate
uvicorn server.app.main:app --host 0.0.0.0 --port 8000
```

Confirme que esta acessivel:

```bash
curl http://localhost:8000/ota/latest | python3 -m json.tool
```

### 2. IP do servidor no WSL

```bash
ip addr show eth0 | grep "inet "
# exemplo: inet 172.24.144.1
```

Use esse IP no `wifi_config.h`:

```cpp
#define OTA_SERVER_HOST "172.24.144.1"
```

### 3. Firmware

```bash
# Copiar template
cp include/wifi_config.h.template include/wifi_config.h
# Editar com suas credenciais
nano include/wifi_config.h

# Compilar e gravar
cd ~/tcc_atual/TCC/PlatformIO/Projects/TCC
PLATFORMIO_RUN_JOBS=1 pio run -e esp32dev -t upload
pio device monitor --port /dev/ttyUSB0 --baud 115200
```

### 4. Saida esperada no serial

```
[OTA] Iniciando verificacao de atualizacao...
[OTA] Versao local: builtin_float32_v1
[WiFi] Conectando a 'MinhaRede'...
.....
[WiFi] IP: 192.168.1.107
[OTA] GET http://192.168.1.100:8000/ota/latest
[OTA] Versao servidor : seismic_edge_v1_tiny_cnn_20260614
[OTA] Versao local    : builtin_float32_v1
[OTA] Nova versao disponivel — iniciando download...
[OTA] Baixados: 67768 bytes
[OTA] SHA-256 OK.
[OTA] Metadados salvos no SPIFFS.
[OTA] Atualizacao aplicada com sucesso.
[OTA] POST /ota/report -> HTTP 200
[OTA] Modelo SPIFFS carregado na RAM: 67768 bytes
[OTA] Usando modelo do SPIFFS (OTA).
================================
Modelo ativo : PIPELINE_TINY_CNN_FLOAT32
Threshold    : 0.72419429
================================
```

Na segunda inicializacao (modelo ja atualizado):

```
[OTA] Versao local: seismic_edge_v1_tiny_cnn_20260614
[OTA] Versao servidor : seismic_edge_v1_tiny_cnn_20260614
[OTA] Ja esta na versao mais recente.
[OTA] POST /ota/report -> HTTP 200
[OTA] Modelo SPIFFS carregado na RAM: 67768 bytes
[OTA] Usando modelo do SPIFFS (OTA).
```

---

## Fluxo Completo Integrado (MLOps → ESP32)

```
MLOps (PC)                         ESP32
──────────────────────────────     ──────────────────────────
1. train_all.py                    (inferindo com modelo anterior)
   └─ novo tiny_cnn treinado

2. promote_model.py
   └─ quality gate aprovado

3. export_tflite.py
   └─ tiny_cnn_float32.tflite

4. build_ota_manifest.py
5. build_ota_package.py
   └─ SHA-256 calculado
6. validate_ota_package.py
7. publish_local_release.py
   └─ releases/latest.json  ──────────────────────────────────
                                              │
8. uvicorn server/app/main:app                │ (proximo boot ou reset)
   └─ GET /ota/latest exposto  ◄──────────────┤
   └─ GET /ota/artifact exposto ─────────────►│
                                         ota_check_and_download()
                                         SHA-256 validado
                                         modelo salvo no SPIFFS
                                         POST /ota/report "success"
                                              │
                                         ota_load_model_into_ram()
                                         inferencia com novo modelo
```

---

## Proximas Evolucoes

| Item | Descricao |
|---|---|
| Reset automatico | Apos OTA bem-sucedida, chamar `ESP.restart()` para garantir ambiente limpo |
| Verificacao de assinatura | Validar HMAC/RSA do pacote no firmware, nao apenas SHA-256 do arquivo |
| OTA de firmware | Atualizar o binario Arduino inteiro via `esp_ota_ops.h` (particionamento ota_0/ota_1) |
| OTA em background | Baixar em task FreeRTOS separada enquanto inferencia continua |
| Rollback automatico | Se modelo novo causar erro no AllocateTensors, deletar SPIFFS e reverter para builtin |

---

## Glossario Rapido

| Termo | Significado |
|---|---|
| SPIFFS | SPI Flash File System — sistema de arquivos na flash do ESP32 |
| mbedtls | Biblioteca de criptografia do SDK ESP-IDF (inclusa automaticamente) |
| SHA-256 | Hash criptografico de 256 bits — garante integridade do arquivo |
| HMAC | Hash-based Message Authentication Code — garante origem do dado |
| heap_caps | API do ESP32 para alocar RAM com capacidades especificas |
| MALLOC_CAP_INTERNAL | Flag: usar apenas RAM interna (nao PSRAM) |
| streaming | Processar dados em pedacos pequenos sem carregar tudo na RAM |
| ArduinoJson | Biblioteca para parse e serializacao de JSON no Arduino/ESP32 |
| WL_CONNECTED | Status do WiFi: conectado com IP atribuido |
