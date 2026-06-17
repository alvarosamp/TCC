#include "ota_http.h"

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <SPIFFS.h>
#include <ArduinoJson.h>
#include <mbedtls/md.h>
#include <esp_heap_caps.h>

// ============================================================
//  Estado interno do modulo
// ============================================================

static uint8_t* s_model_buf  = nullptr;
static size_t   s_model_size = 0;

// ============================================================
//  Utilitarios internos
// ============================================================

// Calcula SHA-256 do arquivo no SPIFFS e preenche hex[65].
static bool sha256_of_file(const char* path, char hex_out[65]) {
    File f = SPIFFS.open(path, "r");
    if (!f) return false;

    mbedtls_md_context_t ctx;
    const mbedtls_md_info_t* info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
    mbedtls_md_init(&ctx);
    mbedtls_md_setup(&ctx, info, 0);
    mbedtls_md_starts(&ctx);

    uint8_t chunk[512];
    while (f.available()) {
        size_t n = f.read(chunk, sizeof(chunk));
        mbedtls_md_update(&ctx, chunk, n);
    }
    f.close();

    uint8_t hash[32];
    mbedtls_md_finish(&ctx, hash);
    mbedtls_md_free(&ctx);

    for (int i = 0; i < 32; i++)
        snprintf(hex_out + i * 2, 3, "%02x", hash[i]);
    hex_out[64] = '\0';
    return true;
}

// Baixa URL por streaming direto para um arquivo no SPIFFS.
// Retorna numero de bytes escritos, ou -1 em caso de erro.
static int stream_to_spiffs(const String& url, const char* path) {
    HTTPClient http;
    http.begin(url);
    http.setTimeout(30000);
    int code = http.GET();
    if (code != 200) {
        Serial.printf("[OTA] download HTTP %d: %s\n", code, url.c_str());
        http.end();
        return -1;
    }

    File f = SPIFFS.open(path, "w");
    if (!f) {
        Serial.println("[OTA] Falha ao abrir SPIFFS para escrita");
        http.end();
        return -1;
    }

    WiFiClient* stream = http.getStreamPtr();
    int content_len   = http.getSize();
    uint8_t buf[512];
    int written = 0;
    uint32_t t0 = millis();

    while (http.connected() &&
           (content_len < 0 || written < content_len) &&
           (millis() - t0 < 30000)) {

        size_t avail = stream->available();
        if (avail == 0) { delay(1); continue; }

        size_t n = stream->readBytes(buf, min(avail, (size_t)sizeof(buf)));
        f.write(buf, n);
        written += (int)n;
    }

    f.close();
    http.end();

    if (content_len > 0 && written != content_len) {
        Serial.printf("[OTA] Incompleto: %d/%d bytes\n", written, content_len);
        return -1;
    }
    return written;
}

// Salva metadados da versao instalada no SPIFFS.
static void save_meta(const char* version, const char* sha256) {
    File f = SPIFFS.open(OTA_SPIFFS_META_PATH, "w");
    if (!f) return;
    JsonDocument doc;
    doc["version"] = version;
    doc["sha256"]  = sha256;
    serializeJson(doc, f);
    f.close();
}

// ============================================================
//  WiFi
// ============================================================

bool wifi_connect(const char* ssid, const char* password, uint32_t timeout_ms) {
    Serial.printf("[WiFi] Conectando a '%s'...\n", ssid);
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - t0) < timeout_ms) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("[WiFi] IP: ");
        Serial.println(WiFi.localIP());
        return true;
    }

    Serial.println("[WiFi] Timeout — sem conexao.");
    return false;
}

// ============================================================
//  OTA — verificacao e download
// ============================================================

OtaResult ota_check_and_download(const char* host, uint16_t port,
                                  const char* current_version) {
    OtaResult result = {};

    // --- 1. Consulta /ota/latest ---
    String url_latest = String("http://") + host + ":" + port + "/ota/latest";
    Serial.printf("[OTA] GET %s\n", url_latest.c_str());

    HTTPClient http;
    http.begin(url_latest);
    http.setTimeout(10000);
    int code = http.GET();
    if (code != 200) {
        snprintf(result.error, sizeof(result.error),
                 "/ota/latest retornou HTTP %d", code);
        http.end();
        return result;
    }

    String body = http.getString();
    http.end();

    // --- 2. Parse JSON ---
    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, body);
    if (err) {
        snprintf(result.error, sizeof(result.error),
                 "JSON invalido: %s", err.c_str());
        return result;
    }

    const char* server_version = doc["model"]["version"] | "";
    const char* sha256         = doc["artifact"]["sha256"] | "";

    strncpy(result.version, server_version, sizeof(result.version) - 1);
    strncpy(result.sha256,  sha256,         sizeof(result.sha256)  - 1);

    Serial.printf("[OTA] Versao servidor : %s\n", server_version);
    Serial.printf("[OTA] Versao local    : %s\n", current_version);

    // --- 3. Compara versoes ---
    if (strcmp(server_version, current_version) == 0) {
        result.already_current = true;
        Serial.println("[OTA] Ja esta na versao mais recente.");
        return result;
    }

    result.update_available = true;
    Serial.println("[OTA] Nova versao disponivel — iniciando download...");

    // --- 4. Download do artefato ---
    String url_artifact = String("http://") + host + ":" + port + "/ota/artifact";
    int bytes = stream_to_spiffs(url_artifact, OTA_SPIFFS_MODEL_PATH);
    if (bytes < 0) {
        snprintf(result.error, sizeof(result.error),
                 "Falha no download do artefato");
        return result;
    }
    Serial.printf("[OTA] Baixados: %d bytes\n", bytes);
    result.download_ok = true;

    // --- 5. Valida SHA-256 ---
    char actual_sha256[65];
    if (!sha256_of_file(OTA_SPIFFS_MODEL_PATH, actual_sha256)) {
        snprintf(result.error, sizeof(result.error),
                 "Falha ao calcular SHA-256 do arquivo baixado");
        SPIFFS.remove(OTA_SPIFFS_MODEL_PATH);
        return result;
    }

    if (strcmp(actual_sha256, sha256) != 0) {
        snprintf(result.error, sizeof(result.error),
                 "SHA-256 diverge!\n  esperado: %s\n  obtido  : %s",
                 sha256, actual_sha256);
        SPIFFS.remove(OTA_SPIFFS_MODEL_PATH);
        return result;
    }

    Serial.println("[OTA] SHA-256 OK.");
    result.sha256_ok = true;

    // --- 6. Salva metadados ---
    save_meta(server_version, sha256);
    Serial.println("[OTA] Metadados salvos no SPIFFS.");

    return result;
}

// ============================================================
//  Carregamento do modelo do SPIFFS para RAM
// ============================================================

bool ota_spiffs_model_exists() {
    return SPIFFS.exists(OTA_SPIFFS_MODEL_PATH);
}

bool ota_get_installed_version(char* buf, size_t buf_size) {
    if (!SPIFFS.exists(OTA_SPIFFS_META_PATH)) return false;

    File f = SPIFFS.open(OTA_SPIFFS_META_PATH, "r");
    if (!f) return false;

    JsonDocument doc;
    if (deserializeJson(doc, f)) { f.close(); return false; }
    f.close();

    const char* v = doc["version"] | "";
    strncpy(buf, v, buf_size - 1);
    buf[buf_size - 1] = '\0';
    return strlen(buf) > 0;
}

bool ota_load_model_into_ram() {
    if (!SPIFFS.exists(OTA_SPIFFS_MODEL_PATH)) {
        Serial.println("[OTA] Nenhum modelo no SPIFFS.");
        return false;
    }

    File f = SPIFFS.open(OTA_SPIFFS_MODEL_PATH, "r");
    if (!f) return false;

    size_t sz = f.size();
    if (sz == 0) { f.close(); return false; }

    // Libera buffer anterior se existir
    if (s_model_buf) {
        heap_caps_free(s_model_buf);
        s_model_buf  = nullptr;
        s_model_size = 0;
    }

    // Aloca com alinhamento de 16 bytes (exigido pelo TFLite)
    s_model_buf = (uint8_t*)heap_caps_aligned_alloc(16, sz,
                    MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    if (!s_model_buf) {
        Serial.printf("[OTA] Sem RAM para modelo SPIFFS (%u bytes)\n",
                      (unsigned)sz);
        f.close();
        return false;
    }

    size_t lido = f.read(s_model_buf, sz);
    f.close();

    if (lido != sz) {
        heap_caps_free(s_model_buf);
        s_model_buf = nullptr;
        return false;
    }

    s_model_size = sz;
    Serial.printf("[OTA] Modelo SPIFFS carregado na RAM: %u bytes\n",
                  (unsigned)sz);
    return true;
}

const uint8_t* ota_get_model_data() {
    return s_model_buf;
}

// ============================================================
//  Reporte de status ao servidor
// ============================================================

void ota_post_report(const char* host, uint16_t port,
                     const char* device_id,
                     const char* prev_version,
                     const char* new_version,
                     const char* status,
                     const char* message) {

    String url = String("http://") + host + ":" + port + "/ota/report";

    JsonDocument doc;
    doc["device_id"]             = device_id;
    doc["previous_model_version"] = prev_version;
    doc["new_model_version"]      = new_version;
    doc["status"]                 = status;
    doc["message"]                = message;

    String body;
    serializeJson(doc, body);

    HTTPClient http;
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(10000);
    int code = http.POST(body);
    Serial.printf("[OTA] POST /ota/report -> HTTP %d\n", code);
    http.end();
}
