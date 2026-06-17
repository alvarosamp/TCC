#pragma once

#include <stdint.h>
#include <stddef.h>

// ============================================================
//  OTA HTTP — Modulo de atualizacao de modelo via servidor
//
//  Fluxo:
//    1. wifi_connect()              — conecta ao AP
//    2. ota_check_and_download()    — consulta /ota/latest,
//                                    baixa artefato se versao nova,
//                                    valida SHA-256
//    3. ota_load_model_into_ram()   — le do SPIFFS para buffer
//    4. ota_get_model_data()        — ponteiro usado pelo TFLite
//    5. ota_post_report()           — reporta resultado ao servidor
// ============================================================

// Caminho do modelo baixado via OTA dentro do SPIFFS
#define OTA_SPIFFS_MODEL_PATH  "/ota_model.tflite"
// Metadados da versao instalada (JSON)
#define OTA_SPIFFS_META_PATH   "/ota_meta.json"

// ----------------------------------------
// Resultado de uma verificacao OTA
// ----------------------------------------
struct OtaResult {
    bool update_available;   // servidor tem versao mais nova
    bool download_ok;        // download concluido sem erro HTTP
    bool sha256_ok;          // integridade confirmada
    bool already_current;    // versao local ja e a mais recente
    char version[64];        // versao baixada (ou disponivel)
    char sha256[65];         // hash esperado (hex, 64 chars + \0)
    char error[256];         // mensagem de erro se algo falhou
};

// ----------------------------------------
// WiFi
// ----------------------------------------

/**
 * Conecta ao WiFi. Retorna true se conectou dentro do timeout.
 * timeout_ms: tempo maximo de espera (padrao 15 s).
 */
bool wifi_connect(const char* ssid, const char* password,
                  uint32_t timeout_ms = 15000);

// ----------------------------------------
// OTA
// ----------------------------------------

/**
 * Consulta GET /ota/latest no servidor. Se ha versao nova,
 * baixa o artefato via GET /ota/artifact, salva no SPIFFS
 * e valida SHA-256. Retorna OtaResult com detalhes.
 *
 * host: IP ou hostname do servidor (ex: "192.168.1.100")
 * port: porta do servidor (ex: 8000)
 * current_version: versao atualmente em uso (para comparar)
 */
OtaResult ota_check_and_download(const char* host, uint16_t port,
                                  const char* current_version);

/**
 * Le o modelo salvo no SPIFFS para um buffer em RAM.
 * Deve ser chamada apos ota_check_and_download() bem-sucedida
 * ou quando SPIFFS ja tiver um modelo de download anterior.
 * Retorna true se conseguiu carregar.
 */
bool ota_load_model_into_ram();

/**
 * Retorna ponteiro para o buffer do modelo carregado do SPIFFS.
 * Retorna nullptr se ota_load_model_into_ram() nao foi chamada
 * ou falhou.
 */
const uint8_t* ota_get_model_data();

/**
 * Verifica se existe modelo baixado via OTA no SPIFFS.
 */
bool ota_spiffs_model_exists();

/**
 * Le a versao instalada no SPIFFS (armazenada em ota_meta.json).
 * Preenche buf com a string da versao. Retorna false se nao ha
 * metadado salvo.
 */
bool ota_get_installed_version(char* buf, size_t buf_size);

// ----------------------------------------
// Reporte de status ao servidor
// ----------------------------------------

/**
 * Envia POST /ota/report informando ao servidor o resultado
 * da atualizacao. Deve ser chamada apos tentativa de OTA.
 *
 * status:  "success" | "failed" | "skipped" | "rollback"
 * message: descricao livre do resultado
 */
void ota_post_report(const char* host, uint16_t port,
                     const char* device_id,
                     const char* prev_version,
                     const char* new_version,
                     const char* status,
                     const char* message);
