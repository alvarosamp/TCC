//<|include <Arduino.h>
//#include <WiFi.h>
//#include <HTTPClient.h>
//#include <ArduinoJson.h>
//#include "secrets.h"
//#include "config.h"
#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ArduinoJson.h>

#include "config.h"
#include "secrets.h"

String currentModelVersion = "unknown";

unsigned long lastEventTime = 0;
unsigned long lastStatusTime = 0;
unsigned long lastOtaCheckTime = 0;

bool wifiConnected() {
    return WiFi.status() == WL_CONNECTED;
}

void printFreeHeap(const String& label) {
    Serial.print("[HEAP] ");
    Serial.print(label);
    Serial.print(": ");
    Serial.print(ESP.getFreeHeap());
    Serial.println(" bytes");
}

void connectWiFi() {
    Serial.println();
    Serial.println("Conectando ao Wi-Fi");
    Serial.print("SSID: ");
    Serial.println(WIFI_SSID);

    WiFi.mode(WIFI_STA);
    WiFi.persistent(false);
    WiFi.setAutoReconnect(true);

    if (WiFi.status() != WL_CONNECTED) {
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    }

    int attempts = 0;

    while (!wifiConnected() && attempts < 40) {
        delay(500);
        Serial.print(".");
        yield();
        attempts++;
    }

    Serial.println();

    if (wifiConnected()) {
        Serial.println("Conectado ao Wi-Fi");
        Serial.print("IP: ");
        Serial.println(WiFi.localIP());
        Serial.print("RSSI: ");
        Serial.println(WiFi.RSSI());
    } else {
        Serial.println("Falha ao conectar ao Wi-Fi");
        Serial.print("WiFi.status(): ");
        Serial.println(WiFi.status());
    }

    printFreeHeap("Depois do Wi-Fi");
}

bool postJson(const String& endpoint, const String& payload, String* responseBody = nullptr) {
    if (!wifiConnected()) {
        Serial.println("Wi-Fi desconectado. POST cancelado.");
        return false;
    }

    printFreeHeap("Antes do POST " + endpoint);

    WiFiClient client;
    HTTPClient http;

    String url = String(SERVER_URL) + endpoint;

    Serial.println();
    Serial.println("POST " + url);
    Serial.println("Payload: " + payload);

    if (!http.begin(client, url)) {
        Serial.println("Falha ao iniciar HTTPClient no POST.");
        return false;
    }

    http.setTimeout(5000);
    http.setReuse(false);
    http.addHeader("Content-Type", "application/json");
    http.addHeader("Connection", "close");

    int httpCode = http.POST(payload);

    Serial.println("HTTP Code: " + String(httpCode));

    if (httpCode > 0) {
        if (responseBody != nullptr) {
            *responseBody = http.getString();

            Serial.println("Resposta:");
            Serial.println(*responseBody);
        } else {
            Serial.println("POST enviado com sucesso. Resposta ignorada para economizar memoria.");
        }
    } else {
        Serial.print("Erro no POST: ");
        Serial.println(http.errorToString(httpCode));
    }

    http.end();
    client.stop();

    yield();
    delay(100);

    printFreeHeap("Depois do POST " + endpoint);

    return httpCode >= 200 && httpCode < 300;
}

bool getJson(const String& endpoint, String& responseBody) {
    if (!wifiConnected()) {
        Serial.println("Wi-Fi desconectado. GET cancelado.");
        return false;
    }

    printFreeHeap("Antes do GET " + endpoint);

    WiFiClient client;
    HTTPClient http;

    String url = String(SERVER_URL) + endpoint;

    Serial.println();
    Serial.println("GET " + url);

    if (!http.begin(client, url)) {
        Serial.println("Falha ao iniciar HTTPClient no GET.");
        return false;
    }

    http.setTimeout(5000);
    http.setReuse(false);
    http.addHeader("Connection", "close");

    int httpCode = http.GET();

    Serial.println("HTTP Code: " + String(httpCode));

    if (httpCode > 0) {
        responseBody = http.getString();

        Serial.println("Resposta:");
        Serial.println(responseBody);
    } else {
        Serial.print("Erro no GET: ");
        Serial.println(http.errorToString(httpCode));
    }

    http.end();
    client.stop();

    yield();
    delay(100);

    printFreeHeap("Depois do GET " + endpoint);

    return httpCode >= 200 && httpCode < 300;
}

void registerDevice() {
    Serial.println();
    Serial.println("[ETAPA] registerDevice - inicio");
    printFreeHeap("Antes de montar JSON registerDevice");

    StaticJsonDocument<384> doc;

    doc["device_id"] = DEVICE_ID;
    doc["device_type"] = DEVICE_TYPE;
    doc["location"] = DEVICE_LOCATION;
    doc["firmware_version"] = FIRMWARE_VERSION;
    doc["model_version"] = currentModelVersion;

    String payload;
    payload.reserve(256);
    serializeJson(doc, payload);

    bool ok = postJson("/devices/register", payload);

    if (ok) {
        Serial.println("[ETAPA] registerDevice - sucesso");
    } else {
        Serial.println("[ETAPA] registerDevice - falhou");
    }

    printFreeHeap("Depois de registerDevice");
    Serial.println("[ETAPA] registerDevice - fim");
}

void sendDeviceStatus() {
    Serial.println();
    Serial.println("[ETAPA] sendDeviceStatus - inicio");
    printFreeHeap("Antes de montar JSON sendDeviceStatus");

    StaticJsonDocument<640> doc;

    doc["device_id"] = DEVICE_ID;
    doc["firmware_version"] = FIRMWARE_VERSION;
    doc["model_version"] = currentModelVersion;
    doc["battery_level"] = 100.0;
    doc["free_memory_kb"] = ESP.getFreeHeap() / 1024.0;
    doc["signal_quality"] = WiFi.RSSI();

    JsonObject extra = doc.createNestedObject("extra");
    extra["wifi_ip"] = WiFi.localIP().toString();
    extra["chip_model"] = "ESP8266";
    extra["cpu_freq_mhz"] = ESP.getCpuFreqMHz();
    extra["flash_size_mb"] = ESP.getFlashChipRealSize() / (1024 * 1024);
    extra["sdk_version"] = ESP.getSdkVersion();

    String payload;
    payload.reserve(512);
    serializeJson(doc, payload);

    bool ok = postJson("/devices/status", payload);

    if (ok) {
        Serial.println("[ETAPA] sendDeviceStatus - sucesso");
    } else {
        Serial.println("[ETAPA] sendDeviceStatus - falhou");
    }

    printFreeHeap("Depois de sendDeviceStatus");
    Serial.println("[ETAPA] sendDeviceStatus - fim");
}

void sendOtaReport(
    const String& previousVersion,
    const String& newVersion,
    const String& status,
    const String& message
) {
    Serial.println();
    Serial.println("[ETAPA] sendOtaReport - inicio");

    StaticJsonDocument<512> doc;

    doc["device_id"] = DEVICE_ID;
    doc["previous_model_version"] = previousVersion;
    doc["new_model_version"] = newVersion;
    doc["status"] = status;
    doc["message"] = message;

    String payload;
    payload.reserve(512);
    serializeJson(doc, payload);

    bool ok = postJson("/ota/report", payload);

    if (ok) {
        Serial.println("[ETAPA] sendOtaReport - sucesso");
    } else {
        Serial.println("[ETAPA] sendOtaReport - falhou");
    }

    Serial.println("[ETAPA] sendOtaReport - fim");
}

void checkOtaLatest() {
    Serial.println();
    Serial.println("[ETAPA] checkOtaLatest - inicio");
    printFreeHeap("Antes de consultar OTA");

    String response;
    response.reserve(2048);

    bool ok = getJson("/ota/latest", response);

    if (!ok) {
        Serial.println("Nao foi possivel consultar /ota/latest.");
        Serial.println("[ETAPA] checkOtaLatest - fim com erro HTTP");
        return;
    }

    Serial.print("Tamanho da resposta OTA: ");
    Serial.println(response.length());

    if (response.length() == 0) {
        Serial.println("Resposta OTA vazia.");
        Serial.println("[ETAPA] checkOtaLatest - fim resposta vazia");
        return;
    }

    if (response.length() > 1800) {
        Serial.println("Resposta OTA muito grande para o buffer atual.");
        Serial.println("Aumente o DynamicJsonDocument ou reduza o manifesto no servidor.");
        Serial.println("[ETAPA] checkOtaLatest - fim resposta grande");
        return;
    }

    DynamicJsonDocument doc(2048);

    DeserializationError error = deserializeJson(doc, response);

    if (error) {
        Serial.print("Erro ao interpretar JSON do OTA: ");
        Serial.println(error.c_str());
        Serial.println("[ETAPA] checkOtaLatest - fim erro JSON");
        return;
    }

    const char* newVersion = doc["model"]["version"] | "unknown";
    const char* targetDevice = doc["target"]["device"] | "unknown";
    const char* runtime = doc["target"]["runtime"] | "unknown";
    const char* artifactSha = doc["artifact"]["sha256"] | "unknown";

    Serial.println();
    Serial.println("Manifesto OTA recebido:");
    Serial.print("Versao: ");
    Serial.println(newVersion);
    Serial.print("Target: ");
    Serial.println(targetDevice);
    Serial.print("Runtime: ");
    Serial.println(runtime);
    Serial.print("SHA256: ");
    Serial.println(artifactSha);

    if (String(targetDevice) != DEVICE_TYPE) {
        Serial.println("Manifesto OTA ignorado: target incompativel.");
        Serial.println("[ETAPA] checkOtaLatest - fim target incompativel");
        return;
    }

    if (currentModelVersion != String(newVersion)) {
        Serial.println("Nova versao detectada.");

        String previousVersion = currentModelVersion;
        currentModelVersion = String(newVersion);

        delay(500);
        yield();

        sendOtaReport(
            previousVersion,
            currentModelVersion,
            "success",
            "Manifesto OTA consultado com sucesso. Download real ainda nao aplicado neste MVP."
        );
    } else {
        Serial.println("Modelo ja esta atualizado.");
    }

    printFreeHeap("Depois de checkOtaLatest");
    Serial.println("[ETAPA] checkOtaLatest - fim");
}

float simulateTinyMLScore() {
    return random(0, 1000) / 1000.0f;
}

String classifyPrediction(float score) {
    if (score >= 0.7f) {
        return "anomaly";
    }

    if (score >= 0.4f && score <= 0.6f) {
        return "uncertain";
    }

    return "normal";
}

void fillSimulatedFeatures(JsonObject features) {
    features["mean"] = random(-200, 200) / 1000.0f;
    features["std"] = random(500, 2000) / 1000.0f;
    features["max"] = random(800, 4000) / 1000.0f;
    features["min"] = random(-4000, -800) / 1000.0f;
    features["free_heap_kb"] = ESP.getFreeHeap() / 1024.0f;
    features["rssi"] = WiFi.RSSI();
}

void sendInferenceEvent() {
    Serial.println();
    Serial.println("[ETAPA] sendInferenceEvent - inicio");
    printFreeHeap("Antes de montar JSON sendInferenceEvent");

    float score = simulateTinyMLScore();
    String prediction = classifyPrediction(score);

    StaticJsonDocument<768> doc;

    doc["device_id"] = DEVICE_ID;
    doc["model_version"] = currentModelVersion;
    doc["prediction"] = prediction;
    doc["score"] = score;
    doc["threshold"] = MODEL_THRESHOLD;
    doc["window_size"] = WINDOW_SIZE;
    doc["sampling_rate"] = SAMPLING_RATE;

    JsonObject features = doc.createNestedObject("features");
    fillSimulatedFeatures(features);

    JsonObject extra = doc.createNestedObject("extra");
    extra["source"] = "esp8266_platformio_physical_test";
    extra["firmware_version"] = FIRMWARE_VERSION;
    extra["tinyml_mode"] = "simulated_inference";

    String payload;
    payload.reserve(768);
    serializeJson(doc, payload);

    bool ok = postJson("/events", payload);

    if (ok) {
        Serial.println("[ETAPA] sendInferenceEvent - sucesso");
    } else {
        Serial.println("[ETAPA] sendInferenceEvent - falhou");
    }

    printFreeHeap("Depois de sendInferenceEvent");
    Serial.println("[ETAPA] sendInferenceEvent - fim");
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    randomSeed(analogRead(A0));

    Serial.println();
    Serial.println("=======================================");
    Serial.println("TCC TinyML ESP8266 Client - PlatformIO");
    Serial.println("=======================================");

    printFreeHeap("Inicio do setup");

    connectWiFi();

    if (wifiConnected()) {
        Serial.println();
        Serial.println("[SETUP] ANTES registerDevice");
        registerDevice();
        Serial.println("[SETUP] DEPOIS registerDevice");

        delay(1000);
        yield();

        Serial.println("[SETUP] ANTES checkOtaLatest");
        checkOtaLatest();
        Serial.println("[SETUP] DEPOIS checkOtaLatest");

        delay(1000);
        yield();

        Serial.println("[SETUP] ANTES sendDeviceStatus");
        sendDeviceStatus();
        Serial.println("[SETUP] DEPOIS sendDeviceStatus");

        delay(1000);
        yield();
    }

    lastEventTime = millis();
    lastStatusTime = millis();
    lastOtaCheckTime = millis();

    printFreeHeap("Fim do setup");

    Serial.println();
    Serial.println("[SETUP] Finalizado. Entrando no loop.");
}

void loop() {
    if (!wifiConnected()) {
        Serial.println("Wi-Fi caiu. Tentando reconectar...");
        connectWiFi();
        delay(1000);
        yield();
        return;
    }

    unsigned long now = millis();

    if (now - lastStatusTime >= STATUS_INTERVAL_MS) {
        lastStatusTime = now;
        sendDeviceStatus();
        delay(500);
        yield();
    }

    if (now - lastEventTime >= EVENT_INTERVAL_MS) {
        lastEventTime = now;
        sendInferenceEvent();
        delay(500);
        yield();
    }

    if (now - lastOtaCheckTime >= OTA_CHECK_INTERVAL_MS) {
        lastOtaCheckTime = now;
        checkOtaLatest();
        delay(500);
        yield();
    }

    delay(10);
    yield();
}