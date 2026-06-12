#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "secrets.h"
#include "config.h"

String currentModelVersion = "unknown";
unsigned long lastEventTime = 0;
unsigned long lastStatusTime = 0;
unsigned long lastOtaCheckTime = 0;

bool wifiConnected(){
  return WiFi.status() == WL_CONNECTED;
}
void connectWiFi() {
  Serial.println("Conectando ao wifi");
  Serial.print("SSID: ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;

  while (!wifiConnected() && attempts < 40){
    delay(500);
    Serial.print('.');
    attempts++;
  }
  Serial.println();

  if (wifiConnected()){
    Serial.println("Conectado ao wifi");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    Serial.println("RSSI: " + String(WiFi.RSSI()));
  } else {
    Serial.println("Falha ao conectar ao wifi");
  }
}

bool postJson(const String& endpoint, const String& payload, String* responseBody = nullptr) {
    if (!wifiConnected()) {
        Serial.println("Wifi desconectado. POST cancelado");
        return false;
    }

    HTTPClient http;
    String url = String(SERVER_URL) + endpoint;

    Serial.println();
    Serial.println("POST " + url);
    Serial.println("Payload: " + payload);

    http.begin(url);
    http.addHeader("Content-Type", "application/json");

    int httpCode = http.POST(payload);

    Serial.println("HTTP Code: " + String(httpCode));

    if (httpCode > 0) {
        String response = http.getString();
        Serial.println("Resposta:");
        Serial.println(response);

        if (responseBody != nullptr) {
            *responseBody = response;
        }
    } else {
        Serial.print("Erro no POST: ");
        Serial.println(http.errorToString(httpCode));
    }

    http.end();

    return httpCode >= 200 && httpCode < 300;
}
bool getJson(const String& endpoint, String& responseBody) {
    if (!wifiConnected()) {
        Serial.println("Wifi desconectado. GET cancelado");
        return false;
    }

    HTTPClient http;
    String url = String(SERVER_URL) + endpoint;

    Serial.println();
    Serial.println("GET " + url);

    http.begin(url);

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

    return httpCode >= 200 && httpCode < 300;
}

void registerDevice() {
    StaticJsonDocument<512> doc;

    doc["device_id"] = DEVICE_ID;
    doc["device_type"] = DEVICE_TYPE;
    doc["location"] = DEVICE_LOCATION;
    doc["firmware_version"] = FIRMWARE_VERSION;
    doc["model_version"] = currentModelVersion;

    String payload;
    serializeJson(doc, payload);

    postJson("/devices/register", payload);
}

void sendDeviceStatus() {
    StaticJsonDocument<768> doc;

    doc["device_id"] = DEVICE_ID;
    doc["firmware_version"] = FIRMWARE_VERSION;
    doc["model_version"] = currentModelVersion;
    doc["battery_level"] = 100.0;
    doc["free_memory_kb"] = ESP.getFreeHeap() / 1024.0;
    doc["signal_quality"] = WiFi.RSSI();

    JsonObject extra = doc.createNestedObject("extra");
    extra["wifi_ip"] = WiFi.localIP().toString();
    extra["chip_model"] = ESP.getChipModel();
    extra["cpu_freq_mhz"] = ESP.getCpuFreqMHz();
    extra["flash_size_mb"] = ESP.getFlashChipSize() / (1024 * 1024);
    extra["sdk_version"] = ESP.getSdkVersion();

    String payload;
    serializeJson(doc, payload);

    postJson("/devices/status", payload);
}

void sendOtaReport(
    const String& previousVersion,
    const String& newVersion,
    const String& status,
    const String& message
) {
    StaticJsonDocument<512> doc;

    doc["device_id"] = DEVICE_ID;
    doc["previous_model_version"] = previousVersion;
    doc["new_model_version"] = newVersion;
    doc["status"] = status;
    doc["message"] = message;

    String payload;
    serializeJson(doc, payload);

    postJson("/ota/report", payload);
}
void checkOtaLatest() {
    String response;

    bool ok = getJson("/ota/latest", response);

    if (!ok) {
        Serial.println("Nao foi possivel consultar /ota/latest.");
        return;
    }

    StaticJsonDocument<4096> doc;

    DeserializationError error = deserializeJson(doc, response);

    if (error) {
        Serial.print("Erro ao interpretar JSON do OTA: ");
        Serial.println(error.c_str());
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
        Serial.println("Manifesto OTA ignorado: target incompatível.");
        return;
    }

    if (currentModelVersion != String(newVersion)) {
        Serial.println("Nova versao detectada.");

        String previousVersion = currentModelVersion;
        currentModelVersion = String(newVersion);

        sendOtaReport(
            previousVersion,
            currentModelVersion,
            "success",
            "Manifesto OTA consultado com sucesso. Download real ainda nao aplicado neste MVP."
        );
    } else {
        Serial.println("Modelo ja esta atualizado.");
    }
}

float simulateTinyMLScore() {
    /*
      Aqui ainda estamos simulando a inferencia.

      Depois voce troca esta funcao por:
      1. coleta da janela temporal
      2. preprocessamento
      3. chamada do TensorFlow Lite Micro
      4. retorno do score real do modelo
    */

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
    float score = simulateTinyMLScore();
    String prediction = classifyPrediction(score);

    StaticJsonDocument<1024> doc;

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
    extra["source"] = "esp32_platformio_physical_test";
    extra["firmware_version"] = FIRMWARE_VERSION;
    extra["tinyml_mode"] = "simulated_inference";

    String payload;
    serializeJson(doc, payload);

    postJson("/events", payload);
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    randomSeed(analogRead(0));

    Serial.println();
    Serial.println("=====================================");
    Serial.println("TCC TinyML ESP32 Client - PlatformIO");
    Serial.println("=====================================");

    connectWiFi();

    if (wifiConnected()) {
        registerDevice();
        checkOtaLatest();
        sendDeviceStatus();
    }

    lastEventTime = millis();
    lastStatusTime = millis();
    lastOtaCheckTime = millis();
}

void loop() {
    if (!wifiConnected()) {
        Serial.println("Wi-Fi caiu. Tentando reconectar...");
        connectWiFi();
        delay(1000);
        return;
    }

    unsigned long now = millis();

    if (now - lastStatusTime >= STATUS_INTERVAL_MS) {
        lastStatusTime = now;
        sendDeviceStatus();
    }

    if (now - lastEventTime >= EVENT_INTERVAL_MS) {
        lastEventTime = now;
        sendInferenceEvent();
    }

    if (now - lastOtaCheckTime >= OTA_CHECK_INTERVAL_MS) {
        lastOtaCheckTime = now;
        checkOtaLatest();
    }
}