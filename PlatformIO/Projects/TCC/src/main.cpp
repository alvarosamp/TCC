#include <Arduino.h>
#include <esp_bt.h>
#include <esp_heap_caps.h>
#include <float.h>
#include <math.h>
#include <SPIFFS.h>
#include <WiFi.h>

// ============================================================
//  Para trocar o modelo: edite apenas model_config.h
//  Mude ACTIVE_MODEL para MODEL_TCN_INT8 / FLOAT16 / FLOAT32
// ============================================================
#include "model_config.h"
#include "preprocessing.h"
#include "ota_http.h"

// wifi_config.h define WIFI_SSID, WIFI_PASSWORD, OTA_SERVER_HOST,
// OTA_SERVER_PORT e DEVICE_ID.
// Copie include/wifi_config.h.template para include/wifi_config.h
// e preencha com suas credenciais antes de compilar.
#if __has_include("wifi_config.h")
  #include "wifi_config.h"
  #define HAS_WIFI_CONFIG 1
#else
  #define HAS_WIFI_CONFIG 0
  #define WIFI_SSID       ""
  #define WIFI_PASSWORD   ""
  #define OTA_SERVER_HOST ""
  #define OTA_SERVER_PORT  8000
  #define DEVICE_ID       "esp32_sem_config"
#endif

#if __has_include("real_dataset.h")
  #include "real_dataset.h"
  #define HAS_REAL_DATASET 1
#else
  #define HAS_REAL_DATASET 0
#endif

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// =============================
// Configuracoes do experimento
// =============================

// Janela: 800 amostras = 20 s a 40 Hz.
constexpr int kWindowSize = 800;

#if HAS_REAL_DATASET
static_assert(kRealDatasetWindowSize == kWindowSize,
              "real_dataset.h precisa ter janelas com 800 amostras.");
#endif

// Threshold: vem automaticamente do model_config.h.
// Para sobrescrever manualmente, comente a linha abaixo e
// defina o valor desejado diretamente:
//   constexpr float kThreshold = 0.75f;
constexpr float kThreshold = MODEL_THRESHOLD;

// Pipeline de preprocessamento na borda.
//   true  → pipeline completo: detrend + taper + bandpass + zscore
//           (use com --raw no export — dado bruto ou nao normalizado)
//   false → apenas zscore  (comportamento legado)
constexpr bool kEdgeFullPipeline = true;

// Intervalo de decisao (usado para estimar %CPU).
// Se a janela anda de 10 em 10 s, use 10000 ms.
constexpr float kDecisionIntervalMs = 10000.0f;

// Estimativa de consumo eletrico (ajuste apos medir corrente real).
constexpr float kVoltage  = 3.3f;
constexpr float kCurrentA = 0.08f;

// Tensor arena. Alocada ANTES do WiFi para garantir bloco continuo.
constexpr int kTensorArenaSize = 80 * 1024;
uint8_t* tensor_arena = nullptr;
size_t tensor_arena_size = kTensorArenaSize;

// Numero de inferencias por rodada.
constexpr int kNumRuns = 100;

// =============================
// Variaveis globais TFLM
// =============================

// Versao do modelo compilado no firmware (fallback quando nao ha OTA).
// Deve ser atualizada manualmente quando o .h mudar.
static const char kBuiltinModelVersion[] = "builtin_float32_v1";

const tflite::Model* model      = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

float raw_window[kWindowSize];
float norm_window[kWindowSize];

float input_scale      = 1.0f;
int   input_zero_point = 0;
float output_scale      = 1.0f;
int   output_zero_point = 0;

struct MetricStats {
  float sum;
  float min_value;
  float max_value;
};

extern "C" void DebugLog(const char* s) {
  Serial.print(s);
}

MetricStats novaMetrica() {
  return {0.0f, FLT_MAX, -FLT_MAX};
}

void atualizarMetrica(MetricStats& stats, float value) {
  stats.sum += value;
  if (value < stats.min_value) stats.min_value = value;
  if (value > stats.max_value) stats.max_value = value;
}

void imprimirResumoLinha(const char* nome, const MetricStats& stats, int count) {
  float avg = stats.sum / count;
  Serial.print("# summary,");
  Serial.print(nome);
  Serial.print(",");
  Serial.print(avg, 6);
  Serial.print(",");
  Serial.print(stats.min_value, 6);
  Serial.print(",");
  Serial.println(stats.max_value, 6);
}

void imprimirResumoRodada(
  int count,
  const MetricStats& preprocess_stats,
  const MetricStats& inference_stats,
  const MetricStats& total_stats,
  const MetricStats& fps_stats,
  const MetricStats& cpu_stats,
  const MetricStats& energy_stats
) {
  Serial.println("# summary,metric,avg,min,max");
  imprimirResumoLinha("preprocess_ms", preprocess_stats, count);
  imprimirResumoLinha("inference_ms", inference_stats, count);
  imprimirResumoLinha("total_ms", total_stats, count);
  imprimirResumoLinha("fps", fps_stats, count);
  imprimirResumoLinha("cpu_percent_est", cpu_stats, count);
  imprimirResumoLinha("energy_mj_est", energy_stats, count);
}

size_t maiorBlocoLivreInterno() {
  return heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
}

void liberarMemoriaNaoUsada() {
  Serial.print("Maior bloco livre antes BT release: ");
  Serial.print(maiorBlocoLivreInterno());
  Serial.println(" bytes");

  esp_err_t bt_status = esp_bt_mem_release(ESP_BT_MODE_BTDM);
  Serial.print("BT memory release: ");
  Serial.println(bt_status == ESP_OK ? "OK" : "nao liberada");

  Serial.print("Maior bloco livre apos BT release : ");
  Serial.print(maiorBlocoLivreInterno());
  Serial.println(" bytes");
}

// =============================
// Funcoes auxiliares
// =============================

void preencherJanelaSintetica() {
  for (int i = 0; i < kWindowSize; i++) {
    float t = i / 40.0f;
    raw_window[i] =
      0.3f  * sinf(2.0f * PI *  2.0f * t) +
      0.1f  * sinf(2.0f * PI *  7.0f * t) +
      0.02f * sinf(2.0f * PI * 13.0f * t);
  }
  // Pulso artificial para simular trecho mais energetico.
  for (int i = 360; i < 430; i++) {
    float t = (i - 360) / 40.0f;
    raw_window[i] += 1.0f * sinf(2.0f * PI * 6.0f * t);
  }
}

void normalizarZScore() {
  float soma = 0.0f;
  for (int i = 0; i < kWindowSize; i++) soma += raw_window[i];
  float media = soma / kWindowSize;

  float soma_q = 0.0f;
  for (int i = 0; i < kWindowSize; i++) {
    float d = raw_window[i] - media;
    soma_q += d * d;
  }
  float desvio = sqrtf(soma_q / kWindowSize) + 1e-6f;

  for (int i = 0; i < kWindowSize; i++)
    norm_window[i] = (raw_window[i] - media) / desvio;
}

int prepararJanelaEntrada(int run) {
#if HAS_REAL_DATASET
  int window_index = run % kRealDatasetWindowCount;

  if (kRealDatasetAlreadyPreprocessed) {
    // Dado ja normalizado: copia direto para norm_window.
    for (int i = 0; i < kWindowSize; i++)
      norm_window[i] = real_dataset_windows[window_index][i];
  } else if (kEdgeFullPipeline) {
    // Pipeline completo na borda: detrend + taper + bandpass + zscore.
    for (int i = 0; i < kWindowSize; i++)
      raw_window[i] = real_dataset_windows[window_index][i];
    preprocessarBorda(raw_window, norm_window, kWindowSize);
  } else {
    // Legado: apenas zscore.
    for (int i = 0; i < kWindowSize; i++)
      raw_window[i] = real_dataset_windows[window_index][i];
    normalizarZScore();
  }

  return real_dataset_labels[window_index];
#else
  preencherJanelaSintetica();
  normalizarZScore();
  return -1;
#endif
}

int8_t quantizarFloatParaInt8(float x) {
  int q = (int)roundf(x / input_scale) + input_zero_point;
  if (q < -128) q = -128;
  if (q >  127) q =  127;
  return (int8_t)q;
}

float dequantizarInt8ParaFloat(int8_t y) {
  return ((int)y - output_zero_point) * output_scale;
}

void copiarEntradaParaTensor() {
  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < kWindowSize; i++)
      input->data.f[i] = norm_window[i];
    return;
  }
  if (input->type == kTfLiteInt8) {
    for (int i = 0; i < kWindowSize; i++)
      input->data.int8[i] = quantizarFloatParaInt8(norm_window[i]);
    return;
  }
  Serial.println("ERRO: tipo de entrada nao suportado");
}

float lerScoreSaida() {
  if (output->type == kTfLiteFloat32) return output->data.f[0];
  if (output->type == kTfLiteInt8)    return dequantizarInt8ParaFloat(output->data.int8[0]);
  Serial.println("ERRO: tipo de saida nao suportado");
  return 0.0f;
}

const char* tipoTensor(TfLiteType type) {
  if (type == kTfLiteFloat32) return "float32";
  if (type == kTfLiteInt8)    return "int8";
  if (type == kTfLiteUInt8)   return "uint8";
  return "outro";
}

void imprimirCabecalhoCsv() {
  Serial.println(
    "run,"
    "score,"
    "pred,"
    "expected_label,"
    "correct,"
    "preprocess_ms,"
    "inference_ms,"
    "total_ms,"
    "fps,"
    "cpu_percent_est,"
    "energy_mj_est,"
    "tensor_arena_kb,"
    "heap_free_before_kb,"
    "heap_free_after_kb,"
    "input_type,"
    "output_type"
  );
}

void imprimirLinhaCsv(
  int   run,
  float score,
  int   pred,
  int   expected_label,
  float preprocess_ms,
  float inference_ms,
  float total_ms,
  float heap_before_kb,
  float heap_after_kb
) {
  float fps          = 1000.0f / inference_ms;
  float cpu_percent  = (total_ms / kDecisionIntervalMs) * 100.0f;
  float energy_mj    = kVoltage * kCurrentA * (total_ms / 1000.0f) * 1000.0f;
  float arena_kb     = tensor_arena_size / 1024.0f;
  int correct         = expected_label < 0 ? -1 : (pred == expected_label ? 1 : 0);

  Serial.print(run);             Serial.print(",");
  Serial.print(score, 6);        Serial.print(",");
  Serial.print(pred);            Serial.print(",");
  Serial.print(expected_label);  Serial.print(",");
  Serial.print(correct);         Serial.print(",");
  Serial.print(preprocess_ms,4); Serial.print(",");
  Serial.print(inference_ms, 4); Serial.print(",");
  Serial.print(total_ms,     4); Serial.print(",");
  Serial.print(fps,          4); Serial.print(",");
  Serial.print(cpu_percent,  6); Serial.print(",");
  Serial.print(energy_mj,    6); Serial.print(",");
  Serial.print(arena_kb,     2); Serial.print(",");
  Serial.print(heap_before_kb,2);Serial.print(",");
  Serial.print(heap_after_kb, 2);Serial.print(",");
  Serial.print(tipoTensor(input->type));  Serial.print(",");
  Serial.println(tipoTensor(output->type));
}

// =============================
// Setup
// =============================

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("================================");
  Serial.print  ("Modelo ativo : "); Serial.println(MODEL_NAME);
  Serial.print  ("Threshold    : "); Serial.println(kThreshold, 8);
  Serial.println("================================");
  Serial.println("Inicializando TFLite Micro...");

#if HAS_REAL_DATASET
  Serial.print("Dataset real : "); Serial.print(kRealDatasetWindowCount);
  Serial.println(" janelas");
  Serial.print("Preprocessamento: ");
  if (kRealDatasetAlreadyPreprocessed) {
    Serial.println("dataset ja preprocessado (copia direta)");
  } else if (kEdgeFullPipeline) {
    Serial.println("borda completo (detrend+taper+bandpass+zscore)");
  } else {
    Serial.println("borda zscore-only");
  }
#else
  Serial.println("Entrada: janela sintetica");
#endif

  liberarMemoriaNaoUsada();

  // ============================================================
  //  FASE 1: OTA via HTTP (WiFi ligado)
  //  Tenta conectar ao WiFi e verificar atualizacao de modelo.
  //  Se nao houver wifi_config.h ou conexao falhar, continua
  //  normalmente com o modelo compilado no firmware.
  // ============================================================
  if (!SPIFFS.begin(true)) {
    Serial.println("[OTA] Falha ao montar SPIFFS — OTA desabilitado.");
  }

#if HAS_WIFI_CONFIG
  Serial.println("[OTA] Iniciando verificacao de atualizacao...");

  // Descobre versao local (SPIFFS ou builtin)
  char local_version[64];
  if (!ota_get_installed_version(local_version, sizeof(local_version))) {
    strncpy(local_version, kBuiltinModelVersion, sizeof(local_version) - 1);
  }
  Serial.printf("[OTA] Versao local: %s\n", local_version);

  bool wifi_ok = wifi_connect(WIFI_SSID, WIFI_PASSWORD, 15000);
  if (wifi_ok) {
    OtaResult ota = ota_check_and_download(OTA_SERVER_HOST, OTA_SERVER_PORT,
                                            local_version);
    if (ota.already_current) {
      Serial.println("[OTA] Modelo ja e o mais recente.");
      ota_post_report(OTA_SERVER_HOST, OTA_SERVER_PORT,
                      DEVICE_ID, local_version, local_version,
                      "skipped", "Versao ja atual");
    } else if (ota.sha256_ok) {
      Serial.println("[OTA] Atualizacao aplicada com sucesso.");
      ota_post_report(OTA_SERVER_HOST, OTA_SERVER_PORT,
                      DEVICE_ID, local_version, ota.version,
                      "success", "Modelo atualizado via HTTP OTA");
    } else if (ota.update_available) {
      Serial.printf("[OTA] Falha na atualizacao: %s\n", ota.error);
      ota_post_report(OTA_SERVER_HOST, OTA_SERVER_PORT,
                      DEVICE_ID, local_version, ota.version,
                      "failed", ota.error);
    }
  } else {
    Serial.println("[OTA] Sem WiFi — usando modelo local.");
  }
#else
  Serial.println("[OTA] wifi_config.h nao encontrado — OTA pulado.");
  Serial.println("      Copie include/wifi_config.h.template para include/wifi_config.h");
#endif

  // ============================================================
  //  FASE 2: Desliga WiFi e aloca arena grande para inferencia
  // ============================================================
  WiFi.disconnect(true);
  delay(200);
  {
    size_t free_now = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    Serial.print("[MEM] RAM livre apos WiFi off: "); Serial.print(free_now); Serial.println(" bytes");
    // Usa o maximo disponivel, reservando 16KB para stack e sistema
    tensor_arena_size = (free_now > 20 * 1024) ? (free_now - 16 * 1024) : free_now;
    tensor_arena_size = tensor_arena_size & ~15u;
    tensor_arena = static_cast<uint8_t*>(
      heap_caps_aligned_alloc(16, tensor_arena_size, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL)
    );
  }
  if (tensor_arena == nullptr) {
    Serial.println("ERRO: falha ao alocar tensor_arena");
    while (true) delay(1000);
  }
  Serial.print("[MEM] tensor_arena: "); Serial.print(tensor_arena_size / 1024); Serial.println(" KB");

  // Escolhe fonte do modelo: SPIFFS (OTA) ou header compilado (fallback)
  const uint8_t* model_data = MODEL_DATA;
  if (ota_load_model_into_ram()) {
    model_data = ota_get_model_data();
    Serial.println("[OTA] Usando modelo do SPIFFS (OTA).");
  } else {
    Serial.println("[OTA] Usando modelo compilado no firmware (builtin).");
  }

  model = tflite::GetModel(model_data);   // model_data = SPIFFS (OTA) ou MODEL_DATA (builtin)

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERRO: versao do modelo incompativel com TFLite Micro");
    while (true) delay(1000);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, tensor_arena_size
  );
  interpreter = &static_interpreter;

  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERRO: falha em AllocateTensors()");
    Serial.println("Aumente kTensorArenaSize.");
    while (true) delay(1000);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  if (input->type  == kTfLiteInt8) {
    input_scale      = input->params.scale;
    input_zero_point = input->params.zero_point;
  }
  if (output->type == kTfLiteInt8) {
    output_scale      = output->params.scale;
    output_zero_point = output->params.zero_point;
  }

  Serial.println("Modelo carregado com sucesso.");
  Serial.print("Input  type: "); Serial.println(tipoTensor(input->type));
  Serial.print("Output type: "); Serial.println(tipoTensor(output->type));
  Serial.print("Input  scale: ");      Serial.println(input_scale, 8);
  Serial.print("Input  zero_point: "); Serial.println(input_zero_point);
  Serial.print("Output scale: ");      Serial.println(output_scale, 8);
  Serial.print("Output zero_point: "); Serial.println(output_zero_point);

  Serial.println("Iniciando medicoes em CSV...");
  imprimirCabecalhoCsv();
}

// =============================
// Loop
// =============================

void loop() {
  MetricStats preprocess_stats = novaMetrica();
  MetricStats inference_stats = novaMetrica();
  MetricStats total_stats = novaMetrica();
  MetricStats fps_stats = novaMetrica();
  MetricStats cpu_stats = novaMetrica();
  MetricStats energy_stats = novaMetrica();
  int labeled_count = 0;
  int correct_count = 0;

  for (int run = 0; run < kNumRuns; run++) {
    float heap_before_kb = ESP.getFreeHeap() / 1024.0f;

    uint32_t t0 = micros();
    int expected_label = prepararJanelaEntrada(run);
    uint32_t t1 = micros();

    copiarEntradaParaTensor();
    uint32_t t2 = micros();

    TfLiteStatus status = interpreter->Invoke();
    uint32_t t3 = micros();

    if (status != kTfLiteOk) {
      Serial.println("ERRO: falha em Invoke()");
      delay(1000);
      continue;
    }

    float score = lerScoreSaida();
    int   pred  = (score >= kThreshold) ? 1 : 0;
    if (expected_label >= 0) {
      labeled_count++;
      if (pred == expected_label) correct_count++;
    }
    float heap_after_kb = ESP.getFreeHeap() / 1024.0f;

    float preprocess_ms = (t1 - t0) / 1000.0f;
    float inference_ms  = (t3 - t2) / 1000.0f;
    float total_ms      = (t3 - t0) / 1000.0f;
    float fps           = 1000.0f / inference_ms;
    float cpu_percent   = (total_ms / kDecisionIntervalMs) * 100.0f;
    float energy_mj     = kVoltage * kCurrentA * (total_ms / 1000.0f) * 1000.0f;

    atualizarMetrica(preprocess_stats, preprocess_ms);
    atualizarMetrica(inference_stats, inference_ms);
    atualizarMetrica(total_stats, total_ms);
    atualizarMetrica(fps_stats, fps);
    atualizarMetrica(cpu_stats, cpu_percent);
    atualizarMetrica(energy_stats, energy_mj);

    imprimirLinhaCsv(
      run, score, pred,
      expected_label,
      preprocess_ms, inference_ms, total_ms,
      heap_before_kb, heap_after_kb
    );

    delay(100);
  }

  imprimirResumoRodada(
    kNumRuns,
    preprocess_stats,
    inference_stats,
    total_stats,
    fps_stats,
    cpu_stats,
    energy_stats
  );
  if (labeled_count > 0) {
    Serial.print("# summary,accuracy,");
    Serial.print((float)correct_count / labeled_count, 6);
    Serial.print(",");
    Serial.print(correct_count);
    Serial.print(",");
    Serial.println(labeled_count);
  }
  Serial.println("# Fim da rodada de medicoes.");
  delay(5000);
}
