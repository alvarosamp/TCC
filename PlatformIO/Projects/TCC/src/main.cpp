#include <Arduino.h>
#if defined(ARDUINO_ARCH_ESP32)
  #include <esp_bt.h>
  #include <esp_heap_caps.h>
#endif
#include <float.h>
#include <math.h>

// ============================================================
//  Para trocar o modelo: edite apenas model_config.h
//  Mude ACTIVE_MODEL para MODEL_TCN_INT8 / FLOAT16 / FLOAT32
// ============================================================
#include "model_config.h"
#include "preprocessing.h"

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

// Tensor arena. Precisa caber no maior bloco continuo livre do microcontrolador.
// ESP8266 tem bem menos RAM que ESP32; esta arena e apenas para teste de viabilidade.
#if defined(ARDUINO_ARCH_ESP8266)
constexpr int kTensorArenaSize = 48 * 1024;
#else
constexpr int kTensorArenaSize = 100 * 1024;
#endif
uint8_t* tensor_arena = nullptr;
size_t tensor_arena_size = kTensorArenaSize;

// Numero de inferencias por rodada.
constexpr int kNumRuns = 100;

// =============================
// Variaveis globais TFLM
// =============================

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
#if defined(ARDUINO_ARCH_ESP32)
  return heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
#else
  return ESP.getFreeHeap();
#endif
}

void liberarMemoriaNaoUsada() {
  Serial.print("Maior bloco livre inicial: ");
  Serial.print(maiorBlocoLivreInterno());
  Serial.println(" bytes");

#if defined(ARDUINO_ARCH_ESP32)
  esp_err_t bt_status = esp_bt_mem_release(ESP_BT_MODE_BTDM);
  Serial.print("BT memory release: ");
  Serial.println(bt_status == ESP_OK ? "OK" : "nao liberada");

  Serial.print("Maior bloco livre apos BT release : ");
  Serial.print(maiorBlocoLivreInterno());
  Serial.println(" bytes");
#else
  Serial.println("BT memory release: nao aplicavel no ESP8266");
#endif
}

uint8_t* alocarTensorArena(size_t bytes) {
#if defined(ARDUINO_ARCH_ESP32)
  return static_cast<uint8_t*>(
    heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL)
  );
#else
  return static_cast<uint8_t*>(malloc(bytes));
#endif
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

  tensor_arena = alocarTensorArena(tensor_arena_size);
  if (tensor_arena == nullptr) {
    Serial.println("ERRO: falha ao alocar tensor_arena");
    Serial.print("Maior bloco livre interno: ");
    Serial.print(maiorBlocoLivreInterno());
    Serial.println(" bytes");
    while (true) delay(1000);
  }

  model = tflite::GetModel(MODEL_DATA);   // <-- usa o modelo selecionado

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
