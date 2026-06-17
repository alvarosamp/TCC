#pragma once

// ============================================================
//  SELECAO DO MODELO
//  Troque ACTIVE_MODEL para mudar o modelo testado no ESP32.
// ============================================================

#define MODEL_TCN_INT8                  1
#define MODEL_TCN_FLOAT16               2
#define MODEL_TCN_FLOAT32               3
#define MODEL_PIPELINE_TINY_CNN_INT8    4
#define MODEL_PIPELINE_TINY_CNN_FLOAT32 5

// INT8 falhou no TFLite Micro por incompatibilidade no kernel REDUCE_MAX.
// Para validacao funcional embarcada, usamos float32.
#define ACTIVE_MODEL MODEL_PIPELINE_TINY_CNN_FLOAT32

// ============================================================
//  Thresholds por modelo
// ============================================================

#if ACTIVE_MODEL == MODEL_PIPELINE_TINY_CNN_FLOAT32
  #include "tiny_cnn_float32.h"
  #define MODEL_DATA      artefacts_edge_tiny_cnn_float32_tflite
  #define MODEL_NAME      "PIPELINE_TINY_CNN_FLOAT32"
  #define MODEL_THRESHOLD 0.7241942882537842f

#elif ACTIVE_MODEL == MODEL_PIPELINE_TINY_CNN_INT8
  #include "tiny_cnn_int8.h"
  #define MODEL_DATA      tiny_cnn_int8_model_data
  #define MODEL_NAME      "PIPELINE_TINY_CNN_INT8"
  #define MODEL_THRESHOLD 0.7241942882537842f

#elif ACTIVE_MODEL == MODEL_TCN_INT8
  #include "TCN/Keras/optuna_tiny_tcn_classifier_int8.h"
  #define MODEL_DATA      g_optuna_tiny_tcn_classifier_int8_model
  #define MODEL_NAME      "TCN_INT8"
  #define MODEL_THRESHOLD 0.83984375f

#elif ACTIVE_MODEL == MODEL_TCN_FLOAT16
  #include "TCN/Keras/optuna_tiny_tcn_classifier_float16.h"
  #define MODEL_DATA      g_optuna_tiny_tcn_classifier_float16_model
  #define MODEL_NAME      "TCN_FLOAT16"
  #define MODEL_THRESHOLD 0.73828125f

#elif ACTIVE_MODEL == MODEL_TCN_FLOAT32
  #include "TCN/Keras/optuna_tiny_tcn_classifier_float32.h"
  #define MODEL_DATA      g_optuna_tiny_tcn_classifier_float32_model
  #define MODEL_NAME      "TCN_FLOAT32"
  #define MODEL_THRESHOLD 0.73828125f

#else
  #error "ACTIVE_MODEL invalido."
#endif
