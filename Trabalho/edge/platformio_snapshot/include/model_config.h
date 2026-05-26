#pragma once

// ============================================================
//  SELECAO DO MODELO
//  Troque o valor de ACTIVE_MODEL para mudar o modelo testado.
//
//  Opcoes disponiveis:
//    MODEL_TCN_INT8     -> optuna_tiny_tcn_classifier_int8
//    MODEL_TCN_FLOAT16  -> optuna_tiny_tcn_classifier_float16
//    MODEL_TCN_FLOAT32  -> optuna_tiny_tcn_classifier_float32
// ============================================================

#define MODEL_TCN_INT8    1
#define MODEL_TCN_FLOAT16 2
#define MODEL_TCN_FLOAT32 3

// >>>  MUDE AQUI  <<<
#define ACTIVE_MODEL MODEL_TCN_INT8

// ============================================================
//  Thresholds por modelo (ajuste conforme validacao)
// ============================================================

#if ACTIVE_MODEL == MODEL_TCN_INT8
  #include "TCN/Keras/optuna_tiny_tcn_classifier_int8.h"
  #define MODEL_DATA      g_optuna_tiny_tcn_classifier_int8_model
  #define MODEL_NAME      "TCN_INT8"
  #define MODEL_THRESHOLD 0.83984375f   // threshold validacao int8

#elif ACTIVE_MODEL == MODEL_TCN_FLOAT16
  #include "TCN/Keras/optuna_tiny_tcn_classifier_float16.h"
  #define MODEL_DATA      g_optuna_tiny_tcn_classifier_float16_model
  #define MODEL_NAME      "TCN_FLOAT16"
  #define MODEL_THRESHOLD 0.73828125f   // threshold validacao float16

#elif ACTIVE_MODEL == MODEL_TCN_FLOAT32
  #include "TCN/Keras/optuna_tiny_tcn_classifier_float32.h"
  #define MODEL_DATA      g_optuna_tiny_tcn_classifier_float32_model
  #define MODEL_NAME      "TCN_FLOAT32"
  #define MODEL_THRESHOLD 0.73828125f   // threshold validacao float32

#else
  #error "ACTIVE_MODEL invalido. Use MODEL_TCN_INT8, MODEL_TCN_FLOAT16 ou MODEL_TCN_FLOAT32."
#endif
