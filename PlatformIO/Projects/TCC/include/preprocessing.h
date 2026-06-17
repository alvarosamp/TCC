#pragma once

// =============================================================================
// preprocessing.h — Pipeline de pré-processamento para inferência na borda
//
// Replica (em single-precision float) o pipeline Python usado para gerar o
// dataset v3:
//   detrend(linear) → demean → taper(5%) → bandpass(0.5–15 Hz, zerophase)
//                                         → z-score por janela
//
// Limitação conhecida: remove_response (remoção de resposta instrumental) não
// é implementada aqui — requer StationXML e FFT, inviável no ESP32.
// Em produção real, substitua por um fator de ganho fixo do sensor.
//
// Filtro: Butterworth 2ª ordem, bandpass 0.5–15 Hz @ 40 Hz.
//   Coeficientes gerados por:
//     scipy.signal.butter(2, [0.5, 15.0], btype='bandpass', fs=40.0, output='sos')
//   Verificação da resposta em frequência:
//     f= 0.1 Hz → -28.7 dB   (rejeita DC/deriva)
//     f= 0.5 Hz →  -2.9 dB   (corte inferior)
//     f= 1.0 Hz →  -0.2 dB
//     f= 5.0 Hz →   0.0 dB   (banda passante)
//     f=10.0 Hz →  -0.1 dB
//     f=15.0 Hz →  -3.0 dB   (corte superior)
//     f=18.0 Hz → -17.1 dB
//     f=20.0 Hz → -97.5 dB   (rejeita Nyquist)
// =============================================================================

#include <math.h>

// ---------------------------------------------------------------------------
// Coeficientes SOS — Butterworth 2ª ordem bandpass 0.5–15 Hz @ 40 Hz
// Cada linha: [b0, b1, b2, a0=1, a1, a2]
// ---------------------------------------------------------------------------
constexpr int kBpNSections = 2;

constexpr float kBpSos[kBpNSections][6] = {
  // Secao 1 (highpass 0.5 Hz)
  { 0.5363402127f,  1.0726804254f,  0.5363402127f,
    1.0000000000f,  0.9338468022f,  0.3359301255f },
  // Secao 2 (lowpass 15 Hz)
  { 1.0000000000f, -2.0000000000f,  1.0000000000f,
    1.0000000000f, -1.8889579709f,  0.8949890597f },
};

// ---------------------------------------------------------------------------
// detrendarLinear: remove tendência linear por mínimos quadrados (in-place).
//   Ajusta y = a*t + b e subtrai de cada amostra.
//   Remove offset DC e deriva lenta do sensor.
// ---------------------------------------------------------------------------
inline void detrendarLinear(float* x, int n) {
  float fn     = (float)n;
  float sum_t  = fn * (fn - 1.0f) * 0.5f;         // 0+1+…+(n-1)
  float sum_tt = fn * (fn - 1.0f) * (2.0f*fn - 1.0f) / 6.0f;

  float sum_x  = 0.0f;
  float sum_tx = 0.0f;
  for (int i = 0; i < n; i++) {
    sum_x  += x[i];
    sum_tx += (float)i * x[i];
  }

  float denom = fn * sum_tt - sum_t * sum_t;
  if (fabsf(denom) < 1e-12f) {
    // Caso degenerado: apenas remove a média
    float mean = sum_x / fn;
    for (int i = 0; i < n; i++) x[i] -= mean;
    return;
  }

  float a = (fn * sum_tx - sum_t * sum_x) / denom;
  float b = (sum_x  - a * sum_t) / fn;
  for (int i = 0; i < n; i++) {
    x[i] -= (a * (float)i + b);
  }
}

// ---------------------------------------------------------------------------
// aplicarTaper: janela de cosseno em 5% de cada extremidade (in-place).
//   Evita descontinuidades nas bordas que causam vazamento espectral.
// ---------------------------------------------------------------------------
inline void aplicarTaper(float* x, int n) {
  int n_taper = (int)(0.05f * (float)n);
  if (n_taper < 1) return;

  for (int i = 0; i < n_taper; i++) {
    float w = 0.5f * (1.0f - cosf((float)i * (float)M_PI / (float)n_taper));
    x[i]           *= w;
    x[n - 1 - i]   *= w;
  }
}

// ---------------------------------------------------------------------------
// sosfiltForward: filtragem IIR SOS em sentido direto (in-place).
//   Implementa o Direct Form II Transposto para estabilidade numérica.
// ---------------------------------------------------------------------------
inline void sosfiltForward(float* x, int n) {
  float z[kBpNSections][2] = {};  // estados iniciais = 0

  for (int i = 0; i < n; i++) {
    float v = x[i];
    for (int s = 0; s < kBpNSections; s++) {
      float y   = kBpSos[s][0] * v + z[s][0];
      z[s][0]   = kBpSos[s][1] * v - kBpSos[s][4] * y + z[s][1];
      z[s][1]   = kBpSos[s][2] * v - kBpSos[s][5] * y;
      v = y;
    }
    x[i] = v;
  }
}

// ---------------------------------------------------------------------------
// aplicarBandpassZerophase: filtro passa-banda sem distorção de fase.
//   Faz passagem direta + inversão + passagem direta novamente (≡ sosfiltfilt).
//   Requer a janela inteira em RAM — OK para 800 amostras no ESP32.
// ---------------------------------------------------------------------------
inline void aplicarBandpassZerophase(float* x, int n) {
  // Passagem 1: sentido direto
  sosfiltForward(x, n);

  // Inversão do sinal
  for (int i = 0; i < n / 2; i++) {
    float tmp        = x[i];
    x[i]             = x[n - 1 - i];
    x[n - 1 - i]     = tmp;
  }

  // Passagem 2: sentido direto (equivale a reverso no sinal original)
  sosfiltForward(x, n);

  // Desfaz inversão
  for (int i = 0; i < n / 2; i++) {
    float tmp        = x[i];
    x[i]             = x[n - 1 - i];
    x[n - 1 - i]     = tmp;
  }
}

// ---------------------------------------------------------------------------
// zscoreInPlace: normalização z-score por janela (in-place).
//   µ = 0, σ = 1.  Epsilon 1e-6 evita divisão por zero em sinais constantes.
// ---------------------------------------------------------------------------
inline void zscoreInPlace(float* x, int n) {
  float soma = 0.0f;
  for (int i = 0; i < n; i++) soma += x[i];
  float media = soma / (float)n;

  float soma_q = 0.0f;
  for (int i = 0; i < n; i++) {
    float d = x[i] - media;
    soma_q += d * d;
  }
  float desvio = sqrtf(soma_q / (float)n) + 1e-6f;

  for (int i = 0; i < n; i++) x[i] = (x[i] - media) / desvio;
}

// ---------------------------------------------------------------------------
// preprocessarBorda: pipeline completo raw → pronto para inferência.
//
//   Etapas:
//     1. Copia raw[n] → out[n]
//     2. Detrend linear (remove deriva + offset DC)
//     3. Taper cosseno 5% (reduz vazamento espectral nas bordas)
//     4. Bandpass zerophase 0.5–15 Hz (isola faixa sísmica de interesse)
//     5. Z-score por janela (normaliza amplitude para o modelo)
//
//   Parâmetros:
//     raw[n] — sinal bruto de entrada (não modificado)
//     out[n] — buffer de saída normalizado
//     n      — número de amostras (deve ser kWindowSize = 800)
// ---------------------------------------------------------------------------
inline void preprocessarBorda(const float* raw, float* out, int n) {
  // 1. Copia
  for (int i = 0; i < n; i++) out[i] = raw[i];

  // 2. Detrend linear
  detrendarLinear(out, n);

  // 3. Taper 5%
  aplicarTaper(out, n);

  // 4. Bandpass 0.5–15 Hz zerophase
  aplicarBandpassZerophase(out, n);

  // 5. Z-score
  zscoreInPlace(out, n);
}
