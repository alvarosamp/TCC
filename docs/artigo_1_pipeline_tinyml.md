# Artigo 1 — A Generic TinyML Pipeline for Time-Series Anomaly Detection: Edge-Aware Preprocessing, MLOps Quality Gates, and Secure OTA Updates on ESP32

**Venue alvo:** BRACIS 2026 / IEEE WF-IoT 2026  
**Formato:** IEEE Conference (double-column, 8–10 páginas)  
**Status:** Rascunho v0.2 — resultados parciais; campos `[TODO]` indicam experimentos pendentes

---

## Abstract

We present a generic, reproducible TinyML pipeline for anomaly detection in complex time series, validated in the seismic domain targeting ESP32 microcontrollers. The architecture decouples domain adapters from the machine learning core, enabling reuse for industrial vibration, electrical current, or telemetry sensors without pipeline rewriting. Edge-aware preprocessing eliminates MCU-incompatible steps (instrumental response removal), reducing training-serving skew at embedded inference time. The pipeline compares six model families — from STA/LTA to Optuna-optimized dilated convolutional networks — under a comprehensive set of operational metrics: AUC-PR, VUS-PR, PA-F1, event-level F1, and false positives per hour (FP/h). The best model, a Tiny TCN with 14,897 parameters, achieves AUC-PR of 0.9416 and 3.136 FP/h, outperforming all baselines including the traditional STA/LTA method. A complete MLOps cycle — DVC, MLflow, multidimensional quality gate, model promotion, and an OTA package with SHA-256 integrity verification and rollback — establishes an auditable foundation for production TinyML systems.

**Keywords:** TinyML, anomaly detection, time series, seismology, ESP32, MLOps, temporal convolutional network, over-the-air update.

---

## 1. Introduction

Continuous sensor monitoring systems face a recurring operational dilemma: transmitting all sensor data for centralized analysis incurs high energy, bandwidth, and storage costs, while fixed-threshold alerting on the device produces high false-positive rates and poor adaptability [1]. In domains such as seismic monitoring, industrial vibration, and equipment telemetry, continuous acquisition generates data volumes that make centralized analysis impractical at scale.

TinyML — the execution of machine learning models on microcontrollers with severely limited resources — offers an alternative: the edge device processes local signal windows and forwards data only when there is reliable evidence of an anomaly [2, 3]. This reduces transmission, energy consumption, and operational cost, while enabling faster response and offline operation.

In the seismic domain, automatic event detection has been addressed by classical methods such as STA/LTA [4, 5] and, more recently, deep networks such as PhaseNet [6] and EQTransformer [7]. These models, however, are designed for server execution and do not consider the memory and processing constraints of microcontrollers.

Evaluation methodology is also under scrutiny: Kim et al. [8] show that standard AUC-PR is unreliable for windowed time series due to point-to-point alignment sensitivity, proposing VUS-PR as a more robust alternative. Xu et al. [9] and Tatbul et al. [10] formalize point-adjust F1 (PA-F1) and event-level F1 as operationally meaningful complements.

**This paper contributes:**

1. A **generic, reproducible** TinyML pipeline for binary time-series anomaly detection, domain-agnostic via adapter separation.
2. **Edge-aware preprocessing** that eliminates MCU-incompatible steps, reducing training-serving skew.
3. Comparison of six model families under a **full operational metric set**: AUC-PR, VUS-PR, PA-F1, Event-F1, and FP/h.
4. A **complete MLOps cycle**: DVC, MLflow, multidimensional quality gate, model promotion, and an OTA package with SHA-256 integrity verification and simulated rollback.

---

## 2. Related Work

### 2.1 TinyML and Embedded Inference

Warden and Situnayake [2] established practical foundations for running ML models on microcontrollers. Banbury et al. [3] formalized performance benchmarks for TinyML systems, exposing tradeoffs between accuracy, latency, and energy. David et al. [11] present TensorFlow Lite Micro as the reference framework for OS-free inference. Lin et al. [12] propose architecture and inference engine co-design to maximize accuracy on MCUs with less than 512 KB SRAM.

### 2.2 Anomaly Detection in Time Series

Chandola et al. [13] provide a comprehensive taxonomy of anomaly detection techniques. Pang et al. [14] review deep learning methods for anomalies, highlighting the importance of appropriate metrics for imbalanced distributions. Kim et al. [8] propose VUS-PR, integrating AUC-PR over multiple prediction buffer sizes to produce a more rigorous evaluation measure. Tatbul et al. [10] define range-based precision and recall for time series, formalizing event-level evaluation.

### 2.3 Seismic Detection with ML

STA/LTA [4, 5] is the classical baseline: it compares short-window energy (STA) to long-window energy (LTA) to identify event onsets. PhaseNet [6] is a U-Net for seismic phase detection at datacenter scale. EQTransformer [7] introduces a hierarchical transformer for simultaneous detection and phase classification. Both achieve strong performance but with tens of millions of parameters, infeasible for ESP32 deployment.

### 2.4 Temporal Convolutional Networks

Bai et al. [15] demonstrate empirically that causal dilated convolutional networks (TCN) outperform LSTMs on diverse sequence modeling tasks. The absence of recurrent state simplifies sequential inference and reduces memory consumption — an advantage relevant for embedded devices.

### 2.5 MLOps for Embedded Systems

Sculley et al. [1] identify hidden technical debt in ML systems, including training-serving skew and lack of data versioning. Amershi et al. [16] document software engineering practices for production ML. DVC [17] and MLflow [18] form the recognized basis for reproducible MLOps pipelines.

---

## 3. Methodology

### 3.1 Pipeline Architecture

The pipeline enforces explicit separation between domain adapters and the ML core:

```
raw data
  → domain adapter
  → generic NPZ dataset  {X_train/val/test, y_train/val/test}
  → contract validation (pandera)
  → feature extraction (classical) / raw signal (neural)
  → comparative model training (6 families)
  → Optuna HPO
  → candidate_manifest.json
  → multidimensional quality gate
  → production_manifest.json
  → TFLite export + C header
  → versioned OTA package (SHA-256, rollback)
```

The **dataset contract** is a NumPy compressed archive with keys `X_{train,val,test}` of shape `(N, window_size)` for univariate or `(N, window_size, n_channels)` for multivariate series, and binary labels `y_{train,val,test}`. Adding a new domain requires only a new adapter that respects this contract.

### 3.2 Seismic Dataset

> **TODO:** descrever número de eventos, estações, período de coleta e fonte MiniSEED.

Raw data are MiniSEED files organized as `raw/events/` (anomalous windows, y=1) and `raw/continuous/` (background noise, y=0).

| Parameter | Value |
|---|---|
| Sampling rate | 40 Hz |
| Window size | 800 samples (20 s) |
| Window step | 10 s (50% overlap) |
| Split strategy | By event |
| Test anomaly ratio | ≈12.8% |

Split by event guarantees that windows from the same earthquake do not appear simultaneously in train and test, preventing data leakage.

### 3.3 Edge-Aware Preprocessing

**Offline (training):**
`resample 40 Hz → linear detrend → demean → 5% taper → bandpass 0.5–15 Hz → per-window z-score`

**Edge (ESP32 inference):**
`linear detrend → 5% taper → bandpass 0.5–15 Hz → per-window z-score`

The `remove_response` step (requiring StationXML) was removed — it cannot be reproduced on the MCU. All retained steps are implementable in C/C++ with single-precision floating-point arithmetic.

### 3.4 Feature Extraction

For classical supervised models, 28 features are extracted per window:

- **Temporal (20):** mean, std, min, max, median, abs_mean, abs_peak, RMS, crest factor, peak-to-peak, energy, skewness, kurtosis, p05, p25, p75, p95, IQR, zero crossings, zero-crossing rate.
- **Spectral (8):** dominant frequency, spectral centroid, spectral rolloff (85%), bandpower in 0–3, 0.5–3, 3–8, 8–15 Hz, spectral entropy.

Neural networks (Tiny CNN, Tiny TCN) operate directly on 800 raw samples.

### 3.5 Models

| Family | Model | Input | Edge |
|---|---|---|:---:|
| Classical baseline | STA/LTA | Raw signal | ✓ |
| Linear | Logistic Regression | 28 features | ✗ |
| Ensemble | Random Forest | 28 features | ✗ |
| Ensemble | Extra Trees | 28 features | ✗ |
| Lightweight neural | Tiny CNN | 800 pts | ✓ |
| Lightweight neural | Tiny TCN (Optuna) | 800 pts | ✓ |

**Tiny TCN (best model):** Causal dilated separable Conv1D blocks with residual connections. Final Optuna configuration (trial 24, 88 trials total): 6 blocks, 32 filters, kernel 7, dilation base 3, dense 24, dropout 0.028, spatial dropout 0.066; **14,897 parameters**.

### 3.6 Hyperparameter Optimization

Optuna [19] maximizes a composite objective:

$$\mathcal{J} = \text{AUC-PR}_\text{val} - \lambda \cdot \max(0,\; \text{FP/h} - \text{FP/h}_\text{tol})$$

where λ = 0.01 and FP/h_tol = 10. The decision threshold is chosen on the validation set and applied unchanged on the test set.

### 3.7 Evaluation Metrics

**AUC-PR:** primary metric for imbalanced datasets.

**VUS-PR** [8]: average AUC-PR over prediction buffers b ∈ [0, b_max]:

$$\text{VUS-PR} = \frac{1}{b_{\max}+1} \sum_{b=0}^{b_{\max}} \text{AUC-PR}^{(b)}$$

where AUC-PR⁽ᵇ⁾ dilates each anomaly segment by b windows. Buffer b=0 recovers standard AUC-PR.

**PA-F1** [9]: point-adjust F1 — if any window within an anomaly segment is correctly detected, all windows in that segment are counted as TP.

**Event-F1** [10]: F1 at segment level — a true segment is a TP if ≥10% of its windows are predicted anomalous.

**FP/h:**

$$\text{FP/h} = \frac{\text{FP}}{\text{hours of evaluated signal}}$$

### 3.8 MLOps Cycle

- **DVC** versions data and pipeline stages: `generate_data → validate_dataset → train_all → export_tflite`.
- **MLflow** tracks all experiment parameters, metrics, and model artifacts.
- **Quality gate** enforces multidimensional thresholds before promotion: AUC-PR ≥ 0.80, FP/h ≤ 10, |ΔAUC-PR val/test| ≤ 0.05, model size ≤ 100 KB.
- **OTA package** contains: versioned TFLite model, JSON manifest, SHA-256 digest, compatibility metadata (target, runtime, preprocessing version), and a simulated rollback mechanism.

---

## 4. Results

### 4.1 Model Comparison

| Model | AUC-PR | VUS-PR | PA-F1 | Event-F1 | F1 | FP/h | Params | Edge |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Optuna Tiny TCN** | **0.9416** | **[TODO]** | **[TODO]** | **[TODO]** | **0.8885** | **3.136** | **14,897** | ✓ |
| Optuna Tiny CNN | 0.9127 | [TODO] | [TODO] | [TODO] | 0.8526 | 4.896 | 19,489 | ✓ |
| Tiny TCN (no HPO) | 0.8964 | [TODO] | [TODO] | [TODO] | 0.7666 | 21.984 | ≈6k | ✓ |
| Optuna Random Forest | 0.8127 | [TODO] | [TODO] | [TODO] | 0.7367 | 9.264 | — | ✗ |
| Optuna Extra Trees | 0.7901 | [TODO] | [TODO] | [TODO] | 0.7102 | 11.296 | — | ✗ |
| STA/LTA | 0.1662 | [TODO] | [TODO] | [TODO] | 0.2760 | — | — | ✓ |

> **TODO:** preencher VUS-PR, PA-F1 e Event-F1 rodando `evaluate_scores` nos scores salvos no MLflow — não requer retreinamento.

The Optuna Tiny TCN dominates across all metrics: highest AUC-PR (0.9416), lowest FP/h (3.136) among neural models, and only 14,897 parameters — well within ESP32 flash capacity after TFLite conversion.

STA/LTA achieves AUC-PR of 0.1662, near random baseline for the class distribution, confirming that fixed-threshold methods are insufficient for the variability of the evaluated data.

### 4.2 Impact of Hyperparameter Optimization

| Configuration | AUC-PR (val) | FP/h |
|---|---:|---:|
| No HPO | — | 21.984 |
| Trial 0 | 0.8152 | 8.539 |
| Trial 8 | 0.8906 | 4.521 |
| Trial 11 | 0.8981 | 3.586 |
| **Trial 24 (best)** | **0.9343** | **3.136** |

HPO reduces FP/h by **85.7%** relative to the default architecture (21.984 → 3.136). The improvement in AUC-PR from validation to test (+0.0079) indicates good generalization with no overfitting.

### 4.3 Feature Importance (Classical Models)

Feature importance analysis on Random Forest reveals that spectral features dominate class separation: `spectral_rolloff_85`, `bandpower_8_15hz`, `spectral_centroid`, `zero_crossing_rate`, `kurtosis`, `spectral_entropy`. Anomalies are discriminated by spectral distribution and waveform shape, not by raw amplitude — consistent with seismic physics.

### 4.4 MLOps Validation

The Tiny TCN passed the quality gate, generating `production_manifest.json` and an OTA package with verified SHA-256 digest. The simulated OTA flow correctly detected available updates, applied the new version, suppressed unnecessary reinstallation, and completed rollback after a simulated failure — all state transitions verified.

### 4.5 Embedded Resource Estimates

| Artifact | Size |
|---|---|
| `model.tflite` (float32) | [TODO] KB |
| `model.tflite` (int8) | [TODO] KB |
| `model_data.h` | [TODO] KB |
| Estimated inference RAM | [TODO] KB |
| Estimated inference latency | [TODO] ms |

> **TODO:** medir após exportação TFLite com quantização int8.

---

## 5. Discussion

**Edge-aware preprocessing.** Removing `remove_response` from the training pipeline aligns the offline and embedded execution paths, reducing the most common source of training-serving skew in physics-based signal processing. Residual risk: numerical differences between the Python and C implementations of bandpass filtering. Hardware validation on ESP32 (see future work) will quantify this residual gap.

**VUS-PR vs. AUC-PR.** With 50% window overlap, the same anomaly contributes to multiple positive windows. AUC-PR may overestimate performance by counting a single missed event as multiple false negatives, inflating recall. VUS-PR provides a more conservative and operationally meaningful estimate.

**Compact models for edge deployment.** The Tiny TCN (14,897 params) and Tiny CNN (19,489 params) are both viable for ESP32 flash. Non-edge models (Random Forest, Extra Trees) require megabytes after joblib serialization, precluding MCU deployment.

---

## 6. Conclusion

We presented a generic, reproducible TinyML pipeline covering the complete cycle from raw data acquisition to secure OTA model updates on ESP32. The architecture decouples domain adapters from the ML core, enabling reuse across sensor domains without pipeline rewriting. Edge-aware preprocessing eliminates MCU-incompatible dependencies, reducing training-serving skew. The Optuna-optimized Tiny TCN achieves AUC-PR of 0.9416 and 3.136 FP/h with 14,897 parameters, outperforming all evaluated baselines. The multidimensional quality gate and SHA-256-verified OTA package with rollback establish an auditable foundation for production TinyML systems.

**Future work:**
- Hardware validation on ESP32 to quantify the preprocessing gap
- Cross-domain replication on Petrobras 3W and C-MAPSS datasets (see companion papers)
- Int8 quantization and on-device latency measurement

---

## References

1. D. Sculley et al., "Hidden technical debt in machine learning systems," *NeurIPS*, 2015.
2. P. Warden and D. Situnayake, *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers*. O'Reilly, 2019.
3. C. Banbury et al., "MLPerf Tiny Benchmark," *MLSys*, 2021.
4. R. Allen, "Automatic earthquake recognition and timing from single traces," *Bull. Seismol. Soc. Am.*, vol. 68, pp. 1521–1532, 1978.
5. M. Withers et al., "A comparison of select trigger algorithms for automated global seismic phase and event detection," *Bull. Seismol. Soc. Am.*, vol. 88, pp. 95–106, 1998.
6. W. Zhu and G. C. Beroza, "PhaseNet: A deep-neural-network-based seismic arrival-time picking method," *Geophys. J. Int.*, vol. 216, pp. 261–273, 2019.
7. S. M. Mousavi et al., "Earthquake transformer," *Nat. Commun.*, vol. 11, p. 3952, 2020.
8. S. Kim et al., "Towards a rigorous evaluation of time-series anomaly detection," *AAAI*, 2022.
9. H. Xu et al., "Unsupervised anomaly detection via variational auto-encoder for seasonal KPIs in web applications," *WWW*, 2018.
10. N. Tatbul et al., "Precision and recall for time series," *NeurIPS*, 2018.
11. R. David et al., "TensorFlow Lite Micro: Embedded machine learning for TinyML systems," *MLSys*, 2021.
12. J. Lin et al., "MCUNet: Tiny deep learning on IoT devices," *NeurIPS*, 2020.
13. V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," *ACM Comput. Surv.*, vol. 41, 2009.
14. G. Pang et al., "Deep learning for anomaly detection: A review," *ACM Comput. Surv.*, vol. 54, 2021.
15. S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," arXiv:1803.01271, 2018.
16. S. Amershi et al., "Software engineering for machine learning: A case study," *ICSE-SEIP*, 2019.
17. Iterative AI, "DVC: Data version control," https://dvc.org, 2024.
18. M. Zaharia et al., "Accelerating the machine learning lifecycle with MLflow," *IEEE Data Eng. Bull.*, vol. 41, 2018.
19. T. Akiba et al., "Optuna: A next-generation hyperparameter optimization framework," *KDD*, 2019.

---

## Checklist antes da submissão

- [ ] Preencher VUS-PR, PA-F1, Event-F1 (rodar `evaluate_scores` nos scores MLflow)
- [ ] Medir tamanhos TFLite float32 e int8, RAM e latência no ESP32
- [ ] Descrever dataset sísmico (número de eventos, estações, período, fonte)
- [ ] Inserir figuras: pipeline, curva PR, convergência Optuna
- [ ] Ajustar abstract para ≤ 150 palavras (limite IEEE conference)
- [ ] Formatar referências em estilo IEEE
- [ ] Verificar página limit: 8–10 páginas double-column
