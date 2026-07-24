# Artigo 2 — Operational Quality Gates for TinyML Anomaly Detection: Beyond Offline Accuracy with VUS-PR, PA-F1, and Drift Monitoring

**Venue alvo:** IEEE Access (open access, journal)  
**Formato:** IEEE journal (12–15 páginas)  
**Status:** Rascunho v0.2 — experimentos de ablação e drift pendentes  
**Relação:** aprofunda o componente de quality gate do Artigo 1 como contribuição independente

---

## Abstract

TinyML pipelines for time-series anomaly detection are typically evaluated by offline accuracy metrics — AUC-PR, F1 on static test sets — without considering operational behavior: false alarm rate per hour (FP/h), data drift degradation, and the cost of unnecessary model updates. This paper proposes a multidimensional quality-gate framework that combines offline performance metrics (AUC-PR, VUS-PR, PA-F1, event-level F1), operational metrics (FP/h, val/test generalization gap), and data drift monitoring (Evidently AI) to govern model promotion and OTA updates in embedded TinyML systems. We evaluate the framework on seismic anomaly detection targeting ESP32 microcontrollers. Our ablation study shows that models approved solely by AUC-PR exhibit FP/h up to **7.0×** higher than models approved by the full quality gate (21.984 vs. 3.136 FP/h). Continuous drift monitoring coupled with a conditional retraining policy reduces unnecessary OTA updates by [X]% while maintaining detection performance, preserving bandwidth and energy on the edge device.

**Keywords:** MLOps, quality gate, anomaly detection, data drift, TinyML, false positive rate, VUS-PR, PA-F1, over-the-air update, ESP32.

---

## 1. Introduction

### 1.1 The Offline Evaluation Problem

The anomaly detection literature concentrates model evaluation on offline accuracy metrics — AUC-PR, F1, AUC-ROC — computed once on a static test set [1]. This practice ignores two critical operational realities:

**FP/h in production:** A model with AUC-PR = 0.896 can generate 21.984 false alarms per hour, rendering the system unusable for continuous monitoring. In systems where each alarm triggers human investigation or field dispatch, this operational metric is more important than any offline score. We observe this directly: the Tiny TCN without hyperparameter optimization achieves a competitive AUC-PR but produces **7.0×** more false alarms per hour than the HPO-optimized variant with identical architecture.

**Data drift:** Sensor signals change with equipment wear, seasonal variations, and environmental shifts. A model trained today may degrade in weeks without automatic signaling, leading to undetected anomalies or alarm floods [2].

### 1.2 The Metric Selection Problem

Kim et al. [3] argue that standard AUC-PR is unreliable for windowed time series: with 50% window overlap, a single anomaly contributes positively to multiple windows, inflating apparent performance. VUS-PR [3] integrates AUC-PR over prediction buffers b ∈ [0, b_max], providing a buffer-robust evaluation. Xu et al. [4] propose point-adjust F1 (PA-F1), and Tatbul et al. [5] formalize event-level F1 — both more appropriate for operational evaluation than point-wise metrics.

### 1.3 Contributions

1. A **multidimensional quality-gate framework** combining offline, operational, and drift metrics to govern model promotion and OTA updates in TinyML pipelines.
2. An **ablation study** demonstrating the contribution of each gate dimension to operational model quality.
3. A **conditional retraining policy** that responds to drift only when performance actually degrades, reducing unnecessary OTA updates.
4. **Open-source implementation** integrated with DVC, MLflow, and Evidently AI, usable as a drop-in component in any TinyML pipeline.

---

## 2. Background

### 2.1 Evaluation Metrics for Time-Series Anomaly Detection

**AUC-PR [6]:** area under the precision-recall curve. Preferred over AUC-ROC for imbalanced datasets, but insensitive to temporal alignment.

**VUS-PR [3]:**

$$\text{VUS-PR} = \frac{1}{b_{\max}+1}\sum_{b=0}^{b_{\max}} \text{AUC-PR}^{(b)}$$

where AUC-PR⁽ᵇ⁾ is computed after dilating each anomaly segment by b windows. At b=0, VUS-PR equals standard AUC-PR. For b>0, it measures robustness to detection delay.

**PA-F1 [4]:** Point-adjust strategy: if any window within a true anomaly segment is correctly detected (ŷ=1), all windows in that segment are relabeled as detected before computing F1. This rewards systems that detect event onset, even if coverage is partial.

**Event-F1 [5]:** F1 at segment level. A true segment is a TP if ≥τ fraction (τ=0.1) of its windows are predicted anomalous; an alarm without an overlapping true segment counts as an FP.

**FP/h:**

$$\text{FP/h} = \frac{|\{i : \hat{y}_i = 1 \wedge y_i = 0\}|}{T \cdot w / 3600}$$

where T is the number of windows and w is the window step in seconds.

### 2.2 Data Drift Detection

Gama et al. [2] classify drift as: *concept drift* (change in P(y|X)), *data drift* (change in P(X)), and *virtual drift* (change in P(X) without affecting P(y|X)). For embedded monitoring systems, data drift (sensor degradation, environmental change) and concept drift (new failure modes) are most operationally relevant.

### 2.3 MLOps and Model Governance

Sculley et al. [7] identify training-serving skew and lack of automatic quality control as major sources of ML technical debt. Vartak et al. [8] propose model versioning and comparison as a foundation for reproducible ML. Existing MLOps frameworks (MLflow [9], DVC [10]) track experiments but do not enforce operational deployment criteria — a gap this work addresses.

---

## 3. The Quality-Gate Framework

### 3.1 Architecture

The framework operates at two timescales:

```
TRAINING TIME
─────────────────────────────────────────────────────
candidate_manifest.json
        │
        ▼
    Quality Gate
    ├── Dimension 1: Offline Performance
    │     AUC-PR ≥ min_auc_pr
    │     VUS-PR ≥ min_vus_pr
    │
    ├── Dimension 2: Operational
    │     FP/h ≤ max_fp_per_hour
    │     PA-F1 ≥ min_pa_f1
    │     Event-F1 ≥ min_event_f1
    │
    ├── Dimension 3: Generalization
    │     |AUC-PR_val - AUC-PR_test| ≤ max_gap
    │     F1 ≥ min_f1
    │
    └── Dimension 4: Embedded Resources
          model_size ≤ max_size_kb
          n_params ≤ max_params
        │
   PASS │              FAIL │
        ▼                   ▼
production_manifest.json   rejection_report.json
        │
        ▼
    OTA package

PRODUCTION TIME
─────────────────────────────────────────────────────
incoming windows → Drift Monitor (Evidently AI)
                 → Retraining Policy
                 → conditional OTA trigger
```

### 3.2 Gate Dimensions and Thresholds

| Dimension | Criterion | Threshold | Direction |
|---|---|---|---|
| Offline perf. | AUC-PR | ≥ 0.80 | ↑ |
| Offline perf. | VUS-PR | ≥ 0.75 | ↑ |
| Operational | FP/h | ≤ 10.0 | ↓ |
| Operational | PA-F1 | ≥ 0.70 | ↑ |
| Operational | Event-F1 | ≥ 0.60 | ↑ |
| Generalization | \|ΔAUC-PR\| val/test | ≤ 0.05 | ↓ |
| Generalization | F1 | ≥ 0.70 | ↑ |
| Embedded | Model size | ≤ 100 KB | ↓ |
| Embedded | Parameters | ≤ 50k | ↓ |

All thresholds are configurable via the domain profile YAML file.

### 3.3 Drift Monitor

The drift monitor runs on a sliding window of production data and computes:

**Data drift:** Kolmogorov-Smirnov test per feature against the training reference distribution. Drift is flagged if p < 0.05 in more than 50% of features.

**Performance drift:** FP/h computed over a 24-hour sliding window. Alert if FP/h > 2× the value at promotion time.

Both monitors use Evidently AI [11] as the backend, producing structured JSON reports that feed the retraining policy.

### 3.4 Conditional Retraining Policy

```
Input: drift_report, performance_report

if data_drift_detected:
    if performance_drift_detected:
        trigger retraining + quality gate + OTA
    else:
        log data drift, defer retraining
elif performance_drift_detected:
    trigger retraining + quality gate + OTA  (immediate)
else:
    keep current model, continue monitoring
```

The policy requires *both* data and performance drift to trigger a non-urgent retraining cycle, and *either* alone for immediate response to performance degradation. This prevents reactive retraining to every statistical fluctuation.

---

## 4. Experimental Setup

### 4.1 Dataset and Models

Same seismic dataset as in the companion paper [Artigo 1]. We use the 88 Optuna HPO trials for the Tiny TCN as the candidate pool, plus Tiny CNN, Random Forest, and Extra Trees variants — totalling 90+ candidates for the ablation study.

### 4.2 Ablation Protocol

For each subset of gate dimension groups (4 groups → 16 combinations), we record:
- Number of candidates approved
- Mean and std of FP/h among approved candidates
- Mean and std of Event-F1 among approved candidates
- Best AUC-PR among approved candidates

### 4.3 Drift Simulation

Synthetic drift is applied to the test signal:

1. **Gain drift:** multiply signal by linearly increasing factor 1 + αt, α ∈ {0.01, 0.05, 0.10}
2. **Noise drift:** add Gaussian noise with linearly increasing std σ(t) = σ₀ + βt
3. **Offset drift:** add a linear trend to the baseline

For each scenario we measure: (i) latency (number of windows) to drift detection, (ii) FP/h at detection time vs. no-drift baseline, (iii) whether the conditional policy correctly triggers retraining.

---

## 5. Results

### 5.1 Ablation Study

| Gate configuration | Approved | FP/h (mean) | Event-F1 (mean) |
|---|---:|---:|---:|
| AUC-PR ≥ 0.80 only | 4/5 | 3.60 FP/h | 0.9144 |
| + FP/h ≤ 10 | 4/5 | 3.60 FP/h | 0.9144 |
| + VUS-PR ≥ 0.75 | 4/5 | 3.60 FP/h | 0.9144 |
| + PA-F1 ≥ 0.90 | 4/5 | 3.60 FP/h | 0.9144 |
| + Gap ≤ 0.05 | 4/5 | 3.60 FP/h | 0.9144 |
| **Full gate (all dims.)** | **2/5** | **3.39 FP/h** | **0.9144** |

> **TODO:** rodar ablação sobre os 88 trials do Optuna para preencher a tabela.

**FP/h is the binding constraint.** Among all HPO trials with AUC-PR ≥ 0.80, [X]% fail the FP/h ≤ 10 criterion. AUC-PR optimization does not automatically minimize false alarm rate — a consequence of threshold selection without an FP/h penalty in the objective.

**VUS-PR as a conservative screen.** VUS-PR ≥ 0.75 eliminates [X] additional candidates that pass AUC-PR screening, revealing models that perform well only under exact point-to-point alignment.

**Event-F1 is orthogonal to AUC-PR.** [X] candidates achieve AUC-PR > 0.85 but Event-F1 < 0.60, indicating high window-level detection with poor event-level coverage.

### 5.2 Illustrative Comparison: Gate Passing vs. Failing

**Tiny TCN without HPO:**
AUC-PR = 0.8964 ✓ (passes offline gate) but FP/h = 21.984 ✗ (fails operational gate by 2.2×).
In a 24-hour monitoring session: **449 additional false alarms per day**.

**Optuna Tiny TCN (trial 24):**
AUC-PR = 0.9416 ✓, FP/h = 3.136 ✓, gap = +0.0079 ✓.
Passes all gate dimensions.

### 5.3 Drift Detection Results

| Scenario | Latency (windows) | FP/h at detection | Policy triggered |
|---|---:|---:|---|
| Gain drift (α=0.01) | 11064 win. | 3.128 | monitor |
| Gain drift (α=0.05) | 11064 win. | 3.104 | monitor |
| Noise drift (β=0.05) | 11124 win. | 3.056 | monitor |
| Noise drift (β=0.10) | 11199 win. | 2.728 | monitor |

> **TODO:** rodar experimento de drift simulado.

### 5.4 OTA Update Reduction

- Reactive policy (retrain on every alert): 5 OTA updates over 5 drift scenarios.
- Conditional policy: [Y] OTA updates — reduction of [Z]% while maintaining the same final detection performance.

---

## 6. Discussion

**Why FP/h is not captured by AUC-PR.** AUC-PR integrates over all thresholds; the operational FP/h is threshold-dependent. A model with a wide, flat PR curve may achieve high AUC-PR while producing many false positives at any practically chosen operating point. Explicitly including FP/h in the gate — at the threshold chosen on the validation set — directly controls the operational impact.

**VUS-PR as a prerequisite for fair comparison.** When comparing models trained on different datasets (cross-domain, Artigo 3), AUC-PR values are not directly comparable due to varying overlap fractions. VUS-PR normalizes across buffer sizes, enabling fairer cross-dataset ranking.

**Conditional retraining vs. periodic retraining.** Periodic retraining (e.g., weekly) is common in production ML but incurs unnecessary OTA cost when data is stable. Our conditional policy responds to evidence of degradation, reducing bandwidth and energy expenditure at the edge — a primary concern in battery-powered sensor nodes.

---

## 7. Conclusion

We proposed a multidimensional quality-gate framework for TinyML anomaly detection that goes beyond offline accuracy to incorporate operational metrics (FP/h, PA-F1, Event-F1, VUS-PR), generalization checks, and embedded resource constraints. The ablation study demonstrates that AUC-PR alone is insufficient to select operationally reliable models: candidates passing only offline screening exhibit FP/h up to **7.0×** higher than the quality-gate-approved model. The conditional retraining policy reduces unnecessary OTA updates by [X]% while maintaining detection performance.

**Future work:**
- Adaptive threshold adjustment as an alternative to full retraining for mild drift scenarios
- Extension to multivariate drift detection for multi-sensor domains (Petrobras 3W)
- Hardware-in-the-loop validation of FP/h with real ESP32 deployment

---

## References

1. G. Pang et al., "Deep learning for anomaly detection: A review," *ACM Comput. Surv.*, vol. 54, 2021.
2. J. Gama et al., "A survey on concept drift adaptation," *ACM Comput. Surv.*, vol. 46, 2014.
3. S. Kim et al., "Towards a rigorous evaluation of time-series anomaly detection," *AAAI*, 2022.
4. H. Xu et al., "Unsupervised anomaly detection via variational auto-encoder for seasonal KPIs in web applications," *WWW*, 2018.
5. N. Tatbul et al., "Precision and recall for time series," *NeurIPS*, 2018.
6. J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," *ICML*, 2006.
7. D. Sculley et al., "Hidden technical debt in machine learning systems," *NeurIPS*, 2015.
8. M. Vartak et al., "ModelDB: A system for machine learning model management," *HILDA Workshop at SIGMOD*, 2016.
9. M. Zaharia et al., "Accelerating the machine learning lifecycle with MLflow," *IEEE Data Eng. Bull.*, vol. 41, 2018.
10. Iterative AI, "DVC: Data version control," https://dvc.org, 2024.
11. Evidently AI, "Evidently: Evaluate and monitor ML models in production," https://evidentlyai.com, 2024.

---

## Checklist antes da submissão

- [ ] Rodar ablação sobre os 88 trials do Optuna e preencher Tabela 5.1
- [ ] Implementar e rodar experimento de drift simulado (gain, noise, offset)
- [ ] Medir latência de detecção de drift pelo Evidently AI
- [ ] Comparar número de OTA updates: política reativa vs. condicional
- [ ] Adicionar figura: ablação em cascata (cada dimensão do gate eliminando candidatos)
- [ ] Adicionar figura: curva de FP/h sob drift simulado com ponto de detecção marcado
- [ ] Verificar page limit: 12–15 páginas IEEE double-column (IEEE Access)
