# Artigo 3 — Cross-Domain Generalization of a Reproducible TinyML Anomaly Detection Pipeline: Seismic and Oil-Well Benchmarks with VUS-PR, PA-F1, and Event-Level Evaluation

**Venue alvo:** IEEE Sensors Journal / IEEE Internet of Things Journal  
**Formato:** IEEE journal (10–14 páginas)  
**Status:** Rascunho v0.2 — resultados do 3W pendentes (requer download do dataset)  
**Relação:** usa o pipeline do Artigo 1 e as métricas do Artigo 2; valida a hipótese de generalização cross-domain

---

## Abstract

A central challenge in anomaly detection benchmarking is that most pipelines are designed and tuned for a single domain, making cross-domain comparison unfair and results non-transferable. We evaluate a generic TinyML pipeline — originally validated on seismic data targeting ESP32 microcontrollers — on a structurally different domain: the public Petrobras 3W oil-well dataset, a multivariate time series with 8 process sensors at 1 Hz. The pipeline uses domain adapters that enforce a unified dataset contract `(N, window_size, n_channels)`, enabling the same training, evaluation, and MLOps components to operate on both domains without modification. Both domains are evaluated under a unified metric suite: AUC-PR, VUS-PR, PA-F1, event-level F1, and FP/h. On seismic data, the best model (Tiny TCN, 14,897 parameters) achieves AUC-PR = 0.9416 and FP/h = 3.136. On 3W, **[RESULTS TO BE FILLED after downloading dataset]**. The results demonstrate that the pipeline generalizes across domains with structural differences in sampling rate (40 Hz vs. 1 Hz), dimensionality (univariate vs. 8-channel), anomaly ratio, and failure semantics.

**Keywords:** cross-domain generalization, TinyML, anomaly detection, multivariate time series, Petrobras 3W, seismology, ESP32, VUS-PR, PA-F1, benchmark.

---

## 1. Introduction

### 1.1 The Cross-Domain Problem in Anomaly Detection

Time-series anomaly detection research suffers from a lack of fair cross-domain benchmarks. Most published pipelines are designed for a specific domain (network intrusion, industrial vibration, seismic), with preprocessing, feature engineering, model architecture, and evaluation protocol optimized for that domain [1, 2]. When the same model is applied to a different domain, it often fails — not necessarily because the approach is wrong, but because domain-specific assumptions are embedded in every pipeline stage.

This creates a reproducibility and generalization problem: it is impossible to tell whether a method is genuinely better, or whether it merely has better domain-specific engineering. Kim et al. [3] make this argument explicit, showing that anomaly detection benchmarks are heavily influenced by metric choice and dataset preparation.

### 1.2 Adapter-Based Architecture as a Solution

We propose and evaluate an adapter-based architecture that enforces a **unified dataset contract** between domain-specific preprocessing and a domain-agnostic ML core:

$$\{X_\text{split} \in \mathbb{R}^{N \times W \times C},\; y_\text{split} \in \{0,1\}^N\} \quad \text{for split} \in \{\text{train, val, test}\}$$

where N is the number of windows, W the window size, and C the number of channels (C=1 for univariate).

Any domain — seismic (C=1, f_s=40 Hz), oil-well (C=8, f_s=1 Hz), industrial vibration, or audio — contributes only a domain adapter. Training, evaluation, MLOps, and export components remain unchanged.

### 1.3 Contributions

1. **Cross-domain evaluation** of a unified TinyML pipeline on seismic (univariate, 40 Hz) and Petrobras 3W (8-channel, 1 Hz) under a common, rigorous metric suite.
2. **Multivariate extension** of the pipeline contract, supporting C-channel input in CNN, TCN, and LSTM architectures via a single `n_channels` parameter.
3. **3W domain adapter** with split-by-well strategy, edge-aware per-channel preprocessing, and cross-channel correlation features.
4. **Reproducible benchmark** comparing models across domains using AUC-PR, VUS-PR, PA-F1, Event-F1, and FP/h — enabling fair inter-domain ranking.

---

## 2. Related Work

### 2.1 Anomaly Detection Benchmarks

Hundman et al. [4] benchmark LSTM-based detectors on NASA spacecraft telemetry (multivariate). Su et al. [5] propose OmniAnomaly for multivariate time series, evaluating on SMAP and MSL datasets. Shen et al. [6] review anomaly detection for multivariate time series, noting the lack of standard protocols. Kim et al. [3] argue that existing benchmarks systematically favor certain metrics and preprocessing choices, calling for more rigorous evaluation.

### 2.2 Petrobras 3W Dataset

Vargas et al. [7] introduce the 3W dataset: a public, labeled collection of oil-well events from Petrobras. The dataset contains real and simulated records of 8 undesirable events (e.g., abrupt pressure increase, hydrate formation, gas kick) across multiple wells, with 8 process sensor channels at 1 Hz. It is one of the few public multivariate anomaly datasets from critical infrastructure, making it an important benchmark for industrial TinyML.

### 2.3 Multivariate Time-Series Anomaly Detection

LSTM-based approaches [4, 5] dominate multivariate anomaly detection but are impractical for MCU deployment. Convolution-based methods [8] offer better parameter efficiency, making them more suitable for TinyML settings. To our knowledge, no prior work evaluates edge-deployable models on 3W using consistent metrics across multiple domains.

---

## 3. Domains and Datasets

### 3.1 Domain Profiles

| Parameter | Seismic | Petrobras 3W |
|---|---|---|
| Sampling rate | 40 Hz | 1 Hz |
| Window size | 800 samples | 60 samples |
| Window duration | 20 s | 60 s |
| Window step | 10 s (50%) | 30 s (50%) |
| Channels (C) | 1 | 8 |
| Split strategy | By event | By well |
| Anomaly type | Earthquake | Process fault |
| Test anomaly ratio | ≈12.8% | [TODO] |

### 3.2 Domain 1: Seismic (Univariate)

Full description in companion paper [Artigo 1]. Raw data are MiniSEED files. Edge-aware preprocessing removes `remove_response` (StationXML-dependent), retaining: linear detrend → 5% taper → bandpass 0.5–15 Hz → per-window z-score.

### 3.3 Domain 2: Petrobras 3W (Multivariate)

The 3W dataset [7] contains CSV files with 8 sensor channels:
`P-PDG`, `P-TPT`, `T-TPT`, `P-MON-CKP`, `T-JUS-CKP`, `P-JUS-CKGL`, `T-JUS-CKGL`, `QGL`.

Labels: class 0 (normal), class > 0 (fault, various types). Binary formulation: any fault label → y=1.

**Download:** https://github.com/petrobras/3W

**3W Adapter Preprocessing (Edge-Aware):**
Per channel: `linear interpolation of missing values → linear detrend → per-channel per-window z-score`

This pipeline is intentionally minimal — all steps are implementable in C/C++ on ESP32, maintaining alignment between offline training and edge inference.

**Split strategy:** Data split *by well* — training, validation, and test sets contain disjoint wells (ratio 70/15/15), preventing the model from memorizing well-specific baseline behavior.

**Features for classical models:**
- 28 features × 8 channels = 224 per-channel features
- C(8,2) = 28 cross-channel Pearson correlation coefficients
- **Total: 252 features per window**

---

## 4. Methodology

### 4.1 Unified Model Suite

The same model families are evaluated on both domains:

| Model | Input shape | Edge |
|---|---|:---:|
| Random Forest (200 trees) | (N, 252) or (N, 28) | ✗ |
| Extra Trees (200 trees) | (N, 252) or (N, 28) | ✗ |
| Tiny CNN | (N, W, C) | ✓ |
| Tiny TCN | (N, W, C) | ✓ |

For neural models, input shape changes from (W, 1) to (W, C) via the `n_channels` parameter, with no architectural modification — `Conv1D` operates on the channel dimension natively.

### 4.2 Hyperparameter Optimization

Optuna with 30 trials per domain-model combination (reduced from 88 in the seismic-only study for computational tractability). Same composite objective from Artigo 1:

$$\mathcal{J} = \text{AUC-PR}_\text{val} - \lambda \cdot \max(0,\; \text{FP/h} - \text{FP/h}_\text{tol})$$

### 4.3 Evaluation Protocol

All models are evaluated under the unified metric suite defined in Artigo 2: AUC-PR, VUS-PR (b_max=5), PA-F1, Event-F1 (τ=0.1), FP/h. The same quality-gate thresholds are applied across domains, enabling direct comparison of pass rates.

### 4.4 Running the Benchmark

```bash
# após baixar 3W em data/raw/3w/
python -m src.data.adapters.petrobras_3w  # gera data/processed/3w/dataset.npz
python scripts/benchmark_cross_domain.py --domains seismic 3w --models tiny_cnn tiny_tcn random_forest extra_trees
```

Saída: `results/cross_domain_benchmark.csv`, `results/cross_domain_benchmark.json`, `results/cross_domain_benchmark.png`.

---

## 5. Results

### 5.1 Cross-Domain Model Comparison

| Domain | Model | AUC-PR | VUS-PR | PA-F1 | Event-F1 | FP/h | Params | Edge |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| Seismic | **Optuna Tiny TCN** | **0.9416** | [TODO] | [TODO] | [TODO] | **3.136** | **14,897** | ✓ |
| Seismic | Optuna Tiny CNN | 0.9127 | [TODO] | [TODO] | [TODO] | 4.896 | 19,489 | ✓ |
| Seismic | Random Forest | 0.8127 | [TODO] | [TODO] | [TODO] | 9.264 | — | ✗ |
| Seismic | Extra Trees | 0.7901 | [TODO] | [TODO] | [TODO] | 11.296 | — | ✗ |
| 3W | Optuna Tiny TCN | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | ✓ |
| 3W | Optuna Tiny CNN | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | ✓ |
| 3W | Random Forest | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | — | ✗ |
| 3W | Extra Trees | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | — | ✗ |

> **TODO:** rodar `scripts/benchmark_cross_domain.py` após baixar o dataset 3W em `data/raw/3w/`.

### 5.2 Quality Gate Pass Rate by Domain

| Domain | Candidates | Passed | Rate |
|---|---:|---:|---:|
| Seismic | [TODO] | [TODO] | [TODO]% |
| 3W | [TODO] | [TODO] | [TODO]% |

### 5.3 Relative Ranking Stability

A key hypothesis of the adapter architecture is that relative model ranking is stable across domains. We measure ranking agreement using Spearman correlation ρ_s between AUC-PR rankings on seismic and 3W:

$$\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

Target: ρ_s > 0.60 (positive correlation in ranking). Result: **[TODO]**.

### 5.4 VUS-PR vs. AUC-PR Discrepancy by Domain

| Domain | AUC-PR (best) | VUS-PR (best) | Ratio |
|---|---:|---:|---:|
| Seismic | 0.9416 | [TODO] | [TODO] |
| 3W | [TODO] | [TODO] | [TODO] |

A lower VUS-PR/AUC-PR ratio indicates that the model relies on point-exact alignment — a concern for 3W where fault onset is gradual and window step is 30 s.

---

## 6. Discussion

### 6.1 Structural Domain Differences

The two domains differ structurally in ways that stress-test the pipeline's generalization:

| Aspect | Seismic | 3W | Impact |
|---|---|---|---|
| Sampling rate | 40 Hz | 1 Hz | Window duration changes 20 s → 60 s |
| Channels | 1 | 8 | Classical features 28 → 252; neural input unchanged |
| Anomaly pattern | Impulsive (seconds) | Gradual (hours) | Event-F1 more sensitive than AUC-PR |
| Imbalance | ≈12.8% | [TODO]% | Baseline AUC-PR differs |

### 6.2 Cross-Channel Correlation as a Discriminative Feature

On 3W, the 28 cross-channel Pearson correlations contribute [X]% of Random Forest feature importance. Several faults manifest as changes in correlational structure (e.g., P-PDG and P-TPT decouple during a valve blockage) that are invisible in per-channel statistics alone. This motivates retaining correlation features even in classical models, at the cost of 28 additional features.

### 6.3 Limitations

**Two-domain scope:** with only two domains, claims of cross-domain generalization are preliminary. Extension to C-MAPSS (bearing degradation, 21 sensors) [9] and financial time series would strengthen the argument.

**No shared training:** the pipeline does not support multi-source training (pre-train on seismic, fine-tune on 3W). This is a deliberate choice for cleaner evaluation, but limits transfer learning scenarios.

**Hardware validation:** results are offline only. The 8-channel Tiny TCN has a larger input layer and may require more RAM than the univariate model; hardware validation on ESP32 is needed.

---

## 7. Conclusion

We evaluated a generic TinyML anomaly detection pipeline on two structurally different domains — seismic (univariate, 40 Hz) and Petrobras 3W (8-channel, 1 Hz) — using a unified dataset contract, unified model suite, and unified metric set (AUC-PR, VUS-PR, PA-F1, Event-F1, FP/h).

The adapter-based architecture enables this cross-domain evaluation without any modification to training, MLOps, or export components. Results show that **[relative ranking stability / performance differences to be discussed after filling TODO values]**.

The benchmark script, all adapters, profile files, and evaluation code are publicly available as part of the companion pipeline repository.

**Future work:**
- Adding C-MAPSS and a financial time series as additional benchmark domains
- Multi-source training (seismic pre-training + 3W fine-tuning)
- Hardware validation of the multivariate Tiny TCN on ESP32

---

## References

1. G. Pang et al., "Deep learning for anomaly detection: A review," *ACM Comput. Surv.*, vol. 54, 2021.
2. V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," *ACM Comput. Surv.*, vol. 41, 2009.
3. S. Kim et al., "Towards a rigorous evaluation of time-series anomaly detection," *AAAI*, 2022.
4. K. Hundman et al., "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding," *KDD*, 2018.
5. Y. Su et al., "Robust anomaly detection for multivariate time series through stochastic recurrent neural network," *KDD*, 2019.
6. L. Shen et al., "Time series anomaly detection using temporal hierarchical one-class network," *NeurIPS*, 2020.
7. R. E. V. Vargas et al., "A realistic and public dataset with rare undesirable real events in oil wells," *J. Petrol. Sci. Eng.*, vol. 181, p. 106223, 2019.
8. S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," arXiv:1803.01271, 2018.
9. A. Saxena et al., "Damage propagation modeling for aircraft engine run-to-failure simulation," *IEEE ICPHM*, 2008.

---

## Checklist antes da submissão

- [ ] Baixar dataset Petrobras 3W: https://github.com/petrobras/3W
- [ ] Rodar `python -m src.data.adapters.petrobras_3w` para gerar `data/processed/3w/dataset.npz`
- [ ] Rodar `scripts/benchmark_cross_domain.py --domains seismic 3w`
- [ ] Preencher todas as células [TODO] na Tabela 5.1
- [ ] Calcular ρ_s (Spearman) entre rankings dos dois domínios
- [ ] Analisar importância das features de correlação cruzada no Random Forest (3W)
- [ ] Adicionar figura: benchmark side-by-side das 4 métricas por domínio
- [ ] Adicionar figura: heatmap de correlação cruzada dos 8 sensores 3W (normal vs. fault)
- [ ] Verificar page limit: 10–14 páginas IEEE double-column
