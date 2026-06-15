# Relatorio Final - Pipeline TinyML/MLOps

## Resumo Executivo

Este relatorio consolida treino, quality gate, drift, exportacao edge, OTA e observabilidade.

## Modelo Selecionado

- Modelo: tiny_cnn
- AUC-PR teste: 0.8775
- F1 teste: 0.8109
- Precision: 0.8431
- Recall: 0.7810
- FP/h: 6.7459
- Threshold: 0.7242

## Drift

- Nivel: high
- Max PSI: 0.3463
- Max z-shift: 0.0176
- Min KS p-value: 0.0000

## Decisao OTA

- Acao: build_and_publish_ota
- Candidato aprovado: True
- Motivo: Drift recomendou retreino e existe candidato aprovado.

## Exportacao Edge

- float32: 66.1800 KB - `/home/vish8/tcc_atual/TCC/artefacts/edge/tiny_cnn_float32.tflite`
- float16: 37.9400 KB - `/home/vish8/tcc_atual/TCC/artefacts/edge/tiny_cnn_float16.tflite`
- int8: 25.7100 KB - `/home/vish8/tcc_atual/TCC/artefacts/edge/tiny_cnn_int8.tflite`

## Manifests Detalhados

## Candidate Manifest

```json
{
  "selected_by": {
    "metric": "auc_pr",
    "split": "test",
    "mode": "maximize"
  },
  "model_name": "tiny_cnn",
  "family": "neural_classifier",
  "priority": "main_candidate",
  "edge_candidate": true,
  "export_tflite": true,
  "profile": {
    "profile_name": "seismic_edge_v1",
    "profile_version": "1.0.0",
    "task": "binary_anomaly_detection",
    "domain": "seismic",
    "description": null,
    "sampling_rate": 40.0,
    "window_size": 800,
    "window_seconds": 20.0,
    "step_seconds": 10.0,
    "overlap": 0.5,
    "normal_label": 0,
    "anomaly_label": 1,
    "normal_name": "normal",
    "anomaly_name": "anomalia",
    "split_name": "evento",
    "primary_metric": "auc_pr",
    "secondary_metrics": [
      "f1",
      "auc_roc",
      "precision",
      "recall"
    ],
    "preprocessing": {
      "offline_pipeline": [
        "resample_40hz",
        "detrend-linear",
        "demean",
        "taper_5pct",
        "bandpass_0p5_15hz_zerophase",
        "zscore_per_window"
      ],
      "edge_pipeline": [
        "detrend-linear",
        "taper-5pct",
        "bandpass-0p5_15hz_zerophase",
        "zscore-per_window"
      ],
      "remove_response_offline": false,
      "remove_response_edge": false,
      "stationxml_required": false,
      "bandpass_hz": [
        0.5,
        15.0
      ],
      "normalization": "zscore_per_window"
    },
    "embedded": {
      "target": "esp32",
      "runtime": "tensorflow_lite_micro",
      "decision_interval_ms": 10000,
      "preprocessing_version": "seismic_edge_prepoc_v1",
      "ota_strategy_initial": "firmware_full_image",
      "ota_strategy_future": "separate_model_update"
    }
  },
  "dataset": "/mnt/d/PipelineGenerico/data/processed_seismic_edge_v1-20260604T184353Z-3-001/processed_seismic_edge_v1/dataset_seismic_edge_v1_split_evento.npz",
  "model_path": "/home/vish8/tcc_atual/TCC/artefacts/models/tiny_cnn.keras",
  "threshold": 0.7241942882537842,
  "summary_metrics": {
    "model_name": "tiny_cnn",
    "family": "neural_classifier",
    "priority": "main_candidate",
    "edge_candidate": true,
    "export_tflite": true,
    "used_optuna": false,
    "parameter_count": 15377,
    "val_auc_pr": 0.8597729907721775,
    "val_auc_roc": 0.9546285079749647,
    "val_f1": 0.7908236418671339,
    "val_precision": 0.8342857142857143,
    "val_recall": 0.7516656571774682,
    "val_fp_per_hour": 6.878536547554454,
    "test_auc_pr": 0.8775302765225134,
    "test_auc_roc": 0.9575893194486164,
    "test_f1": 0.8108529549352878,
    "test_precision": 0.8430609597924773,
    "test_recall": 0.7810153199158907,
    "test_fp_per_hour": 6.745905764837973,
    "threshold_from_val": 0.7241942882537842,
    "model_path": "/home/vish8/tcc_atual/TCC/artefacts/models/tiny_cnn.keras"
  },
  "metrics": {
    "threshold_from_val": 0.7241942882537842,
    "val": {
      "auc_pr": 0.8597729907721775,
      "auc_roc": 0.9546285079749647,
      "threshold": 0.7241942882537842,
      "best_threshold_by_f1": {
        "threshold": 0.7241942882537842,
        "precision": 0.8342857142857143,
        "recall": 0.7516656571774682,
        "f1": 0.7908236368807035
      },
      "f1": 0.7908236418671339,
      "precision": 0.8342857142857143,
      "recall": 0.7516656571774682,
      "confusion_matrix": {
        "tn": 22007,
        "fp": 493,
        "fn": 820,
        "tp": 2482
      }
    },
    "test": {
      "auc_pr": 0.8775302765225134,
      "auc_roc": 0.9575893194486164,
      "threshold": 0.7241942882537842,
      "best_threshold_by_f1": {
        "threshold": 0.7456919550895691,
        "precision": 0.8575717144763175,
        "recall": 0.7723039951937519,
        "f1": 0.8127074393000762
      },
      "f1": 0.8108529549352878,
      "precision": 0.8430609597924773,
      "recall": 0.7810153199158907,
      "confusion_matrix": {
        "tn": 22016,
        "fp": 484,
        "fn": 729,
        "tp": 2600
      }
    }
  },
  "hpo": {
    "used_optuna": false,
    "best_params": {
      "n_blocks": 3,
      "base_filters": 16,
      "kernel_first": 21,
      "kernel_other": 5,
      "dense_units": 48,
      "dropout": 0.240293,
      "spatial_dropout": 0.007913,
      "learning_rate": 0.0027376,
      "l2_reg": 8.261e-06,
      "use_batch_norm": false,
      "conv_type": "conv",
      "head_pooling": "avgmax",
      "label_smoothing": 0.008688,
      "pos_multiplier": 1.0,
      "epochs": 20,
      "batch_size": 128,
      "patience": 5
    },
    "best_value": null,
    "n_trials": 0
  },
  "params": {
    "n_blocks": 3,
    "base_filters": 16,
    "kernel_first": 21,
    "kernel_other": 5,
    "dense_units": 48,
    "dropout": 0.240293,
    "spatial_dropout": 0.007913,
    "learning_rate": 0.0027376,
    "l2_reg": 8.261e-06,
    "use_batch_norm": false,
    "conv_type": "conv",
    "head_pooling": "avgmax",
    "label_smoothing": 0.008688,
    "pos_multiplier": 1.0,
    "epochs": 20,
    "batch_size": 128,
    "patience": 5
  },
  "parameter_count": 15377
}
```
## Promotion Report

```json
{
  "approved": true,
  "candidate_model": "tiny_cnn",
  "candidate_family": "neural_classifier",
  "model_path": "/home/vish8/tcc_atual/TCC/artefacts/models/tiny_cnn.keras",
  "checks": [
    {
      "name": "min_auc_pr",
      "passed": true,
      "value": 0.8775302765225134,
      "rule": ">= 0.8"
    },
    {
      "name": "min_f1",
      "passed": true,
      "value": 0.8108529549352878,
      "rule": ">= 0.7"
    },
    {
      "name": "max_fp_per_hour",
      "passed": true,
      "value": 6.745905764837973,
      "rule": "<= 10.0"
    },
    {
      "name": "max_val_test_auc_pr_gap",
      "passed": true,
      "value": 0.01775728575033586,
      "rule": "<= 0.08"
    }
  ],
  "reasons": []
}
```
## Drift Report

```json
{
  "dataset": "/mnt/d/PipelineGenerico/data/processed_seismic_edge_v1-20260604T184353Z-3-001/processed_seismic_edge_v1/dataset_seismic_edge_v1_split_evento.npz",
  "split": "test",
  "reference_path": "artefacts/monitoring/drift_reference.json",
  "n_reference_samples": 120429,
  "n_current_samples": 25829,
  "n_features": 28,
  "summary": {
    "drift_level": "high",
    "max_abs_z_shift": 0.017554350197315216,
    "max_psi": 0.3462861952044981,
    "min_ks_pvalue": 3.321462997836135e-05
  },
  "thresholds": {
    "moderate_abs_z_shift": 0.75,
    "high_abs_z_shift": 1.5,
    "moderate_psi": 0.1,
    "high_psi": 0.25,
    "moderate_ks_pvalue": 0.05,
    "high_ks_pvalue": 0.001
  },
  "features": [
    {
      "feature": "mean",
      "reference_mean": 3.293403744164536e-11,
      "current_mean": -2.8759746320550228e-11,
      "reference_std": 9.02212438091965e-09,
      "current_std": 9.057165684112078e-09,
      "z_shift": -0.0032432645093649626,
      "abs_z_shift": 0.0032432645093649626,
      "psi": 0.15464193634591694,
      "ks_statistic": 0.05176352162298192,
      "ks_pvalue": 0.011065581987489545
    },
    {
      "feature": "std",
      "reference_mean": 1.0,
      "current_mean": 1.0,
      "reference_std": 3.901507028558626e-08,
      "current_std": 4.407162279562726e-08,
      "z_shift": 0.0,
      "abs_z_shift": 0.0,
      "psi": 0.02109698626811138,
      "ks_statistic": 0.04476363777149714,
      "ks_pvalue": 0.040891463746000056
    },
    {
      "feature": "min",
      "reference_mean": -3.544682741165161,
      "current_mean": -3.5457537174224854,
      "reference_std": 1.4212646484375,
      "current_std": 1.3858174085617065,
      "z_shift": -0.0007535375189036131,
      "abs_z_shift": 0.0007535375189036131,
      "psi": 0.22230004255125618,
      "ks_statistic": 0.04955669983352051,
      "ks_pvalue": 0.01705344584496927
    },
    {
      "feature": "max",
      "reference_mean": 3.476884365081787,
      "current_mean": 3.4728665351867676,
      "reference_std": 0.9727787375450134,
      "current_std": 0.9645397067070007,
      "z_shift": -0.004130261018872261,
      "abs_z_shift": 0.004130261018872261,
      "psi": 0.21398632536075424,
      "ks_statistic": 0.051861705834527094,
      "ks_pvalue": 0.010849936611562932
    },
    {
      "feature": "median",
      "reference_mean": 0.0008675580029375851,
      "current_mean": 0.0010837108129635453,
      "reference_std": 0.029684215784072876,
      "current_std": 0.029621722176671028,
      "z_shift": 0.007281740196049213,
      "abs_z_shift": 0.007281740196049213,
      "psi": 0.18763606370242134,
      "ks_statistic": 0.051879670138216705,
      "ks_pvalue": 0.010810894630158283
    },
    {
      "feature": "abs_mean",
      "reference_mean": 0.7835810780525208,
      "current_mean": 0.7834722399711609,
      "reference_std": 0.05206476151943207,
      "current_std": 0.05086439102888107,
      "z_shift": -0.0020904361736029387,
      "abs_z_shift": 0.0020904361736029387,
      "psi": 0.30820366705080227,
      "ks_statistic": 0.05491768941887032,
      "ks_pvalue": 0.005771769098735922
    },
    {
      "feature": "abs_peak",
      "reference_mean": 3.789064884185791,
      "current_mean": 3.7870752811431885,
      "reference_std": 1.4648396968841553,
      "current_std": 1.4304171800613403,
      "z_shift": -0.001358239445835352,
      "abs_z_shift": 0.001358239445835352,
      "psi": 0.2262330012644303,
      "ks_statistic": 0.04963413217701029,
      "ks_pvalue": 0.01680195038296989
    },
    {
      "feature": "rms",
      "reference_mean": 1.0,
      "current_mean": 1.0,
      "reference_std": 3.902077239104074e-08,
      "current_std": 4.4073182436932257e-08,
      "z_shift": 0.0,
      "abs_z_shift": 0.0,
      "psi": 0.021001212712896344,
      "ks_statistic": 0.04472492159975219,
      "ks_pvalue": 0.041166480410796735
    },
    {
      "feature": "crest_factor",
      "reference_mean": 3.789064884185791,
      "current_mean": 3.7870755195617676,
      "reference_std": 1.4648396968841553,
      "current_std": 1.4304171800613403,
      "z_shift": -0.0013580766972154379,
      "abs_z_shift": 0.0013580766972154379,
      "psi": 0.2262330012644303,
      "ks_statistic": 0.04963413217701029,
      "ks_pvalue": 0.01680195038296989
    },
    {
      "feature": "peak_to_peak",
      "reference_mean": 7.021566867828369,
      "current_mean": 7.018635272979736,
      "reference_std": 2.136026382446289,
      "current_std": 2.099888563156128,
      "z_shift": -0.0013724525924772024,
      "abs_z_shift": 0.0013724525924772024,
      "psi": 0.2356292597647286,
      "ks_statistic": 0.050624801579619794,
      "ks_pvalue": 0.013864979480788417
    },
    {
      "feature": "energy",
      "reference_mean": 800.0,
      "current_mean": 800.0,
      "reference_std": 6.375187513185665e-05,
      "current_std": 6.372528150677681e-05,
      "z_shift": 0.0,
      "abs_z_shift": 0.0,
      "psi": 0.048929457026483686,
      "ks_statistic": 0.04472492159975219,
      "ks_pvalue": 0.041166480410796735
    },
    {
      "feature": "skewness",
      "reference_mean": -0.05860995873808861,
      "current_mean": -0.055810268968343735,
      "reference_std": 0.9570907354354858,
      "current_std": 0.9189397692680359,
      "z_shift": 0.002925208304077387,
      "abs_z_shift": 0.002925208304077387,
      "psi": 0.3026108783109835,
      "ks_statistic": 0.054452088737465654,
      "ks_pvalue": 0.006369206276687657
    },
    {
      "feature": "kurtosis",
      "reference_mean": 1.9117703437805176,
      "current_mean": 1.8190226554870605,
      "reference_std": 20.78434944152832,
      "current_std": 19.996850967407227,
      "z_shift": -0.0044623808935284615,
      "abs_z_shift": 0.0044623808935284615,
      "psi": 0.3462861952044981,
      "ks_statistic": 0.06517344844941741,
      "ks_pvalue": 0.0005329737534970441
    },
    {
      "feature": "p05",
      "reference_mean": -1.6125205755233765,
      "current_mean": -
```
## Prediction/Performance Drift

```json
{
  "n_events": 138,
  "n_reference_events": 38,
  "n_recent_events": 100,
  "prediction_drift": {
    "level": "moderate",
    "score_psi": 0.17722072881308246,
    "reference_anomaly_rate": 0.15789473684210525,
    "recent_anomaly_rate": 0.27,
    "anomaly_rate_delta": 0.11210526315789476
  },
  "performance_drift": {
    "level": "unknown",
    "n_labeled_events": 0,
    "precision": null,
    "recall": null,
    "f1": null,
    "confusion": {
      "tp": 0,
      "fp": 0,
      "fn": 0,
      "tn": 0
    }
  }
}
```
## OTA Manifest

```json
{
  "schema_version": "1.0.0",
  "created_at_utc": "2026-06-14T21:30:02.103269+00:00",
  "status": "ready_for_packaging",
  "target": {
    "device": "esp32",
    "runtime": "tensorflow_lite_micro"
  },
  "ota": {
    "strategy_initial": "firmware_full_image",
    "strategy_future": "separate_model_update",
    "rollback_required": true,
    "integrity_check": "sha256",
    "signature_required_future": true
  },
  "artifact": {
    "type": "tflite_int8",
    "path": "/home/vish8/tcc_atual/TCC/artefacts/edge/tiny_cnn_int8.tflite",
    "sha256": "f71ac3b1ec6b9207e459176f1bb86623ae23ed9573b51a1c9c4d15d6987c6bc0",
    "size_bytes": 26328,
    "size_kb": 25.71,
    "quantization": "int8",
    "source_model_path": "/home/vish8/tcc_atual/TCC/artefacts/models/tiny_cnn.keras"
  },
  "model": {
    "name": "tiny_cnn",
    "family": "neural_classifier",
    "version": "seismic_edge_v1_tiny_cnn_20260614",
    "threshold": 0.7241942882537842,
    "parameter_count": 15377
  },
  "profile": {
    "name": "seismic_edge_v1",
    "version": "1.0.0",
    "domain": "seismic",
    "sampling_rate": 40.0,
    "window_size": 800,
    "window_seconds": 20.0,
    "step_seconds": 10.0,
    "preprocessing": {
      "offline_pipeline": [
        "resample_40hz",
        "detrend-linear",
        "demean",
        "taper_5pct",
        "bandpass_0p5_15hz_zerophase",
        "zscore_per_window"
      ],
      "edge_pipeline": [
        "detrend-linear",
        "taper-5pct",
        "bandpass-0p5_15hz_zerophase",
        "zscore-per_window"
      ],
      "remove_response_offline": false,
      "remove_response_edge": false,
      "stationxml_required": false,
      "bandpass_hz": [
        0.5,
        15.0
      ],
      "normalization": "zscore_per_window"
    }
  },
  "quality": {
    "summary_metrics": {
      "model_name": "tiny_cnn",
      "family": "neural_classifier",
      "priority": "main_candidate",
      "edge_candidate": true,
      "export_tflite": true,
      "used_optuna": false,
      "parameter_count": 15377,
      "val_auc_pr": 0.8597729907721775,
      "val_auc_roc": 0.9546285079749647,
      "val_f1": 0.7908236418671339,
      "val_precision": 0.8342857142857143,
      "val_recall": 0.7516656571774682,
      "val_fp_per_hour": 6.878536547554454,
      "test_auc_pr": 0.8775302765225134,
      "test_auc_roc": 0.9575893194486164,
      "test_f1": 0.8108529549352878,
      "test_precision": 0.8430609597924773,
      "test_recall": 0.7810153199158907,
      "test_fp_per_hour": 6.745905764837973,
      "threshold_from_val": 0.7241942882537842,
      "model_path": "/home/vish8/tcc_atual/TCC/artefacts/models/tiny_cnn.keras"
    },
    "quality_gate": {
      "approved": true,
      "candidate_model": "tiny_cnn",
      "candidate_family": "neural_classifier",
      "model_path": "/home/vish8/tcc_atual/TCC/artefacts/models/tiny_cnn.keras",
      "checks": [
        {
          "name": "min_auc_pr",
          "passed": true,
          "value": 0.8775302765225134,
          "rule": ">= 0.8"
        },
        {
          "name": "min_f1",
          "passed": true,
          "value": 0.8108529549352878,
          "rule": ">= 0.7"
        },
        {
          "name": "max_fp_per_hour",
          "passed": true,
          "value": 6.745905764837973,
          "rule": "<= 10.0"
        },
        {
          "name": "max_val_test_auc_pr_gap",
          "passed": true,
          "value": 0.01775728575033586,
          "rule": "<= 0.08"
        }
      ],
      "reasons": []
    }
  }
}
```