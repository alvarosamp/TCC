# Relatorio Final - Pipeline TinyML/MLOps

## Resumo Executivo

Este relatorio consolida treino, quality gate, drift, exportacao edge, OTA e observabilidade.

## Modelo Selecionado

- Modelo: tiny_tcn
- AUC-PR teste: 0.9144
- F1 teste: 0.8537
- Precision: 0.8761
- Recall: 0.8324
- FP/h: 5.4636
- Threshold: 0.6696

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


## Manifests Detalhados

## Candidate Manifest

```json
{
  "selected_by": {
    "metric": "auc_pr",
    "split": "test",
    "mode": "maximize",
    "groups": {
      "best_overall": {
        "enabled": true,
        "metric": "auc_pr",
        "split": "test",
        "mode": "maximize",
        "constraints": {}
      },
      "best_edge": {
        "enabled": true,
        "metric": "auc_pr",
        "split": "test",
        "mode": "maximize",
        "constraints": {
          "edge_candidate": true,
          "export_tflite": true,
          "max_params": 20000
        }
      },
      "best_interpretable": {
        "enabled": true,
        "metric": "auc_pr",
        "split": "test",
        "mode": "maximize",
        "constraints": {
          "families": [
            "classical_supervised"
          ]
        }
      }
    }
  },
  "model_name": "tiny_tcn",
  "family": "neural_classifier",
  "priority": "main_edge_candidate",
  "edge_candidate": true,
  "export_tflite": true,
  "profile": {
    "profile_name": "seismic_edge_v1",
    "profile_version": "1.0.0",
    "task": "binary_anomaly_detection",
    "domain": "seismic_edge",
    "description": null,
    "sampling_rate": 40.0,
    "window_size": 800,
    "window_seconds": 20.0,
    "step_seconds": 10.0,
    "overlap": 0.5,
    "normal_label": 0,
    "anomaly_label": 1,
    "normal_name": "normal",
    "anomaly_name": "fault",
    "split_name": "temporal",
    "primary_metric": "auc_pr",
    "secondary_metrics": [
      "f1",
      "auc_roc",
      "precision",
      "recall"
    ],
    "preprocessing": {
      "offline_pipeline": [
        "load_vibration_signal",
        "select_channel",
        "replace_nan_inf",
        "detrend_linear",
        "demean",
        "zscore_per_window"
      ],
      "edge_pipeline": [
        "select_channel",
        "detrend_linear",
        "demean",
        "zscore_per_window"
      ],
      "filtering": {
        "enabled": false,
        "type": "none",
        "bandpass_hz": null
      },
      "normalization": "zscore_per_window"
    },
    "embedded": {
      "target": "esp32",
      "runtime": "tensorflow_lite_micro",
      "preprocessing_version": "seismic_edge_preproc_v1",
      "ota_strategy_initial": "firmware_full_image",
      "ota_strategy_future": "separate_model_update"
    }
  },
  "dataset": "/mnt/d/PipelineGenerico/data/processed_seismic_edge_v1-20260604T184353Z-3-001/processed_seismic_edge_v1/dataset_seismic_edge_v1_split_evento.npz",
  "model_path": "/home/vish8/tcc_atual/TCC/artefacts/models/tiny_tcn.keras",
  "threshold": 0.6695716381072998,
  "summary_metrics": {
    "model_name": "tiny_tcn",
    "family": "neural_classifier",
    "priority": "main_edge_candidate",
    "edge_candidate": true,
    "export_tflite": true,
    "used_optuna": false,
    "parameter_count": 5273,
    "val_auc_pr": 0.9022069688443869,
    "val_auc_roc": 0.9668699508715256,
    "val_f1": 0.8425838820947006,
    "val_precision": 0.8735370611183355,
    "val_recall": 0.8137492428831011,
    "val_fp_per_hour": 5.427486241376638,
    "test_auc_pr": 0.9144131646435841,
    "test_auc_roc": 0.9694001668836154,
    "test_f1": 0.8536660505237215,
    "test_precision": 0.8760670249762883,
    "test_recall": 0.8323820967257435,
    "test_fp_per_hour": 5.463626156645631,
    "threshold_from_val": 0.6695716381072998,
    "model_path": "/home/vish8/tcc_atual/TCC/artefacts/models/tiny_tcn.keras"
  },
  "metrics": {
    "threshold_from_val": 0.6695716381072998,
    "val": {
      "auc_pr": 0.9022069688443869,
      "auc_roc": 0.9668699508715256,
      "threshold": 0.6695716381072998,
      "best_threshold_by_f1": {
        "threshold": 0.6695716381072998,
        "precision": 0.8735370611183355,
        "recall": 0.8137492428831011,
        "f1": 0.8425838771009785
      },
      "f1": 0.8425838820947006,
      "precision": 0.8735370611183355,
      "recall": 0.8137492428831011,
      "confusion_matrix": {
        "tn": 22111,
        "fp": 389,
        "fn": 615,
        "tp": 2687
      }
    },
    "test": {
      "auc_pr": 0.9144131646435841,
      "auc_roc": 0.9694001668836154,
      "threshold": 0.6695716381072998,
      "best_threshold_by_f1": {
        "threshold": 0.6695478558540344,
        "precision": 0.8761061946902655,
        "recall": 0.8326824872334034,
        "f1": 0.8538425947260074
      },
      "f1": 0.8536660505237215,
      "precision": 0.8760670249762883,
      "recall": 0.8323820967257435,
      "confusion_matrix": {
        "tn": 22108,
        "fp": 392,
        "fn": 558,
        "tp": 2771
      }
    }
  },
  "hpo": {
    "used_optuna": false,
    "best_params": {
      "batch_size": 64,
      "pos_multiplier": 1.274375423547668,
      "filters": 24,
      "kernel_size": 11,
      "n_blocks": 3,
      "dilation_base": 2,
      "dropout": 0.03544702294960163,
      "spatial_dropout": 0.1142415531888814,
      "dense_units": 32,
      "learning_rate": 0.0019901844880576103,
      "l2_reg": 8.596266772391992e-07,
      "head_pooling": "avg",
      "label_smoothing": 0.012938050702971032,
      "padding": "same",
      "conv_type": "separable",
      "use_batch_norm": false,
      "epochs": 40,
      "patience": 8
    },
    "best_value": null,
    "n_trials": 0
  },
  "params": {
    "batch_size": 64,
    "pos_multiplier": 1.274375423547668,
    "filters": 24,
    "kernel_size": 11,
    "n_blocks": 3,
    "dilation_base": 2,
    "dropout": 0.03544702294960163,
    "spatial_dropout": 0.1142415531888814,
    "dense_units": 32,
    "learning_rate": 0.0019901844880576103,
    "l2_reg": 8.596266772391992e-07,
    "head_pooling": "avg",
    "label_smoothing": 0.012938050702971032,
    "padding": "same",
    "conv_type": "separable",
    "use_batch_norm": false,
    "epochs": 40,
    "patience": 8
  },
  "parameter_count": 5273
}
```
## Promotion Report

```json
{
  "approved": true,
  "candidate_model": "tiny_tcn",
  "candidate_family": "neural_classifier",
  "model_path": "/home/vish8/tcc_atual/TCC/artefacts/models/tiny_tcn.keras",
  "checks": [
    {
      "name": "min_auc_pr",
      "passed": true,
      "value": 0.9144131646435841,
      "rule": ">= 0.8"
    },
    {
      "name": "min_f1",
      "passed": true,
      "value": 0.8536660505237215,
      "rule": ">= 0.7"
    },
    {
      "name": "max_fp_per_hour",
      "passed": true,
      "value": 5.463626156645631,
      "rule": "<= 10.0"
    },
    {
      "name": "max_val_test_auc_pr_gap",
      "passed": true,
      "value": 0.01220619579919724,
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
  "created_at_utc": "2026-06-18T18:37:15.086042+00:00",
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
    "path": "/home/vish8/tcc_atual/TCC/artefacts/edge/tiny_tcn_int8.tflite",
    "sha256": "53e8316f7fe0d12b95d426484a750ebb18b04304da1931cc6712214fc0ba89e3",
    "size_bytes": 27776,
    "size_kb": 27.12,
    "quantization": "int8",
    "source_model_path": "/home/vish8/tcc_atual/TCC/artefacts/models/tiny_tcn.keras"
  },
  "model": {
    "name": "tiny_tcn",
    "family": "neural_classifier",
    "version": "seismic_edge_v1_tiny_tcn_20260618",
    "threshold": 0.6695716381072998,
    "parameter_count": 5273
  },
  "profile": {
    "name": "seismic_edge_v1",
    "version": "1.0.0",
    "domain": "seismic_edge",
    "sampling_rate": 40.0,
    "window_size": 800,
    "window_seconds": 20.0,
    "step_seconds": 10.0,
    "preprocessing": {
      "offline_pipeline": [
        "load_vibration_signal",
        "select_channel",
        "replace_nan_inf",
        "detrend_linear",
        "demean",
        "zscore_per_window"
      ],
      "edge_pipeline": [
        "select_channel",
        "detrend_linear",
        "demean",
        "zscore_per_window"
      ],
      "filtering": {
        "enabled": false,
        "type": "none",
        "bandpass_hz": null
      },
      "normalization": "zscore_per_window"
    }
  },
  "quality": {
    "summary_metrics": {
      "model_name": "tiny_tcn",
      "family": "neural_classifier",
      "priority": "main_edge_candidate",
      "edge_candidate": true,
      "export_tflite": true,
      "used_optuna": false,
      "parameter_count": 5273,
      "val_auc_pr": 0.9022069688443869,
      "val_auc_roc": 0.9668699508715256,
      "val_f1": 0.8425838820947006,
      "val_precision": 0.8735370611183355,
      "val_recall": 0.8137492428831011,
      "val_fp_per_hour": 5.427486241376638,
      "test_auc_pr": 0.9144131646435841,
      "test_auc_roc": 0.9694001668836154,
      "test_f1": 0.8536660505237215,
      "test_precision": 0.8760670249762883,
      "test_recall": 0.8323820967257435,
      "test_fp_per_hour": 5.463626156645631,
      "threshold_from_val": 0.6695716381072998,
      "model_path": "/home/vish8/tcc_atual/TCC/artefacts/models/tiny_tcn.keras"
    },
    "quality_gate": {
      "approved": true,
      "candidate_model": "tiny_tcn",
      "candidate_family": "neural_classifier",
      "model_path": "/home/vish8/tcc_atual/TCC/artefacts/models/tiny_tcn.keras",
      "checks": [
        {
          "name": "min_auc_pr",
          "passed": true,
          "value": 0.9144131646435841,
          "rule": ">= 0.8"
        },
        {
          "name": "min_f1",
          "passed": true,
          "value": 0.8536660505237215,
          "rule": ">= 0.7"
        },
        {
          "name": "max_fp_per_hour",
          "passed": true,
          "value": 5.463626156645631,
          "rule": "<= 10.0"
        },
        {
          "name": "max_val_test_auc_pr_gap",
          "passed": true,
          "value": 0.01220619579919724,
          "rule": "<= 0.08"
        }
      ],
      "reasons": []
    }
  }
}
```