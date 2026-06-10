| Modelo | Familia | AUC-PR | AUC-ROC | F1 | Precision | Recall | FP/h | Params | Edge | Optuna |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| tiny_tcn | neural_classifier | 0.9416 | 0.9799 | 0.8885 | 0.9266 | 0.8534 | 3.136 | 14897 | True | False |
| extra_trees | classical_supervised | 0.9862 | 0.9841 | 0.9554 | 0.9740 | 0.9375 | 3.600 |  | False | True |
| tiny_cnn | neural_classifier | 0.8940 | 0.9629 | 0.8284 | 0.8544 | 0.8038 | 6.356 | 15129 | True | True |
| logistic_regression | classical_supervised | 0.9948 | 0.9961 | 0.9565 | 0.9506 | 0.9625 | 7.200 |  | False | True |
| random_forest | classical_supervised | 0.9868 | 0.9854 | 0.9625 | 0.9625 | 0.9625 | 5.400 |  | False | True |
