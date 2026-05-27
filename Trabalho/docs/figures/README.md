# Figures for the IEEE Article

This folder keeps editable figure sources for the article. The main architecture figure is available both as Mermaid source and as SVG vector artwork.

## Included figures

| File | Purpose |
|---|---|
| `system_architecture.svg` | Main architecture figure: offline pipeline, ESP32 inference, monitoring lifecycle. |
| `system_architecture.svg.png` | PNG preview generated from the SVG. |
| `system_architecture.mmd` | Editable Mermaid source for the architecture. |
| `model_comparison.svg` | Results figure comparing AUC-PR and F1 across candidate models. |
| `model_comparison.svg.png` | PNG preview generated from the model comparison SVG. |
| `experimental_workflow.mmd` | Methodology figure: data preparation, model selection, evaluation, export, edge benchmark. |
| `preprocessing_comparison.mmd` | Technical figure: offline preprocessing versus embedded preprocessing and the `remove_response` limitation. |

## Recommended additional figures

1. PR curve and confusion matrix for the selected Tiny TCN classifier.
2. Quantization and edge benchmark table: model size, latency, RAM/Flash estimate and metric delta for float32, float16 and int8.
3. Dataset distribution figure: train/validation/test counts and anomaly ratio after event-level split.
4. OTA/MLOps lifecycle diagram, but only if presented as future work or implemented contribution.

For IEEE-style two-column papers, prefer short labels, vector formats, and captions that state the contribution directly.
