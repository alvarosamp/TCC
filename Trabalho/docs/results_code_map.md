# Mapa Resultado -> Codigo

Este arquivo conecta cada resultado usado no TCC ao codigo que o gerou. Ele tambem marca lacunas para evitar resultado solto sem rastreabilidade.

## Status rapido

| Resultado / artefato | Status | Codigo de origem | Observacao |
|---|---|---|---|
| Dataset v3 processado | Parcial | `Trabalho/notebook/Dados/passo_02_v3_pipeline.py` | Pipeline offline original: preprocessing + janelamento + splits estacao/temporal. |
| Split v4 por evento | Coberto | `Trabalho/scripts/create_split_evento_v4.py` | Versao canonica em script. Veio do codigo Colab. |
| Inspecao de dataset/profile | Coberto | `Trabalho/scripts/inspect_dataset.py` | Valida NPZ contra `profiles/seismic_v1.json`. |
| STA/LTA | Coberto | `Trabalho/notebook/Treinamento/passo_03_v2_staltalvta_corrigido.ipynb`, `Trabalho/notebook/Treinamento/Sta-lta/` | Resultado resumido em `metrics_summary.md`. |
| Random Forest / ExtraTrees | Coberto | `Trabalho/scripts/train_ml_models.py`, `Trabalho/notebook/Treinamento/RandomForest/rf.py`, `Trabalho/notebooks/Metricas_Alvaro.ipynb` | Usar script canonico quando possivel. |
| Dense AE | Coberto historico | `Trabalho/notebook/Treinamento/DenseAE/`, `Trabalho/scripts/train_autoencoders.py` | Modelos `.h5/.keras` nao devem ir ao Git. |
| CNN 1D AE / LSTM AE | Coberto historico | `Trabalho/notebook/Treinamento/Cnn-1d/treino.py`, `Trabalho/notebook/Treinamento/LSTM/treino.py`, `Trabalho/notebook/Treinamento/Optuna/` | Verificar quais resultados finais entram no texto. |
| Tiny CNN classifier | Coberto no notebook | `Trabalho/notebooks/Metricas_Alvaro.ipynb` | Falta extrair para script canonico se virar resultado final. |
| Tiny TCN classifier | Coberto no notebook + edge | `Trabalho/notebooks/Metricas_Alvaro.ipynb`, `Trabalho/edge/platformio_snapshot/` | Melhor candidato observado. |
| Conversao `.tflite` | Parcial | `Trabalho/notebooks/Metricas_Alvaro.ipynb`, `Trabalho/edge/platformio_snapshot/include/TCN/` | Falta script canonico de conversao/export se for usado na banca. |
| Quantizacao float32/float16/int8 | Pendente consolidar | Ainda falta apontar arquivo final | Usuario vai enviar resultados/codigos. Deve virar `scripts/convert_and_evaluate_tflite.py` ou notebook documentado. |
| Teste embarcado ESP32 | Parcial | `Trabalho/edge/platformio_snapshot/src/main.cpp` | Falta anexar logs CSV/resultados finais e script de analise. |
| Preprocessing embarcado | Coberto | `Trabalho/edge/platformio_snapshot/include/preprocessing.h` | Aproxima pipeline offline sem `remove_response`. |
| Export de `real_dataset.h` | Coberto | `Trabalho/edge/platformio_snapshot/scripts/export_real_dataset.py`, `export_balanced_dataset.py`, `create_cru_dataset.py` | Usado para testar dados reais/simulados no ESP32. |

## Lacunas que precisam ser fechadas

1. Resultado de quantizacao: falta consolidar metricas finais, tamanho do modelo, latencia e arquivo/script que gerou cada variante.
2. Teste do modelo no ESP32: falta salvar o log CSV e um script de analise que gere tabela final.
3. Conversao TFLite: precisa sair do notebook ou ser documentada em um script reproducivel.
4. Drift/MLOps: ainda nao implementado; proximo passo apos consolidar pipeline generico.
5. Generalidade em outro dataset: ainda pendente.

## Regra de produto/engenharia

Nenhum resultado deve entrar no texto final do TCC sem responder:

```text
Qual dado entrou?
Qual codigo gerou?
Qual profile/preprocessing foi usado?
Qual versao do modelo/threshold foi usada?
Onde esta o arquivo de resultado?
```

Isso evita resultado bonito, mas impossivel de auditar.
