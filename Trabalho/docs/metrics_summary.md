# Resumo de Metricas

Este resumo consolida os resultados registrados no notebook `notebooks/Metricas_Alvaro.ipynb`.

## Dataset

- Dataset principal: `dataset_v4_split_evento.npz`
- Split: por evento
- Janela: 800 amostras
- Taxa de amostragem: 40 Hz
- Duracao da janela: 20 s
- Baseline AUC-PR aproximado: 0.128

Distribuicao observada no notebook:

| Split | Total | Normal | Anomalo | Baseline AUC-PR |
|---|---:|---:|---:|---:|
| train | 120429 | 105000 | 15429 | 0.1281 |
| val | 25802 | 22500 | 3302 | 0.1280 |
| test | 25829 | 22500 | 3329 | 0.1289 |

## Comparacao de modelos

| Modelo | Tipo | AUC-PR | F1 | Observacao |
|---|---|---:|---:|---|
| STA/LTA | classico | 0.1642 | 0.2721 | baseline classico |
| Random Forest | ML features | 0.7641 | 0.6899 | usa features estatisticas/frequenciais |
| ExtraTrees | ML features | 0.7767 | 0.6964 | melhor ensemble classico do notebook |
| Tiny CNN classifier | DL compacto | 0.9020 | 0.8189 | candidato TinyML |
| Tiny TCN classifier | DL compacto | 0.9186 | 0.8247 | melhor modelo observado |

## Conclusoes tecnicas

1. O split por evento e importante porque reduz vazamento entre treino, validacao e teste.
2. AUC-PR e a metrica primaria, pois o dataset e desbalanceado.
3. STA/LTA serve como baseline classico, mas ficou distante dos modelos supervisionados.
4. O Tiny TCN e o melhor candidato para o firmware embarcado, pois combina desempenho alto com viabilidade TinyML.
5. Threshold, modelo e preprocessing devem ser versionados juntos.

## Relacao com TinyML

O objetivo embarcado nao e apenas classificar uma janela. A aplicacao proposta usa TinyML como filtro inteligente:

```text
sinal continuo -> janela -> preprocessing -> modelo -> score -> decisao local
```

Quando o score indicar anomalia com confianca, o dispositivo pode salvar ou transmitir a janela completa. Isso reduz armazenamento, energia, banda e custo de nuvem.
