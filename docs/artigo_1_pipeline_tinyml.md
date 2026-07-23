# Artigo 1 — Pipeline Genérico de TinyML para Detecção de Anomalias em Séries Temporais com Ciclo MLOps e Atualização OTA Segura

**Venue alvo:** BRACIS 2026 ou IEEE WF-IoT 2026  
**Tipo:** Full paper (8–10 páginas IEEE double-column)  
**Status:** Rascunho v0.1 — preencher experimentos reais  

---

## Título (EN)

*A Generic TinyML Pipeline for Time-Series Anomaly Detection: Edge-Aware Preprocessing, MLOps Quality Gates, and Secure OTA Updates on ESP32*

---

## Resumo (PT)

Este artigo apresenta um pipeline genérico e reprodutível de TinyML para detecção de anomalias em séries temporais complexas, validado no domínio sísmico com alvo de implantação em microcontroladores ESP32. A arquitetura separa adaptadores de domínio do núcleo de aprendizado de máquina, permitindo reuso em outros sensores — vibração industrial, corrente elétrica, telemetria — sem reescrita do pipeline. O preprocessamento *edge-aware* elimina etapas incompatíveis com microcontroladores (remoção de resposta instrumental), reduzindo o *training-serving skew* na inferência embarcada. O pipeline compara seis famílias de modelos — de STA/LTA a redes convolucionais dilatadas otimizadas via Optuna — usando métricas operacionais: AUC-PR, PA-F1, VUS-PR, F1 orientado a eventos e falsos positivos por hora (FP/h). O melhor modelo, uma Tiny TCN com 14.897 parâmetros, alcança AUC-PR de 0,9416 e 3,136 FP/h. O ciclo MLOps completo — DVC, MLflow, *quality gate*, promoção e pacote OTA com integridade SHA-256 e rollback — estabelece base auditável para sistemas TinyML em produção.

**Palavras-chave:** TinyML, detecção de anomalias, séries temporais, ESP32, MLOps, TCN, OTA, sismologia.

---

## Abstract (EN)

We present a generic, reproducible TinyML pipeline for anomaly detection in complex time series, validated in the seismic domain targeting ESP32 microcontrollers. The architecture decouples domain adapters from the machine learning core, enabling reuse for industrial vibration, electrical current, or telemetry sensors without pipeline rewriting. Edge-aware preprocessing eliminates MCU-incompatible steps (instrumental response removal), reducing training-serving skew at embedded inference. The pipeline compares six model families — from STA/LTA to Optuna-optimized dilated convolutional networks — under operational metrics: AUC-PR, PA-F1, VUS-PR, event-level F1, and false positives per hour (FP/h). The best model, a 14,897-parameter Tiny TCN, achieves AUC-PR of 0.9416 and 3.136 FP/h. A complete MLOps cycle — DVC, MLflow, quality gate, model promotion, and an OTA package with SHA-256 integrity and rollback — establishes an auditable foundation for production TinyML systems.

**Keywords:** TinyML, anomaly detection, time series, ESP32, MLOps, TCN, OTA, seismology.

---

## 1. Introdução

Sistemas embarcados de monitoramento contínuo enfrentam um dilema operacional: transmitir todos os dados do sensor para análise centralizada eleva custo de energia e banda, enquanto limiares fixos no dispositivo produzem alta taxa de alarmes falsos e baixa adaptabilidade [CITAR: Sculley2015]. TinyML — execução de modelos de ML em microcontroladores com recursos severamente limitados — surge como alternativa: o dispositivo processa janelas do sinal localmente e encaminha dados apenas quando há evidência confiável de anomalia [CITAR: Warden2019, Banbury2021].

No domínio sísmico, métodos clássicos como STA/LTA [CITAR: Allen1978] e modelos profundos como PhaseNet [CITAR: Zhu2019] e EQTransformer [CITAR: Mousavi2020] alcançam bom desempenho, mas são projetados para servidores e ignoram as restrições de MCUs. Trabalhos recentes questionam também as métricas utilizadas: Kim et al. [CITAR: Kim2022] demonstram que AUC-PR isolado é sensível ao alinhamento ponto a ponto em séries com janelamento, propondo VUS-PR como alternativa mais robusta. Xu et al. [CITAR: Xu2018] e Tatbul et al. [CITAR: Tatbul2018] formalizam PA-F1 e F1 orientado a eventos como métricas mais adequadas para avaliação operacional.

Este artigo contribui com:
1. Pipeline genérico e reprodutível de TinyML para séries temporais binárias, agnóstico de domínio.
2. Preprocessamento *edge-aware* que elimina dependências não reproduzíveis em MCUs, reduzindo *training-serving skew*.
3. Comparação de seis famílias de modelos sob métricas operacionais completas: AUC-PR, VUS-PR, PA-F1, Event-F1, FP/h.
4. Ciclo MLOps completo: DVC, MLflow, *quality gate* multidimensional, promoção auditável e pacote OTA com SHA-256 e rollback.

---

## 2. Trabalhos Relacionados

### 2.1 TinyML e Inferência Embarcada
[CITAR: Warden2019, Banbury2021, David2021, Lin2022]

### 2.2 Detecção de Anomalias em Séries Temporais
[CITAR: Chandola2009, Pang2021, Kim2022, Xu2018, Tatbul2018]

### 2.3 Detecção Sísmica com ML
[CITAR: Allen1978, Zhu2019, Mousavi2020, Bai2018]

### 2.4 MLOps para Sistemas Edge
[CITAR: Sculley2015, Amershi2019, DVC, MLflow, Optuna2019]

---

## 3. Metodologia

### 3.1 Arquitetura do Pipeline

```
raw data
  → adapter de domínio
  → dataset genérico NPZ  {X_train/val/test, y_train/val/test}
  → validação de contrato (pandera)
  → extração de features (modelos clássicos) / sinal bruto (redes)
  → treinamento comparativo (6 famílias)
  → otimização Optuna
  → candidate_manifest.json
  → quality gate multidimensional
  → production_manifest.json
  → exportação TFLite + header C
  → pacote OTA versionado (SHA-256, rollback)
```

**Contrato genérico:** arrays numpy `(N, window_size)` univariado ou `(N, window_size, n_channels)` multivariado. Qualquer domínio precisa apenas de um adapter que respeite este contrato.

### 3.2 Dataset Sísmico

> [Descrever: número de eventos, estações, período, fonte MiniSEED]

| Parâmetro | Valor |
|---|---|
| Taxa de amostragem | 40 Hz |
| Janela | 800 amostras (20 s) |
| Passo | 10 s (sobreposição 50%) |
| Split | Por evento |
| Desbalanceamento (teste) | ~12,8% anômalo |

### 3.3 Preprocessamento Edge-Aware

Pipeline offline (treino): `resample 40 Hz → detrend linear → demean → taper 5% → bandpass 0,5–15 Hz → zscore por janela`

Pipeline edge (ESP32): `detrend linear → taper 5% → bandpass 0,5–15 Hz → zscore por janela`

A etapa `remove_response` (StationXML) foi removida — não reproduzível no MCU.

### 3.4 Features (Modelos Clássicos)

28 features por janela: 20 temporais (média, std, RMS, kurtosis, zero-crossing rate, percentis, energia...) + 8 espectrais (centroide, rolloff 85%, potência por banda, entropia espectral).

### 3.5 Modelos

| Família | Modelo | Entrada | Edge |
|---|---|---|---|
| Baseline | STA/LTA | Sinal bruto | Sim |
| Linear | Logistic Regression | 28 features | Não |
| Ensemble | Random Forest | 28 features | Não |
| Ensemble | Extra Trees | 28 features | Não |
| Neural leve | Tiny CNN | 800 pts | Sim |
| Neural leve | Tiny TCN (Optuna) | 800 pts | Sim |

**Tiny TCN (melhor):** 6 blocos TCN causais dilatados, separable conv, 32 filtros, kernel 7, dilatação base 3, dense 24, 14.897 parâmetros.

### 3.6 Métricas

- **AUC-PR**: métrica primária para datasets desbalanceados.
- **VUS-PR** [CITAR: Kim2022]: média de AUC-PR sobre buffers de 0 a `max_buf` janelas — mais robusto a desalinhamentos temporais.
- **PA-F1** [CITAR: Xu2018]: F1 com point-adjust — se qualquer ponto de um segmento anômalo é detectado, todo o segmento conta como TP.
- **Event-F1** [CITAR: Tatbul2018]: F1 no nível de evento — detectar ≥10% das janelas de um evento = TP.
- **FP/h**: falsos positivos por hora de sinal avaliado.

### 3.7 Quality Gate

Antes da promoção, o candidato passa por limiares mínimos:

| Critério | Limiar |
|---|---|
| AUC-PR mínima | 0,80 |
| F1 mínimo | 0,70 |
| FP/h máximo | 10,0 |
| Gap val/test AUC-PR | ≤ 0,05 |
| Tamanho máximo | 100 KB |

### 3.8 Fluxo OTA

`candidate → quality gate → production_manifest.json → pacote OTA (modelo + manifesto + SHA-256) → publicação local → verificação dispositivo → aplicação → rollback simulado`

---

## 4. Resultados

### 4.1 Comparação de Modelos

| Modelo | AUC-PR | VUS-PR | PA-F1 | Event-F1 | FP/h | Params | Edge |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **Optuna Tiny TCN** | **0,9416** | **[MEDIR]** | **[MEDIR]** | **[MEDIR]** | **3,136** | **14.897** | Sim |
| Optuna Tiny CNN | 0,9127 | [MEDIR] | [MEDIR] | [MEDIR] | 4,896 | 19.489 | Sim |
| Tiny TCN (sem HPO) | 0,8964 | [MEDIR] | [MEDIR] | [MEDIR] | 21,984 | ~6k | Sim |
| Random Forest | 0,8127 | [MEDIR] | [MEDIR] | [MEDIR] | 9,264 | — | Não |
| Extra Trees | 0,7901 | [MEDIR] | [MEDIR] | [MEDIR] | 11,296 | — | Não |
| STA/LTA | 0,1662 | [MEDIR] | [MEDIR] | [MEDIR] | — | — | Sim |

> **TODO:** rodar `evaluate_scores` com as novas métricas nos scores já existentes (scores_val/test salvos no MLflow) para preencher VUS-PR, PA-F1 e Event-F1 sem retreinar.

### 4.2 Impacto do HPO

| Configuração | AUC-PR val | AUC-PR teste | FP/h |
|---|---:|---:|---:|
| Tiny TCN sem HPO | — | 0,8964 | 21,984 |
| Trial 0 | 0,8152 | — | 8,539 |
| Trial 11 | 0,8981 | — | 3,586 |
| **Trial 24 (melhor)** | **0,9343** | **0,9416** | **3,136** |

Redução de FP/h: **85,7%** com HPO.

### 4.3 Importância de Features (Modelos Clássicos)

Top-6: `spectral_rolloff_85`, `bandpower_8_15hz`, `spectral_centroid`, `zero_crossing_rate`, `kurtosis`, `spectral_entropy`. Anomalias são discriminadas pela distribuição espectral, não por amplitude — coerente com física sísmica.

### 4.4 Validação do Ciclo MLOps

Quality gate aprovado. Pacote OTA gerado com SHA-256 verificado. Simulação: detecção de atualização → aplicação → re-verificação (sem reinstalação) → rollback em falha simulada — todos os estados corretos.

### 4.5 Recursos para ESP32

| Artefato | Tamanho |
|---|---|
| model.tflite (float32) | [MEDIR] KB |
| model.tflite (int8) | [MEDIR] KB |
| model_data.h | [MEDIR] KB |
| RAM estimada (inferência) | [MEDIR] KB |

> **TODO:** medir após exportação TFLite com quantização int8.

---

## 5. Discussão

**Vantagem do preprocessamento edge-aware:** elimina `remove_response` do treino, alinhando offline e embedded. Risco residual: implementação C do bandpass pode divergir numericamente da versão Python — etapa de validação no hardware real é necessária.

**VUS-PR vs AUC-PR:** com janelas sobrepostas (50%), AUC-PR pode superestimar desempenho ao contar a mesma anomalia múltiplas vezes. VUS-PR com buffers 0–20 fornece estimativa mais conservadora e realista. [COMPLETAR com valores reais.]

**Escalabilidade:** o pipeline foi projetado para suportar novos domínios via adapter. A hipótese de generalização cross-domain é objeto do Artigo 3.

---

## 6. Conclusão

Apresentamos um pipeline genérico de TinyML que cobre todo o ciclo — da aquisição à atualização segura do modelo embarcado. O Tiny TCN otimizado por Optuna supera todos os baselines avaliados com apenas 14.897 parâmetros, viável para ESP32. O *quality gate* multidimensional e o fluxo OTA com SHA-256 e rollback estabelecem base confiável para implantação em campo. Trabalhos futuros incluem validação no hardware real e replicação em outros domínios.

---

## Referências

[Lista completa em artigo.md — consolidar aqui ao finalizar]

- Kim et al. (2022). Towards a Rigorous Evaluation of Time-Series Anomaly Detection. *AAAI 2022*.
- Xu et al. (2018). Unsupervised Anomaly Detection via Variational Auto-Encoder. *WWW 2018*.
- Tatbul et al. (2018). Precision and Recall for Time Series. *NeurIPS 2018*.
- [demais referências: ver artigo.md]

---

## Checklist de Conclusão

- [ ] Preencher VUS-PR, PA-F1, Event-F1 na tabela de resultados (rodar evaluate_scores nos scores existentes)
- [ ] Preencher tamanhos TFLite e estimativa de RAM
- [ ] Adicionar figura: curva PR do Tiny TCN + baselines
- [ ] Adicionar figura: convergência Optuna (trial vs AUC-PR)
- [ ] Descrever dataset sísmico com números reais
- [ ] Revisar abstract para 150 palavras (limite IEEE)
- [ ] Formatar referências em IEEE style
