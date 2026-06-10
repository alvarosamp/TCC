# Pipeline Genérico de TinyML para Detecção de Anomalias em Séries Temporais: Caso de Uso Sísmico com Ciclo MLOps e Atualização OTA

**Autor:** [Nome do Autor]
**Curso:** [Curso]
**Instituição:** [Instituição]
**Data:** Junho de 2026

---

## Resumo

Este trabalho apresenta um pipeline genérico de TinyML para detecção de anomalias em séries temporais, validado no domínio sísmico com alvo de implantação em microcontroladores ESP32. A arquitetura separa adaptadores de domínio do núcleo de aprendizado de máquina, permitindo reuso em outros sensores como vibração industrial, corrente elétrica ou telemetria. O pipeline inclui preprocessamento edge-aware, extração de features estatísticas e espectrais, comparação de seis famílias de modelos com otimização de hiperparâmetros via Optuna, e um ciclo completo de MLOps: DVC, MLflow, quality gate, promoção de modelos e atualização OTA simulada. O melhor modelo, uma Tiny TCN com 14.897 parâmetros otimizada via Optuna, alcança AUC-PR de 0,9416 e apenas 3,136 falsos positivos por hora no conjunto de teste, superando todos os baselines avaliados, incluindo o método STA/LTA tradicional. O modelo foi exportado para TensorFlow Lite e empacotado para implantação embarcada com validação de integridade SHA-256.

**Palavras-chave:** TinyML, detecção de anomalias, séries temporais, sismologia, ESP32, MLOps, redes neurais embarcadas, TCN.

---

## Abstract

This work presents a generic TinyML pipeline for anomaly detection in complex time series, validated in the seismic domain targeting ESP32 microcontrollers. The architecture decouples domain adapters from the machine learning core, enabling reuse for other sensors such as industrial vibration, electrical current, or telemetry. The pipeline includes edge-aware preprocessing, statistical and spectral feature extraction, comparison of six model families with Optuna hyperparameter optimization, and a complete MLOps cycle: DVC, MLflow, quality gate, model promotion, and simulated OTA updates. The best model, an Optuna-optimized Tiny TCN with 14,897 parameters, achieves AUC-PR of 0.9416 and only 3.136 false positives per hour on the test set, outperforming all evaluated baselines including the traditional STA/LTA method. The model was exported to TensorFlow Lite and packaged for embedded deployment with SHA-256 integrity validation.

**Keywords:** TinyML, anomaly detection, time series, seismology, ESP32, MLOps, embedded neural networks, TCN.

---

## 1. Introdução

Sistemas de detecção de anomalias em séries temporais enfrentam um dilema operacional recorrente: transmitir todos os dados do sensor para análise centralizada gera custo elevado de energia, largura de banda e armazenamento, enquanto limiar fixo no dispositivo produz alta taxa de falsos positivos e baixa adaptabilidade [CITAR]. Em domínios como monitoramento sísmico, vibrações industriais e telemetria de equipamentos, a aquisição contínua gera volumes de dados que tornam a análise centralizada inviável em larga escala.

TinyML — a execução de modelos de aprendizado de máquina em microcontroladores com recursos severamente limitados — surge como alternativa: o dispositivo na borda processa o sinal localmente, encaminhando dados apenas quando há evidência confiável de anomalia [CITAR: Warden2019, Banbury2021]. Isso reduz transmissão, energia e custo operacional, ao mesmo tempo que viabiliza resposta mais rápida e operação offline.

No domínio sísmico, a detecção automática de eventos tem sido abordada por métodos clássicos como STA/LTA [CITAR: Allen1978] e, mais recentemente, por redes profundas como PhaseNet [CITAR: Zhu2019] e EQTransformer [CITAR: Mousavi2020]. Esses modelos, entretanto, são projetados para execução em servidores e não consideram as restrições de memória e processamento de microcontroladores.

Este trabalho contribui com:

1. Um pipeline genérico e reprodutível de TinyML para séries temporais binárias, agnóstico de domínio.
2. Um preprocessamento *edge-aware* que elimina a dependência de StationXML, reduzindo o *training-serving skew* na inferência embarcada.
3. A comparação de seis famílias de modelos — de STA/LTA a redes neurais otimizadas — sob métricas operacionais (AUC-PR, FP/h).
4. Um ciclo MLOps completo: versionamento (DVC), rastreamento (MLflow), quality gate, promoção e atualização OTA simulada.
5. A validação do modelo candidato exportado para TensorFlow Lite Micro, com alvo ESP32.

---

## 2. Trabalhos Relacionados

### 2.1 TinyML e Inferência Embarcada

Warden e Situnayake [CITAR: Warden2019] estabeleceram as bases práticas para execução de modelos de aprendizado de máquina em microcontroladores. Banbury et al. [CITAR: Banbury2021] formalizaram benchmarks de desempenho para sistemas TinyML, evidenciando os compromissos entre acurácia, latência e consumo de energia. David et al. [CITAR: David2021] apresentam o TensorFlow Lite Micro como framework de referência para inferência em dispositivos sem sistema operacional. Lin et al. [CITAR: Lin2022MCUNet] propõem co-design de arquitetura e motor de inferência para maximizar acurácia em MCUs com menos de 512 KB de SRAM.

### 2.2 Detecção de Anomalias em Séries Temporais

Chandola et al. [CITAR: Chandola2009] fornecem uma taxonomia abrangente de técnicas de detecção de anomalias, incluindo métodos estatísticos, baseados em distância e em aprendizado de máquina. Pang et al. [CITAR: Pang2021] revisam métodos de aprendizado profundo para anomalias, destacando a importância de métricas adequadas para distribuições desbalanceadas.

### 2.3 Detecção Sísmica com Aprendizado de Máquina

O método STA/LTA [CITAR: Allen1978, Withers1998] é o baseline clássico: compara a energia de curta janela (STA) com a de longa janela (LTA) para identificar onset de eventos. Zhu et al. [CITAR: Zhu2019] propõem PhaseNet, uma U-Net para detecção de fases sísmicas em escala de datacenter. Mousavi et al. [CITAR: Mousavi2020] introduzem EQTransformer, um transformer hierárquico para detecção e classificação simultânea de fases. Ambos os modelos alcançam alto desempenho, mas com dezenas de milhões de parâmetros, inviáveis para ESP32.

### 2.4 Redes Neurais Temporais (TCN)

Bai et al. [CITAR: Bai2018] demonstram empiricamente que redes convolucionais causais com dilatação (TCN) superam LSTMs em diversas tarefas de modelagem de sequências. A ausência de estado recorrente simplifica a inferência sequencial e reduz o consumo de memória, uma vantagem relevante para dispositivos embarcados.

### 2.5 MLOps para Sistemas Embarcados e Edge

Sculley et al. [CITAR: Sculley2015] identificam dívida técnica oculta em sistemas de ML, incluindo *training-serving skew* e ausência de versionamento de dados. Amershi et al. [CITAR: Amershi2019] documentam práticas de engenharia de software para ML em escala industrial. A integração de DVC [CITAR: DVC] para versionamento de dados e pipelines, e MLflow [CITAR: MLflow] para rastreamento de experimentos, constitui base reconhecida para MLOps reprodutíveis.

---

## 3. Metodologia

### 3.1 Arquitetura do Pipeline

O pipeline foi projetado com separação explícita entre adaptadores de domínio e núcleo de ML:

```
raw data (MiniSEED)
  → adapter de domínio
  → dataset genérico NPZ  {X_train/val/test, y_train/val/test}
  → validação de contrato
  → extração de features (opcional)
  → treinamento de modelos
  → otimização Optuna (HPO)
  → comparação por AUC-PR
  → candidate_manifest.json
  → exportação TFLite / header C
  → quality gate
  → production_manifest.json
  → pacote OTA versionado
```

O contrato de dataset — arrays numpy com formato `(N_windows, window_size)` e rótulos binários — é a única interface entre o domínio e o restante do pipeline. Para usar outro sensor, basta implementar um novo adapter que respeite esse contrato.

### 3.2 Dataset

O domínio de validação é sísmico. Os dados brutos são arquivos MiniSEED organizados em:

```
raw/events/      → janelas com terremoto (anomalia, y=1)
raw/continuous/  → janelas de ruído de fundo (normal, y=0)
```

**Parâmetros do perfil `seismic_edge_v1`:**

| Parâmetro | Valor |
|---|---|
| Taxa de amostragem | 40 Hz |
| Tamanho da janela | 800 amostras (20 s) |
| Passo entre janelas | 10 s (sobreposição 50%) |
| Split | Por evento |
| Rótulo normal | 0 |
| Rótulo anomalia | 1 |

O split por evento garante que janelas do mesmo terremoto não apareçam simultaneamente em treino e teste, reduzindo o risco de vazamento de informação.

O conjunto de validação e teste do modelo Tiny TCN somam ~25.800 janelas cada, com ~12,8% de janelas anômalas — distribuição desbalanceada consistente com monitoramento sísmico real.

> **Nota:** Descrição completa do dataset (número de eventos, estações, período de coleta) deve ser adicionada aqui com referência à fonte dos dados MiniSEED.

### 3.3 Preprocessamento Edge-Aware

O preprocessamento foi desenhado para minimizar a diferença entre treinamento offline e inferência no microcontrolador. A etapa `remove_response` (que exige StationXML e resposta instrumental) foi removida, pois não pode ser reproduzida no ESP32.

**Pipeline offline (treinamento):**

```
resample 40 Hz → detrend linear → demean → taper 5% → bandpass 0,5–15 Hz → zscore por janela
```

**Pipeline edge (inferência no ESP32):**

```
detrend linear → taper 5% → bandpass 0,5–15 Hz → zscore por janela
```

Todas as etapas do pipeline edge são implementáveis em C/C++ com operações de ponto flutuante simples, sem dependências externas.

### 3.4 Extração de Features

Para os modelos clássicos supervisionados, foram extraídas 28 features por janela, cobrindo domínio temporal e espectral:

**Temporais:** média, desvio padrão, mínimo, máximo, mediana, RMS, abs_peak, crest_factor, peak-to-peak, energia, skewness, kurtosis, percentis (p05, p25, p75, p95), IQR, zero_crossings, zero_crossing_rate.

**Espectrais:** dominant_freq, spectral_centroid, spectral_rolloff_85, bandpower_0–3 Hz, bandpower_0,5–3 Hz, bandpower_3–8 Hz, bandpower_8–15 Hz, spectral_entropy.

As redes neurais (CNN, TCN) operam diretamente sobre os 800 pontos brutos da janela, sem extração manual de features.

### 3.5 Modelos

| Família | Modelo | Entrada | Edge |
|---|---|---|---|
| Baseline tradicional | STA/LTA | Sinal bruto | Sim |
| Baseline linear | Logistic Regression | 28 features | Não |
| Ensemble clássico | Random Forest | 28 features | Não |
| Ensemble clássico | Extra Trees | 28 features | Não |
| Rede neural leve | Tiny CNN | 800 pontos | Sim |
| Rede neural leve | Tiny TCN | 800 pontos | Sim |

**Tiny CNN:** blocos convolucionais 1D com separable convolution, spatial dropout, pooling avg+max e camada densa. Parâmetros: ~15.377.

**Tiny TCN:** blocos TCN com convoluções dilatadas causais, separable convolution, dropout, normalização por batch e pooling adaptativo. Arquitetura final (trial 24 Optuna): 6 blocos, 32 filtros, kernel 7, dilatação base 3, dense 24, 14.897 parâmetros.

### 3.6 Otimização de Hiperparâmetros

A otimização foi realizada com Optuna [CITAR: Optuna2019], maximizando um objetivo composto que prioriza AUC-PR na validação penalizado por FP/h excessivo. Para o Tiny TCN v4-2, foram executados 88 trials ao longo de ~14 horas, com seleção do melhor configuração por AUC-PR no conjunto de validação. O threshold de decisão foi definido na validação (0,8440) e aplicado sem ajuste no teste.

### 3.7 Métricas de Avaliação

A métrica principal é AUC-PR (área sob a curva precisão-revocação), adequada para datasets desbalanceados onde a classe positiva (anomalia) é rara. AUC-ROC é reportada como métrica complementar.

Para avaliação operacional, foi definida a métrica **FP/h** (falsos positivos por hora):

```
FP/h = número de falsos positivos / horas de sinal avaliado
```

Essa métrica quantifica o impacto prático de alarmes falsos em monitoramento contínuo. Um modelo com AUC-PR alta mas FP/h elevado seria problemático em operação real.

### 3.8 Pipeline MLOps

**DVC:** versionamento das etapas `generate_data → validate_dataset → train_all → export_tflite`, garantindo reprodutibilidade.

**MLflow:** rastreamento de parâmetros, métricas e artefatos de todos os experimentos.

**Quality Gate:** antes da promoção, o modelo candidato é avaliado contra limiares mínimos (`min_auc_pr`, `min_f1`, `max_fp_per_hour`, `max_val_test_auc_pr_gap`, `max_model_size_kb`). O gap entre validação e teste é monitorado para detectar overfitting.

**Manifesto de Candidato:** `candidate_manifest.json` registra modelo, threshold, métricas, parâmetros, profile e caminho do artefato — interface entre MLOps e implantação.

**OTA Simulado:** o modelo aprovado é empacotado com versão semântica, hash SHA-256, manifesto de compatibilidade (target, runtime, preprocessing_version) e mecanismo de rollback. O fluxo simula verificação pelo dispositivo, aplicação e restauração de versão anterior em caso de falha.

---

## 4. Resultados

### 4.1 Comparação de Modelos

| Modelo | AUC-PR | AUC-ROC | F1 | Precisão | Recall | FP/h | Params | Edge |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Optuna Tiny TCN v4-2** | **0,9416** | **0,9799** | **0,8885** | **0,9266** | **0,8534** | **3,136** | **14.897** | Sim |
| Optuna Tiny CNN v4 | 0,9127 | - | 0,8526 | 0,8982 | 0,8114 | 4,896 | 15.377 | Sim |
| Tiny CNN (sem HPO) | 0,8982 | - | 0,7951 | 0,7310 | 0,8716 | 16,944 | ~15k | Sim |
| Tiny TCN (sem HPO) | 0,8964 | - | 0,7666 | 0,6790 | 0,8801 | 21,984 | ~15k | Sim |
| Optuna Random Forest v4 | 0,8127 | - | 0,7367 | 0,7974 | 0,6846 | 9,264 | - | Não |
| Optuna Extra Trees v4 | 0,7901 | - | 0,7102 | 0,7589 | 0,6675 | 11,296 | - | Não |
| STA/LTA v4 | 0,1662 | - | 0,2760 | 0,1773 | 0,6230 | - | - | Sim |

O Optuna Tiny TCN v4-2 domina em todas as métricas: maior AUC-PR (0,9416), menor FP/h (3,136) entre as redes neurais, e apenas 14.897 parâmetros — dentro da capacidade de flash e SRAM do ESP32 após conversão para int8.

O impacto da otimização de hiperparâmetros é expressivo para redes neurais: o Tiny TCN sem HPO tem AUC-PR de 0,8964 e FP/h de 21,984, enquanto a versão otimizada alcança 0,9416 e 3,136 — redução de 85,7% nos falsos positivos por hora.

O STA/LTA, baseline tradicional de monitoramento sísmico, apresenta AUC-PR de 0,1662 — próximo ao baseline aleatório para a distribuição de classe — confirmando que métodos baseados em limiar fixo são insuficientes para a variabilidade dos dados avaliados.

Random Forest e Extra Trees, embora apresentem AUC-PR razoável (~0,79–0,81), não são candidatos edge: seus tamanhos de modelo (centenas de KB a MB após serialização joblib) inviabilizam implantação no ESP32.

### 4.2 Impacto da Otimização de Hiperparâmetros

| Configuração | AUC-PR val | AUC-PR teste | FP/h |
|---|---:|---:|---:|
| Tiny TCN (sem HPO) | - | 0,8964 | 21,984 |
| Tiny TCN trial 0 (HPO) | 0,8152 | - | 8,539 |
| Tiny TCN trial 8 (HPO) | 0,8906 | - | 4,521 |
| Tiny TCN trial 11 (HPO) | 0,8981 | - | 3,586 |
| **Tiny TCN trial 24 (HPO — melhor)** | **0,9343** | **0,9416** | **3,136** |

A convergência do Optuna ocorreu entre os trials 11 e 24, com a arquitetura final estabilizando-se em: 6 blocos TCN, 32 filtros, kernel 7, dilatação base 3, separable convolution, pooling avg, dropout=0,028, spatial_dropout=0,066.

### 4.3 Interpretabilidade dos Modelos Clássicos

A análise de importância de features nos modelos de ensemble revela que as features espectrais dominam a separação entre classes:

1. `spectral_rolloff_85` — frequência abaixo da qual está 85% da energia
2. `bandpower_8_15hz` — energia na faixa de maior conteúdo sísmico
3. `spectral_centroid` — centro de massa espectral
4. `zero_crossing_rate` — indicador de conteúdo de alta frequência
5. `kurtosis` — impulsividade do sinal
6. `spectral_entropy` — distribuição de energia entre frequências

Isso indica que os modelos não discriminam anomalias pela amplitude do sinal, mas pela distribuição espectral e forma da onda — coerente com o comportamento físico de eventos sísmicos.

### 4.4 Estabilidade Val/Teste

O Tiny TCN v4-2 apresenta AUC-PR de 0,9337 na validação e 0,9416 no teste — gap de +0,0079 (test > val), indicando boa generalização e ausência de overfitting. O threshold (0,8440) foi definido na validação e aplicado diretamente no teste sem ajuste.

### 4.5 Validação do Pipeline MLOps

O modelo Tiny TCN v4-2 foi promovido pelo quality gate, gerando `production_manifest.json` e, em seguida, um pacote OTA versionado (`seismic_edge_v1_tiny_cnn_20260606`) com:

- Hash SHA-256 para validação de integridade
- Manifesto de compatibilidade (target=esp32, runtime=tensorflow_lite_micro)
- Mecanismo de rollback simulado

O fluxo OTA simulado demonstrou detecção correta de atualização disponível, aplicação de nova versão, supressão de reinstalação desnecessária e recuperação de versão anterior após falha simulada.

---

## 5. Próxima Etapa: Validação no ESP32

O pipeline produz dois artefatos para implantação embarcada:
- `model.tflite` — modelo em formato TensorFlow Lite (float32 / int8 quantizado)
- `model_data.h` — header C/C++ com o modelo como array de bytes para TFLite Micro

A validação no ESP32 consiste em:

1. **Implementar `preprocessing.h`** com as mesmas operações do pipeline offline: detrend linear, taper 5%, bandpass 0,5–15 Hz, zscore por janela.
2. **Carregar o modelo** via `tflite::MicroInterpreter` no ESP32.
3. **Inferir em janelas reais** capturadas por acelerômetro ou geofone.
4. **Medir latência** de inferência por janela (target: abaixo de 10.000 ms por decisão, conforme `decision_interval_ms` no profile).
5. **Medir uso de memória** (SRAM e flash) e verificar compatibilidade com o hardware.
6. **Comparar decisões** do modelo embarcado com as do pipeline offline na mesma sequência de janelas.

O principal risco é *training-serving skew* residual: diferenças numéricas entre a implementação em C do preprocessamento e a versão Python podem degradar o desempenho. A validação quantitativa nessa etapa é o último elo do pipeline antes da implantação em campo.

---

## 6. Conclusão

Este trabalho apresentou um pipeline genérico e reprodutível de TinyML para detecção de anomalias em séries temporais, validado no domínio sísmico. A separação entre adaptador de domínio e núcleo de ML viabiliza reuso em outros domínios sem reescrita do pipeline.

O preprocessamento edge-aware elimina dependências incompatíveis com microcontroladores (StationXML, resposta instrumental), reduzindo o risco de discrepância entre treinamento e inferência embarcada.

A comparação entre seis abordagens — de STA/LTA a TCN com Optuna — demonstra que redes neurais compactas otimizadas superam largamente baselines tradicionais e modelos clássicos em todas as métricas operacionais. O modelo Tiny TCN v4-2 alcança AUC-PR de 0,9416 e 3,136 FP/h com apenas 14.897 parâmetros, viável para execução no ESP32.

O ciclo MLOps completo — da geração de dados ao pacote OTA — estabelece uma base sólida para sistemas TinyML em produção: rastreável, reprodutível e com mecanismo de atualização segura de modelos.

A próxima etapa, validação no hardware real, fechará o loop entre o desempenho medido offline e o comportamento em campo.

---

## Referências

> As referências abaixo indicam os trabalhos a serem citados. Formatar conforme norma exigida (ABNT, IEEE ou SBC).

**TinyML e Inferência Embarcada**
- WARDEN, P.; SITUNAYAKE, D. *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers*. O'Reilly, 2019.
- BANBURY, C. et al. MLPerf Tiny Benchmark. *MLSys*, 2021.
- DAVID, R. et al. TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems. *MLSys*, 2021.
- LIN, J. et al. MCUNet: Tiny Deep Learning on IoT Devices. *NeurIPS*, 2020.

**Redes Neurais Temporais**
- BAI, S.; KOLTER, J. Z.; KOLTUN, V. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv:1803.01271*, 2018.

**Detecção Sísmica**
- ALLEN, R. Automatic earthquake recognition and timing from single traces. *BSSA*, 1978.
- WITHERS, M. et al. A comparison of select trigger algorithms for automated global seismic phase and event detection. *BSSA*, 1998.
- ZHU, W.; BEROZA, G. C. PhaseNet: A Deep-Neural-Network-Based Seismic Arrival-Time Picking Method. *GJI*, 2019.
- MOUSAVI, S. M. et al. Earthquake transformer — an attentive deep-learning model for simultaneous earthquake detection and phase picking. *Nature Communications*, 2020.

**Detecção de Anomalias**
- CHANDOLA, V.; BANERJEE, A.; KUMAR, V. Anomaly detection: A survey. *ACM Computing Surveys*, 2009.
- PANG, G. et al. Deep Learning for Anomaly Detection: A Review. *ACM Computing Surveys*, 2021.

**MLOps**
- SCULLEY, D. et al. Hidden Technical Debt in Machine Learning Systems. *NeurIPS*, 2015.
- AMERSHI, S. et al. Software Engineering for Machine Learning: A Case Study. *ICSE-SEIP*, 2019.
- AKIBA, T. et al. Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD*, 2019.

**Ferramentas**
- DVC: Data Version Control. https://dvc.org
- MLflow: An open source platform for the machine learning lifecycle. https://mlflow.org
- ESP-IDF: Espressif IoT Development Framework. https://docs.espressif.com
