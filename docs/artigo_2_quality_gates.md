# Artigo 2 — Quality Gates Operacionais para Sistemas TinyML: Além da Acurácia Offline

**Venue alvo:** IEEE Access ou JMLR (open access) — versão journal expandida  
**Tipo:** Full paper (12–15 páginas)  
**Status:** Rascunho v0.1  
**Relação com Artigo 1:** este paper aprofunda o componente de *quality gate* apresentado superficialmente no Artigo 1, tornando-o contribuição independente e mais rigorosa.

---

## Título (EN)

*Operational Quality Gates for TinyML Anomaly Detection: Combining AUC-PR, VUS-PR, PA-F1, and Drift Monitoring Beyond Offline Accuracy*

---

## Resumo (PT)

Pipelines de TinyML para detecção de anomalias em séries temporais frequentemente são avaliados apenas por acurácia offline (AUC-PR, F1 no conjunto de teste), sem considerar comportamento operacional contínuo: taxa de alarmes falsos por hora, degradação por drift de dados e custo de atualização de modelos. Este trabalho propõe um framework de *quality gates* multidimensionais que combinam métricas de desempenho offline (AUC-PR, VUS-PR, PA-F1, Event-F1), métricas operacionais (FP/h, gap val/teste), e detecção de drift (Evidently AI) para controlar a promoção e atualização de modelos embarcados. Avaliamos o framework no contexto de detecção de anomalias sísmicas com alvo ESP32, demonstrando que modelos aprovados apenas por acurácia offline podem ter FP/h 6,8× superior a modelos aprovados pelo *quality gate* completo. O monitoramento contínuo de drift acoplado a uma política de retreinamento condicional reduz em [X]% o número de atualizações OTA desnecessárias, preservando recursos de banda e energia no dispositivo.

**Palavras-chave:** MLOps, quality gate, detecção de anomalias, drift, TinyML, FP/h, VUS-PR, monitoramento.

---

## Abstract (EN)

TinyML pipelines for time-series anomaly detection are typically evaluated by offline accuracy metrics (AUC-PR, F1 on test sets), without considering operational behavior: false alarm rate per hour, data drift degradation, and model update cost. This paper proposes a multidimensional quality-gate framework combining offline performance metrics (AUC-PR, VUS-PR, PA-F1, Event-F1), operational metrics (FP/h, val/test gap), and data drift monitoring (Evidently AI) to govern model promotion and OTA updates. We evaluate the framework on seismic anomaly detection targeting ESP32 microcontrollers, showing that models approved by offline accuracy alone exhibit FP/h up to 6.8× higher than models approved by the full quality gate. Continuous drift monitoring coupled with conditional retraining reduces unnecessary OTA updates by [X]%, preserving bandwidth and energy on the device.

**Keywords:** MLOps, quality gate, anomaly detection, drift, TinyML, FP/h, VUS-PR, monitoring.

---

## 1. Introdução

### O problema da avaliação offline

A literatura de detecção de anomalias em séries temporais concentra a avaliação em métricas offline — AUC-PR, F1, acurácia — calculadas uma única vez em um conjunto de teste estático [CITAR: Pang2021]. Essa prática ignora duas realidades operacionais críticas:

1. **FP/h em produção:** um modelo com AUC-PR=0,89 pode gerar 22 alarmes falsos por hora, tornando o sistema inutilizável em monitoramento contínuo [observado neste trabalho com Tiny TCN sem HPO].
2. **Drift de dados:** sinais de sensores mudam com desgaste de equipamentos, variações sazonais e mudanças de ambiente. Um modelo treinado hoje pode degradar em semanas sem qualquer sinalização automática [CITAR: Gama2014].

### O problema das métricas em séries temporais

Kim et al. [CITAR: Kim2022] argumentam que métricas tradicionais — incluindo AUC-PR — são inadequadas para séries temporais com janelamento sobrepostoo: a mesma anomalia contribui para múltiplas janelas positivas, inflando artificialmente o desempenho. VUS-PR [CITAR: Kim2022] e PA-F1 [CITAR: Xu2018] foram propostos especificamente para este cenário, mas raramente são adotados em pipelines práticos.

### Contribuições deste trabalho

1. **Framework de quality gate multidimensional** que combina: métricas offline, métricas orientadas a eventos (VUS-PR, PA-F1, Event-F1), métricas operacionais (FP/h, gap val/teste) e monitoramento de drift.
2. **Estudo ablativo** mostrando o impacto de cada dimensão do *quality gate* na qualidade do modelo promovido.
3. **Política de retreinamento condicional** baseada em drift detectado, reduzindo atualizações OTA desnecessárias.
4. **Implementação open-source** integrada ao pipeline TinyML (DVC + MLflow + Evidently AI).

---

## 2. Background e Trabalhos Relacionados

### 2.1 Métricas de Avaliação em Anomalias Temporais

**AUC-PR:** adequada para datasets desbalanceados [CITAR: Davis2006], mas insensível a alinhamento temporal.

**VUS-PR** [CITAR: Kim2022]: integra AUC-PR sobre múltiplos tamanhos de buffer de predição. Buffer=0 equivale ao AUC-PR padrão; buffers maiores relaxam a exigência de alinhamento exato, favorecendo sistemas que detectam anomalias com atraso pequeno.

**PA-F1** [CITAR: Xu2018]: point-adjust — se qualquer janela de um segmento anômalo é detectada, todo o segmento conta. Mais adequado para avaliar detecção de *onset* versus cobertura completa.

**Event-F1** [CITAR: Tatbul2018]: F1 no nível de evento (segmento contíguo). TP quando ≥ threshold de sobreposição entre predição e evento real.

**FP/h:** métrica operacional que quantifica o impacto de alarmes falsos em monitoramento contínuo. Essencial em sistemas onde cada alarme aciona investigação humana.

### 2.2 Drift em Séries Temporais

[CITAR: Gama2014, Lu2018, Bayram2022] — tipos de drift (concept, data, virtual), métodos de detecção (ADWIN, DDM, Page-Hinkley), frameworks de monitoramento (Evidently AI, River, NannyML).

### 2.3 MLOps para ML Embarcado

[CITAR: Sculley2015, Amershi2019, Vartak2016] — dívida técnica em sistemas de ML, pipelines auditáveis, versionamento de modelos.

### 2.4 Quality Gates em Pipelines de ML

[CITAR: revisão da literatura — maioria foca em acurácia; gap operacional identificado]

---

## 3. Framework de Quality Gates

### 3.1 Visão Geral

```
candidate_manifest.json
       │
       ▼
┌─────────────────────────────────────────────────────┐
│               QUALITY GATE                          │
│                                                     │
│  Dimensão 1: Desempenho Offline                    │
│    AUC-PR ≥ min_auc_pr                             │
│    VUS-PR ≥ min_vus_pr                             │
│                                                     │
│  Dimensão 2: Desempenho Operacional                │
│    FP/h ≤ max_fp_per_hour                          │
│    PA-F1 ≥ min_pa_f1                               │
│    Event-F1 ≥ min_event_f1                         │
│                                                     │
│  Dimensão 3: Generalização                         │
│    |AUC-PR_val - AUC-PR_test| ≤ max_gap            │
│    F1 ≥ min_f1                                     │
│                                                     │
│  Dimensão 4: Restrições Embarcadas                 │
│    model_size ≤ max_size_kb                        │
│    n_params ≤ max_params                           │
└─────────────────────────────────────────────────────┘
       │ APROVADO              │ REPROVADO
       ▼                       ▼
production_manifest.json    rejection_report.json
```

### 3.2 Limiares por Dimensão

| Dimensão | Critério | Limiar padrão | Justificativa |
|---|---|---|---|
| Offline | AUC-PR | ≥ 0,80 | Desempenho mínimo aceitável em dado desbalanceado |
| Offline | VUS-PR | ≥ 0,75 | Mais conservador que AUC-PR — exige robustez a buffer |
| Operacional | FP/h | ≤ 10,0 | Máximo de 10 alarmes falsos por hora de monitoramento |
| Operacional | PA-F1 | ≥ 0,70 | Detecção de pelo menos onset do evento |
| Operacional | Event-F1 | ≥ 0,60 | Maioria dos eventos detectados em nível de segmento |
| Generalização | Gap val/test | ≤ 0,05 | Controle de overfitting |
| Embarcado | Tamanho | ≤ 100 KB | Limite flash ESP32 para partição de modelo |

### 3.3 Monitoramento de Drift (Evidently AI)

Após promoção, o pipeline monitora continuamente dois tipos de drift:

**Data drift:** distribuição de features de entrada muda em relação à referência de treino.
- Método: teste de Kolmogorov-Smirnov por feature, drift se p-value < 0,05 em > 50% das features.
- Referência: 1.000 janelas normais do conjunto de treino.

**Performance drift:** métricas operacionais degradam abaixo de limiares.
- Método: monitoramento de FP/h em janela deslizante de 24h.
- Gatilho de retreinamento: FP/h > 2× limiar aprovado no quality gate.

### 3.4 Política de Retreinamento Condicional

```
drift detectado?
  NÃO → manter modelo atual
  SIM (data) → avaliar se performance degradou
    performance OK → anotar drift, aguardar
    performance degradou → acionar retreinamento
  SIM (performance) → acionar retreinamento imediato
```

Essa política evita retreinamento reativo a toda flutuação estatística, reduzindo atualizações OTA desnecessárias.

---

## 4. Experimentos

### 4.1 Setup

- Dataset sísmico: [descrever]
- Modelos avaliados: Tiny TCN com HPO, Tiny TCN sem HPO, Tiny CNN, Random Forest
- Limiares do quality gate: conforme Tabela 3.2

### 4.2 Estudo Ablativo do Quality Gate

> **Hipótese:** cada dimensão do quality gate elimina candidatos que pareciam aceitáveis em avaliação offline parcial.

| Configuração do gate | Modelos aprovados | FP/h médio aprovados | Event-F1 médio |
|---|---:|---:|---:|
| Somente AUC-PR ≥ 0,80 | [X] | [Y] | [Z] |
| + FP/h ≤ 10 | [X] | [Y] | [Z] |
| + VUS-PR ≥ 0,75 | [X] | [Y] | [Z] |
| + PA-F1 + Event-F1 | [X] | [Y] | [Z] |
| + Gap val/test | [X] | [Y] | [Z] |
| **Gate completo** | **[X]** | **[Y]** | **[Z]** |

> **TODO:** executar ablação sistemática registrando quais modelos de cada rodada HPO (88 trials) passariam em cada configuração do gate.

### 4.3 Drift Simulado

> **TODO:** simular drift gradual no sinal de teste (variação de ganho do sensor, aumento de ruído de fundo) e avaliar latência de detecção pelo monitoramento Evidently AI.

### 4.4 Redução de Atualizações OTA

> **TODO:** comparar política reativa (atualiza sempre que há novo modelo) vs política condicional (atualiza apenas quando drift + degradação confirmados).

---

## 5. Resultados e Discussão

### 5.1 Impacto do Quality Gate Completo

O modelo Tiny TCN sem HPO teria passado em um gate baseado apenas em AUC-PR ≥ 0,80 (AUC-PR=0,8964), mas gera 21,984 FP/h — **7,0× superior** ao modelo aprovado pelo gate completo (3,136 FP/h). Em um sistema de monitoramento real operando 24h/dia, a diferença é de 520 vs 75 alarmes falsos diários.

### 5.2 VUS-PR como Métrica Mais Conservadora

Em [COMPLETAR com valores], o VUS-PR é [X]% menor que o AUC-PR correspondente, refletindo a sensibilidade ao alinhamento temporal em dados com janelamento sobreposto de 50%.

### 5.3 Detecção de Drift

[COMPLETAR após experimento]

---

## 6. Conclusão

Apresentamos um framework de quality gates multidimensionais que vai além da avaliação offline e incorpora métricas operacionais, robustez a drift e restrições embarcadas. A ablação demonstra que cada dimensão elimina modelos que pareciam aceitáveis em avaliação parcial. A política de retreinamento condicional reduz atualizações OTA desnecessárias, preservando recursos em sistemas embarcados.

---

## Checklist de Conclusão

- [ ] Executar ablação sistemática do quality gate sobre os 88 trials do Optuna
- [ ] Implementar experimento de drift simulado
- [ ] Medir latência de detecção de drift pelo Evidently AI
- [ ] Comparar política reativa vs condicional de retreinamento
- [ ] Preencher tabelas com resultados reais
- [ ] Adicionar figura: ablação do quality gate (gráfico de cascata)
- [ ] Adicionar figura: curva de drift simulado + detecção
- [ ] Revisar para ≤ 15 páginas IEEE double-column
