# 🗺️ Roadmap TCC — Detecção de Eventos Sísmicos com ML + Edge Computing
**Inatel — Engenharia de Software | Março → Setembro**

---

## 🎯 Objetivo do Projeto 5 Estrelas

Desenvolver e avaliar um pipeline completo de detecção de anomalias em séries temporais sísmicas,
comparando STA/LTA clássico com três arquiteturas de autoencoder (Dense, CNN 1D, LSTM),
com análise de viabilidade em edge computing via TFLite e quantização.

**Contribuição técnica central:**
> Tabela de trade-off desempenho × eficiência computacional para 4 métodos de detecção,
> com benchmarking real em ambiente de edge computing.

---

## 📅 Cronograma Geral

| Período | Foco |
|---|---|
| Março | Fundação — conceitos + pipeline base fechado |
| Abril | Detecção clássica — STA/LTA avaliado com métricas |
| Maio | Autoencoders — 3 arquiteturas treinadas e comparadas |
| Junho | Edge computing — benchmarking real |
| Julho | Escrita do TCC |
| Agosto | Revisão, simulação de defesa, ajustes finais |
| Setembro | Defesa |

---

## 📚 MÓDULO 1 — FUNDAÇÃO
> **Meta:** entender séries temporais com seus dados reais e fechar o pipeline do zero.

### Sessão 1 — Séries Temporais: o que muda tudo
- Estacionariedade e não-estacionariedade
- Autocorrelação: o sinal tem memória?
- Domínio do tempo vs. frequência vs. espectrograma
- **Entregável:** notebook com análise exploratória do sinal real do SCEDC

### Sessão 2 — Janelas: decisão técnica com dados reais
- Por que janelas existem (localização temporal)
- Trade-off tamanho × sobreposição × resolução
- Impacto do tamanho da janela nas métricas
- **Entregável:** notebook comparando janelas de 10s, 30s e 60s

### Sessão 3 — Pipeline fechado e reproduzível
- Corrigir bugs do pipeline.py
- Padronizar framework (TensorFlow/Keras)
- Estrutura profissional do repositório
- **Entregável:** pipeline.py final, rodando do zero com 1 comando

---

## 📚 MÓDULO 2 — DETECÇÃO CLÁSSICA
> **Meta:** entender e avaliar o STA/LTA com rigor científico.

### Sessão 4 — STA/LTA: como funciona por dentro
- Derivação matemática da razão STA/LTA
- Efeito de cada parâmetro (sta, lta, thresh, detrigger)
- Quando o STA/LTA falha e por quê
- **Entregável:** notebook com experimentos de parâmetros em dados reais

### Sessão 5 — Avaliando STA/LTA com métricas reais
- Construindo ground truth com eventos conhecidos do SCEDC
- Calculando Precisão, Recall, F1-score
- Curva ROC do STA/LTA (variando threshold)
- **Entregável:** notebook de avaliação — primeira linha da tabela comparativa

---

## 📚 MÓDULO 3 — AUTOENCODERS
> **Meta:** implementar, entender e comparar 3 arquiteturas. Saber ler todos os gráficos.

### Sessão 6a — Autoencoder Denso (baseline)
- O que o encoder realmente aprende
- Por que treinar só com ruído
- Visualizando o espaço latente (PCA/t-SNE)
- **Entregável:** Dense AE treinado, avaliado, salvo

### Sessão 6b — Autoencoder CNN 1D
- Por que convoluções capturam padrões locais em séries temporais
- Arquitetura: Conv1D + Pooling + UpSampling
- Comparação com Dense: tamanho, velocidade, F1
- **Entregável:** CNN AE treinado, avaliado, comparado com Dense

### Sessão 6c — Autoencoder LSTM
- Memória temporal explícita: como o LSTM funciona
- Arquitetura: LSTM Encoder + RepeatVector + LSTM Decoder
- Trade-off: melhor F1, maior custo computacional
- **Entregável:** LSTM AE treinado, avaliado, comparado

### Sessão 7 — Lendo os gráficos: o que cada visualização diz
- Curva de loss: overfitting, underfitting, convergência
- Histograma de erros: separabilidade ruído × evento
- Reconstrução visual: o que o modelo vê vs. o que falha
- Espaço latente: os clusters fazem sentido?
- **Entregável:** guia pessoal de interpretação de gráficos

### Sessão 8 — Threshold, ROC, F1: escolha com critério
- Três estratégias de threshold e quando usar cada uma
- Curva ROC das 3 arquiteturas no mesmo gráfico
- Como escolher o ponto operacional para sua aplicação
- **Entregável:** notebook de análise de threshold — 3 arquiteturas comparadas

---

## 📚 MÓDULO 4 — COMPARAÇÃO E EDGE COMPUTING
> **Meta:** produzir a tabela central do TCC e benchmarking real no edge.

### Sessão 9 — Notebook de Comparação Final
- STA/LTA × Dense AE × CNN AE × LSTM AE
- Mesmas métricas, mesmos dados, mesma divisão temporal
- Análise de erro qualitativa: o que cada método erra e por quê
- **Entregável:** notebook comparativo — o coração do TCC

### Sessão 10 — Edge Computing: benchmark real
- Conversão para TFLite das 3 arquiteturas
- Quantização int8: impacto no F1 vs. ganho em velocidade
- Medição de tempo de inferência por janela
- Tabela final: F1 × Tamanho × Inferência × Quantizado
- **Entregável:** notebook de benchmarking edge — tabela publicável

---

## 📚 MÓDULO 5 — TCC FECHADO
> **Meta:** transformar os notebooks em um trabalho acadêmico sólido.

### Sessão 11 — Estrutura do texto e narrativa
- Como escrever introdução que prende a banca
- Metodologia: descrevendo o pipeline como um paper
- Resultados: tabelas limpas, gráficos que contam história
- Discussão honesta das limitações

### Sessão 12 — Simulação de defesa
- 10 perguntas difíceis que a banca vai fazer
- Como responder sobre limitações sem se afundar
- O que fazer se não souber responder algo

---

## 🏆 Tabela Alvo do TCC

| Método | F1 | AUC | Tamanho | Inferência | F1 Quantizado | Inf. Quant. |
|---|---|---|---|---|---|---|
| STA/LTA | - | - | N/A | Xms | N/A | N/A |
| Dense AE | - | - | XMB | Xms | - | Xms |
| CNN 1D AE | - | - | XMB | Xms | - | Xms |
| LSTM AE | - | - | XMB | Xms | - | Xms |

*Preencher com resultados reais ao longo do projeto.*

---

## 📁 Estrutura do Repositório (Meta)

```
tcc-seismic-detection/
├── README.md                  ← Como rodar tudo do zero
├── data/
│   ├── raw/                   ← Arquivos .ms do SCEDC
│   ├── processed/             ← Janelas .npz
│   └── inventory/             ← XMLs das estações
├── src/
│   ├── pipeline.py            ← Pré-processamento
│   ├── windowing.py           ← Criação de janelas
│   └── evaluation.py         ← Métricas e plots
├── models/
│   ├── dense_ae/
│   ├── cnn_ae/
│   └── lstm_ae/
├── notebooks/
│   ├── sessao_01_series_temporais.ipynb
│   ├── sessao_02_janelas.ipynb
│   ├── sessao_03_pipeline.ipynb
│   ├── sessao_04_sta_lta.ipynb
│   ├── sessao_05_avaliacao_sta_lta.ipynb
│   ├── sessao_06a_dense_ae.ipynb
│   ├── sessao_06b_cnn_ae.ipynb
│   ├── sessao_06c_lstm_ae.ipynb
│   ├── sessao_07_graficos.ipynb
│   ├── sessao_08_threshold_roc.ipynb
│   ├── sessao_09_comparacao.ipynb
│   └── sessao_10_edge_benchmark.ipynb
└── results/
    ├── figures/
    └── tabela_comparativa.csv
```
