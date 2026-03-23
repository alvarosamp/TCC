# 🛰️ Anomaly Detection in Continuous Time Series
## Edge AI + MLOps — Seismic Sensor Data

> Pipeline completo de detecção de anomalias em séries temporais com
> Deep Learning, Edge Computing e MLOps.

---

## 📌 Motivação

Redes de sensores que operam continuamente — como redes sísmicas, sistemas de
monitoramento de dutos ou equipamentos industriais — geram volumes massivos de
dados, grande parte desnecessária, com custos elevados de armazenamento e
transmissão.

Este projeto ataca esse problema na origem: **processar o sinal diretamente no
dispositivo de aquisição e transmitir apenas o que é relevante**, combinando
sensoriamento remoto, deep learning e edge computing num pipeline reproduzível
e genérico.

Os dados sísmicos do [SCEDC](https://scedc.caltech.edu/) são usados como
estudo de caso — não com foco em sismologia, mas como representante de qualquer
problema de monitoramento contínuo por sensores.

---

## 🎯 Objetivo

Desenvolver, avaliar e comparar um pipeline de detecção de anomalias em séries
temporais não-estacionárias, com foco em:

- Comparação sistemática entre métodos clássicos e deep learning
- Viabilidade de execução em dispositivos edge (TFLite + quantização int8)
- Ciclo completo de MLOps: versionamento, drift detection, retreinamento e OTA
- Generalização para múltiplos domínios industriais

---

## 🧠 Métodos Comparados

| Método | Tipo | Descrição |
|---|---|---|
| STA/LTA | Clássico | Baseline da sismologia — detector de energia |
| Dense Autoencoder | Deep Learning | Detecção por erro de reconstrução |
| CNN 1D Autoencoder | Deep Learning | Captura padrões locais no sinal temporal |
| LSTM Autoencoder | Deep Learning | Memória temporal explícita de longo prazo |

Todos avaliados nos **mesmos dados**, com as **mesmas métricas**,
respeitando a **mesma divisão temporal**.

---

## 📊 Tabela Comparativa (meta)

| Método | F1 | AUC | Tamanho | Inf. (ms) | F1 Quant. | Inf. Quant. |
|---|---|---|---|---|---|---|
| STA/LTA | — | — | N/A | — | N/A | N/A |
| Dense AE | — | — | — | — | — | — |
| CNN 1D AE | — | — | — | — | — | — |
| LSTM AE | — | — | — | — | — | — |

*Preenchida com resultados reais ao longo do projeto.*

---

## 🏗️ Pipeline

```
DADOS BRUTOS (.ms + XML)
        ↓
PRÉ-PROCESSAMENTO
  detrend → taper → remoção de resposta → filtro passa-banda
        ↓
SEGMENTAÇÃO EM JANELAS
  30s · 50% sobreposição · normalização z-score por janela
        ↓
DATASET VERSIONADO
  windows_noise.npz · windows_events.npz · dataset_info.json
        ↓
DETECÇÃO
  STA/LTA · Dense AE · CNN 1D AE · LSTM AE
        ↓
AVALIAÇÃO
  F1 · AUC-ROC · Precisão · Recall · Análise de erro
        ↓
EDGE COMPUTING
  TFLite · Quantização int8 · Benchmark de inferência
        ↓
MLOps
  Drift Detection (KS test) · Retreinamento · OTA simulado
```

---

## ⚡ Edge Computing + MLOps

### Detecção de Concept Drift
O modelo embarcado coleta estatísticas locais do erro de reconstrução.
Um teste KS (Kolmogorov-Smirnov) compara a distribuição atual com a de
referência do treinamento. Se `p < 0.05` → drift detectado → retreinamento.

### Retreinamento Automático
Quando drift é detectado, o pipeline coleta janelas recentes, retreina o
autoencoder, converte para TFLite, quantiza e valida métricas.

### Atualização OTA Simulada
O novo modelo `.tflite` é enviado ao dispositivo simulado. A substituição é
atômica — ou funciona completamente ou o modelo anterior continua rodando.

---

## 🌐 Generalização

A metodologia é validada em dois domínios:

- **SCEDC** — séries temporais sísmicas (domínio principal)
- **CWRU Bearing Dataset** — vibração de rolamentos industriais

---

## 📁 Estrutura do Repositório

```
tcc-anomaly-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                        ← Arquivos .ms do SCEDC (não versionados)
│   ├── processed/
│   │   ├── inventario_dados.csv    ← Mapa completo dos dados (Passo 1)
│   │   └── windows/
│   │       ├── windows_noise.npz   ← Janelas de ruído (treino/val/teste)
│   │       ├── windows_events.npz  ← Janelas de evento (avaliação)
│   │       └── dataset_info.json   ← Metadados e rastreabilidade
│   └── inventory/                  ← XMLs das estações (STATIONXML)
│
├── src/
│   ├── pipeline.py                 ← Pré-processamento modular
│   ├── windowing.py                ← Criação e normalização de janelas
│   └── evaluation.py              ← Métricas, ROC, visualizações
│
├── models/
│   ├── dense_ae/                   ← Modelo + config + histórico
│   ├── cnn_ae/
│   └── lstm_ae/
│
├── edge/
│   ├── dense_ae.tflite
│   ├── dense_ae_quant.tflite
│   ├── cnn_ae.tflite
│   ├── cnn_ae_quant.tflite
│   └── benchmark.py               ← Medição de inferência
│
├── mlops/
│   ├── drift_detection.py          ← KS test sobre erro de reconstrução
│   ├── retrain.py                  ← Pipeline de retreinamento automático
│   └── ota_simulation.py           ← Simulação de atualização Over-The-Air
│
├── notebooks/
│   ├── passo_01_verificar_dados.ipynb
│   ├── passo_02_pipeline_dataset.ipynb
│   ├── passo_03_sta_lta.ipynb
│   ├── passo_04_dense_ae.ipynb
│   ├── passo_05_cnn_ae.ipynb
│   ├── passo_06_lstm_ae.ipynb
│   ├── passo_07_threshold_roc.ipynb
│   ├── passo_08_edge_benchmark.ipynb
│   ├── passo_09_drift_mlops.ipynb
│   └── passo_10_comparacao_final.ipynb
│
└── results/
    ├── figures/                    ← Gráficos em alta resolução
    └── tabela_comparativa.csv      ← Tabela central do paper
```

---

## 🚀 Como Rodar

### 1. Clonar e instalar dependências

```bash
git clone https://github.com/seu-usuario/tcc-anomaly-detection.git
cd tcc-anomaly-detection
pip install -r requirements.txt
```

### 2. Configurar caminhos

Abra o notebook `passo_01_verificar_dados.ipynb` e ajuste `PASTA_RAIZ`
na Célula 1 para o caminho dos seus dados SCEDC.

### 3. Rodar os passos em ordem

Abra cada notebook no VS Code com Jupyter e execute célula por célula:

```
passo_01 → passo_02 → passo_03 → ... → passo_10
```

Cada passo gera arquivos que o próximo consome.
O dataset versionado em `data/processed/windows/` é gerado no Passo 2.

---

## 📦 Dependências Principais

```
obspy>=1.4.0
tensorflow>=2.13.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
optuna>=3.3.0
matplotlib>=3.7.0
pandas>=2.0.0
seaborn>=0.12.0
```

Instale tudo com:
```bash
pip install -r requirements.txt
```

---

## 🔬 Parâmetros do Pipeline

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Canal | BHZ | Vertical banda-larga — mais sensível a ondas P |
| Taxa | 40 Hz | Taxa nativa dos canais BH; zero reamostragem |
| Janela | 30s | Cobre chegada de ondas P e S regionais |
| Sobreposição | 50% | Garante cobertura total do sinal |
| Filtro | 0.5–15 Hz | Banda de eventos sísmicos regionais |
| Output | VEL (m/s) | Unidade física após remoção de resposta |

---
Artigo em preparação para submissão no padrão **IEEE**.
Congressos alvo: SBMO · SBrT · MOMAG

Candidato ao **32º Prêmio Jovem Cientista — CNPq 2026**
Tema: "Inteligência Artificial para o Bem Comum"
Linha: IA & Meio Ambiente — Sensoriamento Remoto e Monitoramento

---

## 🏭 Aplicações Industriais

| Setor | Aplicação |
|---|---|
| Óleo e Gás | Monitoramento de integridade de dutos e poços |
| Mineração | Detecção de eventos sísmicos induzidos |
| Estrutural | Anomalias em pontes, barragens e edifícios |
| Industrial | Manutenção preditiva de equipamentos rotativos |

---

## 👨‍💻 Autor

**[Seu Nome]**
Engenharia de Software com Ênfase em IA — Inatel
[seu-email] · [linkedin] · [github]

---

## 📝 Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.
