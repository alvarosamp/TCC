# Detecção de Eventos Sísmicos com Autoencoders para Edge Computing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ObsPy](https://img.shields.io/badge/ObsPy-1.4+-orange.svg)](https://obspy.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-ff69b4.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Sobre o Projeto

Este repositório contém o código-fonte do meu Trabalho de Conclusão de Curso (TCC) em Ciência da Computação, que aborda a **detecção de eventos sísmicos em séries temporais não estacionárias utilizando aprendizado não supervisionado otimizado para edge computing**.

**Autores:** [Seu Nome]  
**Orientador:** [Nome do Orientador]  
**Instituição:** [Sua Universidade]  
**Ano:** 2024/2025

### 🎯 Objetivo

Desenvolver um sistema capaz de detectar eventos sísmicos (terremotos) em dados de forma de onda contínua utilizando autoencoders, com as seguintes características:

- **Não supervisionado**: não depende de dados rotulados para treinamento
- **Robusto à não-estacionariedade**: lida com a natureza variável dos sinais sísmicos
- **Otimizado para edge**: passível de execução em dispositivos com recursos limitados

### 🔬 Contexto Acadêmico

Este trabalho se insere na linha de pesquisa de **Aprendizado de Máquina para Séries Temporais** e **Sistemas Embarcados Inteligentes**, com aplicação direta em geofísica e monitoramento sísmico.

## 🗂️ Estrutura do Repositório

```
.
├── data/                           # Dados (ignorado pelo git)
│   ├── scedc-pds/                  # Dados brutos do SCEDC
│   │   ├── event_waveforms/         # Eventos catalogados (para teste)
│   │   └── continuous_waveforms/    # Dados contínuos (para treino)
│   └── processed/                   # Dados processados
│       ├── windows_40hz_60s/        # Janelas de eventos
│       └── continuous_windows/       # Janelas de dados contínuos
│
├── notebooks/                       # Jupyter notebooks para análise
│   ├── 01_exploracao_dados.ipynb
│   ├── 02_preprocessamento.ipynb
│   └── 03_autoencoder_experimentos.ipynb
│
├── src/                             # Código-fonte principal
│   ├── data/                        # Módulos de aquisição e processamento
│   │   ├── download_scedc.py        # Download do bucket S3 do SCEDC
│   │   ├── build_windows.py         # Janelamento de eventos
│   │   ├── build_continuous_windows.py # Janelamento de dados contínuos
│   │   └── label_windows.py         # Rotulagem opcional para validação
│   │
│   ├── models/                       # Arquiteturas de autoencoders
│   │   ├── autoencoder_basico.py     # Autoencoder denso (baseline)
│   │   ├── conv1d_autoencoder.py     # Autoencoder convolucional 1D
│   │   ├── lstm_autoencoder.py       # Autoencoder com LSTM
│   │   └── variational_autoencoder.py # VAE (opcional)
│   │
│   ├── detection/                     # Módulos de detecção
│   │   ├── threshold.py               # Métodos de thresholding (percentil, KDE)
│   │   └── evaluate.py                 # Métricas de avaliação
│   │
│   ├── edge/                          # Otimização para edge computing
│   │   ├── quantize.py                 # Quantização de modelos
│   │   ├── prune.py                     # Poda de pesos
│   │   └── tflite_convert.py            # Conversão para TensorFlow Lite
│   │
│   └── visualization/                   # Visualizações
│       ├── plot_windows.py
│       ├── plot_reconstructions.py
│       └── plot_latent_space.py
│
├── tests/                            # Testes unitários
├── docs/                             # Documentação adicional
├── requirements.txt                   # Dependências do projeto
├── setup.py                           # Instalação do pacote
└── README.md                          # Este arquivo
```

## 🚀 Começando

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes)
- Git
- (Opcional) Ambiente virtual (conda ou venv)

### Instalação

1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/deteccao-sismica-autoencoder.git
cd deteccao-sismica-autoencoder
```

2. Crie e ative um ambiente virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências
```bash
pip install -r requirements.txt
```

4. (Opcional) Instale o pacote em modo desenvolvimento
```bash
pip install -e .
```

## 📊 Dados

### Fonte

Os dados são obtidos do **Southern California Earthquake Data Center (SCEDC)**, disponíveis publicamente no bucket S3: `s3://scedc-pds/`

### Tipos de Dados Utilizados

| Tipo | Descrição | Uso |
|------|-----------|-----|
| **Event Waveforms** | Janelas em torno de terremotos catalogados | Teste e validação |
| **Continuous Waveforms** | Gravações contínuas de estações sísmicas | Treino (ruído de fundo) |

### Download

Para baixar os dados:

```bash
# Download de dados contínuos (para treino)
python src/data/download_scedc.py --type continuous --station PASC --year 2016 --days 7

# Download de eventos catalogados (para teste)
python src/data/download_scedc.py --type event --year 2016 --max-files 50
```

### Pré-processamento

```bash
# Processar dados contínuos para janelas deslizantes
python src/data/build_continuous_windows.py --input-dir data/scedc-pds/continuous_waveforms --output-dir data/processed/continuous_windows

# Processar eventos para janelas fixas
python src/data/build_windows.py --input-dir data/scedc-pds/event_waveforms --output-dir data/processed/windows_40hz_60s --max-files 100
```

## 🧠 Modelos

### Arquiteturas Implementadas

| Modelo | Descrição | Status |
|--------|-----------|--------|
| **Dense Autoencoder** | Baseline com camadas densas | ✅ Implementado |
| **Conv1D Autoencoder** | Autoencoder convolucional 1D | 🚧 Em desenvolvimento |
| **LSTM Autoencoder** | Autoencoder com camadas recorrentes | 📅 Planejado |
| **Variational Autoencoder** | Versão probabilística | 📅 Planejado |

### Exemplo de Uso

```python
from src.models.conv1d_autoencoder import Conv1DAutoencoder
import numpy as np

# Carregar dados (assumindo X_train com shape [n_samples, 2400])
X_train = np.load('data/processed/continuous_windows/continuous_windows.npz')['X']

# Inicializar e treinar
model = Conv1DAutoencoder(input_dim=2400, latent_dim=128)
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1)

# Reconstruir e calcular erro
X_reconstructed = model.predict(X_train)
reconstruction_error = np.mean(np.square(X_train - X_reconstructed), axis=1)
```

## 🔍 Detecção de Eventos

### Métodos de Threshold Implementados

1. **Percentil**: Define threshold no percentil `p` da distribuição de erros no treino
2. **Média + k*σ**: Threshold = μ + k*σ (onde μ e σ são média e desvio do erro no treino)
3. **Kernel Density Estimation (KDE)**: Threshold baseado na densidade do espaço latente (baseado em )

### Exemplo

```python
from src.detection.threshold import KDEThresholdDetector

# Treinar detector com erros do conjunto de treino (ruído)
detector = KDEThresholdDetector(contamination=0.05)  # espera 5% de anomalias
detector.fit(reconstruction_errors_train)

# Detectar anomalias
anomalies = detector.predict(reconstruction_errors_test)
```

## 📈 Experimentos e Resultados

### Pipeline de Avaliação

1. **Treino**: Autoencoder com janelas de ruído (dados contínuos sem eventos)
2. **Threshold**: Calculado a partir da distribuição de erros no treino
3. **Teste**: Aplicação em janelas com eventos conhecidos
4. **Métricas**: Precisão, Recall, F1-Score, Curva ROC

### Resultados Preliminares

*[A serem preenchidos conforme experimentos]*

| Modelo | Precisão | Recall | F1-Score | Tamanho (MB) | Inferência (ms) |
|--------|----------|--------|----------|---------------|------------------|
| Dense Autoencoder | - | - | - | - | - |
| Conv1D Autoencoder | - | - | - | - | - |
| LSTM Autoencoder | - | - | - | - | - |

## ⚡ Otimização para Edge Computing

### Técnicas Implementadas

| Técnica | Descrição | Redução Esperada |
|---------|-----------|------------------|
| **Quantização** | Conversão float32 → int8 | 4x menos memória |
| **Poda (Pruning)** | Remoção de conexões pouco importantes | 2-3x inferência |
| **Destilação** | Modelo pequeno imita modelo grande | 5-10x inferência |

### Exemplo de Otimização

```bash
# Quantizar modelo treinado
python src/edge/quantize.py --model models/conv1d_autoencoder.h5 --output models/conv1d_autoencoder_quantized.tflite

# Testar performance na borda (simulado)
python src/edge/benchmark.py --model models/conv1d_autoencoder_quantized.tflite --input-shape 2400
```

## 📓 Notebooks de Análise

Recomendo explorar os notebooks na seguinte ordem:

1. `01_exploracao_dados.ipynb`: Análise exploratória dos dados sísmicos
2. `02_preprocessamento.ipynb`: Visualização do pipeline de janelamento
3. `03_autoencoder_experimentos.ipynb`: Experimentos com diferentes arquiteturas
4. `04_deteccao_validacao.ipynb`: Validação dos métodos de detecção
5. `05_edge_otimizacao.ipynb`: Testes de otimização para edge

## 📚 Referências

### Artigos Científicos

1. Omojola, J., & Persaud, P. (2025). Detecting Urban Earthquakes com the San Fernando Valley Nodal Array and Machine Learning. *Seismological Research Letters*.

2. MLESmap: Machine Learning Estimator for ground-shaking maps (2024). *Communications Earth & Environment*.

3. *Estudo do NIOSH sobre detecção de microeventos sísmicos com autoencoder convolucional e KDE*. (Fonte: CDC)

4. *Tese da Unesp sobre caracterização de sismos vulcânicos com Dual Feature Autoencoder (DAF)*. (2024)

### Bases de Dados

- [SCEDC - Southern California Earthquake Data Center](https://scedc.caltech.edu/)
- [USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/)

### Bibliotecas Utilizadas

- [ObsPy](https://obspy.org/) - Processamento de dados sísmicos
- [TensorFlow/Keras](https://tensorflow.org) - Implementação dos autoencoders
- [scikit-learn](https://scikit-learn.org) - Métricas e pré-processamento
- [Matplotlib/Seaborn](https://matplotlib.org/) - Visualizações

## 👥 Contribuição

Este é um trabalho acadêmico individual, mas sugestões e discussões são bem-vindas através de issues.

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## ✉️ Contato

- **Autor:** [Seu Nome]
- **Email:** [seu.email@exemplo.com]
- **LinkedIn:** [Perfil no LinkedIn]
- **GitHub:** [@seu-usuario]

---

**Citação sugerida:**

```
@misc{seunome2025deteccao,
	author = {Seu Nome},
	title = {Detecção de Eventos Sísmicos com Autoencoders para Edge Computing},
	year = {2025},
	publisher = {GitHub},
	url = {https://github.com/seu-usuario/deteccao-sismica-autoencoder}
}
```

---

⭐ Se este projeto foi útil para você, considere dar uma estrela no GitHub!
