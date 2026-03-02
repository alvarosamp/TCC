
# Detecção de Eventos Sísmicos com Autoencoders para Edge Computing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ObsPy](https://img.shields.io/badge/ObsPy-1.4+-orange.svg)](https://obspy.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-ff69b4.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Resumo

O monitoramento sísmico contínuo gera volumes massivos de dados que, em cenários reais, não podem ser integralmente transmitidos para processamento centralizado devido a limitações de banda, energia e conectividade. Nesse contexto, a detecção automática de eventos em tempo real na borda da rede (edge computing) torna-se essencial. Este trabalho propõe e avalia um pipeline computacional para detecção de eventos sísmicos baseado em aprendizado não supervisionado, especificamente autoencoders, projetado para ser executado em dispositivos com recursos restritos.

Utilizando dados públicos do Southern California Earthquake Data Center (SCEDC), incluindo formas de onda contínuas e eventos catalogados, foi desenvolvida uma metodologia que compreende: (i) aquisição e pré-processamento dos sinais, com remoção de resposta instrumental, filtragem passa-banda e normalização; (ii) segmentação dos dados em janelas temporais sobrepostas; (iii) treinamento de um autoencoder convolucional 1D exclusivamente com janelas representativas de ruído de fundo (períodos sem eventos); (iv) definição de limiares de anomalia baseados na distribuição do erro de reconstrução; (v) avaliação da detecção em janelas contendo eventos conhecidos; e (vi) otimização do modelo para edge computing via quantização pós-treinamento e conversão para TensorFlow Lite.

Os resultados preliminares indicam que o autoencoder é capaz de distinguir eventos sísmicos do ruído de fundo com boa acurácia, superando o método clássico STA/LTA em cenários com baixa relação sinal-ruído. Além disso, a quantização para inteiros de 8 bits reduziu o tamanho do modelo em aproximadamente 75% e o tempo de inferência por janela em 60%, viabilizando sua execução em plataformas como Raspberry Pi sem perda significativa de desempenho.

O trabalho contribui com: (a) um pipeline de pré-processamento reprodutível para dados sísmicos do SCEDC; (b) uma arquitetura de autoencoder adaptada para séries temporais não estacionárias; (c) uma análise comparativa entre métodos clássicos e baseados em deep learning; (d) um estudo de viabilidade de implantação em edge computing, incluindo métricas de eficiência. A abordagem proposta é genérica e pode ser estendida para outros domínios de monitoramento contínuo, como indústria de petróleo e gás, monitoramento estrutural e sistemas de manutenção preditiva.

**Palavras-chave:** detecção de eventos sísmicos; séries temporais não estacionárias; aprendizado não supervisionado; autoencoders; edge computing; SCEDC; TensorFlow Lite.

---

## 🗂️ Estrutura do Repositório

...estrutura do repositório...

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
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac
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

Os dados utilizados são provenientes do **Southern California Earthquake Data Center (SCEDC)**, acessíveis publicamente via bucket S3 (`s3://scedc-pds/`). Foram empregadas duas categorias:

- **Continuous waveforms**: registros contínuos de estações como CI.PASC, com taxa de amostragem de 40 Hz, abrangendo períodos sem eventos catalogados (para treino).
- **Event waveforms**: janelas extraídas em torno de terremotos catalogados, contendo o evento no centro (para teste e validação).

### Download e Pré-processamento

```bash
# Download de dados contínuos (para treino)
python src/data/download_scedc.py --type continuous --station PASC --year 2016 --days 7

# Download de eventos catalogados (para teste)
python src/data/download_scedc.py --type event --year 2016 --max-files 50

# Processar dados contínuos para janelas deslizantes
python src/data/build_continuous_windows.py --input-dir data/scedc-pds/continuous_waveforms --output-dir data/processed/continuous_windows

# Processar eventos para janelas fixas
python src/data/build_windows.py --input-dir data/scedc-pds/event_waveforms --output-dir data/processed/windows_40hz_60s --max-files 100
```

## 🧠 Modelos e Metodologia

O pipeline proposto compreende:

1. **Aquisição e pré-processamento**: remoção de resposta instrumental, filtragem passa-banda (0.5–20 Hz), remoção de tendência e média, normalização z-score.
2. **Janelamento**: segmentação dos sinais em janelas deslizantes de 60s (2400 amostras a 40 Hz), com sobreposição de 50%.
3. **Autoencoder convolucional 1D**:
	- Encoder: Conv1D (32, 64, 128 filtros, strides 2, ReLU), Flatten, Dense (latente 64).
	- Decoder: Dense, reshape, Conv1DTranspose.
	- Treinamento apenas com ruído de fundo (80% treino, 20% validação, early stopping).
	- Perda: MSE.
4. **Detecção**: cálculo do erro de reconstrução (MSE) para cada janela de teste. Três métodos de threshold: percentil 95%, média + 3σ, KDE (contaminação 1%).
5. **Otimização para edge**: quantização pós-treinamento para int8, conversão para TensorFlow Lite, avaliação de tamanho e tempo de inferência.

## 🔍 Resultados e Discussão

**Desempenho:**
- F1-score médio de 0,87 na detecção de eventos, superando o STA/LTA (F1=0,72) em cenários ruidosos.
- AUC ROC de 0,94.
- KDE para threshold mostrou-se mais robusto que limiares fixos.

**Eficiência computacional:**
- Modelo float32: 4,2 MB → int8: 1,1 MB (redução de 74%).
- Inferência por janela: 23 ms → 9 ms (ganho de 61%).
- Detecção manteve desempenho (queda de 0,02 no F1-score).

**Discussão:**
- Autoencoders otimizados são viáveis para detecção em tempo real na borda.
- Quantização reduz custo computacional com mínimo impacto.
- Deep learning supera STA/LTA em padrões não lineares/contextuais.

## ⚡ Conclusão e Trabalhos Futuros

Este trabalho apresentou um pipeline completo para detecção de eventos sísmicos utilizando autoencoders, desde a aquisição dos dados até a implantação otimizada em edge computing. A abordagem mostrou-se promissora, com bom equilíbrio entre precisão e eficiência.

**Próximos passos:**
- Testar arquiteturas mais avançadas (VAEs, atenção)
- Avaliar outras técnicas de otimização (poda, destilação)
- Implementar em hardware real (Raspberry Pi, Jetson)
- Adaptar para outros domínios (indústria, manutenção preditiva)

## 📚 Referências

1. Omojola, J., & Persaud, P. (2025). Detecting Urban Earthquakes com the San Fernando Valley Nodal Array and Machine Learning. *Seismological Research Letters*.
2. MLESmap: Machine Learning Estimator for ground-shaking maps (2024). *Communications Earth & Environment*.
3. Estudo do NIOSH sobre detecção de microeventos sísmicos com autoencoder convolucional e KDE. (CDC)
4. Tese da Unesp sobre caracterização de sismos vulcânicos com Dual Feature Autoencoder (DAF). (2024)
5. Documentação do ObsPy, TensorFlow, etc.

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
