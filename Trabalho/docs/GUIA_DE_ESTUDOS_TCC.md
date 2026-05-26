# Guia de Estudos do TCC

Este guia organiza a teoria e a prática necessárias para dominar o projeto:

> Pipeline reutilizável de IA para detecção de anomalias em séries temporais, usando sinais sísmicos como estudo de caso.

A ideia é você conseguir explicar o projeto como engenheiro de software e também como alguém que entende o problema técnico: dados, sinais, pré-processamento, modelos, avaliação, implantação e comunicação.

---

## 1. Visão Geral do Projeto

### O que o projeto faz

O projeto recebe sinais temporais de sensores, transforma esses sinais em janelas, treina modelos para distinguir comportamento normal de comportamento anômalo e compara os métodos por desempenho e custo computacional.

No estudo de caso atual:

- o sinal é sísmico;
- a série temporal representa amplitude ao longo do tempo;
- janelas normais representam ruído ou comportamento sem evento relevante;
- janelas anômalas representam regiões associadas a eventos sísmicos catalogados;
- os modelos produzem um score de anomalia;
- um threshold transforma o score em decisão: normal ou evento.

### Frase central para a banca

> O objetivo do trabalho é avaliar uma metodologia reutilizável para detecção de anomalias em séries temporais de sensores, usando sinais sísmicos reais como estudo de caso e comparando métodos clássicos, machine learning e deep learning.

### O que o TCC não promete

O TCC não promete encontrar petróleo diretamente, nem ser um sistema operacional de alerta sísmico. Ele cria uma base técnica para problemas desse tipo.

Uma resposta madura:

> Encontrar petróleo exige dados sísmicos de reflexão, interpretação geológica e validação especializada. Este TCC atua em uma camada anterior: detecção de padrões anômalos em séries temporais de sensores. Essa metodologia pode ser adaptada futuramente para problemas geofísicos mais específicos.

---

## 2. Fundamentos de Séries Temporais

### O que é uma série temporal

Uma série temporal é uma sequência de observações ordenadas no tempo.

Exemplos:

- amplitude sísmica ao longo do tempo;
- vibração de uma máquina;
- consumo de energia;
- batimento cardíaco;
- tráfego de rede;
- temperatura;
- pressão em uma tubulação.

Formalmente:

```text
x[0], x[1], x[2], ..., x[t]
```

O ponto essencial é que a ordem importa. Em uma tabela comum, as linhas podem ser independentes. Em uma série temporal, o valor atual pode depender dos valores anteriores.

### Taxa de amostragem

A taxa de amostragem diz quantos pontos são medidos por segundo.

Exemplo:

```text
40 Hz = 40 amostras por segundo
```

Então:

```text
10 segundos = 400 amostras
20 segundos = 800 amostras
30 segundos = 1200 amostras
```

No seu projeto, os datasets novos usam janelas de 10 segundos com 400 pontos.

### Conceitos importantes

**Tendência:** movimento de longo prazo no sinal.

**Sazonalidade:** padrão que se repete periodicamente.

**Ruído:** variação aleatória ou indesejada.

**Evento:** mudança relevante localizada no tempo.

**Anomalia:** comportamento que foge do padrão esperado.

**Autocorrelação:** relação entre valores atuais e valores passados.

**Domínio do tempo:** o sinal visto como amplitude ao longo do tempo.

**Domínio da frequência:** o sinal visto pelas frequências que o compõem.

### Perguntas que você precisa saber responder

1. Qual é a taxa de amostragem dos seus dados?
2. Quantos pontos há em uma janela?
3. O que representa uma janela normal?
4. O que representa uma janela anômala?
5. O evento é detectado por amplitude, forma, frequência ou combinação disso?

---

## 3. Sinais Sísmicos e Sinais Físicos

### O que é um sinal sísmico

Um sinal sísmico mede vibrações no solo. O sensor registra uma amplitude ao longo do tempo.

Em termos de engenharia de dados, ele é uma série temporal.

### Evento versus ruído

Em sismologia:

- ruído pode vir de ambiente, instrumento, atividade humana, vento, pequenas vibrações;
- evento pode ser um terremoto ou outra ocorrência detectável no solo.

O desafio é que nem sempre evento significa apenas amplitude alta. Às vezes a forma do sinal, a frequência, o início súbito e a energia ao longo do tempo importam mais.

### Relação com petróleo e geofísica

Na exploração de petróleo, a sísmica frequentemente envolve ondas emitidas artificialmente e reflexão em camadas geológicas. O objetivo é inferir estruturas no subsolo.

Seu TCC não faz essa inferência, mas trabalha com uma competência base:

> usar IA para interpretar padrões em sinais geofísicos temporais.

---

## 4. Pré-processamento

Pré-processamento é a etapa que transforma o dado bruto em uma representação mais útil, comparável e estável.

### Por que pré-processar

Dados reais têm:

- ruído;
- escalas diferentes;
- offsets;
- sensores diferentes;
- frequências irrelevantes;
- falhas de medição;
- formatos heterogêneos.

O pré-processamento reduz parte dessa bagunça para o modelo aprender o que importa.

### Etapas comuns

#### Remoção de tendência

Remove deslocamentos lentos ou inclinações artificiais no sinal.

#### Filtragem

Mantém apenas frequências relevantes.

Exemplo:

```text
Filtro passa-faixa 0.5 Hz a 15 Hz
```

Isso remove frequências muito baixas e muito altas.

#### Normalização

Coloca os dados em escala comparável.

Tipos:

- z-score: subtrai média e divide pelo desvio padrão;
- min-max: coloca entre 0 e 1;
- max-abs: divide pelo maior valor absoluto;
- por janela: normaliza cada janela individualmente;
- por arquivo: normaliza cada arquivo;
- global: usa estatísticas do treino.

Ponto crítico:

> Normalização por janela pode remover informação de amplitude. Isso pode ser bom ou ruim, dependendo do problema.

#### Reamostragem

Padroniza diferentes taxas de amostragem.

Exemplo:

```text
100 Hz -> 40 Hz
```

#### Janelamento

Divide um sinal longo em pedaços menores.

Exemplo:

```text
0s-10s
5s-15s
10s-20s
```

Isso transforma o sinal em exemplos de treino.

### Onde isso aparece no projeto

- `notebook/Preprocessamento/`
- `artefacts/data/windows/`
- `src/tcc_seismic/data.py`

### Perguntas de banca

1. Por que você usa janelas?
2. Por que 10 segundos?
3. O que acontece se a janela for muito pequena?
4. O que acontece se a janela for muito grande?
5. Você normalizou por janela, arquivo ou globalmente?

---

## 5. Janelas, Labels e Vazamento de Dados

### O que é uma janela

Uma janela é um trecho do sinal.

Se o sinal tem 40 Hz e a janela tem 10 segundos:

```text
40 * 10 = 400 pontos
```

Então cada exemplo é um vetor:

```text
X = [x1, x2, ..., x400]
```

### O que é label

Label é a resposta esperada:

```text
0 = normal
1 = evento/anomalia
```

### O maior perigo: vazamento de dados

Vazamento acontece quando informação do teste aparece no treino.

Em séries temporais, isso é muito fácil.

Exemplo ruim:

```text
janela 0s-10s do evento A -> treino
janela 5s-15s do evento A -> teste
```

Essas janelas compartilham metade dos pontos. O modelo pode parecer bom sem generalizar de verdade.

### Splits melhores

- por evento;
- por arquivo;
- por estação;
- por tempo;
- por região.

### Regra de ouro

> O teste precisa representar dados que o modelo realmente não viu.

---

## 6. Tipos de Problemas em Séries Temporais

### Previsão

Prever valores futuros.

```text
entrada: últimos 60 minutos
saída: próximo minuto
```

### Classificação

Classificar uma janela.

```text
entrada: janela de 10s
saída: evento ou ruído
```

### Detecção de anomalias

Detectar comportamento incomum.

```text
entrada: janela
saída: normal ou anômala
```

### Segmentação

Marcar início e fim do evento no tempo.

```text
0s-12s normal
12s-18s evento
18s-30s normal
```

### Seu TCC

Seu projeto está entre:

- detecção de anomalias;
- classificação de janelas;
- detecção de eventos.

---

## 7. Features para Machine Learning Clássico

Modelos clássicos geralmente não recebem a janela bruta. Eles recebem features.

Exemplo:

```text
janela com 400 pontos
  -> média
  -> desvio padrão
  -> RMS
  -> energia
  -> frequência dominante
  -> curtose
  -> skewness
```

### Features do tempo

- média;
- desvio padrão;
- mínimo;
- máximo;
- pico a pico;
- RMS;
- energia;
- valor absoluto médio;
- crest factor;
- zero-crossing rate;
- skewness;
- curtose.

### Features de frequência

Usam a transformada de Fourier.

- frequência dominante;
- centroide espectral;
- energia em bandas de frequência.

### Onde aparece no projeto

- `src/tcc_seismic/features.py`
- `scripts/train_ml_models.py`

### Por que isso importa

Essas features testam uma hipótese:

> Talvez características estatísticas simples já consigam separar ruído e evento.

Se isso acontecer, o TCC fica mais forte, porque mostra que você comparou modelos complexos contra baselines simples.

---

## 8. STA/LTA

STA/LTA é um método clássico da sismologia.

### Ideia

Comparar energia recente com energia de fundo.

```text
STA = Short Time Average
LTA = Long Time Average
score = STA / LTA
```

Se a energia curta aumenta muito em relação ao fundo, pode haver evento.

### Parâmetros

- tamanho da STA;
- tamanho da LTA;
- threshold de disparo;
- threshold de desligamento.

### Vantagens

- simples;
- rápido;
- interpretável;
- bom baseline.

### Limitações

- sensível a ruído;
- pode falhar em eventos fracos;
- depende muito dos parâmetros;
- não aprende padrões complexos.

### Como explicar

> Usei STA/LTA como baseline clássico, pois ele representa uma abordagem tradicional de detecção sísmica baseada em aumento relativo de energia.

---

## 9. Modelos de Machine Learning

### Não supervisionados ou one-class

Treinam principalmente com dados normais.

Modelos:

- Isolation Forest;
- One-Class SVM;
- Local Outlier Factor;
- Autoencoder treinado só com ruído.

Eles aprendem:

> como é o comportamento normal.

### Supervisionados

Treinam com normal e evento.

Modelos:

- Logistic Regression;
- Random Forest;
- Extra Trees;
- SVM supervisionado;
- XGBoost;
- redes neurais classificadoras.

Eles aprendem:

> qual fronteira separa normal de evento.

### Isolation Forest

Anomalias são mais fáceis de isolar por cortes aleatórios.

Bom para:

- baseline rápido;
- treino sem labels de evento;
- dados tabulares com features.

### One-Class SVM

Aprende uma fronteira ao redor da classe normal.

Bom para:

- detecção de novidade;
- treino só com normal.

Precisa de normalização.

### Local Outlier Factor

Compara densidade local dos pontos.

Um ponto é anômalo se está em região menos densa que seus vizinhos.

### Logistic Regression

Classificador linear.

Bom para:

- baseline simples;
- interpretação;
- treino rápido.

Precisa de exemplos das duas classes.

### Random Forest

Conjunto de árvores de decisão.

Bom para:

- features tabulares;
- não linearidade;
- importância de features;
- baseline forte.

### Extra Trees

Parecido com Random Forest, mas com mais aleatoriedade nas divisões.

Pode generalizar bem e treinar rápido.

---

## 10. Deep Learning para Séries Temporais

### Dense Autoencoder

Recebe a janela como vetor.

```text
400 pontos -> encoder -> espaço latente -> decoder -> 400 pontos
```

Aprende a reconstruir ruído. Se a janela for evento, o erro de reconstrução tende a aumentar.

### CNN 1D Autoencoder

Usa convoluções ao longo do tempo.

Boa para:

- padrões locais;
- início súbito;
- formas de onda;
- oscilações curtas.

### LSTM Autoencoder

Usa memória temporal.

Boa para:

- dependências sequenciais;
- evolução do sinal dentro da janela.

Limitações:

- mais lenta;
- pode overfitar;
- exige cuidado com reshape.

### Transformer Autoencoder

Usa atenção para relacionar diferentes partes da janela.

Boa para:

- dependências de longo alcance;
- arquitetura moderna;
- comparação com modelos recentes.

Limitações:

- custo computacional maior;
- precisa de mais dados;
- pode ser exagerado para dataset pequeno.

### Onde aparece no projeto

- `src/tcc_seismic/tf_models.py`
- `scripts/train_autoencoders.py`

---

## 11. Scores, Threshold e Decisão

Modelos de anomalia normalmente produzem um score.

Exemplo:

```text
score baixo = normal
score alto = anomalia
```

Para transformar score em classe, usa-se um threshold.

```text
se score >= threshold:
    evento
senão:
    normal
```

### Regra importante

O threshold deve ser escolhido na validação, não no teste.

Fluxo correto:

```text
treino -> aprende modelo
validação -> escolhe threshold
teste -> avalia resultado final
```

### Onde aparece no projeto

- `src/tcc_seismic/metrics.py`

---

## 12. Métricas de Avaliação

### Matriz de confusão

```text
TP = previu evento e era evento
FP = previu evento, mas era normal
TN = previu normal e era normal
FN = previu normal, mas era evento
```

### Precision

Das vezes que o modelo disse evento, quantas estavam certas?

```text
precision = TP / (TP + FP)
```

### Recall

Dos eventos reais, quantos o modelo encontrou?

```text
recall = TP / (TP + FN)
```

### F1

Equilíbrio entre precision e recall.

```text
F1 = 2 * precision * recall / (precision + recall)
```

### AUC-ROC

Mede separação geral entre classes variando o threshold.

```text
0.5 = aleatório
1.0 = perfeito
```

### AUC-PR

Mede relação precision-recall variando o threshold.

É muito útil em problemas desbalanceados.

### Qual métrica priorizar

Para seu TCC:

1. AUC-PR;
2. F1;
3. precision;
4. recall;
5. AUC-ROC;
6. matriz de confusão;
7. tempo de inferência;
8. tamanho do modelo.

---

## 13. Overfitting, Underfitting e Generalização

### Underfitting

Modelo simples demais. Não aprende nem o treino.

### Overfitting

Modelo decora treino e falha no teste.

### Generalização

Capacidade de funcionar em dados novos.

Níveis:

- nova janela do mesmo evento;
- novo evento da mesma estação;
- nova estação;
- novo período temporal;
- nova região.

O nível mais honesto para o TCC deve ser explicitado.

---

## 14. Edge Computing

Edge computing significa executar próximo ao sensor, não em um servidor grande.

### Por que importa

Em sistemas reais, pode haver:

- baixa conectividade;
- necessidade de baixa latência;
- limitação de energia;
- limitação de memória;
- alto volume de dados.

### Métricas de edge

- tempo de inferência por janela;
- tamanho do modelo;
- uso de memória;
- consumo de CPU/GPU;
- possibilidade de quantização;
- robustez.

### TFLite

TensorFlow Lite permite converter modelos para ambientes mais leves.

### Quantização

Reduz precisão numérica:

```text
float32 -> int8
```

Pode reduzir tamanho e latência, mas pode degradar desempenho.

---

## 15. Protocolos de Comunicação

Se o projeto evoluir para sensores reais ou edge, comunicação vira parte importante.

### HTTP/REST

Modelo simples de requisição e resposta.

Uso:

- APIs web;
- envio eventual de resultados;
- dashboards.

Vantagens:

- simples;
- fácil de integrar;
- bem conhecido.

Limitações:

- menos eficiente para stream contínuo;
- overhead maior.

### MQTT

Protocolo publish/subscribe muito usado em IoT.

Elementos:

- broker;
- publisher;
- subscriber;
- tópico.

Exemplo:

```text
sensor/sismico/estacao01/janela
sensor/sismico/estacao01/alerta
```

Vantagens:

- leve;
- bom para sensores;
- funciona bem com conexões instáveis;
- permite múltiplos consumidores.

### gRPC

Comunicação eficiente baseada em contratos.

Bom para:

- microserviços;
- baixa latência;
- streaming;
- sistemas fortemente tipados.

### WebSocket

Canal persistente bidirecional.

Bom para:

- dashboards em tempo real;
- visualização contínua de alertas.

### OPC-UA

Muito usado em indústria.

Bom para:

- automação industrial;
- interoperabilidade entre equipamentos;
- ambientes industriais.

### Modbus

Protocolo industrial antigo e simples.

Bom para:

- CLPs;
- sensores industriais;
- integração legada.

### LoRaWAN

Comunicação de longo alcance e baixo consumo.

Bom para:

- sensores remotos;
- monitoramento ambiental;
- baixa taxa de dados.

### Arquitetura possível para seu tema

```text
sensor
  -> edge device
  -> pré-processamento local
  -> modelo de detecção
  -> envio de score/alerta via MQTT
  -> backend
  -> dashboard
```

---

## 16. Engenharia de Software do Projeto

Um TCC forte não tem só modelo. Tem organização.

### Boas práticas

- separar código em módulos;
- evitar caminhos absolutos;
- usar variáveis de ambiente;
- ter requirements;
- salvar resultados em JSON/CSV;
- manter scripts reproduzíveis;
- documentar como rodar;
- separar treino, avaliação e comparação.

### Onde isso aparece agora

- `src/tcc_seismic/paths.py`
- `src/tcc_seismic/data.py`
- `scripts/train_autoencoders.py`
- `scripts/train_ml_models.py`
- `scripts/compare_corrected_results.py`

---

## 17. Ordem Recomendada de Estudo

### Semana 1: entender dados e séries temporais

Leia:

- `artefacts/data/windows/dataset_info_v2.json`
- `src/tcc_seismic/data.py`

Você precisa explicar:

- o que é X;
- o que é y;
- quantas amostras existem;
- qual é o shape dos dados;
- por que há treino, validação e teste.

### Semana 2: pré-processamento e janelas

Leia:

- `notebook/Preprocessamento/`
- `notebook/Dados/`

Você precisa explicar:

- como o sinal bruto vira janela;
- por que filtrar;
- por que normalizar;
- por que janelar.

### Semana 3: métricas e avaliação

Leia:

- `src/tcc_seismic/metrics.py`

Você precisa explicar:

- precision;
- recall;
- F1;
- AUC-PR;
- AUC-ROC;
- matriz de confusão;
- threshold na validação.

### Semana 4: ML clássico

Leia:

- `src/tcc_seismic/features.py`
- `scripts/train_ml_models.py`

Você precisa explicar:

- por que extrair features;
- Isolation Forest;
- One-Class SVM;
- LOF;
- Random Forest;
- Extra Trees.

### Semana 5: deep learning

Leia:

- `src/tcc_seismic/tf_models.py`
- `scripts/train_autoencoders.py`

Você precisa explicar:

- autoencoder;
- erro de reconstrução;
- CNN 1D;
- LSTM;
- Transformer;
- overfitting.

### Semana 6: comparação final e escrita

Leia:

- `scripts/compare_corrected_results.py`
- resultados em `artefacts/results/`

Você precisa montar:

- tabela comparativa;
- discussão dos melhores modelos;
- limitações;
- aplicações futuras.

---

## 18. Perguntas que Você Deve Conseguir Responder

1. Por que este problema é uma série temporal?
2. O que é uma janela?
3. Por que treinar autoencoder só com ruído?
4. O que é score de anomalia?
5. O que é threshold?
6. Por que threshold não deve ser escolhido no teste?
7. Por que AUC-PR é importante?
8. Por que acurácia pode enganar?
9. Qual é o papel do STA/LTA?
10. Por que comparar com modelos clássicos?
11. O que CNN 1D captura que Dense AE não captura?
12. O que LSTM tenta capturar?
13. O que Transformer adiciona?
14. O que é edge computing?
15. Qual protocolo você usaria para sensores em campo?
16. Como evitar vazamento de dados?
17. Quais são as limitações do seu dataset?
18. Como esse trabalho se conecta a petróleo sem prometer encontrar petróleo?

---

## 19. Resposta de Defesa Pronta

> Este trabalho não busca criar um sistema operacional de detecção sísmica nem localizar petróleo diretamente. A proposta é desenvolver e avaliar um pipeline reutilizável de IA para detecção de anomalias em séries temporais de sensores. Sinais sísmicos foram escolhidos como estudo de caso por representarem dados físicos reais, ruidosos e dependentes do tempo. A metodologia compara STA/LTA, modelos clássicos de machine learning e autoencoders, incluindo arquiteturas CNN, LSTM e Transformer. A avaliação prioriza AUC-PR, F1, precision, recall e custo computacional, com threshold definido na validação para evitar vazamento. O resultado esperado é uma análise técnica do trade-off entre desempenho e viabilidade computacional, com aplicação futura em geofísica, sensores industriais, infraestrutura crítica e edge AI.

---

## 20. Mentalidade de Engenheiro

O ponto mais importante:

> Modelo bom em avaliação fraca não vale tanto quanto modelo razoável em avaliação honesta.

Prioridades:

1. entender os dados;
2. evitar vazamento;
3. comparar com baselines simples;
4. usar métricas adequadas;
5. explicar erros;
6. documentar limitações;
7. conectar a aplicação sem prometer demais.

