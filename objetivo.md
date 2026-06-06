Claro. Aqui vai um modelo para você anotar no caderno, cobrindo **todas as etapas até agora**.

```md
# Registro Técnico Do Projeto De TCC

## 1. Definição Da Ideia Do Projeto

### Objetivo

O objetivo do projeto é construir um pipeline genérico de TinyML para detecção de anomalias em séries temporais complexas. O primeiro domínio utilizado é o sísmico, mas a estrutura foi pensada para ser reutilizada em outros sinais, como vibração industrial, corrente elétrica, áudio ou telemetria.

### Ideia Central

O modelo TinyML atua como um organizador inteligente na borda. Em vez de transmitir todos os dados continuamente, o dispositivo processa janelas locais do sinal e só envia informações quando identifica uma possível anomalia.

### Benefícios Esperados

- Redução de transmissão de dados;
- menor consumo de energia;
- menor custo de armazenamento;
- menor custo operacional;
- resposta mais rápida na borda.

---

## 2. Construção Do Pipeline Genérico

### Objetivo

Separar o que é específico do domínio do que é genérico no pipeline de machine learning.

### Decisão Técnica

A arquitetura foi organizada em duas partes:

```text
adapter de domínio → dataset genérico → treino genérico → exportação edge
```

O adapter de domínio é responsável por transformar dados específicos, como MiniSEED no caso sísmico, em um formato comum.

### Contrato Genérico De Dataset

O pipeline espera sempre:

```text
X_train, y_train
X_val,   y_val
X_test,  y_test
```

Assim, qualquer domínio pode ser usado desde que respeite esse contrato.

### Importância

Essa separação permite que o projeto não fique preso à sismologia. Para usar outro sensor, basta criar outro adapter que gere o mesmo formato de dataset.

---

## 3. Profile Do Pipeline

### Objetivo

Criar um arquivo de configuração que descreve as características do domínio, do sinal e da implantação embarcada.

### Informações Do Profile

O profile contém:

- nome do domínio;
- taxa de amostragem;
- tamanho da janela;
- duração da janela;
- passo entre janelas;
- labels;
- métricas principais;
- pipeline de preprocessamento;
- configuração embarcada.

### Exemplo Do Caso Sísmico

```text
sampling_rate = 40 Hz
window_size = 800 amostras
window_seconds = 20 s
step_seconds = 10 s
target = ESP32
runtime = TensorFlow Lite Micro
```

### Importância

O profile conecta dataset, treino, métricas, exportação e firmware. Ele funciona como a fonte de verdade técnica do experimento.

---

## 4. Preprocessamento Edge-Aware

### Objetivo

Ajustar o preprocessamento para reduzir a diferença entre treinamento e inferência no microcontrolador.

### Problema Inicial

O pipeline anterior utilizava `remove_response`, que depende de StationXML e da resposta instrumental da estação. Essa etapa é difícil de reproduzir em um ESP32.

### Decisão Técnica

Foi adotado um pipeline sem `remove_response`:

```text
resample 40 Hz
→ detrend linear
→ demean
→ taper 5%
→ bandpass 0.5-15 Hz
→ zscore por janela
```

### Justificativa

Essa versão é mais compatível com a borda, pois suas etapas podem ser aproximadas em C/C++ no microcontrolador.

### Limitação

A remoção de `remove_response` pode reduzir a padronização física entre sensores diferentes. Por isso, a validação em múltiplas estações e datasets é importante.

---

## 5. Splits Do Dataset

### Objetivo

Evitar vazamento de informação entre treino, validação e teste.

### Estratégias Discutidas

Foram considerados splits:

```text
temporal
por estação
por evento
```

### Split Principal

O split por evento foi escolhido como principal, pois impede que janelas do mesmo evento apareçam em mais de um conjunto.

### Importância

Isso reduz a chance de o modelo memorizar um terremoto específico em vez de aprender padrões gerais de anomalia.

---

## 6. Treinamento De Modelos

### Objetivo

Treinar e comparar diferentes famílias de modelos.

### Modelos Considerados

- Random Forest;
- Extra Trees;
- Tiny CNN;
- Tiny TCN;
- LSTM;
- Autoencoders;
- Isolation Forest;
- Logistic Regression.

### Decisão De Produto

Nem todos os modelos precisam rodar sempre. O usuário pode ativar ou desativar modelos no arquivo de configuração.

### Configuração Atual

Os modelos habilitados por padrão são:

```text
random_forest
extra_trees
tiny_cnn
tiny_tcn
```

### Importância

Isso permite comparar modelos interpretáveis com modelos candidatos para TinyML.

---

## 7. Otimização Com Optuna

### Objetivo

Buscar hiperparâmetros melhores automaticamente.

### Ideia

O fluxo desejado é:

```text
treina modelos selecionados
→ roda Optuna quando habilitado
→ avalia em validação
→ treina modelo final
→ avalia em teste
→ escolhe melhor candidato
```

### Resultado

Foram incorporados presets baseados nas melhores rodadas externas de Optuna:

- `tiny_cnn` como melhor candidato completo;
- `tiny_tcn` como candidato ainda em avaliação.

### Importância

A otimização ajuda a encontrar arquiteturas pequenas, mas eficientes, adequadas ao uso em microcontroladores.

---

## 8. Features Estatísticas E Espectrais

### Objetivo

Criar features genéricas para modelos clássicos e para análise interpretável.

### Features Implementadas

Foram extraídas 28 features por janela, incluindo:

```text
mean
std
rms
abs_peak
crest_factor
kurtosis
zero_crossing_rate
dominant_freq
spectral_centroid
spectral_rolloff_85
bandpower_0_3hz
bandpower_8_15hz
spectral_entropy
```

### Resultado Observado

Nos modelos clássicos, features espectrais apareceram como muito importantes, principalmente:

```text
spectral_rolloff_85
bandpower_8_15hz
spectral_centroid
zero_crossing_rate
kurtosis
spectral_entropy
```

### Interpretação

Isso indica que os modelos não dependem apenas de amplitude. Eles capturam características de frequência e forma do sinal.

---

## 9. Métricas De Avaliação

### Objetivo

Avaliar os modelos de forma adequada para detecção de anomalias.

### Métrica Principal

A métrica principal escolhida foi AUC-PR.

### Justificativa

Em problemas desbalanceados, AUC-PR é mais informativa que acurácia, pois foca no desempenho da classe rara/anômala.

### Métricas Complementares

Também foram usadas:

```text
AUC-ROC
F1
precision
recall
FP/h
```

### FP/h

Foi adicionada a métrica de falsos positivos por hora:

```text
FP/h = falsos positivos / horas avaliadas
```

### Importância

Essa métrica aproxima a avaliação do uso real. Um modelo com boa AUC-PR pode ainda ser ruim se gerar alarmes falsos demais.

---

## 10. Seleção Do Melhor Modelo

### Objetivo

Escolher automaticamente o melhor modelo com base na métrica definida.

### Fluxo

```text
model_comparison.csv
model_comparison.json
candidate_manifest.json
```

### Resultado

O pipeline gera um manifesto do melhor candidato, contendo:

- nome do modelo;
- família;
- métricas;
- threshold;
- profile;
- dataset;
- caminho do modelo;
- parâmetros;
- contagem de parâmetros.

### Importância

O `candidate_manifest.json` é a ponte entre treinamento e MLOps.

---

## 11. Prevenção De Overfitting

### Objetivo

Reduzir o risco de o modelo decorar o dataset.

### Técnicas Usadas

- split por evento;
- validação separada;
- threshold escolhido na validação;
- early stopping;
- dropout;
- spatial dropout;
- L2 regularization;
- label smoothing;
- comparação entre validação e teste.

### Regra Importante

O threshold não deve ser escolhido no teste. O correto é:

```text
escolher threshold na validação
aplicar o mesmo threshold no teste
```

### Importância

Isso evita otimizar o resultado final artificialmente.

---

## 12. DVC Como Pipeline Reprodutível

### Objetivo

Permitir a reprodução das etapas do projeto.

### Pipeline DVC

O pipeline atual contém:

```text
generate_data
→ validate_dataset
→ train_all
→ export_tflite
```

### Observação

Atualmente, o `dvc repro` usa um dataset sintético e configuração smoke. Portanto, ele serve para validar comunicação entre etapas, mas não como resultado científico final.

### Importância

O DVC garante que o fluxo completo possa ser reexecutado de forma rastreável.

---

## 13. MLflow E Rastreamento

### Objetivo

Registrar parâmetros, métricas e artefatos dos modelos treinados.

### O Que É Registrado

- nome do modelo;
- família;
- parâmetros;
- métricas de validação;
- métricas de teste;
- threshold;
- artefatos do modelo.

### Importância

O MLflow permite comparar experimentos e manter rastreabilidade entre configurações e resultados.

---

## 14. Quality Gate

### Objetivo

Impedir que qualquer modelo treinado seja automaticamente promovido para produção.

### Funcionamento

O quality gate lê:

```text
candidate_manifest.json
```

e aplica regras como:

```text
min_auc_pr
min_f1
max_fp_per_hour
max_val_test_auc_pr_gap
max_model_size_kb
```

### Saídas

```text
promotion_report.json
production_manifest.json
```

### Resultado Atual

O modelo `tiny_cnn` foi aprovado no quality gate e promovido para produção local.

### Limitação

A aprovação atual ocorreu sobre dataset sintético/fácil. Portanto, valida o fluxo MLOps, mas não representa ainda a conclusão científica do TCC.

---

## 15. Preparação Para OTA

### Objetivo

Criar a ponte entre o modelo aprovado e a atualização do dispositivo embarcado.

### Ideia

Antes de fazer OTA real, é necessário gerar um manifesto OTA.

### Fluxo Planejado

```text
production_manifest.json
→ ota_manifest.json
→ pacote de atualização
→ validação no ESP32
→ aplicação da atualização
→ rollback se falhar
```

### Informações Do Manifesto OTA

O manifesto deve conter:

- nome do modelo;
- versão;
- threshold;
- target;
- runtime;
- caminho do artefato;
- checksum SHA-256;
- estratégia OTA;
- profile;
- métricas;
- regras de qualidade.

### Importância

OTA sem manifesto é arriscado, pois não há rastreabilidade sobre qual modelo foi enviado, com qual threshold e para qual versão de preprocessamento.

---

## 16. Estado Atual Do Projeto

### O Que Já Funciona

```text
dataset sintético
→ validação
→ treino
→ comparação
→ seleção de candidato
→ quality gate
→ produção local
```

### O Que Ainda Precisa Ser Feito

```text
rodar com dataset real
validar splits reais
exportar modelo final
testar no ESP32
gerar manifesto OTA
implementar OTA real
monitorar drift
retreinar quando necessário
```

### Conclusão Parcial

O pipeline técnico está tomando forma como um sistema completo de TinyML/MLOps. A estrutura já permite treinar, comparar, selecionar, promover e preparar modelos para implantação. A próxima etapa crítica é substituir o dataset sintético pelo dataset real e validar se os resultados continuam consistentes.
```