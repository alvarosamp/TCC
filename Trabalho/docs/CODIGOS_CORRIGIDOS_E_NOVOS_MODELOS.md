# Codigos corrigidos e novos modelos

Este pacote adiciona uma base limpa para treinar e comparar modelos sem vazar
informacao do teste.

## O que foi corrigido

Antes, os scripts antigos calculavam o melhor F1 diretamente no teste em alguns
pontos. Isso infla o resultado, porque o teste passa a participar da escolha do
threshold.

Agora o fluxo correto e:

1. Treina o modelo no conjunto de treino.
2. Calcula scores no conjunto de validacao.
3. Escolhe o threshold usando apenas a validacao.
4. Aplica esse threshold fixo no teste.
5. Reporta AUC-PR, AUC-ROC, precision, recall, F1 e matriz de confusao.

## Estrutura nova

```text
src/tcc_seismic/
  data.py        Carregamento dos datasets existentes e splits v3.
  metrics.py     Threshold na validacao e metricas finais no teste.
  features.py    Features explicaveis para ML classico.
  tf_models.py   Dense AE, CNN 1D AE, LSTM AE e Transformer AE.
  paths.py       Caminhos por variavel de ambiente ou pelo repo.

scripts/
  train_autoencoders.py       Treina modelos neurais.
  train_ml_models.py          Treina baselines classicos de ML.
  compare_corrected_results.py Consolida resultados em CSV.
```

## Dependencias

Crie/ative seu ambiente Python e instale:

```bash
pip install -r requirements-models.txt
```

Se voce usa GPU, instale a versao do TensorFlow compativel com seu CUDA.

## Dados

Por padrao, os scripts usam os dados que ja existem no repositorio:

```text
artefacts/data/windows/windows_noise_v2.npz
artefacts/data/windows/windows_events_v2.npz
```

Esse modo aparece como `legacy_v2`.

Se voce tiver os arquivos mais novos:

```text
dataset_v3_split_estacao.npz
dataset_v3_split_temporal.npz
```

aponte a pasta assim:

```bash
set TCC_PROCESSED_DIR=C:\caminho\para\processed
```

No PowerShell:

```powershell
$env:TCC_PROCESSED_DIR="C:\caminho\para\processed"
```

## Rodar autoencoders

Dense Autoencoder:

```bash
python scripts/train_autoencoders.py --model dense --dataset legacy_v2
```

CNN 1D Autoencoder:

```bash
python scripts/train_autoencoders.py --model cnn1d --dataset legacy_v2
```

LSTM Autoencoder:

```bash
python scripts/train_autoencoders.py --model lstm --dataset legacy_v2 --timesteps 50
```

Transformer Autoencoder:

```bash
python scripts/train_autoencoders.py --model transformer --dataset legacy_v2 --patch-size 8
```

Para rodar com os splits v3:

```bash
python scripts/train_autoencoders.py --model transformer --dataset v3
```

## Rodar machine learning classico

```bash
python scripts/train_ml_models.py --dataset legacy_v2
```

No dataset `legacy_v2`, o treino tem apenas ruido. Por isso os modelos
supervisionados sao pulados e entram os modelos de anomalia:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor com `novelty=True`

Se um split tiver `y_train` com ruido e evento, o script tambem treina:

- Logistic Regression
- Random Forest
- Extra Trees

## Comparar resultados

Depois de treinar alguns modelos:

```bash
python scripts/compare_corrected_results.py
```

Saida:

```text
artefacts/results/models_corrected/comparison_corrected.csv
```

## Como interpretar

Use AUC-PR como metrica principal quando houver desbalanceamento ou quando
falso alarme importa.

Use F1 como ponto operacional, mas lembre:

```text
F1 depende do threshold.
AUC-PR e AUC-ROC avaliam o ranking dos scores.
```

Para o TCC, a tabela final recomendada e:

```text
modelo, split, AUC-PR, AUC-ROC, precision, recall, F1, threshold, tamanho, latencia
```

## Observacao importante

Esses codigos corrigem a avaliacao, mas nao resolvem sozinhos a questao mais
cientifica do trabalho: garantir que os splits nao tenham vazamento entre janelas
parecidas do mesmo evento/arquivo/estacao. Para a versao final do TCC, prefira
splits por evento, arquivo, tempo ou estacao.

