# Contrato Generico do Pipeline

Este projeto deve ser tratado como um pipeline generico de deteccao de anomalias em series temporais. O dominio sismico e a primeira instancia validada, nao uma dependencia estrutural do codigo.

## Camadas

```text
profiles/              contrato por dominio
src/tcc_pipeline/      nucleo generico reutilizavel
scripts/               comandos reproduziveis
edge/                  implantacao TinyML/ESP32
docs/                  rastreabilidade e decisoes tecnicas
notebook/              historico exploratorio
artefacts/             resultados locais e arquivos pesados nao versionados
```

## Contrato de entrada

Todo dataset treinavel deve ser entregue em um `.npz` com as chaves:

```text
X_train, y_train
X_val, y_val
X_test, y_test
```

Regras:

- `X_*` deve representar janelas temporais numericas.
- Shapes aceitos: `(n, window_size)`, `(n, window_size, 1)` ou `(n, 1, window_size)`.
- `y_*` deve usar labels inteiros declarados no profile.
- O split deve ser feito antes do treino e documentado no profile.
- O preprocessing usado para gerar as janelas deve estar declarado no profile.

## Contrato de profile

Cada dominio precisa de um arquivo em `profiles/` declarando:

- nome e versao do profile;
- dominio;
- taxa de amostragem;
- tamanho da janela;
- labels normal/anomalo;
- split principal;
- metrica primaria;
- pipeline de preprocessing offline;
- pipeline de preprocessing embarcado;
- estrategia de implantacao edge.

Exemplos:

```text
profiles/seismic_v1.json
profiles/cwru_v1.json
profiles/industrial_motor_v1.json
```

## O que e generico

- Validacao de janela e labels.
- Leitura de NPZ.
- Calculo de metricas.
- Escolha de threshold pela validacao.
- Treino de classificadores compactos.
- Export para TFLite e header C/C++.
- Manifesto de rastreabilidade.

## O que e especifico do dominio

- Como transformar dado bruto em janelas.
- Como rotular normal/anomalo.
- Como evitar vazamento no split.
- Quais filtros/preprocessamentos fazem sentido.
- Como calibrar o sensor.

## Regra de projeto

Codigo novo deve entrar primeiro no nucleo generico se ele puder funcionar para mais de um dominio. Codigo que so faz sentido para sismologia deve ficar nomeado como sismico, documentado como adaptador ou historico.

Essa separacao permite defender que o trabalho nao e apenas um detector sismico, mas uma arquitetura TinyML/MLOps para anomalias em series temporais.