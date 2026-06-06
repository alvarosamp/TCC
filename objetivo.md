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