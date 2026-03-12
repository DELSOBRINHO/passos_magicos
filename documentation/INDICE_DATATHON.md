# 📚 Índice do Datathon — Passos Mágicos

> Documento de mapeamento entre o enunciado do case, os entregáveis esperados e a solução atual do repositório, em alinhamento com `documentation/PLANO_MESTRE.md`.

## 1. Como ler este índice

Este arquivo conecta três camadas:

1. o que o Datathon pede;
2. como o plano mestre orienta a resposta;
3. como o repositório implementa isso hoje.

## 2. Entregáveis obrigatórios

### E1 — Código no GitHub

**Pedido do case:** repositório com limpeza, análise, modelagem e app.

**Atendimento atual:** estrutura versionada com `app/`, `data/`, `documentation/`, `notebooks/` e `tests/`.

### E2 — Notebook(s) de análise e modelagem

**Pedido do case:** feature engineering, treino/teste, modelagem e avaliação.

**Atendimento atual:** notebooks e scripts auxiliares em `notebooks/` cobrindo análise exploratória, normalização e pipeline preditivo.

### E3 — Aplicação Streamlit com deploy

**Pedido do case:** disponibilizar o modelo em ferramenta utilizável.

**Atendimento atual:** app com:

- **Predição Individual**;
- **Análise da Turma**;
- **Sobre o Projeto**.

O deploy usa `streamlit_app.py` como entrypoint na raiz.

### E4 — Apresentação executiva

**Pedido do case:** storytelling gerencial em PPT ou PDF.

**Atendimento atual:** narrativa orientada pelo eixo **“A Jornada da Pedra”** definido no plano mestre.

### E5 — Vídeo de demonstração

**Pedido do case:** vídeo curto com história analítica e demonstração da solução.

**Atendimento atual:** frente ainda pendente de consolidação final, mas já suportada por app, notebooks e documentação.

## 3. Mapeamento das 11 perguntas do case

| Pergunta | Tema | Resposta atual no repositório |
|---|---|---|
| P1 | IAN | IAN integra o conjunto-base da calculadora individual e da leitura de defasagem |
| P2 | IDA | IDA alimenta a dimensão acadêmica e as análises de desempenho |
| P3 | IEG | IEG participa da leitura de engajamento e risco |
| P4 | IAA | IAA compõe a dimensão psicossocial e a leitura de coerência do aluno |
| P5 | IPS | IPS sustenta a análise psicossocial preventiva |
| P6 | IPP | IPP entra na dimensão psicopedagógica; no lote pode usar proxy de `Cf/Ct` |
| P7 | IPV | IPV integra a dimensão psicopedagógica e a leitura do ponto de virada |
| P8 | INDE | O app calcula INDE dinâmico por fase e consolida combinações de indicadores |
| P9 | Predição de risco | O app estima probabilidade de risco com contexto, dimensões e sinais textuais |
| P10 | Efetividade do programa | A solução permite leitura por fase, pedras e evolução contextual |
| P11 | Insights extras | NLP, recomendações e fatores de risco ampliam a análise do case |

## 4. Operacionalização atual da pergunta 9

No estado atual do repositório, a frente preditiva foi operacionalizada com:

- `IAN`, `IDA`, `IEG`, `IAA`, `IPS`, `IPP`, `IPV`;
- `Fase`;
- `Pedra Atual` e `Pedra Anterior`;
- `Nº de Avaliações`;
- observação textual convertida em sinais numéricos.

### Derivações internas do app

- cálculo do **INDE dinâmico por fase**;
- consolidação em dimensões Acadêmica, Psicossocial e Psicopedagógica;
- geração de `sent_score` e `sent_len`;
- aplicação de calibração na probabilidade bruta.

### Saídas entregues

- INDE calculado;
- dimensões consolidadas;
- probabilidade de risco em `%`;
- faixa de risco (`baixo`, `médio`, `alto`);
- recomendação de intervenção.

## 5. Componentes da solução atual

### 5.1 Engenharia de dados

- preparação dos indicadores;
- normalização e consolidação das dimensões;
- cálculo do INDE dinâmico.

### 5.2 NLP

- leitura lexical das observações;
- uso operacional de `sent_score` e `sent_len`.

### 5.3 Predição

- modelo probabilístico tabular serializado em `app/`;
- threshold de risco lido do metadado;
- fallback heurístico para contingência.

### 5.4 Interface gerencial

- simulador individual para triagem;
- análise em lote para turma;
- página de contexto com métricas do modelo, incluindo ROC-AUC quando disponível no metadado.

## 6. Coerência com o plano mestre

Para manter o alinhamento correto, este índice assume sempre que:

- `PLANO_MESTRE.md` define a direção estratégica;
- `PLANO_DESENVOLVIMENTO.md` descreve a execução atual;
- a documentação não deve reintroduzir como padrão vigente arquiteturas ou artefatos legados que não representam o estado atual publicado.

## 7. Observação importante sobre modelagem

O case e a fase do curso enfatizam Deep Learning e dados não estruturados. Em coerência com isso:

- o **plano mestre** preserva essa direção estratégica;
- o **repositório atual** operacionaliza a entrega com modelo probabilístico calibrado, dimensões consolidadas e NLP lexical.

Assim, o projeto permanece fiel ao objetivo do case sem distorcer a implementação realmente presente no repositório.