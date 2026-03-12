# Datathon Passos Mágicos — Contexto do Desafio

> Este arquivo resume o enunciado do case. A direção executiva do projeto está em `documentation/PLANO_MESTRE.md`.

## 📘 Papel deste Documento

Use este arquivo como **contexto do desafio**. Para o estado atual da solução, consulte:

- `documentation/PLANO_MESTRE.md` → direção estratégica
- `documentation/PLANO_DESENVOLVIMENTO.md` → execução atual do repositório
- `documentation/INDICE_DATATHON.md` → mapeamento entre entregáveis, perguntas e solução

## 🎓 Contexto Acadêmico

O Datathon pertence à **Fase 5 – Deep Learning & Unstructured Data** do curso FIAP Postech em Data Analytics, cobrindo especialmente:

- Deep & Reinforcement Learning
- Processamento de texto
- Classificação de texto e análise de sentimentos
- Embeddings e dados gerados por humanos
- Storytelling analítico com foco em impacto social

## 🧩 Caso de Negócio

A Associação Passos Mágicos atua na transformação da vida de crianças e jovens em situação de vulnerabilidade social por meio da educação, apoio psicossocial e desenvolvimento de protagonismo.

O desafio do Datathon é transformar a base PEDE em uma solução analítica capaz de:

- diagnosticar padrões de defasagem educacional
- identificar alavancas de melhoria
- apoiar decisões pedagógicas e psicossociais
- prever risco futuro de defasagem com uma aplicação utilizável pela equipe

## ❓ Perguntas Analíticas do Case

1. **IAN** — Qual é o perfil geral de defasagem dos alunos?
2. **IDA** — O desempenho acadêmico melhora, cai ou se mantém ao longo das fases?
3. **IEG** — O engajamento se relaciona com desempenho e ponto de virada?
4. **IAA** — A autoavaliação é coerente com o desempenho real?
5. **IPS** — Há sinais psicossociais que antecedem quedas?
6. **IPP** — A leitura psicopedagógica confirma a defasagem observada?
7. **IPV** — Quais fatores mais influenciam o ponto de virada?
8. **INDE** — Quais combinações de indicadores elevam ou reduzem o desempenho global?
9. **Predição de risco** — Como estimar a probabilidade de um aluno entrar em risco de defasagem?
10. **Efetividade do programa** — Há melhora consistente ao longo da jornada das pedras?
11. **Insights extras** — Que recomendações adicionais podem ser propostas à associação?

## 📦 Entregáveis Esperados

O case pede:

- repositório GitHub com código de limpeza, análise e modelagem
- apresentação executiva em PPT ou PDF
- notebook(s) com feature engineering, treino/teste, modelagem e avaliação
- aplicação em Streamlit com deploy no Community Cloud
- vídeo curto apresentando storytelling e solução preditiva

## 🔗 Como este Repositório Responde ao Enunciado

Em alinhamento ao plano mestre, o repositório atual organiza a solução em quatro frentes operacionais:

1. **Harmonização dos indicadores do PEDE**
2. **Leitura textual das observações via NLP lexical**
3. **Predição com dimensões consolidadas e probabilidade calibrada**
4. **App Streamlit com predição individual e análise da turma**

Na pergunta 9, a operacionalização atual está centrada em:

- **INDE dinâmico por fase**
- **dimensões Acadêmica, Psicossocial e Psicopedagógica**
- **contexto de fase, pedras e número de avaliações**
- **sinais textuais derivados das observações dos avaliadores**

## 🪨 Storytelling Proposto

O fio narrativo adotado no plano mestre é **“A Jornada da Pedra”**, conectando:

- diagnóstico atual
- fatores de risco e de proteção
- identificação precoce
- intervenções práticas com impacto social
