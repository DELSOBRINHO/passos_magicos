# 🌟 Passos Mágicos – Inteligência Educacional

> Repositório do Datathon FIAP Postech alinhado ao `documentation/PLANO_MESTRE.md`.

## 🎯 Visão Geral

Esta solução foi organizada em torno de cinco eixos do plano mestre:

1. **Engenharia de dados e harmonização do PEDE**
2. **Extração de sinais textuais via NLP**
3. **Predição de risco educacional**
4. **App gerencial em Streamlit**
5. **Storytelling “A Jornada da Pedra”**

No estado atual do repositório, o app operacional usa:

- **INDE dinâmico por fase**
- **dimensões consolidadas** (`dim_academica`, `dim_psicossocial`, `dim_psicopedagogica`)
- **sinais textuais** derivados das observações dos avaliadores (`sent_score`, `sent_len`)
- **probabilidade calibrada de risco**
- **fallback heurístico seguro** quando o artefato do modelo não está utilizável

## 📁 Estrutura Atual do Repositório

```
passos_magicos/
├── app/
│   ├── app.py
│   ├── risk_calibration.py
│   ├── ui_helpers.py
│   ├── modelo_risco_clean.pkl
│   ├── scaler_clean.pkl
│   └── modelo_meta_clean.json
├── data/
│   ├── BASE DE DADOS PEDE 2024 - DATATHON - PEDE2022.csv
│   └── fig_P*.png
├── documentation/
│   ├── PLANO_MESTRE.md
│   ├── PLANO_DESENVOLVIMENTO.md
│   └── INDICE_DATATHON.md
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_modelo_preditivo.ipynb
│   ├── 03_normalizacao.ipynb
│   └── scripts auxiliares de preparação/retreino
├── tests/
│   ├── test_risk_calibration.py
│   └── test_ui_helpers.py
├── streamlit_app.py
├── requirements.txt
└── requirements-dev.txt
```

## 🧭 Documentos de Referência

- `documentation/PLANO_MESTRE.md` → documento autoritativo de direção
- `documentation/PLANO_DESENVOLVIMENTO.md` → estado atual e backlog executivo
- `documentation/INDICE_DATATHON.md` → mapeamento dos entregáveis e perguntas do case
- `datathon.md` → resumo consolidado do enunciado original

## 🚀 Como Executar Localmente

### 1. Instalar dependências da aplicação

```bash
pip install -r requirements.txt
```

### 1.1 Para notebooks e reprodução completa do pipeline

```bash
pip install -r requirements-dev.txt
```

### 2. Rodar o Streamlit localmente

```bash
python -m streamlit run app/app.py
```

Depois acesse: `http://localhost:8501`

### 3. Executar o notebook do modelo preditivo

O notebook `notebooks/02_modelo_preditivo.ipynb` foi alinhado ao pipeline operacional atual.
Ao rodá-lo, ele gera/atualiza os artefatos `clean` usados pelo app e pode ser salvo também como versão executada (`02_modelo_preditivo.executed.ipynb`).

## ☁️ Deploy no Streamlit Community Cloud

O deploy em cloud usa o arquivo da raiz:

- **entrypoint:** `streamlit_app.py`

Esse wrapper apenas encaminha a execução para:

- `app/app.py`

## 🧠 Lógica Atual da Solução Preditiva

### Predição individual

Entradas principais da calculadora:

- `IAN`, `IDA`, `IEG`, `IAA`, `IPS`, `IPP`, `IPV`
- `Fase`
- `Pedra Atual` e `Pedra Anterior`
- `Nº de Avaliações`
- `Observações` dos avaliadores

O app transforma essas entradas em:

- **INDE calculado dinamicamente por fase**
- **dimensões consolidadas** para o modelo
- **probabilidade de risco calibrada**
- **classificação em baixo / médio / alto risco**
- **recomendações de intervenção**

### Análise da turma

No upload em lote, o app:

- aceita CSV com indicadores acadêmicos e de contexto
- deriva automaticamente dimensões consolidadas e INDE
- aplica o modelo calibrado quando disponível
- gera tabela de risco e arquivo de saída para download

## 🤖 Modelo e Artefatos Atuais

Artefatos principais atualmente usados pelo app:

- `app/modelo_risco_clean.pkl`
- `app/scaler_clean.pkl`
- `app/modelo_meta_clean.json`

Observações importantes:

- o **plano mestre** posiciona a frente preditiva dentro do eixo de Deep Learning da fase
- o **estado operacional atual do repositório** está implementado com modelo probabilístico tabular serializado via `joblib`, usando dimensões consolidadas e calibração de probabilidade
- o notebook `notebooks/02_modelo_preditivo.ipynb` foi atualizado para documentar exatamente esse pipeline operacional vigente
- a execução do notebook gera os artefatos `app/modelo_risco_clean.pkl`, `app/scaler_clean.pkl`, `app/modelo_risco_clean_imputer.pkl` e `app/modelo_meta_clean.json`
- o threshold é lido do metadado do modelo; no artefato clean atual, ele está em **0.30**

## 📊 Entregáveis Cobertos no Repositório

- código versionado no GitHub
- notebooks de análise e modelagem
- aplicação Streamlit publicada
- documentação executiva alinhada ao plano mestre
- base para apresentação e vídeo do Datathon

## ✅ Validação Recente

Verificações automatizadas já executadas no repositório:

- `python -m py_compile app/app.py app/ui_helpers.py`
- `python -m pytest tests -q`
- execução end-to-end de `notebooks/02_modelo_preditivo.ipynb`, com geração de `notebooks/02_modelo_preditivo.executed.ipynb`

Resultado mais recente:

- **7 testes passando**
- notebook 02 executado com sucesso sobre a base PEDE (`860 x 42`), AUC de teste `0.9996` e artefatos clean regenerados

## 🛠️ Tecnologias

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `Joblib` · `Streamlit`

## 👥 Contexto da Entrega

**Datathon:** Passos Mágicos  
**Curso:** FIAP Postech – Data Analytics  
**Fase:** 5 – Deep Learning & Unstructured Data

