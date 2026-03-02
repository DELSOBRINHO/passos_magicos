# 🌟 Passos Mágicos – Datathon 2025-2026

> Solução de Data Analytics + Deep Learning + NLP para identificar alunos em risco de defasagem educacional.

## 📁 Estrutura do Repositório

```
passos_magicos/
├── data/                              # Base de dados e figuras geradas
│   └── BASE DE DADOS PEDE 2024 - DATATHON - PEDE2022.csv
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb  # EDA – Perguntas 1-8, 10 e 11
│   ├── 02_modelo_preditivo.ipynb      # Deep Learning MLP – Pergunta 9
│   ├── gerar_notebooks.py             # Script gerador do notebook EDA
│   └── build_model_nb.py              # Script gerador do notebook ML
├── app/
│   ├── app.py                         # Aplicação Streamlit
│   ├── modelo_risco.h5 (ou .pkl)      # Modelo treinado (gerado pelo notebook)
│   ├── scaler.pkl                     # StandardScaler
│   └── modelo_meta.pkl                # Metadados do modelo
├── .streamlit/
│   └── config.toml                    # Tema da aplicação
├── requirements.txt                   # Dependências
└── README.md
```

## 🚀 Como Executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Gerar os Notebooks
```bash
cd notebooks
python gerar_notebooks.py    # cria 01_analise_exploratoria.ipynb
python build_model_nb.py     # cria 02_modelo_preditivo.ipynb
```

### 3. Executar os Notebooks (na ordem)
Abra o Jupyter Lab/Notebook e execute:
1. `notebooks/01_analise_exploratoria.ipynb` — EDA completo
2. `notebooks/02_modelo_preditivo.ipynb` — Treina e salva o modelo em `/app`

### 4. Rodar o Streamlit localmente
```bash
streamlit run app/app.py
```
Acesse: http://localhost:8501

## ☁️ Deploy no Streamlit Community Cloud

1. Faça push deste repositório no GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Clique em **New app**
4. Selecione o repositório e configure:
   - **Main file path:** `app/app.py`
   - **Python version:** 3.10+
5. Clique em **Deploy**

> ⚠️ **Importante:** Execute o notebook `02_modelo_preditivo.ipynb` antes do deploy para gerar os arquivos `modelo_risco.h5`, `scaler.pkl` e `modelo_meta.pkl` na pasta `/app`.

## 📊 Perguntas Respondidas

| # | Pergunta | Indicador | Arquivo |
|---|---------|-----------|---------|
| 1 | Perfil de defasagem | IAN | EDA |
| 2 | Evolução do desempenho | IDA | EDA |
| 3 | Engajamento vs desempenho/virada | IEG | EDA |
| 4 | Coerência da autoavaliação | IAA | EDA |
| 5 | Padrões psicossociais | IPS | EDA |
| 6 | Avaliação psicopedagógica vs IAN | IPP | EDA |
| 7 | Fatores do Ponto de Virada | IPV | EDA |
| 8 | Multidimensionalidade (Clusters) | INDE | EDA |
| 9 | **Modelo Preditivo de Risco** | MLP | ML |
| 10 | Efetividade por fase | INDE | EDA |
| 11 | Insights criativos (NLP + gênero) | — | EDA |

## 🤖 Modelo Preditivo – Detalhes Técnicos

- **Target:** `em_risco` — aluno com defasagem, INDE baixo ou baixo IEG+IDA
- **Arquitetura:** MLP: `Input(11) → Dense(128, ReLU) → Dropout(0.3) → Dense(64) → Dropout(0.3) → Dense(32) → Sigmoid`
- **Otimização:** Adam, EarlyStopping monitorando Recall
- **NLP:** Score de sentimento lexical nas recomendações dos avaliadores
- **Embeddings:** Pedra mapeada ordinalmente (Quartzo=1 → Topázio=4)
- **Métrica foco:** **Recall** — não deixar alunos em risco despercebidos

## 🛠️ Tecnologias

`Python 3.10` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `TensorFlow/Keras` · `Streamlit` · `SciPy`

## 👥 Grupo – Datathon Postech FIAP

**Entrega:** 22/03/2026 | **Fase 5:** Deep Learning & Unstructured Data

