# 📋 Plano de Desenvolvimento – Datathon Passos Mágicos
## Repositório: https://github.com/DELSOBRINHO/passos_magicos
## Prazo final: **22/03/2026** | Fase 5 – Deep Learning & NLP

---

> **Como usar este documento:**
> A cada iteração de trabalho, o agente deve:
> 1. Ler este arquivo **primeiro** para entender o estado atual
> 2. Marcar tarefas concluídas trocando `[ ]` por `[x]`
> 3. Adicionar data de conclusão ao lado da tarefa
> 4. Jamais retrabalhar itens já marcados `[x]`
> 5. Salvar o arquivo atualizado ao final da sessão

---

## 🗂️ ESTRUTURA DE ARQUIVOS ESPERADA

```
passos_magicos/
├── documentation/
│   └── PLANO_DESENVOLVIMENTO.md   ← este arquivo
├── data/
│   └── BASE DE DADOS PEDE 2024 - DATATHON - PEDE2022.csv
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_modelo_preditivo.ipynb
│   ├── gerar_notebooks.py
│   └── build_model_nb.py
├── app/
│   ├── app.py
│   ├── modelo_risco.h5   (gerado pelo notebook 02)
│   ├── scaler.pkl        (gerado pelo notebook 02)
│   └── modelo_meta.pkl   (gerado pelo notebook 02)
├── .streamlit/
│   └── config.toml
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📦 FASE 0 – INFRAESTRUTURA E REPOSITÓRIO

### 0.1 Estrutura local
- [x] Criar pasta `/data` — 2026-03-02
- [x] Criar pasta `/notebooks` — 2026-03-02
- [x] Criar pasta `/app` — 2026-03-02
- [x] Criar pasta `/documentation` — 2026-03-02
- [x] Copiar CSV para `/data` — 2026-03-02
- [x] Criar `.gitignore` (ignorar `__pycache__`, `.h5`, `.pkl`, dados grandes) — 2026-03-02

### 0.4 Devcontainer (VS Code)
- [x] Criar `.devcontainer/Dockerfile` (Python 3.10-bullseye, libgomp1, libhdf5-dev) — 2026-03-02
- [x] Criar `.devcontainer/devcontainer.json` (extensions, portas 8501+8888, postCreateCommand) — 2026-03-02
- [x] Verificar que `.devcontainer/` está rastreado no git (não ignorado) — 2026-03-02

### 0.2 Repositório GitHub
- [x] Fazer o primeiro `git push` com a estrutura base — 2026-03-02
- [x] Verificar que todos os arquivos estão no GitHub: https://github.com/DELSOBRINHO/passos_magicos — 2026-03-02
- [x] Confirmar branch `main` como padrão — 2026-03-02

### 0.3 Dependências
- [x] Instalar: `pandas, numpy, matplotlib, scipy, scikit-learn` — pré-existentes
- [x] Instalar: `seaborn` — 2026-03-02
- [x] Instalar: `nbformat, notebook` — 2026-03-02
- [x] Instalar: `streamlit` — 2026-03-02
- [x] Instalar: `tensorflow` (necessário para treinar modelo MLP completo) — 2026-03-03
- [x] Separar `requirements.txt` (Streamlit Cloud, usa `tensorflow-cpu`) de `requirements-dev.txt` (dev completo) — 2026-03-02
- [x] Verificar `requirements.txt` cobrindo todas as dependências do Streamlit Cloud — 2026-03-02

---

## 📊 FASE 1 – ANÁLISE EXPLORATÓRIA (Notebook 01)

**Arquivo:** `notebooks/01_analise_exploratoria.ipynb`
**Gerado por:** `notebooks/gerar_notebooks.py`

### 1.0 Base de Dados
- [x] Identificar encoding correto (`utf-8-sig`, sep=`,`) — 2026-03-02
- [x] Confirmar shape: **860 alunos × 42 colunas** — 2026-03-02
- [x] Mapear colunas numéricas com vírgula como decimal — 2026-03-02
- [x] Criar `Pedra_XX_num` (ordinal: Quartzo=1, Ágata=2, Ametista=3, Topázio=4) — 2026-03-02
- [x] Criar `Evolucao_Pedra` (Pedra_22 - Pedra_21) — 2026-03-02
- [x] Criar `Nivel_Defasagem` (categórica: Sem/Leve/Moderada/Severa) — 2026-03-02
- [x] Criar `IPP` proxy = (Cf + Ct) / 2 — 2026-03-02
- [ ] **EXECUTAR** o notebook e verificar que todas as células rodam sem erro
- [ ] Verificar que as figuras foram salvas em `/data/fig_P*.png`

### 1.1 Pergunta 1 – IAN (Defasagem)
- [x] Código criado: histograma IAN + pizza por nível de defasagem — 2026-03-02
- [ ] Célula executada com sucesso e figura salva (`fig_P1_ian.png`)
- [ ] Insight documentado na célula de conclusão

### 1.2 Pergunta 2 – IDA (Desempenho Acadêmico)
- [x] Código criado: IDA por Fase (barras+erro) e IDA por Pedra — 2026-03-02
- [ ] Célula executada com sucesso e figura salva (`fig_P2_ida.png`)
- [ ] Insight documentado

### 1.3 Pergunta 3 – IEG (Engajamento)
- [x] Código criado: scatter IEG×IDA e IEG×IPV com regressão linear — 2026-03-02
- [ ] Correlações calculadas e impressas (r e p-value)
- [ ] Figura salva (`fig_P3_ieg.png`)

### 1.4 Pergunta 4 – IAA (Autoavaliação)
- [x] Código criado: scatter IAA×IDA e IAA×IEG — 2026-03-02
- [ ] Figura salva (`fig_P4_iaa.png`)

### 1.5 Pergunta 5 – IPS (Aspectos Psicossociais)
- [x] Código criado: IDA/IEG por quartil de IPS + mapa de calor de correlações — 2026-03-02
- [ ] Figura salva (`fig_P5_ips.png`)
- [ ] Mapa de calor inclui: IPS, IDA, IEG, IAA, IPV, INDE 22

### 1.6 Pergunta 6 – IPP (Psicopedagógico vs IAN)
- [x] Código criado: boxplot IPP por nível de defasagem + scatter IAN×IPP — 2026-03-02
- [ ] Figura salva (`fig_P6_ipp.png`)
- [ ] Correlação calculada e interpretada

### 1.7 Pergunta 7 – IPV (Ponto de Virada)
- [x] Código criado: barras horizontais de correlação com IPV — 2026-03-02
- [ ] Figura salva (`fig_P7_ipv.png`)
- [ ] Top 3 preditores do IPV identificados e documentados

### 1.8 Pergunta 8 – Multidimensionalidade (Clusterização)
- [x] Código criado: Elbow Method + K-Means K=4 + PCA 2D — 2026-03-02
- [ ] Célula de perfil médio por cluster executada
- [ ] Figura salva (`fig_P8_clusters.png`)
- [ ] Nomes/descrições dos 4 perfis de alunos documentados na conclusão

### 1.9 Pergunta 10 – Efetividade do Programa
- [x] Código criado: INDE/IDA/IEG/IPS por Pedra (Quartzo→Topázio) — 2026-03-02
- [ ] Figura salva (`fig_P10_efetividade.png`)
- [ ] Progressão estatística confirmada ou refutada

### 1.10 Pergunta 11 – Insights Criativos
- [x] NLP: frequência de palavras nas recomendações dos avaliadores — 2026-03-02
- [x] Indicados para bolsa vs INDE (teste-t) — 2026-03-02
- [x] Gênero vs INDE — 2026-03-02
- [ ] Figuras salvas (`fig_P11_nlp.png`, `fig_P11_genero.png`)
- [ ] Adicionar análise de: tempo no programa (Ano ingresso → 2022) vs INDE
- [ ] Adicionar análise de: tipo de escola (pública vs privada) vs indicadores

---

## 🤖 FASE 2 – MODELO PREDITIVO (Notebook 02)

**Arquivo:** `notebooks/02_modelo_preditivo.ipynb`
**Gerado por:** `notebooks/build_model_nb.py`

### 2.1 Feature Engineering
- [x] Target `em_risco` criado (multicritério: Defas<0 OR INDE baixo OR IEG+IDA baixos) — 2026-03-02
- [x] NLP: `sent_score` — análise de sentimento lexical nas recomendações — 2026-03-02
- [x] Embedding ordinal de Pedra (`Pedra_22_num`) — 2026-03-02
- [x] Feature `Evolucao_Pedra` — 2026-03-02
- [ ] **EXECUTAR** e verificar taxa de risco no dataset (esperado: 40-60%)
- [ ] Verificar que não há data leakage (target criado apenas com dados de input)

### 2.2 Separação Treino/Teste
- [x] Código criado: `train_test_split` com `stratify=y`, 80/20 — 2026-03-02
- [x] `StandardScaler` fitado apenas no treino — 2026-03-02
- [x] `scaler.pkl` salvo em `/app` — 2026-03-02
- [ ] Verificar balanceamento no treino e no teste após split

### 2.3 Arquitetura MLP
- [x] Arquitetura definida: `Input(11)→Dense(128,ReLU)→BN→Dropout(0.3)→Dense(64)→BN→Dropout(0.3)→Dense(32)→Dropout(0.2)→Sigmoid` — 2026-03-02
- [x] Callbacks: EarlyStopping (Recall), ReduceLROnPlateau — 2026-03-02
- [x] Class weights para balancear classes — 2026-03-02
- [x] Fallback `sklearn.MLPClassifier` quando TensorFlow ausente — 2026-03-02
- [ ] **TREINAR** o modelo (requer TensorFlow instalado)
- [ ] Curva de aprendizado gerada e salva (`fig_learning_curve.png`)

### 2.4 Avaliação
- [x] Código criado: Matriz de Confusão, Curva ROC, Curva Precisão-Recall — 2026-03-02
- [x] Threshold ajustável (padrão: 0.35, favorecendo Recall) — 2026-03-02
- [x] Importância de features por permutação — 2026-03-02
- [ ] **EXECUTAR** avaliação e documentar métricas finais:
  - ROC-AUC esperado: > 0.75
  - Recall esperado: > 0.80 (prioridade: não perder alunos em risco)
- [ ] Figura salva (`fig_model_eval.png`)
- [ ] Figura salva (`fig_feature_importance.png`)

### 2.5 Salvamento do Modelo
- [x] Código criado para salvar `modelo_risco.h5` (TF) ou `.pkl` (sklearn) — 2026-03-02
- [x] `modelo_meta.pkl` com features, threshold e ROC-AUC — 2026-03-02
- [ ] Arquivos gerados e presentes em `/app/`
- [ ] Verificar que o Streamlit carrega os arquivos corretamente

---

## 🚀 FASE 3 – APLICAÇÃO STREAMLIT

**Arquivo:** `app/app.py`

### 3.1 Estrutura do App
- [x] 3 páginas: Predição Individual | Análise da Turma | Sobre o Projeto — 2026-03-02
- [x] Sidebar com navegação e tema visual (cores Passos Mágicos) — 2026-03-02
- [x] CSS customizado para cards de risco (verde/amarelo/vermelho) — 2026-03-02

### 3.2 Página 1 – Predição Individual
- [x] Formulário com todos os 11 indicadores (sliders) — 2026-03-02
- [x] Cálculo de probabilidade via modelo treinado — 2026-03-02
- [x] Gauge chart visual (velocímetro) — 2026-03-02
- [x] Nível de risco: Baixo / Médio / Alto com CSS diferenciado — 2026-03-02
- [x] Recomendações automáticas por nível de risco — 2026-03-02
- [x] Top 3 fatores de risco com barra de progresso — 2026-03-02
- [x] Fallback heurístico quando modelo não está disponível — 2026-03-02
- [ ] **TESTAR** com valores reais de alunos do dataset
- [ ] Verificar que o gauge renderiza corretamente
- [ ] Verificar recomendações para cada nível (Baixo/Médio/Alto)

### 3.3 Página 2 – Análise da Turma
- [x] Upload de CSV — 2026-03-02
- [x] Predições em lote para toda a turma — 2026-03-02
- [x] Resumo por nível de risco (cards de métricas) — 2026-03-02
- [x] Tabela de alunos em Risco Alto — 2026-03-02
- [x] Download do resultado com probabilidades (.csv) — 2026-03-02
- [ ] **TESTAR** upload com o CSV original do datathon
- [ ] Verificar mapeamento de colunas quando faltam features no upload

### 3.4 Página 3 – Sobre o Projeto
- [x] Descrição da Passos Mágicos — 2026-03-02
- [x] Tabela de tecnologias — 2026-03-02
- [x] Indicadores do modelo — 2026-03-02
- [x] Métrica ROC-AUC dinâmica (lida do modelo) — 2026-03-02

### 3.5 Configuração e Tema
- [x] `.streamlit/config.toml` com cores da identidade visual — 2026-03-02
- [ ] Adicionar logo/imagem da Passos Mágicos (se disponível)
- [ ] Testar layout em tela pequena (responsividade)

---

## ☁️ FASE 4 – DEPLOY (Streamlit Community Cloud)

- [ ] Repositório GitHub com todos os arquivos commitados
- [ ] Verificar que `requirements.txt` está correto para o Cloud
  - Remover `pickle5` se Python ≥ 3.8
  - Confirmar versões compatíveis com Streamlit Cloud
- [ ] Acessar https://share.streamlit.io → New app
- [ ] Configurar: repo=`DELSOBRINHO/passos_magicos`, file=`app/app.py`
- [ ] Aguardar build e verificar logs de erro
- [ ] Anotar URL pública da aplicação: `https://________.streamlit.app`
- [ ] Testar app público no celular e no computador

---

## 📊 FASE 5 – APRESENTAÇÃO (STORYTELLING)

**Formato:** PPT ou PDF | Foco: impacto social, menos código, mais insights

### Roteiro sugerido (slides)
- [ ] Slide 1 – Capa: Missão Passos Mágicos + problema da defasagem
- [ ] Slide 2 – Quem são os alunos? (perfil demográfico e distribuição por fase)
- [ ] Slide 3 – P1: Perfil de defasagem (IAN) — quantos e quão graves
- [ ] Slide 4 – P2+P3: Desempenho e Engajamento — tendências e correlações
- [ ] Slide 5 – P4+P5: Autoavaliação e Psicossocial — o aluno sabe quem ele é?
- [ ] Slide 6 – P6+P7: Psicopedagógico e Ponto de Virada — o que muda tudo
- [ ] Slide 7 – P8: Os 4 Perfis de Alunos (Clusters) — quem são eles?
- [ ] Slide 8 – P10: Efetividade — o programa funciona? Quartzo → Topázio
- [ ] Slide 9 – P11: Insights extras (NLP + gênero + bolsas)
- [ ] Slide 10 – P9: O Modelo Preditivo — como funciona e quão confiável é
- [ ] Slide 11 – Demo: A ferramenta Streamlit em ação
- [ ] Slide 12 – Recomendações práticas para a Associação
- [ ] Slide 13 – Conclusão e próximos passos

---

## 🎬 FASE 6 – VÍDEO (até 5 minutos)

### Roteiro (conforme plano mestre)
- [ ] Min 0:00–1:00 – Introdução à missão e o problema da defasagem
- [ ] Min 1:00–2:00 – Insights principais da análise 2022 (gráficos do EDA)
- [ ] Min 2:00–3:00 – Como o Modelo Preditivo identifica o risco
- [ ] Min 3:00–4:00 – Demonstração ao vivo do Streamlit
- [ ] Min 4:00–5:00 – Conclusão e recomendações práticas

### Produção
- [ ] Definir membros do grupo que aparecerão no vídeo
- [ ] Preparar slides/tela para screencast
- [ ] Gravar, editar e exportar (MP4)
- [ ] Hospedar (YouTube/Drive) e anotar link

---

## ✅ CHECKLIST FINAL DE ENTREGA

| # | Entregável | Status | Link/Arquivo |
|---|-----------|--------|-------------|
| 1 | Link GitHub com código | ⏳ Pendente | https://github.com/DELSOBRINHO/passos_magicos |
| 2 | Notebook EDA (limpeza + análise) | ⏳ Pendente execução | `notebooks/01_analise_exploratoria.ipynb` |
| 3 | Notebook ML (modelo preditivo) | ⏳ Pendente execução | `notebooks/02_modelo_preditivo.ipynb` |
| 4 | App Streamlit com deploy | ⏳ Pendente deploy | `app/app.py` |
| 5 | Apresentação storytelling (PPT/PDF) | ❌ Não iniciado | — |
| 6 | Vídeo até 5 minutos | ❌ Não iniciado | — |

**Legenda:** ✅ Concluído | ⏳ Em andamento/pendente execução | ❌ Não iniciado

---

## 🔧 DECISÕES TÉCNICAS REGISTRADAS

> Decisões tomadas e justificadas — **não alterar sem discussão prévia**

| Decisão | Justificativa |
|---------|--------------|
| Encoding `utf-8-sig`, sep=`,` | Único que parseia o CSV corretamente (testado) |
| IPP = (Cf + Ct) / 2 | Não há coluna IPP direta; Cf=critério funcional, Ct=técnico |
| Target `em_risco` multicritério | Mais robusto que critério único; prioriza sensibilidade |
| Threshold 0.35 (vs 0.5 padrão) | Prioriza Recall — não perder alunos em risco |
| K=4 clusters | Elbow method confirmou (validar após execução) |
| Fallback sklearn MLP | TensorFlow não instalado localmente — fallback garante portabilidade |
| Pedra ordinal 1-4 | Hierarquia natural: Quartzo < Ágata < Ametista < Topázio |
| Figuras salvas em `/data/` | Evita conflito com arquivos do modelo em `/app/` |

---

## 🚨 RESTRIÇÕES DE ESCOPO

> Itens **fora do escopo** — não implementar sem aprovação explícita

- ❌ Banco de dados externo (SQLite, PostgreSQL, etc.)
- ❌ Autenticação de usuários no Streamlit
- ❌ Modelos de Linguagem (GPT, LLM) — apenas NLP lexical simples
- ❌ Dados de anos além de 2022 (o CSV cobre apenas 2022, com histórico de pedras 2020/2021)
- ❌ Redes convolucionais ou LSTM (dados tabulares, não sequências longas)
- ❌ Deploy em cloud própria (apenas Streamlit Community Cloud)
- ❌ Criação de banco de dados com novos alunos pelo app

---

## 📅 CRONOGRAMA

| Semana | Período | Meta |
|--------|---------|------|
| Semana 1 | 02/03 – 08/03 | Estrutura + EDA completo executado + GitHub com push |
| Semana 2 | 09/03 – 15/03 | Modelo treinado + Streamlit testado + Deploy online |
| Semana 3 | 16/03 – 22/03 | Apresentação PPT + Vídeo + Revisão final |
| **ENTREGA** | **22/03/2026** | Todos os 6 entregáveis completos |

---

## 📝 LOG DE ITERAÇÕES

| Data | Sessão | O que foi feito |
|------|--------|----------------|
| 2026-03-02 | Sessão 1 | Criação da estrutura completa: pastas, notebooks (.ipynb), app Streamlit, requirements.txt, README.md, .streamlit/config.toml |
| 2026-03-02 | Sessão 1 | Instalação: seaborn, nbformat, notebook, streamlit |
| 2026-03-02 | Sessão 1 | Exploração do dataset: 860×42, encoding confirmado, colunas mapeadas |
| 2026-03-02 | Sessão 2 | Criação da pasta documentation e PLANO_DESENVOLVIMENTO.md |
| 2026-03-02 | Sessão 2 | Criação do `.gitignore` |
| 2026-03-02 | Sessão 2 | Separação requirements.txt (app/Cloud) e requirements-dev.txt (dev) |
| 2026-03-02 | Sessão 2 | Criação do `.devcontainer/Dockerfile` e `devcontainer.json` |
| 2026-03-02 | Sessão 2 | Primeiro commit e push → https://github.com/DELSOBRINHO/passos_magicos |
| 2026-03-03 | Sessão 3 | Instalação: `tensorflow-cpu` no ambiente de desenvolvimento; atualização de `requirements-dev.txt` e commit. |

