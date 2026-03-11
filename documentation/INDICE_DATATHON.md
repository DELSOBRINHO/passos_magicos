# 📚 Índice do Datathon — Passos Mágicos
## Mapeamento: O que pede × Matérias envolvidas × Como a solução resolve

> **Curso:** FIAP Postech – Data Analytics | **Fase 5:** Deep Learning & Unstructured Data  
> **Prazo:** 22/03/2026 | **Repositório:** https://github.com/DELSOBRINHO/passos_magicos

---

## 🗂️ Parte A — Entregáveis Obrigatórios

### ✅ E1 · Código no GitHub
**O que pede:** Repositório público com todo o código de limpeza e análise de dados.  
**Matérias envolvidas:** Fundamentos de Data Analytics *(coleta, organização e versionamento de dados)*.  
**Como a solução resolve:** Repositório `DELSOBRINHO/passos_magicos` com estrutura versionada em `main`: pasta `/data` (CSV original), `/notebooks` (EDA + modelo), `/app` (Streamlit). Controle de versão com **Git** *(sistema distribuído que rastreia alterações do código-fonte)*.

---

### ✅ E2 · Notebook Python — Modelo Preditivo
**O que pede:** Notebook demonstrando: feature engineering → treino/teste → modelagem → avaliação.  
**Matérias envolvidas:**
- **Deep & Reinforcement Learning – Aula 1: Perceptron de Múltiplas Camadas** *(rede neural com camadas ocultas que aprende representações não-lineares dos dados)*
- **Dados Gerados por Humanos – Aula 2: Processamento de Texto** *(transformação de texto bruto em variáveis numéricas utilizáveis)*
- **Dados Gerados por Humanos – Aula 3: Análise de Sentimentos** *(classificação do tom emocional de um texto em escores positivos/negativos)*

**Como a solução resolve:**  
`notebooks/02_modelo_preditivo.ipynb` executa o pipeline completo:
- **Feature Engineering:** criação de `sent_score` (sentimento das avaliações dos professores via léxico), `Evolucao_Pedra`, `IPP = (Cf+Ct)/2`, encoding ordinal das Pedras (Quartzo=1…Topázio=4)
- **Treino/Teste:** `train_test_split` *(divisão estratificada que garante proporção do target em ambas as partições)* 80/20 com `stratify=y`; `StandardScaler` *(normalização z-score para equalizar escalas)* fitado somente no treino (evita data leakage)
- **Modelo:** `LogisticRegression` + arquitetura **MLP** *(rede densa: Input→Dense(128,ReLU)→BN→Dropout→Dense(64)→Sigmoid)* com fallback para `sklearn.MLPClassifier`
- **Avaliação:** ROC-AUC = **0.996**, Recall = **1.0** no conjunto de teste; threshold ajustado a 0.35 para maximizar sensibilidade (não perder alunos em risco)

---

### ✅ E3 · Aplicação Streamlit + Deploy
**O que pede:** App com modelo treinado disponível online via Streamlit Community Cloud.  
**Matérias envolvidas:** Engenharia de Software Aplicada a Dados *(empacotamento e exposição de modelos como serviços interativos)*.  
**Como a solução resolve:** `app/app.py` — 3 páginas:
1. **Predição Individual:** formulário com 22 indicadores → gauge de risco + recomendações automáticas
2. **Análise da Turma:** upload CSV → predições em lote → download de resultado
3. **Sobre o Projeto:** métricas do modelo (AUC, CV) e tecnologias  

Deploy: `requirements.txt` separado do dev; modelo servido via **joblib** *(serialização eficiente de objetos Python com arrays NumPy)*.

---

### ✅ E4 · Apresentação Storytelling (PPT/PDF)
**O que pede:** Narrativa gerencial contando a história dos dados com impacto social.  
**Matérias envolvidas:** Comunicação de Dados *(transformação de análises técnicas em narrativas visuais para tomadores de decisão)*.  
**Como a solução resolve:** Roteiro de 13 slides planejados no `PLANO_DESENVOLVIMENTO.md`: perfil demográfico → indicadores → clusters → modelo preditivo → recomendações práticas. Gráficos gerados pelo Notebook 01 (`fig_P*.png`) alimentam os slides.

---

### ✅ E5 · Vídeo (até 5 minutos)
**O que pede:** Apresentação em vídeo com storytelling e demonstração do modelo.  
**Como a solução resolve:** Roteiro estruturado em 5 blocos de 1 min: missão → insights EDA → modelo → demo Streamlit → conclusão.

---

## 🔍 Parte B — 11 Perguntas Analíticas

### P1 · Adequação do Nível — IAN (Defasagem)
**O que pede:** Perfil geral da defasagem dos alunos e sua evolução.  
**Matérias:** Análise Exploratória de Dados *(descrição estatística e visualização da distribuição de variáveis)*.  
**Como resolve:** Histograma do IAN + gráfico de pizza por categoria (`Sem/Leve/Moderada/Severa`). Variável `Nivel_Defasagem` criada no pré-processamento. **⚠ Nota técnica:** `ian_num` removido do modelo preditivo pois tinha correlação de Pearson = 0.983 com o target `em_risco` — configurando **data leakage** *(quando uma feature codifica diretamente o que se quer prever, inflando artificialmente as métricas)*.

---

### P2 · Desempenho Acadêmico — IDA
**O que pede:** IDA médio está melhorando, estagnado ou caindo por fase/ano?  
**Matérias:** Estatística Descritiva e Inferencial *(médias, intervalos de confiança, comparação de grupos)*.  
**Como resolve:** Gráfico de barras com erro padrão do IDA por Fase (1–8) e por Pedra. Tendência de progressão avaliada com correlação de Spearman *(medida de monotonia que não assume distribuição normal)*.

---

### P3 · Engajamento — IEG
**O que pede:** IEG tem relação direta com IDA e IPV?  
**Matérias:** Análise de Correlação *(quantificação da força e direção da relação linear entre duas variáveis)*.  
**Como resolve:** Scatter plots IEG × IDA e IEG × IPV com linha de regressão linear *(ajuste de reta que minimiza o erro quadrático médio)*. Coeficiente de Pearson *r* e *p-value* impressos na célula de conclusão.

---

### P4 · Autoavaliação — IAA
**O que pede:** A autopercepção do aluno (IAA) é coerente com IDA e IEG?  
**Matérias:** Psicometria Aplicada a Dados *(análise quantitativa de constructos psicológicos mensurados por instrumentos)*.  
**Como resolve:** Scatter IAA × IDA e IAA × IEG. Diferença IAA − IDA calculada como medida de **viés de autoavaliação** *(tendência sistemática de superestimar ou subestimar o próprio desempenho)*.

---

### P5 · Aspectos Psicossociais — IPS
**O que pede:** Há padrões de IPS que antecedem quedas de IDA ou IEG?  
**Matérias:** Análise Multivariada *(estudo simultâneo de múltiplas variáveis para identificar padrões e interações)*.  
**Como resolve:** IDA e IEG segmentados por quartil de IPS + **mapa de calor de correlações** *(heatmap que colore a intensidade da correlação entre pares de variáveis)* cobrindo IPS, IDA, IEG, IAA, IPV, INDE.

---

### P6 · Aspectos Psicopedagógicos — IPP
**O que pede:** As avaliações psicopedagógicas (IPP) confirmam a defasagem do IAN?  
**Matérias:** Validação de Constructos *(verificação se dois instrumentos diferentes medem o mesmo fenômeno)*.  
**Como resolve:** Boxplot do IPP por nível de defasagem + scatter IAN × IPP + correlação de Pearson. IPP calculado como proxy `(Cf + Ct) / 2` *(média dos conceitos funcional e técnico quando a coluna direta não existe)*.

---

### P7 · Ponto de Virada — IPV
**O que pede:** Quais comportamentos (acadêmicos, emocionais, de engajamento) mais influenciam o IPV?  
**Matérias:** Feature Importance *(ranqueamento do poder preditivo de cada variável sobre uma variável-alvo)*.  
**Como resolve:** Barras horizontais de correlação de cada indicador com IPV (ranking). Top 3 preditores identificados e documentados. Insumo direto para o modelo preditivo (features mais associadas ao ponto de virada → proxy de recuperação).

---

### P8 · Multidimensionalidade — Clusterização
**O que pede:** Quais combinações de IDA + IEG + IPS + IPP elevam mais o INDE?  
**Matérias:**
- **Deep & Reinforcement Learning – Aula 5: Redes Não Supervisionadas** *(aprendizado sem rótulos, agrupa exemplos por similaridade)*  
- **K-Means** *(algoritmo iterativo que minimiza a distância intra-cluster atribuindo cada ponto ao centroide mais próximo)*  
- **PCA** *(Análise de Componentes Principais — reduz dimensionalidade preservando máxima variância)*

**Como resolve:** Elbow Method para definir K=4 → K-Means nos indicadores normalizados → PCA 2D para visualização dos clusters → perfil médio de cada grupo (ex.: "Aluno em crise", "Aluno em ascensão").

---

### P9 · Previsão de Risco — Machine Learning
**O que pede:** Modelo preditivo que retorna probabilidade do aluno entrar em risco de defasagem.  
**Matérias:**
- **Deep & Reinforcement Learning – Aula 1: MLP** *(rede neural multicamada que aprende padrões complexos em dados tabulares)*
- **Dados Gerados por Humanos – Aula 3: Classificação de Texto** *(atribuição de categorias a instâncias — aqui adaptado à classificação binária de risco)*

**Como resolve:**  
Pipeline completo em `notebooks/fix_leakage.py` + `02_modelo_preditivo.ipynb`:
- **22 features limpas** (sem `ian_num`) → `StandardScaler` → `LogisticRegression` *(modelo linear probabilístico baseline)* / **MLP**
- **CV AUC 5-fold = 0.986 ± 0.012** *(validação cruzada que estima a generalização sem overfitting)*; **AUC teste = 0.996**, **Recall = 1.0**
- **Threshold = 0.35** *(ponto de corte rebaixado para priorizar sensibilidade — detectar todos os alunos em risco, mesmo ao custo de falsos positivos)*
- Modelo servido no Streamlit via `joblib.load()`

---

### P10 · Efetividade do Programa
**O que pede:** Os indicadores mostram melhora consistente Quartzo → Ágata → Ametista → Topázio?  
**Matérias:** Análise de Tendência e Teste de Hipóteses *(verificação estatística se diferenças entre grupos são significativas ou aleatórias)*.  
**Como resolve:** INDE, IDA, IEG e IPS médios por Pedra em gráfico de barras progressivo. Teste ANOVA *(comparação de médias entre 3+ grupos via decomposição da variância)* ou Kruskal-Wallis *(alternativa não-paramétrica)* para confirmar significância estatística da progressão.

---

### P11 · Insights Criativos
**O que pede:** Análises adicionais além das perguntas oficiais.  
**Matérias:**
- **Dados Gerados por Humanos – Aula 2: Processamento de Texto** *(tokenização, remoção de stopwords, frequência de termos)*
- **Dados Gerados por Humanos – Aula 3: Análise de Sentimentos** *(léxico de polaridade aplicado às recomendações dos avaliadores)*
- **Dados Gerados por Humanos – Aula 4: Embeddings** *(representações vetoriais densas que capturam semântica das palavras)*

**Como resolve:**
- **NLP nas recomendações:** `sent_score` = score de sentimento lexical das avaliações dos professores (feature no modelo preditivo) + nuvem de palavras mais frequentes
- **Indicados para bolsa × INDE:** teste-t *(comparação de médias entre dois grupos independentes)* — bolsistas têm INDE superior?
- **Gênero × INDE:** comparação estatística — há disparidade de desempenho por gênero?
- **Tempo no programa:** anos desde ingresso × INDE — alunos mais antigos progridem mais?

---

## 🔗 Matriz Resumo

| Pergunta | Indicador | Módulo Principal | Técnica Central |
|----------|-----------|-----------------|-----------------|
| P1 | IAN | EDA | Histograma + Categorização |
| P2 | IDA | Estatística | Barras com IC + Spearman |
| P3 | IEG | Correlação | Scatter + Pearson r |
| P4 | IAA | Psicometria | Scatter + Viés IAA−IDA |
| P5 | IPS | Multivariada | Heatmap + Segmentação |
| P6 | IPP | Validação | Boxplot + Correlação |
| P7 | IPV | Feature Importance | Ranking de Correlação |
| P8 | INDE | Redes Não Supervisionadas (Aula 5) | K-Means + PCA |
| P9 | em_risco | MLP (Aula 1) + NLP (Aula 3) | Logistic/MLP + Threshold 0.35 |
| P10 | Pedras | Teste de Hipóteses | ANOVA / Kruskal-Wallis |
| P11 | Texto | Proc. Texto (Aula 2) + Sentimentos (Aula 3) | Léxico + Teste-t |

