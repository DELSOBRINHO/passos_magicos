"""
Passos Mágicos – Ferramenta Preditiva de Risco de Defasagem
Streamlit Community Cloud | Datathon 2025-2026
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Passos Mágicos – Risco de Defasagem",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)
 

# ─── Estilos CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {font-size:2.2rem; font-weight:800; color:#1A3A5C; margin-bottom:0}
    .sub-title  {font-size:1.1rem; color:#666; margin-bottom:1.5rem}
    .risk-alto  {background:#FDECEA; border-left:6px solid #E8562A; padding:1rem; border-radius:6px}
    .risk-medio {background:#FFF8E1; border-left:6px solid #F4A259; padding:1rem; border-radius:6px}
    .risk-baixo {background:#E8F5E9; border-left:6px solid #4CAF9A; padding:1rem; border-radius:6px}
    .metric-card{background:#F5F7FA; border-radius:10px; padding:1rem; text-align:center}
</style>
""", unsafe_allow_html=True)

# ─── Carregar modelo ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    meta_path   = os.path.join(base, 'modelo_meta.pkl')
    scaler_path = os.path.join(base, 'scaler.pkl')

    if not os.path.exists(meta_path):
        return None, None, None

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    if meta.get('use_tf', False):
        try:
            from tensorflow import keras
            model = keras.models.load_model(os.path.join(base, 'modelo_risco.h5'))
        except Exception:
            model = None
    else:
        pkl_path = os.path.join(base, 'modelo_risco.pkl')
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)

    return model, scaler, meta

model, scaler, meta = load_model()

# ─── Header ──────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("# 🌟")
with col_title:
    st.markdown('<p class="main-title">Passos Mágicos</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ferramenta Preditiva de Risco de Defasagem Educacional</p>',
                unsafe_allow_html=True)

st.markdown("---")

# ─── Sidebar: Navegação ───────────────────────────────────────────────────────
st.sidebar.image(
    "https://passosmagicos.org.br/wp-content/uploads/2020/09/logo-passos-magicos.png",
    use_container_width=True
) if False else None  # descomente se tiver acesso à imagem

pagina = st.sidebar.radio(
    "📌 Navegação",
    ["🔮 Predição Individual", "📊 Análise da Turma", "ℹ️ Sobre o Projeto"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Datathon 2025-2026**  \nFase 5 – Deep Learning & NLP")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 – PREDIÇÃO INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🔮 Predição Individual":
    st.header("🔮 Predição de Risco Individual")
    st.markdown("Insira os indicadores atuais do aluno para calcular a probabilidade de entrar em risco de defasagem.")

    with st.form("form_predicao"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📚 Indicadores Acadêmicos**")
            IDA = st.slider("IDA – Desempenho Acadêmico", 0.0, 10.0, 5.5, 0.1)
            Matem = st.slider("Matemática", 0.0, 10.0, 5.0, 0.1)
            Portug = st.slider("Português", 0.0, 10.0, 5.0, 0.1)
        with c2:
            st.markdown("**🧠 Indicadores Comportamentais**")
            IEG = st.slider("IEG – Engajamento", 0.0, 10.0, 6.0, 0.1)
            IAA = st.slider("IAA – Autoavaliação", 0.0, 10.0, 7.0, 0.1)
            IPS = st.slider("IPS – Aspectos Psicossociais", 0.0, 10.0, 6.0, 0.1)
        with c3:
            st.markdown("**📋 Indicadores de Nível**")
            IPV = st.slider("IPV – Ponto de Virada", 0.0, 10.0, 6.5, 0.1)
            IAN = st.slider("IAN – Adequação ao Nível", 0.0, 10.0, 7.5, 0.5)
            IPP = st.slider("IPP – Psicopedagógico (Cf+Ct)/2", 0.0, 10.0, 5.0, 0.5)

        c4, c5 = st.columns(2)
        with c4:
            pedra_atual = st.selectbox("🪨 Pedra Atual (2022)", ['Quartzo','Ágata','Ametista','Topázio'])
            pedra_ant   = st.selectbox("🪨 Pedra Anterior (2021)", ['Quartzo','Ágata','Ametista','Topázio'])
        with c5:
            fase = st.slider("Fase (1-8)", 1, 8, 5)
            sent_score = st.slider("💬 Score de Sentimento dos Avaliadores", -5, 5, 0)

        submitted = st.form_submit_button("🔍 Calcular Risco", use_container_width=True, type="primary")

    if submitted:
        pedra_map = {'Quartzo':1,'Ágata':2,'Ametista':3,'Topázio':4}
        p22 = pedra_map[pedra_atual]
        p21 = pedra_map[pedra_ant]
        evolucao = p22 - p21

        entrada = np.array([[IAA, IEG, IPS, IDA, IPV, IAN, IPP, p22, evolucao, sent_score, fase]])

        if model is not None and scaler is not None:
            entrada_sc = scaler.transform(entrada)
            if meta.get('use_tf', False):
                prob = float(model.predict(entrada_sc)[0][0])
            else:
                prob = float(model.predict_proba(entrada_sc)[0][1])
        else:
            # Fallback heurístico quando modelo ainda não foi treinado
            score = (10 - IDA)*0.25 + (10 - IEG)*0.25 + (10 - IAN)*0.2 + (10 - IPS)*0.15 + (10 - IPP)*0.15
            prob = min(max(score / 10, 0), 1)

        pct = prob * 100
        threshold = meta.get('threshold', 0.35) if meta else 0.35

        st.markdown("---")
        st.subheader("📊 Resultado da Análise")

        col_gauge, col_info = st.columns([1, 2])
        with col_gauge:
            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(aspect='equal'))
            theta = np.linspace(np.pi, 0, 300)
            for i, (start, end, color) in enumerate([(0,.35,'#4CAF9A'),(.35,.65,'#F4A259'),(.65,1,'#E8562A')]):
                sl = theta[int(start*299):int(end*299)+1]
                ax.fill_between(np.cos(sl), np.sin(sl)*0, np.sin(sl), color=color, alpha=0.8)
            ang = np.pi - prob * np.pi
            ax.annotate('', xy=(np.cos(ang)*0.75, np.sin(ang)*0.75), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='black', lw=3))
            ax.text(0, -0.3, f'{pct:.1f}%', ha='center', va='center', fontsize=22, fontweight='bold')
            ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.5, 1.2); ax.axis('off')
            ax.set_title('Probabilidade de Risco', fontsize=12, fontweight='bold')
            st.pyplot(fig)

        with col_info:
            if prob >= 0.65:
                nivel = "ALTO"
                css = "risk-alto"
                emoji = "🔴"
                recomendacao = (
                    "**Ação imediata recomendada:**\n\n"
                    "- 🧠 Encaminhar para avaliação psicopedagógica urgente\n"
                    "- 📞 Contato com responsáveis esta semana\n"
                    "- 📖 Plano de reforço acadêmico personalizado\n"
                    "- 💙 Apoio psicossocial – verificar contexto familiar"
                )
            elif prob >= threshold:
                nivel = "MÉDIO"
                css = "risk-medio"
                emoji = "🟡"
                recomendacao = (
                    "**Monitoramento intensificado:**\n\n"
                    "- 📊 Acompanhar indicadores semanalmente\n"
                    "- 🎯 Reforçar engajamento em atividades extracurriculares\n"
                    "- 💬 Conversa individual com o aluno sobre motivação\n"
                    "- 👀 Observar IPS nas próximas semanas"
                )
            else:
                nivel = "BAIXO"
                css = "risk-baixo"
                emoji = "🟢"
                recomendacao = (
                    "**Manter e potencializar:**\n\n"
                    "- ✅ Aluno apresenta bom desempenho geral\n"
                    "- 🏆 Avaliar indicação para bolsa ou destaque\n"
                    "- 🚀 Desafiar com atividades de liderança\n"
                    "- 📈 Manter monitoramento mensal regular"
                )

            st.markdown(f'<div class="{css}">', unsafe_allow_html=True)
            st.markdown(f"## {emoji} Risco {nivel} — {pct:.1f}%")
            st.markdown(recomendacao)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("\n**Fatores que mais influenciaram esta predição:**")
            fatores = {
                'Engajamento (IEG)': 10 - IEG,
                'Desempenho (IDA)': 10 - IDA,
                'Sentimento Avaliadores': -sent_score,
                'Psicossocial (IPS)': 10 - IPS,
                'Ponto de Virada (IPV)': 10 - IPV
            }
            for fator, val in sorted(fatores.items(), key=lambda x: x[1], reverse=True)[:3]:
                bar_pct = min(max(val/10, 0), 1)
                cor = '#E8562A' if bar_pct > 0.5 else '#F4A259'
                st.markdown(f"**{fator}**: `{val:.1f}/10 de fator de risco`")
                st.progress(bar_pct)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 – ANÁLISE DA TURMA
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Análise da Turma":
    st.header("📊 Análise da Turma – Upload de Dados")
    st.markdown("Faça upload de uma planilha CSV com os indicadores dos alunos para análise coletiva.")

    uploaded = st.file_uploader("📂 Carregar CSV da turma", type=["csv"])

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded, encoding='utf-8-sig', sep=',')
            cols_float = ['IAA','IEG','IPS','IDA','IPV','IAN','INDE 22']
            for c in cols_float:
                if c in df_up.columns:
                    df_up[c] = pd.to_numeric(df_up[c].astype(str).str.replace(',','.'), errors='coerce')

            st.success(f"✅ {len(df_up)} alunos carregados!")
            st.dataframe(df_up.head(10))

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            indicadores_disp = [c for c in ['INDE 22','IDA','IEG','IPS'] if c in df_up.columns]
            metricas = [col_m1, col_m2, col_m3, col_m4]
            for i, ind in enumerate(indicadores_disp[:4]):
                metricas[i].metric(f"Média {ind}", f"{df_up[ind].mean():.2f}")

            if all(c in df_up.columns for c in ['IAA','IEG','IPS','IDA','IPV','IAN']):
                pedra_map = {'Quartzo':1,'Ágata':2,'Ametista':3,'Topázio':4}
                st.subheader("🔮 Predições em Lote")
                FEATURES = ['IAA','IEG','IPS','IDA','IPV','IAN','IPP','Pedra_22_num','Evolucao_Pedra','sent_score','Fase']

                if 'IPP' not in df_up.columns and 'Cf' in df_up.columns and 'Ct' in df_up.columns:
                    df_up['IPP'] = (df_up['Cf'] + df_up['Ct']) / 2
                else:
                    df_up.setdefault('IPP', 5.0)

                for col_def, default in [('Pedra_22_num', 2), ('Evolucao_Pedra', 0),
                                          ('sent_score', 0), ('Fase', 5)]:
                    if col_def not in df_up.columns:
                        df_up[col_def] = default

                feat_avail = [f for f in FEATURES if f in df_up.columns]
                X_lote = df_up[feat_avail].fillna(df_up[feat_avail].median())

                if model is not None and scaler is not None and len(feat_avail) == len(FEATURES):
                    X_sc = scaler.transform(X_lote)
                    if meta.get('use_tf', False):
                        probs = model.predict(X_sc).flatten()
                    else:
                        probs = model.predict_proba(X_sc)[:, 1]
                    df_up['Prob_Risco_%'] = (probs * 100).round(1)
                    df_up['Nivel_Risco'] = pd.cut(probs, bins=[0,.35,.65,1.0],
                                                   labels=['🟢 Baixo','🟡 Médio','🔴 Alto'])
                else:
                    # Fallback heurístico
                    df_up['Prob_Risco_%'] = (
                        (10 - df_up['IDA'].fillna(5))*0.25 +
                        (10 - df_up['IEG'].fillna(5))*0.25 +
                        (10 - df_up.get('IAN', pd.Series([7]*len(df_up))).fillna(7))*0.2 +
                        (10 - df_up.get('IPS', pd.Series([6]*len(df_up))).fillna(6))*0.15 +
                        (10 - df_up.get('IPP', pd.Series([5]*len(df_up))).fillna(5))*0.15
                    ).clip(0, 10) * 10
                    df_up['Nivel_Risco'] = pd.cut(df_up['Prob_Risco_%']/100,
                                                   bins=[0,.35,.65,1.0],
                                                   labels=['🟢 Baixo','🟡 Médio','🔴 Alto'])

                resumo = df_up['Nivel_Risco'].value_counts()
                cols_res = st.columns(3)
                for i, (nivel, cnt) in enumerate(resumo.items()):
                    cols_res[i].metric(str(nivel), f"{cnt} alunos")

                st.markdown("**📋 Alunos em RISCO ALTO (requer atenção imediata):**")
                alto_risco = df_up[df_up['Nivel_Risco'] == '🔴 Alto']
                if 'Nome' in alto_risco.columns:
                    st.dataframe(alto_risco[['Nome','Prob_Risco_%','IDA','IEG','IPS']].sort_values('Prob_Risco_%', ascending=False))
                else:
                    st.dataframe(alto_risco[['Prob_Risco_%','IDA','IEG','IPS']].sort_values('Prob_Risco_%', ascending=False))

                csv_out = df_up.to_csv(index=False, sep=';').encode('utf-8-sig')
                st.download_button("⬇️ Baixar resultado completo (.csv)", csv_out,
                                   "resultado_risco_turma.csv", "text/csv")
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")
    else:
        st.info("👆 Carregue o arquivo CSV com os indicadores da turma.")
        st.markdown("**Colunas esperadas:** `IAA, IEG, IPS, IDA, IPV, IAN, Fase, Pedra 22`")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 – SOBRE O PROJETO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "ℹ️ Sobre o Projeto":
    st.header("ℹ️ Sobre o Projeto")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### 🌟 Associação Passos Mágicos
Fundada em 1992 por Michelle Flues e Dimetri Ivanoff, a Passos Mágicos transforma
a vida de crianças e jovens de baixa renda através da **educação de qualidade**,
**apoio psicossocial** e **ampliação do mundo**.

### 🎯 Objetivo desta Ferramenta
Identificar **preventivamente** alunos em risco de defasagem antes que a queda
de desempenho se torne irreversível — permitindo intervenções precisas e rápidas.
        """)
    with col2:
        st.markdown("""
### 🤖 Tecnologias Utilizadas
| Componente | Tecnologia |
|-----------|-----------|
| Análise exploratória | Pandas, Seaborn, Scipy |
| Clusterização | K-Means + PCA |
| NLP | Análise de sentimento lexical |
| Modelo preditivo | MLP (Keras/TensorFlow) |
| Interface | Streamlit |
| Deploy | Streamlit Community Cloud |

### 📊 Indicadores do Modelo
`IAA · IEG · IPS · IDA · IPV · IAN · IPP · Pedra · Sentimento NLP`
        """)
    st.markdown("---")
    st.markdown("**Datathon 2025-2026 | FIAP Postech – Data Analytics**")
    if meta:
        st.metric("ROC-AUC do Modelo", f"{meta.get('roc_auc', 'N/A'):.4f}" if isinstance(meta.get('roc_auc'), float) else "Modelo não treinado")


