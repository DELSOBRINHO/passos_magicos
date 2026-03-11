"""
Passos Mágicos – Ferramenta Preditiva de Risco de Defasagem
Streamlit Community Cloud | Datathon 2025-2026
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
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
    import json
    base = os.path.dirname(__file__)
    # Prioriza o modelo limpo (sem leakage); faz fallback para o refinado
    candidates = [
        ('modelo_risco_clean.pkl', 'scaler_clean.pkl', 'modelo_meta_clean.json'),
        ('modelo_risco_refined.pkl', 'scaler.pkl',      'modelo_meta_refined.json'),
        ('modelo_risco.pkl',         'scaler.pkl',      None),
    ]
    for model_file, scaler_file, meta_file in candidates:
        model_path  = os.path.join(base, model_file)
        scaler_path = os.path.join(base, scaler_file)
        if not os.path.exists(model_path):
            continue
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        meta = {}
        if meta_file:
            mp = os.path.join(base, meta_file)
            if mp.endswith('.json') and os.path.exists(mp):
                with open(mp, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            elif mp.endswith('.pkl') and os.path.exists(mp):
                meta = joblib.load(mp)
        return model, scaler, meta
    return None, None, {}

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

    # Features do modelo limpo (22 features, sem ian_num)
    FEATURES = meta.get('features', [
        'fase','ano_nasc','idade_22','ano_ingresso','cf','ct','no_av',
        'pedra_22_num','ipp','pedra_21_num','evolucao_pedra','sent_len','sent_score',
        'cg_num','iaa_num','ieg_num','ips_num','ida_num',
        'matem_num','portug_num','ingles_num','ipv_num'
    ])
    pedra_map = {'Quartzo':1,'Ágata':2,'Ametista':3,'Topázio':4}

    with st.form("form_predicao"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📚 Indicadores Acadêmicos**")
            ida_num   = st.slider("IDA – Desempenho Acadêmico", 0.0, 10.0, 5.5, 0.1)
            matem_num = st.slider("Matemática", 0.0, 10.0, 5.0, 0.1)
            portug_num= st.slider("Português",  0.0, 10.0, 5.0, 0.1)
            ingles_num= st.slider("Inglês",     0.0, 10.0, 5.0, 0.1)
        with c2:
            st.markdown("**🧠 Indicadores Comportamentais**")
            ieg_num = st.slider("IEG – Engajamento",           0.0, 10.0, 6.0, 0.1)
            iaa_num = st.slider("IAA – Autoavaliação",         0.0, 10.0, 7.0, 0.1)
            ips_num = st.slider("IPS – Aspectos Psicossociais",0.0, 10.0, 6.0, 0.1)
            ipv_num = st.slider("IPV – Ponto de Virada",       0.0, 10.0, 6.5, 0.1)
        with c3:
            st.markdown("**📋 Indicadores de Nível e Conceito**")
            cf     = st.slider("Cf – Conceito Final",   0.0, 10.0, 5.0, 0.5)
            ct     = st.slider("Ct – Conceito Total",   0.0, 10.0, 5.0, 0.5)
            cg_num = st.slider("Cg – Conceito Global",  0.0, 10.0, 5.0, 0.5)
            no_av  = st.slider("Nº de Avaliações",      1,   4,    3)

        c4, c5 = st.columns(2)
        with c4:
            pedra_atual = st.selectbox("🪨 Pedra Atual (2022)",    ['Quartzo','Ágata','Ametista','Topázio'])
            pedra_ant   = st.selectbox("🪨 Pedra Anterior (2021)", ['Quartzo','Ágata','Ametista','Topázio'])
            fase        = st.slider("Fase (1-8)", 1, 8, 5)
        with c5:
            ano_nasc     = st.number_input("Ano de Nascimento", 2000, 2020, 2010, 1)
            ano_ingresso = st.number_input("Ano de Ingresso",   2014, 2022, 2018, 1)
            sent_score   = st.slider("💬 Score de Sentimento dos Avaliadores", -5, 5, 0)
            sent_len     = st.slider("💬 Comprimento do Texto de Avaliação",    0, 500, 150, 10)

        submitted = st.form_submit_button("🔍 Calcular Risco", use_container_width=True, type="primary")

    if submitted:
        p22      = pedra_map[pedra_atual]
        p21      = pedra_map[pedra_ant]
        evolucao = p22 - p21
        ipp      = (cf + ct) / 2
        idade_22 = 2022 - ano_nasc

        # Montar vetor de entrada na ordem exata das features do modelo
        feature_vals = {
            'fase': fase, 'ano_nasc': ano_nasc, 'idade_22': idade_22,
            'ano_ingresso': ano_ingresso, 'cf': cf, 'ct': ct, 'no_av': no_av,
            'pedra_22_num': p22, 'ipp': ipp, 'pedra_21_num': p21,
            'evolucao_pedra': evolucao, 'sent_len': sent_len, 'sent_score': sent_score,
            'cg_num': cg_num, 'iaa_num': iaa_num, 'ieg_num': ieg_num,
            'ips_num': ips_num, 'ida_num': ida_num, 'matem_num': matem_num,
            'portug_num': portug_num, 'ingles_num': ingles_num, 'ipv_num': ipv_num,
        }
        entrada = np.array([[feature_vals.get(f, 0.0) for f in FEATURES]])

        if model is not None and scaler is not None:
            entrada_sc = scaler.transform(entrada)
            prob = float(model.predict_proba(entrada_sc)[0][1])
        else:
            # Fallback heurístico quando modelo ainda não foi treinado
            score = (10 - ida_num)*0.25 + (10 - ieg_num)*0.25 + (10 - ipp)*0.2 + (10 - ips_num)*0.15 + (10 - cg_num)*0.15
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
                'Engajamento (IEG)': 10 - ieg_num,
                'Desempenho (IDA)': 10 - ida_num,
                'Sentimento Avaliadores': -sent_score,
                'Psicossocial (IPS)': 10 - ips_num,
                'Ponto de Virada (IPV)': 10 - ipv_num,
            }
            for fator, val in sorted(fatores.items(), key=lambda x: x[1], reverse=True)[:3]:
                bar_pct = min(max(val/10, 0), 1)
                st.markdown(f"**{fator}**: `{val:.1f}/10 de fator de risco`")
                st.progress(bar_pct)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 – ANÁLISE DA TURMA
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Análise da Turma":
    st.header("📊 Análise da Turma – Upload de Dados")
    st.markdown("Faça upload de uma planilha CSV com os indicadores dos alunos para análise coletiva.")

    uploaded = st.file_uploader("📂 Carregar CSV da turma", type=["csv"])

    BATCH_FEATURES = meta.get('features', [
        'fase','ano_nasc','idade_22','ano_ingresso','cf','ct','no_av',
        'pedra_22_num','ipp','pedra_21_num','evolucao_pedra','sent_len','sent_score',
        'cg_num','iaa_num','ieg_num','ips_num','ida_num',
        'matem_num','portug_num','ingles_num','ipv_num'
    ])

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded, encoding='utf-8-sig', sep=',')
            for c in df_up.columns:
                df_up[c] = pd.to_numeric(df_up[c].astype(str).str.replace(',','.'), errors='ignore')

            st.success(f"✅ {len(df_up)} alunos carregados!")
            st.dataframe(df_up.head(10))

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            indicadores_disp = [c for c in ['ida_num','ieg_num','ips_num','iaa_num'] if c in df_up.columns]
            metricas = [col_m1, col_m2, col_m3, col_m4]
            for i, ind in enumerate(indicadores_disp[:4]):
                metricas[i].metric(f"Média {ind}", f"{df_up[ind].mean():.2f}")

            st.subheader("🔮 Predições em Lote")

            # Derivar colunas calculadas se ausentes
            if 'ipp' not in df_up.columns and 'cf' in df_up.columns and 'ct' in df_up.columns:
                df_up['ipp'] = (df_up['cf'] + df_up['ct']) / 2
            if 'evolucao_pedra' not in df_up.columns and 'pedra_22_num' in df_up.columns and 'pedra_21_num' in df_up.columns:
                df_up['evolucao_pedra'] = df_up['pedra_22_num'] - df_up['pedra_21_num']
            if 'idade_22' not in df_up.columns and 'ano_nasc' in df_up.columns:
                df_up['idade_22'] = 2022 - df_up['ano_nasc']

            defaults = {'fase':5,'pedra_22_num':2,'pedra_21_num':2,'evolucao_pedra':0,
                        'sent_score':0,'sent_len':150,'ipp':5.0,'no_av':3,
                        'ano_nasc':2010,'idade_22':12,'ano_ingresso':2018,'cg_num':5.0}
            for col, default in defaults.items():
                if col not in df_up.columns:
                    df_up[col] = default

            feat_avail = [f for f in BATCH_FEATURES if f in df_up.columns]
            X_lote = df_up[feat_avail].fillna(df_up[feat_avail].median(numeric_only=True))

            if model is not None and scaler is not None and len(feat_avail) == len(BATCH_FEATURES):
                X_sc = scaler.transform(X_lote)
                probs = model.predict_proba(X_sc)[:, 1]
            else:
                # Fallback heurístico
                probs = (
                    (10 - df_up.get('ida_num', pd.Series([5.5]*len(df_up))))*0.25 +
                    (10 - df_up.get('ieg_num', pd.Series([6.0]*len(df_up))))*0.25 +
                    (10 - df_up.get('ips_num', pd.Series([6.0]*len(df_up))))*0.20 +
                    (10 - df_up.get('ipp',     pd.Series([5.0]*len(df_up))))*0.15 +
                    (10 - df_up.get('cg_num',  pd.Series([5.0]*len(df_up))))*0.15
                ).clip(0, 10) / 10

            df_up['Prob_Risco_%'] = (probs * 100).round(1)
            df_up['Nivel_Risco'] = pd.cut(probs, bins=[0,.35,.65,1.01],
                                           labels=['🟢 Baixo','🟡 Médio','🔴 Alto'])

            resumo = df_up['Nivel_Risco'].value_counts()
            cols_res = st.columns(3)
            for i, (nivel, cnt) in enumerate(resumo.items()):
                cols_res[i % 3].metric(str(nivel), f"{cnt} alunos")

            st.markdown("**📋 Alunos em RISCO ALTO (requer atenção imediata):**")
            alto_risco = df_up[df_up['Nivel_Risco'] == '🔴 Alto']
            cols_disp = ['Prob_Risco_%'] + [c for c in ['Nome','ida_num','ieg_num','ips_num'] if c in alto_risco.columns]
            st.dataframe(alto_risco[cols_disp].sort_values('Prob_Risco_%', ascending=False))

            csv_out = df_up.to_csv(index=False, sep=';').encode('utf-8-sig')
            st.download_button("⬇️ Baixar resultado completo (.csv)", csv_out,
                               "resultado_risco_turma.csv", "text/csv")
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")
    else:
        st.info("👆 Carregue o arquivo CSV com os indicadores da turma.")
        colunas_str = ', '.join(BATCH_FEATURES[:8]) + ', ...'
        st.markdown(f"**Colunas esperadas (modelo limpo):** `{colunas_str}`")

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

### 📊 Indicadores do Modelo (sem leakage)
`IAA · IEG · IPS · IDA · IPV · IPP · Cg · Pedra · Matemática · Português · Inglês · Sentimento NLP`
        """)
    st.markdown("---")
    st.markdown("**Datathon 2025-2026 | FIAP Postech – Data Analytics**")
    if meta:
        auc_val = meta.get('test_auc') or meta.get('roc_auc') or meta.get('auc')
        st.metric("ROC-AUC do Modelo (teste)", f"{auc_val:.4f}" if isinstance(auc_val, float) else "N/A")
        cv_val = meta.get('cv_auc_mean')
        if cv_val:
            st.metric("CV AUC (5-fold)", f"{cv_val:.4f} ± {meta.get('cv_auc_std', 0):.4f}")


