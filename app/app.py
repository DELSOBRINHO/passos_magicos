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
import unicodedata
import matplotlib.pyplot as plt
from risk_calibration import apply_probability_calibration
from ui_helpers import build_risk_progress_html, describe_risk, resolve_sentiment_values

INDIVIDUAL_DEFAULT_FEATURES = [
    'dim_academica', 'dim_psicossocial', 'dim_psicopedagogica',
    'fase', 'pedra_22_num', 'pedra_21_num', 'evolucao_pedra',
    'no_av', 'sent_score', 'sent_len'
]

INDICATOR_LABELS = {
    'ian_num': 'IAN – Adequação de Nível',
    'ida_num': 'IDA – Desempenho Acadêmico',
    'ieg_num': 'IEG – Engajamento',
    'iaa_num': 'IAA – Autoavaliação',
    'ips_num': 'IPS – Psicossocial',
    'ipp_num': 'IPP – Psicopedagógico',
    'ipv_num': 'IPV – Ponto de Virada',
}

PALAVRAS_NEG = {
    'melhorar', 'empenhar', 'dificuldade', 'atraso', 'deficit', 'atencao',
    'problema', 'risco', 'alerta', 'comportamento', 'limitacao', 'preocupa',
    'atenção', 'limitação', 'preocupação'
}
PALAVRAS_POS = {
    'destaque', 'excelente', 'promovido', 'bolsa', 'lider', 'potencial',
    'engajado', 'comprometido', 'aprovado', 'evolucao', 'crescimento',
    'líder', 'evolução'
}

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
    .risk-progress-track {width:100%; height:12px; background:rgba(26,58,92,0.10); border-radius:999px; overflow:hidden; margin:0 0 1rem 0}
    .risk-progress-fill {height:100%; border-radius:999px; transition:width 0.3s ease}
    .metric-card{background:#F5F7FA; border-radius:10px; padding:1rem; text-align:center}
</style>
""", unsafe_allow_html=True)

# ─── Carregar modelo ──────────────────────────────────────────────────────────
def get_model_artifact_signature():
    base = os.path.dirname(__file__)
    candidates = [
        ('modelo_risco_clean.pkl', 'scaler_clean.pkl', 'modelo_meta_clean.json'),
        ('modelo_risco_refined.pkl', 'scaler.pkl', 'modelo_meta_refined.json'),
        ('modelo_risco.pkl', 'scaler.pkl', None),
    ]
    signature = []
    for model_file, scaler_file, meta_file in candidates:
        for file_name in [model_file, scaler_file, meta_file]:
            if not file_name:
                continue
            path = os.path.join(base, file_name)
            exists = os.path.exists(path)
            signature.append((
                file_name,
                exists,
                os.path.getmtime(path) if exists else None,
                os.path.getsize(path) if exists else None,
            ))
    return tuple(signature)


@st.cache_resource
def load_model(artifact_signature):
    import json
    _ = artifact_signature
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

model, scaler, meta = load_model(get_model_artifact_signature())


def normalize_text(value):
    text = unicodedata.normalize('NFKD', str(value).strip().lower())
    return ''.join(ch for ch in text if not unicodedata.combining(ch))


def to_num_series(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')


def map_pedra_value(value):
    return {
        'quartzo': 1,
        'agata': 2,
        'ametista': 3,
        'topazio': 4,
    }.get(normalize_text(value), np.nan)


def dynamic_inde_weights(fase):
    fase_num = float(fase)
    if fase_num >= 8:
        return {
            'ian_num': 0.00,
            'ida_num': 0.40,
            'ieg_num': 0.20,
            'iaa_num': 0.00,
            'ips_num': 0.20,
            'ipp_num': 0.10,
            'ipv_num': 0.10,
        }
    return {
        'ian_num': 0.10,
        'ida_num': 0.20,
        'ieg_num': 0.20,
        'iaa_num': 0.10,
        'ips_num': 0.20,
        'ipp_num': 0.10,
        'ipv_num': 0.10,
    }


def compute_dimensions_and_inde(indicators, fase):
    weights = dynamic_inde_weights(fase)
    acad_den = weights['ian_num'] + weights['ida_num']
    psic_den = weights['ieg_num'] + weights['iaa_num'] + weights['ips_num']
    peda_den = weights['ipp_num'] + weights['ipv_num']

    dim_academica = (
        indicators['ian_num'] * weights['ian_num'] + indicators['ida_num'] * weights['ida_num']
    ) / acad_den
    dim_psicossocial = (
        indicators['ieg_num'] * weights['ieg_num']
        + indicators['iaa_num'] * weights['iaa_num']
        + indicators['ips_num'] * weights['ips_num']
    ) / psic_den
    dim_psicopedagogica = (
        indicators['ipp_num'] * weights['ipp_num'] + indicators['ipv_num'] * weights['ipv_num']
    ) / peda_den
    inde_calc = sum(indicators[key] * weight for key, weight in weights.items())

    return {
        'dim_academica': float(dim_academica),
        'dim_psicossocial': float(dim_psicossocial),
        'dim_psicopedagogica': float(dim_psicopedagogica),
        'inde_calc': float(inde_calc),
        'weights': weights,
    }


def reverse_rank_series(series, bounds=None):
    numeric = to_num_series(series)
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    if bounds:
        lower = bounds.get('min')
        upper = bounds.get('max')
    else:
        lower = float(valid.min())
        upper = float(valid.max())
    if lower is None or upper is None or pd.isna(lower) or pd.isna(upper) or upper == lower:
        return pd.Series(np.nan, index=series.index, dtype=float)
    score = 10 * (1 - (numeric - lower) / (upper - lower))
    return score.clip(0, 10).astype(float)


def score_sentimento(texto):
    if pd.isna(texto):
        return 0
    content = str(texto).lower()
    return sum(1 for word in PALAVRAS_POS if word in content) - sum(1 for word in PALAVRAS_NEG if word in content)


def heuristic_individual_prob(form_values):
    feature_values = build_feature_values(form_values)
    inde_gap = meta.get('inde_threshold', 6.5) - feature_values['inde_calc'] if meta else 6.5 - feature_values['inde_calc']
    prob = 1 / (1 + np.exp(-1.2 * inde_gap))
    prob += 0.04 * max(0.0, feature_values['pedra_21_num'] - feature_values['pedra_22_num'])
    prob += 0.02 * max(0.0, -feature_values['sent_score'])
    return float(np.clip(prob, 0, 1))


def build_feature_values(form_values):
    indicators = {key: float(form_values[key]) for key in INDICATOR_LABELS}
    fase = float(form_values['fase'])
    scores = compute_dimensions_and_inde(indicators, fase)
    pedra_22 = float(form_values['pedra_22_num'])
    pedra_21 = float(form_values['pedra_21_num'])
    return {
        **indicators,
        'fase': fase,
        'pedra_22_num': pedra_22,
        'pedra_21_num': pedra_21,
        'evolucao_pedra': pedra_22 - pedra_21,
        'no_av': float(form_values['no_av']),
        'sent_score': float(form_values['sent_score']),
        'sent_len': float(form_values['sent_len']),
        **scores,
    }


def build_feature_vector(feature_values, meta_obj):
    feature_names = meta_obj.get('features', INDIVIDUAL_DEFAULT_FEATURES)
    medians = meta_obj.get('imputer_medians', {})
    values = []
    for feature in feature_names:
        value = feature_values.get(feature, medians.get(feature, 0.0))
        if pd.isna(value):
            value = medians.get(feature, 0.0)
        values.append(float(value))
    return pd.DataFrame([values], columns=feature_names, dtype=float)


def predict_individual_with_model(form_values, model_obj, scaler_obj, meta_obj):
    feature_values = build_feature_values(form_values)
    entrada = build_feature_vector(feature_values, meta_obj)
    if model_obj is None or scaler_obj is None:
        return None
    if not hasattr(model_obj, 'predict_proba'):
        return None
    try:
        entrada_sc = scaler_obj.transform(entrada)
        proba = model_obj.predict_proba(entrada_sc)
    except Exception:
        return None
    if getattr(proba, 'shape', (0, 0))[1] < 2:
        return None
    return float(proba[0][1])


def model_is_usable(model_obj, scaler_obj, meta_obj):
    strong_profile = {
        'fase': 5, 'no_av': 4, 'pedra_22_num': 4, 'pedra_21_num': 3,
        'sent_score': 2, 'sent_len': 200,
        'ian_num': 8.5, 'ida_num': 9.0, 'ieg_num': 9.0, 'iaa_num': 8.5,
        'ips_num': 8.5, 'ipp_num': 8.5, 'ipv_num': 9.0,
    }
    critical_profile = {
        'fase': 5, 'no_av': 2, 'pedra_22_num': 1, 'pedra_21_num': 3,
        'sent_score': -2, 'sent_len': 80,
        'ian_num': 2.5, 'ida_num': 2.0, 'ieg_num': 2.5, 'iaa_num': 3.0,
        'ips_num': 3.0, 'ipp_num': 2.5, 'ipv_num': 2.0,
    }
    try:
        strong_prob = predict_individual_with_model(strong_profile, model_obj, scaler_obj, meta_obj)
        critical_prob = predict_individual_with_model(critical_profile, model_obj, scaler_obj, meta_obj)
    except Exception:
        return False
    if strong_prob is None or critical_prob is None:
        return False
    return critical_prob > strong_prob and (critical_prob - strong_prob) >= 0.10


def get_matching_column(df, *aliases):
    normalized = {normalize_text(col): col for col in df.columns}
    for alias in aliases:
        match = normalized.get(normalize_text(alias))
        if match:
            return match
    return None


def get_numeric_column(df, *aliases):
    col = get_matching_column(df, *aliases)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return to_num_series(df[col])


def get_pedra_column(df, *aliases):
    col = get_matching_column(df, *aliases)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[col].apply(map_pedra_value).astype(float)


def prepare_batch_features(raw_df, meta_obj):
    prepared = raw_df.copy()
    prepared['fase'] = get_numeric_column(raw_df, 'fase')

    for indicator in ['ian_num', 'ida_num', 'ieg_num', 'iaa_num', 'ips_num', 'ipv_num']:
        prepared[indicator] = get_numeric_column(raw_df, indicator, indicator.replace('_num', ''))

    prepared['ipp_num'] = get_numeric_column(raw_df, 'ipp_num', 'ipp')
    if prepared['ipp_num'].isna().all():
        bounds = (meta_obj or {}).get('ipp_proxy_bounds', {})
        cf_proxy = reverse_rank_series(get_numeric_column(raw_df, 'cf', 'Cf'), bounds.get('cf'))
        ct_proxy = reverse_rank_series(get_numeric_column(raw_df, 'ct', 'Ct'), bounds.get('ct'))
        prepared['ipp_num'] = pd.concat([cf_proxy, ct_proxy], axis=1).mean(axis=1)

    prepared['pedra_22_num'] = get_numeric_column(raw_df, 'pedra_22_num')
    if prepared['pedra_22_num'].isna().all():
        prepared['pedra_22_num'] = get_pedra_column(raw_df, 'Pedra 22')

    prepared['pedra_21_num'] = get_numeric_column(raw_df, 'pedra_21_num')
    if prepared['pedra_21_num'].isna().all():
        prepared['pedra_21_num'] = get_pedra_column(raw_df, 'Pedra 21')

    prepared['no_av'] = get_numeric_column(raw_df, 'no_av', 'Nº Av', 'N° Av', 'N Av')
    prepared['sent_score'] = get_numeric_column(raw_df, 'sent_score')
    prepared['sent_len'] = get_numeric_column(raw_df, 'sent_len')

    text_cols = [
        col for col in raw_df.columns
        if any(tag in normalize_text(col) for tag in ['rec av', 'rec ps', 'destaque'])
    ]
    if prepared['sent_score'].isna().all() and text_cols:
        prepared['sent_score'] = sum(raw_df[col].apply(score_sentimento) for col in text_cols)
    if prepared['sent_len'].isna().all() and text_cols:
        prepared['sent_len'] = sum(raw_df[col].apply(lambda x: len(str(x)) if not pd.isna(x) else 0) for col in text_cols)

    prepared['evolucao_pedra'] = prepared['pedra_22_num'] - prepared['pedra_21_num']

    fase_8 = prepared['fase'].fillna(0) >= 8
    w_ian = np.where(fase_8, 0.00, 0.10)
    w_ida = np.where(fase_8, 0.40, 0.20)
    w_ieg = np.full(len(prepared), 0.20)
    w_iaa = np.where(fase_8, 0.00, 0.10)
    w_ips = np.full(len(prepared), 0.20)
    w_ipp = np.full(len(prepared), 0.10)
    w_ipv = np.full(len(prepared), 0.10)

    prepared['dim_academica'] = (prepared['ian_num'] * w_ian + prepared['ida_num'] * w_ida) / (w_ian + w_ida)
    prepared['dim_psicossocial'] = (
        prepared['ieg_num'] * w_ieg + prepared['iaa_num'] * w_iaa + prepared['ips_num'] * w_ips
    ) / (w_ieg + w_iaa + w_ips)
    prepared['dim_psicopedagogica'] = (prepared['ipp_num'] * w_ipp + prepared['ipv_num'] * w_ipv) / (w_ipp + w_ipv)
    prepared['inde_calc'] = (
        prepared['ian_num'] * w_ian
        + prepared['ida_num'] * w_ida
        + prepared['ieg_num'] * w_ieg
        + prepared['iaa_num'] * w_iaa
        + prepared['ips_num'] * w_ips
        + prepared['ipp_num'] * w_ipp
        + prepared['ipv_num'] * w_ipv
    )

    feature_names = meta_obj.get('features', INDIVIDUAL_DEFAULT_FEATURES)
    medians = meta_obj.get('imputer_medians', {}) if meta_obj else {}
    X = pd.DataFrame(index=prepared.index)
    for feature in feature_names:
        series = pd.to_numeric(prepared.get(feature, pd.Series(np.nan, index=prepared.index)), errors='coerce')
        fill_value = medians.get(feature)
        if fill_value is None or pd.isna(fill_value):
            fill_value = float(series.median()) if series.notna().any() else 0.0
        X[feature] = series.fillna(fill_value)
    return prepared, X

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
    st.markdown(
        "Insira os 7 indicadores do INDE e os dados mínimos de contexto para calcular o "
        "**INDE dinâmico por fase** e a **probabilidade de risco**."
    )

    model_usable = model_is_usable(model, scaler, meta)
    pedra_map = {'Quartzo': 1, 'Ágata': 2, 'Ametista': 3, 'Topázio': 4}
    result_state_key = 'individual_last_inputs'

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📚 Dimensão Acadêmica**")
        ian_num = st.slider("IAN – Adequação de Nível", 0.0, 10.0, 6.0, 0.1)
        ida_num = st.slider("IDA – Desempenho Acadêmico", 0.0, 10.0, 6.0, 0.1)
    with c2:
        st.markdown("**🧠 Dimensão Psicossocial**")
        ieg_num = st.slider("IEG – Engajamento", 0.0, 10.0, 6.0, 0.1)
        iaa_num = st.slider("IAA – Autoavaliação", 0.0, 10.0, 6.5, 0.1)
        ips_num = st.slider("IPS – Psicossocial", 0.0, 10.0, 6.0, 0.1)
    with c3:
        st.markdown("**📋 Dimensão Psicopedagógica**")
        ipp_num = st.slider("IPP – Psicopedagógico", 0.0, 10.0, 6.0, 0.1)
        ipv_num = st.slider("IPV – Ponto de Virada", 0.0, 10.0, 6.5, 0.1)
        no_av = st.slider("Nº de Avaliações", 1, 6, 3)

    c4, c5 = st.columns(2)
    with c4:
        fase = st.slider("Fase do Aluno (0-8)", 0, 8, 5)
        pedra_atual = st.selectbox("🪨 Pedra Atual (2022)", ['Quartzo', 'Ágata', 'Ametista', 'Topázio'])
        pedra_ant = st.selectbox("🪨 Pedra Anterior (2021)", ['Quartzo', 'Ágata', 'Ametista', 'Topázio'])
    with c5:
        st.markdown("**💬 Texto e Ajustes de Avaliação**")
        observacoes = st.text_area(
            "💬 Observações dos avaliadores (opcional)",
            placeholder="Ex.: aluno engajado, com boa evolução e participação constante."
        )
        usar_sentimento_manual = st.checkbox(
            "Informar manualmente score e comprimento do texto",
            value=False,
            help="Útil para simular cenários, revisar a heurística automática ou informar valores quando não houver texto confiável."
        )
        if usar_sentimento_manual:
            sent_score_manual = st.slider("💬 Score de Sentimento dos Avaliadores", -5, 5, 0)
            sent_len_manual = st.slider("💬 Comprimento do Texto de Avaliação", 0, 500, 150, 10)
        else:
            sent_score_manual = None
            sent_len_manual = None

        sentiment_values = resolve_sentiment_values(
            observacoes,
            score_sentimento,
            usar_manual=usar_sentimento_manual,
            sent_score_manual=sent_score_manual,
            sent_len_manual=sent_len_manual,
        )
        if sentiment_values['mode'] == 'manual':
            st.caption(
                f"{sentiment_values['preview']} · Use este modo apenas quando quiser simular ou corrigir manualmente esses sinais."
            )
        else:
            st.caption(f"{sentiment_values['preview']} · Os valores acima são derivados automaticamente das observações.")

    current_inputs = {
        'fase': fase,
        'ian_num': ian_num,
        'ida_num': ida_num,
        'ieg_num': ieg_num,
        'iaa_num': iaa_num,
        'ips_num': ips_num,
        'ipp_num': ipp_num,
        'ipv_num': ipv_num,
        'no_av': no_av,
        'pedra_22_num': pedra_map[pedra_atual],
        'pedra_21_num': pedra_map[pedra_ant],
        'sent_score': sentiment_values['sent_score'],
        'sent_len': sentiment_values['sent_len'],
    }

    submitted = st.button("🔍 Calcular Risco", use_container_width=True, type="primary")
    if submitted:
        st.session_state[result_state_key] = dict(current_inputs)

    last_inputs = st.session_state.get(result_state_key)
    is_current_result = bool(last_inputs) and last_inputs == current_inputs
    if last_inputs and not is_current_result:
        st.info("Os parâmetros foram alterados desde o último cálculo. Clique em **Calcular Risco** para atualizar o resultado.")

    if is_current_result:
        sent_score = sentiment_values['sent_score']
        sent_len = sentiment_values['sent_len']
        feature_vals = {
            'fase': fase,
            'ian_num': ian_num,
            'ida_num': ida_num,
            'ieg_num': ieg_num,
            'iaa_num': iaa_num,
            'ips_num': ips_num,
            'ipp_num': ipp_num,
            'ipv_num': ipv_num,
            'no_av': no_av,
            'pedra_22_num': pedra_map[pedra_atual],
            'pedra_21_num': pedra_map[pedra_ant],
            'sent_score': sent_score,
            'sent_len': sent_len,
        }

        feature_values = build_feature_values(feature_vals)
        raw_prob = None
        if model_usable:
            raw_prob = predict_individual_with_model(feature_vals, model, scaler, meta)

        if raw_prob is not None:
            prob = apply_probability_calibration(raw_prob, meta)
        else:
            prob = heuristic_individual_prob(feature_vals)

        pct = prob * 100
        threshold = meta.get('threshold', 0.35) if meta else 0.35

        st.markdown("---")
        st.subheader("📊 Resultado da Análise")

        if model_usable:
            st.success(
                "Modelo alinhado ao plano mestre carregado com sucesso. "
                "A predição foi feita sem contingência e com calibração para reduzir saturação no topo."
            )
        else:
            st.info(
                "Predição individual em modo de contingência: o modelo não ficou utilizável nesta execução, "
                "então o app aplicou a regra heurística como proteção."
            )

        dim1, dim2, dim3, dim4 = st.columns(4)
        dim1.metric("INDE calculado", f"{feature_values['inde_calc']:.2f}")
        dim2.metric("Dim. Acadêmica", f"{feature_values['dim_academica']:.2f}")
        dim3.metric("Dim. Psicossocial", f"{feature_values['dim_psicossocial']:.2f}")
        dim4.metric("Dim. Psicopedagógica", f"{feature_values['dim_psicopedagogica']:.2f}")

        pesos = feature_values['weights']
        pesos_fmt = ' · '.join(
            f"{sigla.replace('_num', '').upper()}: {peso * 100:.0f}%"
            for sigla, peso in pesos.items() if peso > 0
        )
        st.caption(f"Pesos aplicados para a fase {fase}: {pesos_fmt}")

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
            risk_details = describe_risk(prob, threshold)
            st.markdown(f'<div class="{risk_details["css_class"]}">', unsafe_allow_html=True)
            st.markdown(build_risk_progress_html(prob, risk_details['accent_color']), unsafe_allow_html=True)
            st.markdown(f"## {risk_details['emoji']} Risco {risk_details['level']} — {pct:.1f}%")
            st.markdown(risk_details['recommendation'])
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("\n**Fatores que mais influenciaram esta predição:**")
            fatores = {
                'Dimensão Acadêmica': max(0.0, 10 - feature_values['dim_academica']),
                'Dimensão Psicossocial': max(0.0, 10 - feature_values['dim_psicossocial']),
                'Dimensão Psicopedagógica': max(0.0, 10 - feature_values['dim_psicopedagogica']),
                'Evolução de Pedra': max(0.0, feature_values['pedra_21_num'] - feature_values['pedra_22_num']) * 2.5,
                'Sentimento dos Avaliadores': max(0.0, -feature_values['sent_score']) * 2.0,
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
    st.markdown(
        "Faça upload de uma planilha CSV com os indicadores dos alunos para análise coletiva. "
        "O app deriva automaticamente as **dimensões consolidadas** e o **INDE dinâmico**."
    )

    uploaded = st.file_uploader("📂 Carregar CSV da turma", type=["csv"])

    BATCH_FEATURES = meta.get('features', INDIVIDUAL_DEFAULT_FEATURES)

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded, encoding='utf-8-sig', sep=',')
            st.success(f"✅ {len(df_up)} alunos carregados!")
            st.dataframe(df_up.head(10))

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            prepared_df, X_lote = prepare_batch_features(df_up, meta)
            metricas = [col_m1, col_m2, col_m3, col_m4]
            resumo_metricas = [
                ("Média INDE calc.", prepared_df['inde_calc'].mean()),
                ("Dim. Acadêmica", prepared_df['dim_academica'].mean()),
                ("Dim. Psicossocial", prepared_df['dim_psicossocial'].mean()),
                ("Dim. Psicopedagógica", prepared_df['dim_psicopedagogica'].mean()),
            ]
            for i, (titulo, valor) in enumerate(resumo_metricas):
                metricas[i].metric(titulo, f"{valor:.2f}")

            st.subheader("🔮 Predições em Lote")

            if model is not None and scaler is not None and model_is_usable(model, scaler, meta):
                X_sc = scaler.transform(X_lote)
                probs = apply_probability_calibration(model.predict_proba(X_sc)[:, 1], meta)
            else:
                probs = (
                    (10 - prepared_df['dim_academica']) * 0.35 +
                    (10 - prepared_df['dim_psicossocial']) * 0.35 +
                    (10 - prepared_df['dim_psicopedagogica']) * 0.20 +
                    np.maximum(0, prepared_df['pedra_21_num'] - prepared_df['pedra_22_num']) * 0.5 +
                    np.maximum(0, -prepared_df['sent_score']) * 0.2
                ).clip(0, 10) / 10

            threshold = meta.get('threshold', 0.35) if meta else 0.35
            prepared_df['Prob_Risco_%'] = (probs * 100).round(1)
            prepared_df['Nivel_Risco'] = pd.cut(
                probs,
                bins=[0, threshold, 0.65, 1.01],
                labels=['🟢 Baixo', '🟡 Médio', '🔴 Alto'],
                include_lowest=True,
            )

            resumo = prepared_df['Nivel_Risco'].value_counts()
            cols_res = st.columns(3)
            for i, (nivel, cnt) in enumerate(resumo.items()):
                cols_res[i % 3].metric(str(nivel), f"{cnt} alunos")

            st.markdown("**📋 Alunos em RISCO ALTO (requer atenção imediata):**")
            alto_risco = prepared_df[prepared_df['Nivel_Risco'] == '🔴 Alto']
            cols_disp = ['Prob_Risco_%', 'inde_calc'] + [
                c for c in ['Nome', 'dim_academica', 'dim_psicossocial', 'dim_psicopedagogica']
                if c in alto_risco.columns
            ]
            st.dataframe(alto_risco[cols_disp].sort_values('Prob_Risco_%', ascending=False))

            csv_out = prepared_df.to_csv(index=False, sep=';').encode('utf-8-sig')
            st.download_button("⬇️ Baixar resultado completo (.csv)", csv_out,
                               "resultado_risco_turma.csv", "text/csv")
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")
    else:
        st.info("👆 Carregue o arquivo CSV com os indicadores da turma.")
        colunas_str = 'fase, IAN, IDA, IEG, IAA, IPS, IPP ou Cf/Ct, IPV, Pedra 21, Pedra 22, Nº Av'
        st.markdown(f"**Colunas aceitas/esperadas:** `{colunas_str}`")

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
| NLP | Análise de sentimento lexical |
| Modelo preditivo | Regressão logística com dimensões consolidadas |
| Interface | Streamlit |
| Deploy | Streamlit Community Cloud |

### 📊 Lógica do Modelo Atual
`INDE dinâmico por fase · Dimensão Acadêmica · Dimensão Psicossocial · Dimensão Psicopedagógica · Pedra · Sentimento NLP`
        """)
    st.markdown("---")
    st.markdown("**Datathon 2025-2026 | FIAP Postech – Data Analytics**")
    if meta:
        auc_val = meta.get('test_auc') or meta.get('roc_auc') or meta.get('auc')
        st.metric("ROC-AUC do Modelo (teste)", f"{auc_val:.4f}" if isinstance(auc_val, float) else "N/A")
        cv_val = meta.get('cv_auc_mean')
        if cv_val:
            st.metric("CV AUC (5-fold)", f"{cv_val:.4f} ± {meta.get('cv_auc_std', 0):.4f}")


