"""
Retreina o modelo de risco alinhado ao PLANO_MESTRE.

- calcula o INDE dinamicamente por fase (0–7 vs 8)
- consolida os inputs nas dimensões acadêmica, psicossocial e psicopedagógica
- mantém contexto complementar mínimo: fase, pedra, nº de avaliações e sentimento textual

Observação importante:
como a base 2022 não traz o IPP explícito, o treino usa um proxy conservador
obtido pela inversão dos rankings Cf e Ct para a escala 0–10.

Execute: python notebooks/retrain_model.py
"""
import json
import warnings
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

CSV_BASE = "data/BASE DE DADOS PEDE 2024 - DATATHON - PEDE2022.csv"
OUT_MODEL = "app/modelo_risco_clean.pkl"
OUT_SCALER = "app/scaler_clean.pkl"
OUT_META = "app/modelo_meta_clean.json"
OUT_IMPUTER = "app/modelo_risco_clean_imputer.pkl"

INDE_THRESHOLD = 6.5
FEATURES = [
    'dim_academica',
    'dim_psicossocial',
    'dim_psicopedagogica',
    'fase',
    'pedra_22_num',
    'pedra_21_num',
    'evolucao_pedra',
    'no_av',
    'sent_score',
    'sent_len',
]

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


def to_num(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')


def normalize_text(value):
    text = unicodedata.normalize('NFKD', str(value).strip().lower())
    return ''.join(ch for ch in text if not unicodedata.combining(ch))


def map_pedra(value):
    return {
        'quartzo': 1,
        'agata': 2,
        'ametista': 3,
        'topazio': 4,
    }.get(normalize_text(value), np.nan)


def dynamic_weights(fase_series):
    fase = pd.to_numeric(fase_series, errors='coerce').fillna(0)
    fase_8 = fase >= 8
    return {
        'ian_num': np.where(fase_8, 0.00, 0.10),
        'ida_num': np.where(fase_8, 0.40, 0.20),
        'ieg_num': np.full(len(fase), 0.20),
        'iaa_num': np.where(fase_8, 0.00, 0.10),
        'ips_num': np.full(len(fase), 0.20),
        'ipp_num': np.full(len(fase), 0.10),
        'ipv_num': np.full(len(fase), 0.10),
    }


def reverse_rank_score(series):
    numeric = to_num(series)
    valid = numeric.dropna()
    if valid.empty or valid.max() == valid.min():
        bounds = {'min': np.nan, 'max': np.nan}
        return pd.Series(np.nan, index=series.index, dtype=float), bounds
    lower = float(valid.min())
    upper = float(valid.max())
    score = 10 * (1 - (numeric - lower) / (upper - lower))
    score = score.clip(0, 10)
    return score.astype(float), {'min': lower, 'max': upper}


def score_sentimento(texto):
    if pd.isna(texto):
        return 0
    content = str(texto).lower()
    return sum(1 for p in PALAVRAS_POS if p in content) - sum(1 for p in PALAVRAS_NEG if p in content)


def build_master_features(df):
    df = df.copy()
    ingles_col = next((c for c in df.columns if 'ingl' in c.lower()), None)
    nav_col = next((c for c in df.columns if 'av' in normalize_text(c) and c.upper().startswith('N')), None)

    df['fase'] = to_num(df['Fase'])
    df['ian_num'] = to_num(df['IAN'])
    df['ida_num'] = to_num(df['IDA'])
    df['ieg_num'] = to_num(df['IEG'])
    df['iaa_num'] = to_num(df['IAA'])
    df['ips_num'] = to_num(df['IPS'])
    df['ipv_num'] = to_num(df['IPV'])
    df['inde_num'] = to_num(df['INDE 22'])

    df['pedra_22_num'] = df['Pedra 22'].apply(map_pedra)
    df['pedra_21_num'] = df['Pedra 21'].apply(map_pedra) if 'Pedra 21' in df.columns else np.nan
    df['evolucao_pedra'] = df['pedra_22_num'] - df['pedra_21_num']
    df['no_av'] = to_num(df[nav_col]) if nav_col else np.nan

    cf_score, cf_bounds = reverse_rank_score(df['Cf'])
    ct_score, ct_bounds = reverse_rank_score(df['Ct'])
    df['ipp_num'] = pd.concat([cf_score, ct_score], axis=1).mean(axis=1)

    cols_rec = [c for c in df.columns if any(tag in c for tag in ['Rec Av', 'Rec Ps', 'Destaque'])]
    df['sent_score'] = sum(df[c].apply(score_sentimento) for c in cols_rec) if cols_rec else 0.0
    df['sent_len'] = (
        sum(df[c].apply(lambda x: len(str(x)) if not pd.isna(x) else 0) for c in cols_rec)
        if cols_rec else 0.0
    )

    weights = dynamic_weights(df['fase'])
    w_acad = weights['ian_num'] + weights['ida_num']
    w_psico = weights['ieg_num'] + weights['iaa_num'] + weights['ips_num']
    w_peda = weights['ipp_num'] + weights['ipv_num']

    df['dim_academica'] = (
        df['ian_num'] * weights['ian_num'] + df['ida_num'] * weights['ida_num']
    ) / w_acad
    df['dim_psicossocial'] = (
        df['ieg_num'] * weights['ieg_num']
        + df['iaa_num'] * weights['iaa_num']
        + df['ips_num'] * weights['ips_num']
    ) / w_psico
    df['dim_psicopedagogica'] = (
        df['ipp_num'] * weights['ipp_num'] + df['ipv_num'] * weights['ipv_num']
    ) / w_peda
    df['inde_calc'] = (
        df['ian_num'] * weights['ian_num']
        + df['ida_num'] * weights['ida_num']
        + df['ieg_num'] * weights['ieg_num']
        + df['iaa_num'] * weights['iaa_num']
        + df['ips_num'] * weights['ips_num']
        + df['ipp_num'] * weights['ipp_num']
        + df['ipv_num'] * weights['ipv_num']
    )

    return df, {'cf': cf_bounds, 'ct': ct_bounds, 'source': 'reverse_rank_cf_ct'}


print("1. Carregando CSV base original...")
df = pd.read_csv(CSV_BASE, encoding='utf-8-sig')
df.columns = df.columns.str.strip()
print(f"   Shape: {df.shape}")

print("\n2. Construindo features alinhadas ao plano mestre...")
df, ipp_bounds = build_master_features(df)

print("\n3. Criando target: em_risco = (INDE 22 < 6.5)...")
df['em_risco'] = (df['inde_num'] < INDE_THRESHOLD).astype(int)
n_risco = int(df['em_risco'].sum())
n_total = int(df['em_risco'].notna().sum())
print(f"   Em risco (INDE < {INDE_THRESHOLD}): {n_risco} / {n_total} ({n_risco / n_total * 100:.1f}%)")
print(f"   INDE original: min={df['inde_num'].min():.2f} | mediana={df['inde_num'].median():.2f} | max={df['inde_num'].max():.2f}")
print(f"   INDE calculado: min={df['inde_calc'].min():.2f} | mediana={df['inde_calc'].median():.2f} | max={df['inde_calc'].max():.2f}")

df_model = df[FEATURES + ['em_risco']].copy().dropna(subset=['em_risco'])
for feature in FEATURES:
    df_model[feature] = pd.to_numeric(df_model[feature], errors='coerce')

imputer = SimpleImputer(strategy='median')
df_model[FEATURES] = imputer.fit_transform(df_model[FEATURES])

print(f"\n4. Dataset final: {df_model.shape}")
print("\n   Correlação das features com em_risco:")
for feature in FEATURES:
    corr = df_model[feature].corr(df_model['em_risco'])
    direction = 'neg (coerente)' if corr < 0 else 'pos (inspecionar)'
    print(f"   {feature:22s}: {corr:+.3f} | {direction}")

X = df_model[FEATURES]
y = df_model['em_risco']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

model = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_tr_sc, y_tr, cv=cv, scoring='roc_auc')
print(f"\n5. CV AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

model.fit(X_tr_sc, y_tr)
y_prob = model.predict_proba(X_te_sc)[:, 1]
test_auc = roc_auc_score(y_te, y_prob)

best_thresh = 0.35
best_f1 = 0.0
for threshold in np.arange(0.20, 0.75, 0.05):
    f1 = f1_score(y_te, (y_prob >= threshold).astype(int))
    if f1 > best_f1:
        best_f1 = float(f1)
        best_thresh = float(threshold)

y_pred = (y_prob >= best_thresh).astype(int)
print(f"   AUC teste: {test_auc:.4f} | Threshold otimizado: {best_thresh:.2f}")
print(classification_report(y_te, y_pred, target_names=['Sem Risco', 'Em Risco']))

print("\n   Coeficientes do modelo:")
for name, coef in sorted(zip(FEATURES, model.coef_[0]), key=lambda item: item[1]):
    direction = '✅ neg (esperado)' if coef < 0 else '⚠️ pos (inspecionar)'
    print(f"   {name:22s}: {coef:+.3f} | {direction}")

joblib.dump(model, OUT_MODEL)
joblib.dump(scaler, OUT_SCALER)
joblib.dump(imputer, OUT_IMPUTER)

meta = {
    'features': FEATURES,
    'threshold': round(best_thresh, 2),
    'inde_threshold': INDE_THRESHOLD,
    'target_description': f'INDE 22 < {INDE_THRESHOLD}',
    'probability_calibration': {
        'method': 'logit_temperature',
        'clip_epsilon': 1e-6,
        'temperature': 3.5,
        'bias': 3.8,
        'description': 'Pós-calibração para reduzir saturação e melhorar discriminação entre perfis médio e crítico.',
    },
    'model_family': 'logistic_regression',
    'alignment': 'plano_mestre_dimensoes_consolidadas',
    'input_mode': 'master_plan_indicators',
    'ipp_proxy_description': 'Proxy conservador: média de Cf/Ct invertidos para escala 0-10.',
    'ipp_proxy_bounds': ipp_bounds,
    'inde_weights': {
        'fase_0_7': {'ian_num': 0.10, 'ida_num': 0.20, 'ieg_num': 0.20, 'iaa_num': 0.10, 'ips_num': 0.20, 'ipp_num': 0.10, 'ipv_num': 0.10},
        'fase_8': {'ian_num': 0.00, 'ida_num': 0.40, 'ieg_num': 0.20, 'iaa_num': 0.00, 'ips_num': 0.20, 'ipp_num': 0.10, 'ipv_num': 0.10},
    },
    'dimension_weights': {
        'fase_0_7': {'dim_academica': 0.30, 'dim_psicossocial': 0.50, 'dim_psicopedagogica': 0.20},
        'fase_8': {'dim_academica': 0.40, 'dim_psicossocial': 0.40, 'dim_psicopedagogica': 0.20},
    },
    'cv_auc_mean': round(float(cv_scores.mean()), 4),
    'cv_auc_std': round(float(cv_scores.std()), 4),
    'test_auc': round(float(test_auc), 4),
    'num_features': len(FEATURES),
    'y_distribution': {'1': int(y.sum()), '0': int((y == 0).sum())},
    'imputer_medians': {feature: float(imputer.statistics_[i]) for i, feature in enumerate(FEATURES)},
}

with open(OUT_META, 'w', encoding='utf-8') as fp:
    json.dump(meta, fp, indent=2, ensure_ascii=False)

print(f"\n✅ Modelo salvo:  {OUT_MODEL}")
print(f"✅ Scaler salvo:  {OUT_SCALER}")
print(f"✅ Imputer salvo: {OUT_IMPUTER}")
print(f"✅ Meta salvo:    {OUT_META}")

