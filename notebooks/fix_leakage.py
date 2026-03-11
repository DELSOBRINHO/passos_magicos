"""
Diagnóstico de Data Leakage e Retreinamento do Modelo Limpo
Executa: python notebooks/fix_leakage.py
"""
import json, warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, classification_report
warnings.filterwarnings('ignore')

CSV    = "data/pe_de_clean.csv"
SCALER = "app/scaler.pkl"
MODEL  = "app/modelo_risco_refined.pkl"
OUT_FIG_IMP  = "data/fig_feature_importance.png"
OUT_FIG_CORR = "data/fig_correlacao_target.png"
OUT_JSON_IMP = "app/feature_importance_permutation.json"
OUT_MODEL_CLEAN = "app/modelo_risco_clean.pkl"
OUT_SCALER_CLEAN = "app/scaler_clean.pkl"
OUT_META_CLEAN   = "app/modelo_meta_clean.json"

print("=" * 60)
print("1. Carregando dados e modelo existente...")
df = pd.read_csv(CSV, encoding="utf-8-sig")
print(f"   Shape: {df.shape}")

# Recriar target em_risco
if "em_risco" not in df.columns:
    if "defas" in df.columns:
        df["em_risco"] = (df["defas"] < 0).astype(int)
    elif "ian_num" in df.columns:
        df["em_risco"] = (df["ian_num"] < df["ian_num"].mean() - df["ian_num"].std()).astype(int)
    else:
        raise ValueError("Não foi possível recriar em_risco")
print(f"   Target em_risco: {df['em_risco'].sum()} positivos / {len(df)} total ({df['em_risco'].mean()*100:.1f}%)")

# Features do modelo refined
FEATURES_REFINED = [
    "fase","ano_nasc","idade_22","ano_ingresso","cf","ct","no_av",
    "pedra_22_num","ipp","pedra_21_num","evolucao_pedra","sent_len","sent_score",
    "cg_num","iaa_num","ieg_num","ips_num","ida_num",
    "matem_num","portug_num","ingles_num","ipv_num","ian_num"
]
avail = [c for c in FEATURES_REFINED if c in df.columns]
df_m  = df[avail + ["em_risco"]].dropna()
X_all = df_m[avail]
y_all = df_m["em_risco"]
print(f"\n2. Correlação Pearson com target (detecta leakage direto):")
corr = X_all.corrwith(y_all).abs().sort_values(ascending=False)
for feat, val in corr.items():
    flag = " ← LEAKAGE SUSPEITO" if val > 0.60 else ""
    print(f"   {feat:<20}: {val:.3f}{flag}")

# Gráfico de correlação
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#E8562A" if v > 0.60 else "#1A3A5C" if v > 0.30 else "#aaaaaa" for v in corr.values]
ax.barh(corr.index, corr.values, color=colors)
ax.axvline(0.60, color="red", linestyle="--", alpha=0.7, label="Threshold leakage (0.60)")
ax.axvline(0.30, color="orange", linestyle="--", alpha=0.5, label="Alta correlação (0.30)")
ax.set_title("Correlação Absoluta das Features com o Target em_risco", fontsize=13, fontweight="bold")
ax.set_xlabel("Correlação de Pearson (valor absoluto)")
ax.legend(); plt.tight_layout()
plt.savefig(OUT_FIG_CORR, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n   Gráfico salvo: {OUT_FIG_CORR}")

# Features com leakage (correlação > 0.60)
leak_feats = corr[corr > 0.60].index.tolist()
print(f"\n3. Features removidas por leakage (corr > 0.60): {leak_feats}")

FEATURES_CLEAN = [f for f in avail if f not in leak_feats]
print(f"   Features limpas ({len(FEATURES_CLEAN)}): {FEATURES_CLEAN}")

# Retreinar modelo limpo
X_clean = df_m[FEATURES_CLEAN]
y       = df_m["em_risco"]
X_tr, X_te, y_tr, y_te = train_test_split(X_clean, y, test_size=0.2, stratify=y, random_state=42)
scaler_clean = StandardScaler()
X_tr_sc = scaler_clean.fit_transform(X_tr)
X_te_sc = scaler_clean.transform(X_te)

model_clean = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_clean, X_tr_sc, y_tr, cv=cv, scoring="roc_auc")
print(f"\n4. CV AUC (5-fold, treino): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

model_clean.fit(X_tr_sc, y_tr)
y_prob = model_clean.predict_proba(X_te_sc)[:, 1]
test_auc = roc_auc_score(y_te, y_prob)
y_pred = (y_prob >= 0.35).astype(int)
print(f"   AUC no teste: {test_auc:.3f}")
print("\n   Relatório de classificação (threshold=0.35):")
print(classification_report(y_te, y_pred, target_names=["Sem Risco","Em Risco"]))

# Permutation importance
pi = permutation_importance(model_clean, X_te_sc, y_te, n_repeats=10, scoring="roc_auc", random_state=42)
imp_df = pd.DataFrame({"Feature": FEATURES_CLEAN, "Importance": pi.importances_mean,
                        "Std": pi.importances_std}).sort_values("Importance", ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(imp_df["Feature"], imp_df["Importance"],
        xerr=imp_df["Std"], color="#1A3A5C", alpha=0.85, error_kw={"ecolor":"#E8562A"})
ax.set_title("Importância por Permutação — Modelo Limpo (sem leakage)", fontsize=13, fontweight="bold")
ax.set_xlabel("Queda no AUC (maior = mais importante)")
plt.tight_layout()
plt.savefig(OUT_FIG_IMP, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n   Gráfico importância salvo: {OUT_FIG_IMP}")

# Salvar artefatos
joblib.dump(model_clean, OUT_MODEL_CLEAN)
joblib.dump(scaler_clean, OUT_SCALER_CLEAN)
meta = {
    "features": FEATURES_CLEAN, "threshold": 0.35,
    "cv_auc_mean": round(float(cv_scores.mean()), 4),
    "cv_auc_std":  round(float(cv_scores.std()),  4),
    "test_auc":    round(float(test_auc), 4),
    "leakage_removed": leak_feats,
    "num_features": len(FEATURES_CLEAN),
    "y_distribution": {"1": int(y.sum()), "0": int((y==0).sum())}
}
with open(OUT_META_CLEAN, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
imp_df.to_json(OUT_JSON_IMP, orient="records", indent=2)
print(f"\n5. Artefatos salvos:")
print(f"   {OUT_MODEL_CLEAN}  |  {OUT_SCALER_CLEAN}  |  {OUT_META_CLEAN}")
print("\n✅ Concluído!")

