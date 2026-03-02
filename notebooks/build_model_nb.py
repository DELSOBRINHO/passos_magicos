"""
Gera o notebook 02_modelo_preditivo.ipynb
Execute: python build_model_nb.py
"""
import nbformat as nbf

def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)

nb = nbf.v4.new_notebook()
nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}
cells = []

cells.append(md("""# 🤖 Passos Mágicos – Modelo Preditivo de Risco
## Deep Learning (MLP) + NLP | Datathon 2025-2026

### ❓ Pergunta 9
> *"Quais padrões nos indicadores permitem identificar alunos em risco antes de uma queda no desempenho ou aumento da defasagem?"*

### Fluxo do notebook:
1. **Feature Engineering** – criação da variável target e features derivadas
2. **NLP** – extração de sentimento das recomendações textuais
3. **Pré-processamento** – separação treino/teste, normalização, embeddings de pedra
4. **Arquitetura MLP** – Input → Camadas ocultas (Dropout) → Softmax
5. **Avaliação** – ROC, Precisão/Recall, Matriz de Confusão
6. **Salvamento do modelo** – `.h5` para uso no Streamlit
"""))

cells.append(md("## 0. Instalação e Imports"))
cells.append(code("""# Instale se necessário:
# pip install tensorflow scikit-learn seaborn matplotlib pandas numpy

import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, precision_recall_curve)
import re
from collections import Counter

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f'✅ TensorFlow {tf.__version__}')
    USE_TF = True
except ImportError:
    print('⚠️  TensorFlow não encontrado – usando sklearn MLP como fallback')
    from sklearn.neural_network import MLPClassifier
    USE_TF = False

plt.rcParams.update({'figure.figsize': (10,5), 'axes.titlesize': 13})
CORES = ['#1A3A5C','#E8562A','#4CAF9A','#F4A259']
print('✅ Pronto!')"""))

cells.append(md("## 1. Carregamento e Limpeza"))
cells.append(code("""df = pd.read_csv('../data/BASE DE DADOS PEDE 2024 - DATATHON - PEDE2022.csv',
                 encoding='utf-8-sig', sep=',')
print(f'Shape original: {df.shape}')

cols_float = ['INDE 22','IAA','IEG','IPS','IDA','IPV','IAN','Matem','Portug','Inglês','Cg']
for c in cols_float:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',','.'), errors='coerce')

pedra_ordem = {'Quartzo':1,'Ágata':2,'Ametista':3,'Topázio':4}
for ano in ['20','21','22']:
    df[f'Pedra_{ano}_num'] = df.get(f'Pedra {ano}', pd.Series()).map(pedra_ordem)

df['Evolucao_Pedra'] = df['Pedra_22_num'] - df['Pedra_21_num']
df['IPP'] = (df['Cf'] + df['Ct']) / 2
print('Limpeza concluída.')"""))

cells.append(md("## 2. Feature Engineering – Variável Target e Features"))
cells.append(code("""# ── TARGET: "Em Risco de Defasagem" ──────────────────────────────────────────
# Critério multicritério (conservador para não perder casos → prioriza Recall):
#   1) Defasagem < 0 (já defasado)
#   OU 2) INDE 22 < média - 1 desvio padrão  (INDE baixo)
#   OU 3) IEG < 4 E IDA < 5  (baixo engajamento + baixo desempenho)

inde_media, inde_std = df['INDE 22'].mean(), df['INDE 22'].std()

df['em_risco'] = (
    (df['Defas'] < 0) |
    (df['INDE 22'] < inde_media - inde_std) |
    ((df['IEG'] < 4) & (df['IDA'] < 5))
).astype(int)

print(f'Distribuição do target:')
print(df['em_risco'].value_counts(normalize=True).rename({0:'Sem risco',1:'Em risco'}).round(3))"""))

cells.append(code("""# ── NLP: Sentimento das Recomendações ────────────────────────────────────────
palavras_neg = {'melhorar','empenhar','dificuldade','atraso','déficit','atenção',
                'problema','risco','alerta','comportamento','limitação','preocupa'}
palavras_pos = {'destaque','excelente','promovido','bolsa','líder','potencial',
                'engajado','comprometido','aprovado','evolução','crescimento'}

def score_sentimento(texto):
    if pd.isna(texto): return 0
    t = texto.lower()
    pos = sum(1 for p in palavras_pos if p in t)
    neg = sum(1 for p in palavras_neg if p in t)
    return pos - neg

cols_rec = ['Rec Av1','Rec Av2','Rec Av3','Rec Av4','Rec Psicologia',
            'Destaque IEG','Destaque IDA','Destaque IPV']
df['sent_score'] = sum(df[c].apply(score_sentimento) for c in cols_rec if c in df.columns)

print(f'Score de sentimento – média: {df["sent_score"].mean():.2f} | std: {df["sent_score"].std():.2f}')"""))

cells.append(code("""# ── Embedding ordinal de Pedra ────────────────────────────────────────────────
# Já criado como Pedra_22_num (1=Quartzo → 4=Topázio)

# ── Features finais ───────────────────────────────────────────────────────────
FEATURES = ['IAA','IEG','IPS','IDA','IPV','IAN','IPP',
            'Pedra_22_num','Evolucao_Pedra','sent_score','Fase']
TARGET = 'em_risco'

df_model = df[FEATURES + [TARGET]].dropna()
print(f'Dataset de modelagem: {df_model.shape[0]} amostras, {len(FEATURES)} features')
print(f'Taxa de risco: {df_model[TARGET].mean()*100:.1f}%')
df_model.describe().round(2)"""))

cells.append(md("## 3. Separação Treino / Teste"))
cells.append(code("""X = df_model[FEATURES].values
y = df_model[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f'Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}')
print(f'Taxa de risco no treino: {y_train.mean()*100:.1f}%')
print(f'Taxa de risco no teste:  {y_test.mean()*100:.1f}%')

# Salvar scaler para uso no Streamlit
import pickle, os
os.makedirs('../app', exist_ok=True)
with open('../app/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('../app/features.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)
print('✅ Scaler salvo em ../app/scaler.pkl')"""))

cells.append(md("## 4. Arquitetura do Modelo – MLP (Deep Learning)"))
cells.append(code("""if USE_TF:
    # ── Rede Neural Keras ─────────────────────────────────────────────────────
    tf.random.set_seed(42)
    n_features = X_train_sc.shape[1]

    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')   # probabilidade de risco
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Recall(name='recall')]
    )
    model.summary()
else:
    # ── Fallback: sklearn MLP ─────────────────────────────────────────────────
    model_sk = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                              solver='adam', max_iter=500, random_state=42,
                              early_stopping=True, validation_fraction=0.15)
    print('Usando sklearn MLPClassifier como fallback.')"""))

cells.append(code("""if USE_TF:
    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_recall', patience=15,
                                      restore_best_weights=True, mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=8, min_lr=1e-6)
    ]

    # Peso de classe para priorizar Recall (não perder alunos em risco)
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {0: cw[0], 1: cw[1]}
    print(f'Class weights: {class_weight}')

    history = model.fit(
        X_train_sc, y_train,
        epochs=100, batch_size=32,
        validation_split=0.15,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Plot curva de aprendizado
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    axes[0].plot(history.history['loss'], label='Treino', color='#1A3A5C')
    axes[0].plot(history.history['val_loss'], label='Validação', color='#E8562A')
    axes[0].set_title('Loss (Binary Cross-Entropy)'); axes[0].set_xlabel('Época'); axes[0].legend()

    axes[1].plot(history.history['recall'], label='Treino', color='#4CAF9A')
    axes[1].plot(history.history['val_recall'], label='Validação', color='#E8562A')
    axes[1].set_title('Recall (Treino vs Validação)'); axes[1].set_xlabel('Época'); axes[1].legend()

    plt.suptitle('Curva de Aprendizado – MLP', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig('../data/fig_learning_curve.png', dpi=150, bbox_inches='tight'); plt.show()
else:
    model_sk.fit(X_train_sc, y_train)
    print('Modelo sklearn treinado.')"""))

cells.append(md("## 5. Avaliação do Modelo"))
cells.append(code("""if USE_TF:
    y_prob = model.predict(X_test_sc).flatten()
    # Threshold ajustado para maximizar Recall
    threshold = 0.35  # mais sensível para capturar alunos em risco
    y_pred = (y_prob >= threshold).astype(int)
else:
    y_prob = model_sk.predict_proba(X_test_sc)[:, 1]
    y_pred = model_sk.predict(X_test_sc)

print('='*60)
print('RELATÓRIO DE CLASSIFICAÇÃO')
print('='*60)
print(classification_report(y_test, y_pred, target_names=['Sem Risco','Em Risco']))
print(f'ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}')"""))

cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Sem Risco','Em Risco'], yticklabels=['Sem Risco','Em Risco'])
axes[0].set_title('Matriz de Confusão'); axes[0].set_xlabel('Previsto'); axes[0].set_ylabel('Real')

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_val = roc_auc_score(y_test, y_prob)
axes[1].plot(fpr, tpr, color='#1A3A5C', lw=2, label=f'AUC = {auc_val:.3f}')
axes[1].plot([0,1],[0,1], '--', color='gray')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='#1A3A5C')
axes[1].set_title('Curva ROC'); axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR'); axes[1].legend()

# Curva Precisão-Recall
prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[2].plot(rec, prec, color='#E8562A', lw=2)
axes[2].fill_between(rec, prec, alpha=0.1, color='#E8562A')
axes[2].set_title('Curva Precisão-Recall'); axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precisão')

plt.suptitle('Avaliação do Modelo Preditivo – MLP Passos Mágicos', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('../data/fig_model_eval.png', dpi=150, bbox_inches='tight'); plt.show()"""))

cells.append(code("""# Importância das features via permutação
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier

if not USE_TF:
    pi = permutation_importance(model_sk, X_test_sc, y_test, n_repeats=10, random_state=42)
    imp_df = pd.DataFrame({'Feature': FEATURES, 'Importance': pi.importances_mean}).sort_values('Importance', ascending=True)
else:
    # Para TF: usar gradientes ou método simples de perturbação
    importances = []
    baseline_auc = roc_auc_score(y_test, y_prob)
    for i, feat in enumerate(FEATURES):
        X_perm = X_test_sc.copy(); np.random.shuffle(X_perm[:, i])
        y_perm = model.predict(X_perm).flatten()
        importances.append(baseline_auc - roc_auc_score(y_test, y_perm))
    imp_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importances}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10,6))
colors = ['#E8562A' if v > 0 else '#cccccc' for v in imp_df['Importance']]
ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors, edgecolor='white')
ax.set_title('Importância das Features (Queda de AUC na Permutação)', fontsize=13, fontweight='bold')
ax.set_xlabel('Queda no AUC-ROC')
plt.tight_layout(); plt.savefig('../data/fig_feature_importance.png', dpi=150, bbox_inches='tight'); plt.show()"""))

cells.append(md("## 6. Salvar Modelo para o Streamlit"))
cells.append(code("""import pickle, os
os.makedirs('../app', exist_ok=True)

if USE_TF:
    model.save('../app/modelo_risco.h5')
    print('✅ Modelo Keras salvo: ../app/modelo_risco.h5')
else:
    with open('../app/modelo_risco.pkl', 'wb') as f:
        pickle.dump(model_sk, f)
    print('✅ Modelo sklearn salvo: ../app/modelo_risco.pkl')

# Salvar metadados úteis para o Streamlit
meta = {
    'features': FEATURES,
    'threshold': 0.35,
    'roc_auc': float(roc_auc_score(y_test, y_prob)),
    'use_tf': USE_TF
}
with open('../app/modelo_meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
print('✅ Metadados salvos: ../app/modelo_meta.pkl')
print(f'\\n🎯 ROC-AUC final: {meta["roc_auc"]:.4f}')"""))

cells.append(md("""---
## ✅ Resumo do Modelo Preditivo

| Aspecto | Detalhe |
|---------|---------|
| **Target** | `em_risco` (binária) – defasagem OR INDE baixo OR baixo IEG+IDA |
| **Features** | 11 indicadores numéricos + sentimento NLP |
| **Arquitetura** | MLP: 128→64→32 neurônios, Dropout 0.3, BatchNorm |
| **Otimização** | Adam lr=0.001, EarlyStopping (monitor=Recall) |
| **Métrica foco** | **Recall** – não deixar nenhum aluno em risco despercebido |
| **Threshold** | 0.35 (ajustado para maximizar Recall) |
| **Output** | Probabilidade de risco (0–100%) |
| **Deploy** | Modelo salvo em `.h5` (TF) ou `.pkl` (sklearn) |
"""))

nb['cells'] = cells

with open('02_modelo_preditivo.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('✅ 02_modelo_preditivo.ipynb criado!')

