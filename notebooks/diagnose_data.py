"""Diagnose dataset: check which columns are available, NaN rates, and feature-target correlations."""
import pandas as pd
import numpy as np
import unicodedata

CSV = 'data/BASE DE DADOS PEDE 2024 - DATATHON - PEDE2022.csv'
df = pd.read_csv(CSV, encoding='utf-8-sig')
df.columns = df.columns.str.strip()

print('=== ALL COLUMNS ===')
for c in df.columns:
    s = df[c].astype(str).str.replace(',', '.')
    sn = pd.to_numeric(s, errors='coerce')
    nn = sn.notna().sum()
    print(f'  {c!r:30s}  non-null={nn:4d}  dtype={df[c].dtype}  sample={df[c].dropna().head(3).tolist()}')

print()
print('=== TARGET: em_risco = (Defas < 0) ===')
defas = pd.to_numeric(df['Defas'], errors='coerce')
em_risco = (defas < 0).astype(int)
df['em_risco'] = em_risco
print(f'em_risco: {em_risco.sum()} positive ({em_risco.mean()*100:.1f}%)')

print()
print('=== FEATURE-TARGET CORRELATIONS ===')
pedra_map = {
    'quartzo': 1, 'agata': 2, 'ágata': 2,
    'ametista': 3, 'topazio': 4, 'topázio': 4
}
def map_pedra(x):
    s = unicodedata.normalize('NFC', str(x).strip().lower())
    return pedra_map.get(s, np.nan)

df['pedra_22_n'] = df['Pedra 22'].apply(map_pedra)
df['pedra_21_n'] = df['Pedra 21'].apply(map_pedra)
df['evolucao_pedra'] = df['pedra_22_n'] - df['pedra_21_n']

ingles_col = 'Inglês' if 'Inglês' in df.columns else ('Ingles' if 'Ingles' in df.columns else None)
nav_col = next((c for c in df.columns if 'Av' in c and c.startswith('N')), None)
print(f'Inglês col: {ingles_col!r}, Nº Av col: {nav_col!r}')

def to_num(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')

candidates = {
    'iaa_num': to_num(df['IAA']),
    'ieg_num': to_num(df['IEG']),
    'ips_num': to_num(df['IPS']),
    'ida_num': to_num(df['IDA']),
    'ipv_num': to_num(df['IPV']),
    'matem_num': to_num(df['Matem']),
    'portug_num': to_num(df['Portug']),
    'ingles_num': to_num(df[ingles_col]) if ingles_col else pd.Series(np.nan, index=df.index),
    'pedra_22_n': df['pedra_22_n'],
    'pedra_21_n': df['pedra_21_n'],
    'evolucao_pedra': df['evolucao_pedra'],
    'no_av': to_num(df[nav_col]) if nav_col else pd.Series(np.nan, index=df.index),
    'cf_raw': to_num(df['Cf']),
    'ct_raw': to_num(df['Ct']),
    'cg_raw': to_num(df['Cg']),
    'fase': to_num(df['Fase']),
    'ano_nasc': to_num(df['Ano nasc']),
}

for name, s in candidates.items():
    nn = s.notna().sum()
    comb = pd.DataFrame({'x': s, 'y': em_risco}).dropna()
    corr = comb.corr()['y']['x'] if len(comb) > 10 else np.nan
    print(f'  {name:20s}  non-null={nn:4d}  corr_with_target={corr:+.3f}')

