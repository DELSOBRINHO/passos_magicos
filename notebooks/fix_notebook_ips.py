"""
Fix IPS_quartil cell in notebook 01 to use rank(method='first') to avoid qcut duplicates error.
"""
import json

nb_path = "notebooks/01_analise_exploratoria.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

old_line = "df['IPS_quartil'] = pd.qcut(df['IPS'].dropna(), q=4, labels=['Q1 (baixo)','Q2','Q3','Q4 (alto)'], duplicates='drop')\n"
new_line = "df['IPS_quartil'] = pd.qcut(df['IPS'].rank(method='first'), q=4, labels=['Q1 (baixo)','Q2','Q3','Q4 (alto)'])\n"

fixed = 0
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = cell["source"]
        new_src = [new_line if line == old_line else line for line in src]
        if new_src != src:
            cell["source"] = new_src
            fixed += 1

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Fixed {fixed} cell(s) in {nb_path}")

