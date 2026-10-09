"""Hypotheses: numeric ID, row index, region x city_type groups (run from repo root)."""
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

df = pd.read_csv("hard_train.csv"); y = df.protection_score
num = df.customer_id.str.extract(r"(\d+)", expand=False).astype(int)
row = np.arange(len(df))
print("H1 ID: pearson %.4f p=%.3f | spearman %.4f p=%.3f" % (*pearsonr(num, y), *spearmanr(num, y)))
print("H2 row: pearson %.4f p=%.3f | spearman %.4f p=%.3f" % (*pearsonr(row, y), *spearmanr(row, y)))
best = (0, 0)
for P in range(2, 201):
    for f in [np.sin, np.cos]:
        r = abs(np.corrcoef(f(2 * np.pi * num / P), y)[0, 1])
        if r > best[0]:
            best = (r, P)
print("H1 cyclic: max |r| over periods 2..200 = %.4f at period %d" % best)
g = df.groupby(["region", "city_type"])["protection_score"].agg(["count", "mean", "std", "min", "max"])
print(g.round(2).to_string())
print("groups:", len(g), "min std %.2f max std %.2f" % (g["std"].min(), g["std"].max()))
