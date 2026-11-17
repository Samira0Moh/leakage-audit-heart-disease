"""
step14_friedman_hospital_blocks.py — sensitivity analysis for Section 2.6:
the Friedman test with the four hospitals as blocks (the 3 repetitions inside a hospital are not independent).
Reads results/swarm_folds.csv.  Usage: python step14_friedman_hospital_blocks.py
"""
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, rankdata
S = pd.read_csv("results/swarm_folds.csv"); S = S[S.dataset != "kaggle1025 (duplicated)"]
M = ["All features", "PSO", "GWO", "PSOGWO", "LGWO"]; rows = []
for cn, g in S.groupby("clf"):
    for blocks, keys in [("12 (hospital x repeat)", ["dataset", "repeat"]), ("4 (hospital)", ["dataset"])]:
        B = g.groupby(keys + ["method"]).acc.mean().unstack("method")[M]
        stat, p = friedmanchisquare(*[B[m].values for m in M])
        ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, B.values).mean(0)
        rows.append(dict(clf=cn, blocks=blocks, friedman_p=round(p, 4), **{f"rank_{m}": round(r, 2) for m, r in zip(M, ranks)}))
out = pd.DataFrame(rows); out.to_csv("results/friedman_hospital_blocks.csv", index=False); print(out.to_string(index=False))
