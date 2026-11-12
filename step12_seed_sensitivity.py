"""
step12_seed_sensitivity.py — Seed sensitivity of the duplication effect (L4a) on Cleveland.
For each seed s: the duplicated file is resampled with rng(s) and the 5x10 CV split uses random_state=42+s.
Seed 0 reproduces the configuration reported in the paper.
Output: results/seed_sensitivity.csv (per seed) and results/seed_sensitivity_summary.csv
"""
import sys; sys.argv = [sys.argv[0], "5"]
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
import step2_leakage_experiment as S

SEEDS = [0, 1, 2, 3, 4, 5]
def run(ds, cond, X, y, rs):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=rs)
    out = Parallel(n_jobs=2)(delayed(S.run_fold)(ds, cond, X, y, tr, te, k) for k, (tr, te) in enumerate(cv.split(X, y)))
    return pd.DataFrame([r for o in out for r in o])

Xc, yc = S.load_uci("cleveland")
rows = []
for s in SEEDS:
    idx = np.random.default_rng(s).choice(len(yc), size=1025, replace=True)
    c0 = run("cleveland", "C0", Xc, yc, 42 + s)
    l4 = run("cleveland", "L4a", Xc[idx], yc[idx], 42 + s)
    for cn in ["SVM", "Stacking"]:
        a = l4[l4.clf == cn].acc.values; b = c0[c0.clf == cn].acc.values
        m, lo, hi, p = S.unpaired_corrected(a, b)
        rows.append(dict(seed=s, clf=cn, unique_patients=len(np.unique(idx)), acc_C0=round(b.mean()*100, 2),
                         acc_L4a=round(a.mean()*100, 2), gain=round(m*100, 2), ci_low=round(lo*100, 2),
                         ci_high=round(hi*100, 2), p_value=round(p, 4)))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(S.OUT / "seed_sensitivity.csv", index=False)
summ = df.groupby("clf").agg(gain_mean=("gain", "mean"), gain_sd=("gain", "std"), gain_min=("gain", "min"),
                             gain_max=("gain", "max"), n_sig=("p_value", lambda p: int((p < 0.05).sum())),
                             acc_L4a_mean=("acc_L4a", "mean"), acc_C0_mean=("acc_C0", "mean")).round(2).reset_index()
summ.to_csv(S.OUT / "seed_sensitivity_summary.csv", index=False); print(summ)
