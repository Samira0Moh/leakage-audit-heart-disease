"""
step6_posthoc.py — اختبار لاحق (post-hoc) بعد Friedman
Friedman يقول "في فرق بين الطرق"، لكن ما يقولش وين. هنا نقارنوا "كل الخصائص" مع كل خوارزمية:
  1) Wilcoxon على 12 كتلة (4 مستشفيات × 3 تكرارات) + تصحيح Holm للمقارنات المتعددة
  2) Nemenyi: الفرق الحرج CD = q * sqrt(k(k+1)/(6N)) ، q(0.05, k=5) = 2.728 (Demšar 2006)
الاستعمال: python step6_posthoc.py   (يقرا results/swarm_folds.csv — ما يحتاجش تشغيل التجارب من جديد)
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata, wilcoxon

R = Path("results")
M = ["All features", "PSO", "GWO", "PSOGWO", "LGWO"]
df = pd.read_csv(R / "swarm_folds.csv")
df = df[df.dataset != "kaggle1025 (duplicated)"]
rows = []
for cn, g in df.groupby("clf"):
    B = g.groupby(["dataset", "repeat", "method"]).acc.mean().unstack("method")[M]
    N, k = len(B), len(M)
    ranks = dict(zip(M, np.apply_along_axis(lambda r: rankdata(-r), 1, B.values).mean(0)))
    cd = 2.728 * np.sqrt(k * (k + 1) / (6 * N))
    tests = sorted(((m, wilcoxon(B["All features"], B[m]).pvalue) for m in M[1:]), key=lambda x: x[1])
    adj, running = {}, 0
    for i, (m, p) in enumerate(tests):               # Holm step-down
        running = max(running, min(1, p * (len(tests) - i))); adj[m] = running
    for m in M[1:]:
        rows.append(dict(clf=cn, method=m, diff_vs_all=round((B["All features"] - B[m]).mean() * 100, 2),
                         rank_all=round(ranks["All features"], 2), rank_method=round(ranks[m], 2),
                         rank_gap=round(ranks[m] - ranks["All features"], 2), nemenyi_CD=round(cd, 2),
                         nemenyi_sig=ranks[m] - ranks["All features"] > cd, holm_p=round(adj[m], 4)))
out = pd.DataFrame(rows); out.to_csv(R / "swarm_posthoc.csv", index=False)
print(out.to_string(index=False))
