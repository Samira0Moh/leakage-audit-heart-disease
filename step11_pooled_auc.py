"""
step11_pooled_auc.py — AUC محسوب على كل احتمالات out-of-fold مجتمعة + bootstrap على مستوى المريض

ليش: في Switzerland كل fold فيه مريض سليم واحد على الأكثر، فـ AUC على مستوى الـ fold ما ينحسبش صح.
هنا: لكل تكرار (repeat) نحسبوا AUC على كل المرضى مع بعض، وناخذوا المتوسط عبر التكرارات الخمسة،
وفترة الثقة من 2,000 عينة bootstrap للمرضى (نفس المرضى المختارين في كل التكرارات).
يقرا results/oof_probabilities.csv (من step9).

الاستعمال: python step11_pooled_auc.py
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

R = Path("results")
P = pd.read_csv(R / "oof_probabilities.csv")
rng = np.random.default_rng(0)
rows = []
for (ds, m), g in P.groupby(["dataset", "model"]):
    g = g.assign(rep=g.groupby("idx").cumcount())            # كل مريض يظهر مرة في كل تكرار
    W = g.pivot(index="idx", columns="rep", values="p"); yv = g.groupby("idx").y.first().loc[W.index].values
    auc = np.mean([roc_auc_score(yv, W[r].values) for r in W.columns])
    boots = []
    for _ in range(2000):
        b = rng.integers(0, len(yv), len(yv))
        if len(set(yv[b])) < 2: continue
        boots.append(np.mean([roc_auc_score(yv[b], W[r].values[b]) for r in W.columns]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append(dict(dataset=ds, model=m, n_patients=len(yv), n_healthy=int((yv == 0).sum()),
                     pooled_auc=round(auc * 100, 1), ci_low=round(lo * 100, 1), ci_high=round(hi * 100, 1)))
out = pd.DataFrame(rows); out.to_csv(R / "pooled_auc_bootstrap.csv", index=False)
print(out.to_string(index=False))
