"""
step10_ablation.py — الفرق بين Logistic regression و Stacking (C0): من النموذج ولا من المعالجة (SMOTE و GWO)؟

نجربوا النموذجين مع ثلاث مستويات معالجة، بنفس طيات 5×10 CV:
  none        : تعويض + تطبيع بس (داخل التدريب)
  SMOTE       : + SMOTE على التدريب
  SMOTE + GWO : + اختيار خصائص بـ GWO على التدريب (هذا هو C0)
الحالتين LR/none و Stacking/SMOTE+GWO موجودين قبل (step7 و step2)، لكن نعاودوهم هنا باش الجدول يطلع من سكربت واحد.
المقاييس: الدقة، AUC، Brier، ميل المعايرة (من احتمالات out-of-fold لكل تكرار).

الاستعمال: python step10_ablation.py    (حوالي 30 دقيقة)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from imblearn.over_sampling import SMOTE
import step2_leakage_experiment as s2

R = Path("results")
HOSP = ["cleveland", "hungarian", "switzerland", "va"]
LEVELS = ["none", "SMOTE", "SMOTE + GWO"]


def models():
    return {"Logistic regression": LogisticRegression(max_iter=2000), "Stacking": s2.classifiers()["Stacking"]}


def fold(ds, X, y, k, tr, te):
    f = s2.prep_fit(X[tr]); Xtr, Xte, ytr = f(X[tr]), f(X[te]), y[tr]
    Xs, ys = SMOTE(random_state=k).fit_resample(Xtr, ytr)
    mask = s2.select(Xs, ys, k)
    out = []
    for lvl in LEVELS:
        A, b = (Xtr, ytr) if lvl == "none" else (Xs, ys)
        m = mask if lvl == "SMOTE + GWO" else np.ones(X.shape[1], bool)
        for name, clf in models().items():
            p = clf.fit(A[:, m], b).predict_proba(Xte[:, m])[:, 1]
            out += [(ds, name, lvl, k, i, y[i], pi) for i, pi in zip(te, p)]
    return out


def cal_slope(yy, p):
    p = np.clip(p, 1e-4, 1 - 1e-4); lp = np.log(p / (1 - p))
    return LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], yy).coef_[0][0]


if __name__ == "__main__":
    rows = []
    for ds in HOSP:
        X, y = s2.load_uci(ds)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)   # نفس طيات step2
        out = Parallel(n_jobs=2)(delayed(fold)(ds, X, y, k, tr, te) for k, (tr, te) in enumerate(cv.split(X, y)))
        rows += [r for o in out for r in o]; print(ds, "done", flush=True)
    P = pd.DataFrame(rows, columns=["dataset", "model", "preprocessing", "fold", "idx", "y", "p"])
    P.to_csv(R / "ablation_oof.csv", index=False)
    summ = []
    for (ds, m, lvl), g in P.groupby(["dataset", "model", "preprocessing"], sort=False):
        per_fold = g.groupby("fold").apply(lambda h: pd.Series(dict(
            acc=accuracy_score(h.y, h.p > .5),
            auc=roc_auc_score(h.y, h.p) if h.y.nunique() > 1 else np.nan)))
        g = g.assign(rep=g.fold // 10)
        slope = np.mean([cal_slope(h.y.values, h.p.values) for _, h in g.groupby("rep")])
        summ.append(dict(dataset=ds, model=m, preprocessing=lvl, acc=round(per_fold.acc.mean() * 100, 2),
                         auc=round(per_fold.auc.mean() * 100, 2), brier=round(brier_score_loss(g.y, g.p), 3),
                         cal_slope=round(slope, 2), mean_pred=round(g.p.mean() * 100, 1), observed=round(g.y.mean() * 100, 1)))
    summ = pd.DataFrame(summ); summ.to_csv(R / "ablation_summary.csv", index=False)
    pd.set_option("display.width", 200); print(summ.to_string(index=False))
