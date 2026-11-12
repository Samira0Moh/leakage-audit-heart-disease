"""
step9_extra_stats.py — إحصاءات إضافية (يقرا نتائج step2 و step4 و step7، ويعاود تشغيل C0 بس للمعايرة)

(أ) مقارنة الخطوط المرجعية مع C0 بنفس الطيات: Logistic regression و SVM nested مقابل SVM/Stacking (C0)
    بـ corrected resampled t-test + فترة ثقة 95%
(ب) فترات ثقة للـ AUC لكل نموذج (المتوسط ± corrected CI عبر الطيات)
(ج) Switzerland (8 أصحاء بس): فترة Wilson للـ specificity
(د) تحليل حساسية: Friedman و Holm بدون Switzerland (9 كتل بدل 12)
(هـ) المعايرة (calibration): slope و intercept من احتمالات out-of-fold لـ Logistic regression و Stacking (C0)

الاستعمال: python step9_extra_stats.py     (حوالي 10 دقائق بسبب إعادة C0 للمعايرة)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist, norm, friedmanchisquare, rankdata, wilcoxon
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE
import step2_leakage_experiment as s2
from step7_baselines import pipe

R = Path("results")
HOSP = ["cleveland", "hungarian", "switzerland", "va"]


def corrected(d, ratio=1 / 9):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; k = len(d)
    se = np.sqrt((1 / k + ratio) * d.var(ddof=1)); m = d.mean(); tc = t_dist.ppf(.975, k - 1)
    return m, m - tc * se, m + tc * se, (2 * t_dist.sf(abs(m / se), k - 1) if se > 0 else np.nan)


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


L = pd.read_csv(R / "leakage_folds.csv"); B = pd.read_csv(R / "baselines_folds.csv")
c0 = L[L.condition == "C0 correct"]
fold = {}
for ds in HOSP:
    for cn in ["SVM", "Stacking"]:
        fold[(ds, f"{cn} (C0)")] = c0[(c0.dataset == ds) & (c0.clf == cn)].sort_values("fold")
    for m in ["Logistic regression", "SVM tuned (nested CV)", "SVM tuned on test fold (L5)"]:
        fold[(ds, m)] = B[(B.dataset == ds) & (B.model == m)].sort_values("fold")

# (أ) + (ب)
comp, aucs = [], []
for ds in HOSP:
    for a, b in [("Logistic regression", "Stacking (C0)"), ("Logistic regression", "SVM (C0)"),
                 ("SVM tuned (nested CV)", "Stacking (C0)"), ("Logistic regression", "SVM tuned (nested CV)")]:
        for metric in ["acc", "auc"]:
            m, lo, hi, p = corrected(fold[(ds, a)][metric].values - fold[(ds, b)][metric].values)
            comp.append(dict(dataset=ds, model=a, vs=b, metric=metric, diff=round(m * 100, 2),
                             ci_low=round(lo * 100, 2), ci_high=round(hi * 100, 2), p_value=round(p, 4)))
    for name in ["SVM (C0)", "Stacking (C0)", "Logistic regression", "SVM tuned (nested CV)"]:
        m, lo, hi, _ = corrected(fold[(ds, name)].auc.values - 0)
        aucs.append(dict(dataset=ds, model=name, auc=round(m * 100, 1), ci_low=round(lo * 100, 1), ci_high=round(hi * 100, 1)))
comp = pd.DataFrame(comp); comp.to_csv(R / "baseline_comparisons.csv", index=False)
aucs = pd.DataFrame(aucs); aucs.to_csv(R / "auc_ci.csv", index=False)

# (ج) Switzerland: 8 مرضى أصحاء
X, y = s2.load_uci("switzerland"); n_neg = int((y == 0).sum())
sw = []
for name in ["SVM (C0)", "Stacking (C0)", "Logistic regression", "SVM tuned (nested CV)"]:
    sp = fold[("switzerland", name)].spec.mean(); lo, hi = wilson(sp, n_neg)
    sw.append(dict(model=name, n_healthy=n_neg, spec=round(sp * 100, 1), ci_low=round(lo * 100, 1), ci_high=round(hi * 100, 1)))
for cond in ["L3 SMOTE before split"]:
    for cn in ["SVM", "Stacking"]:
        sp = L[(L.dataset == "switzerland") & (L.condition == cond) & (L.clf == cn)].spec.mean(); lo, hi = wilson(sp, n_neg)
        sw.append(dict(model=f"{cn} ({cond})", n_healthy=n_neg, spec=round(sp * 100, 1), ci_low=round(lo * 100, 1), ci_high=round(hi * 100, 1)))
sw = pd.DataFrame(sw); sw.to_csv(R / "switzerland_specificity_ci.csv", index=False)

# (د) Friedman + Holm بدون Switzerland
S = pd.read_csv(R / "swarm_folds.csv"); S = S[~S.dataset.isin(["kaggle1025 (duplicated)", "switzerland"])]
M = ["All features", "PSO", "GWO", "PSOGWO", "LGWO"]; sens = []
for cn, g in S.groupby("clf"):
    Bk = g.groupby(["dataset", "repeat", "method"]).acc.mean().unstack("method")[M]
    ranks = dict(zip(M, np.apply_along_axis(lambda r: rankdata(-r), 1, Bk.values).mean(0)))
    pf = friedmanchisquare(*[Bk[m] for m in M]).pvalue
    tests = sorted(((m, wilcoxon(Bk["All features"], Bk[m]).pvalue) for m in M[1:]), key=lambda x: x[1])
    run = 0
    for i, (m, p) in enumerate(tests):
        run = max(run, min(1, p * (len(tests) - i)))
        sens.append(dict(clf=cn, method=m, n_blocks=len(Bk), friedman_p=round(pf, 4), rank_all=round(ranks["All features"], 2),
                         rank_method=round(ranks[m], 2), diff_vs_all=round((Bk["All features"] - Bk[m]).mean() * 100, 2), holm_p=round(run, 4)))
sens = pd.DataFrame(sens); sens.to_csv(R / "swarm_sensitivity_no_switzerland.csv", index=False)


# (هـ) المعايرة: احتمالات out-of-fold (5×10 CV) لـ Logistic regression و Stacking (C0)
def oof_fold(ds, X, y, k, tr, te):
    f = s2.prep_fit(X[tr]); Xtr, Xte = f(X[tr]), f(X[te])
    Xs, ys = SMOTE(random_state=k).fit_resample(Xtr, y[tr]); mask = s2.select(Xs, ys, k)
    st = s2.classifiers()["Stacking"].fit(Xs[:, mask], ys)
    lr = pipe(LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
    return [(ds, "Stacking (C0)", i, y[i], p) for i, p in zip(te, st.predict_proba(Xte[:, mask])[:, 1])] + \
           [(ds, "Logistic regression", i, y[i], p) for i, p in zip(te, lr.predict_proba(X[te])[:, 1])]


rows = []
for ds in HOSP:
    X, y = s2.load_uci(ds)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    out = Parallel(n_jobs=2)(delayed(oof_fold)(ds, X, y, k, tr, te) for k, (tr, te) in enumerate(cv.split(X, y)))
    rows += [r for o in out for r in o]; print(ds, "calibration done", flush=True)
P = pd.DataFrame(rows, columns=["dataset", "model", "idx", "y", "p"]); P.to_csv(R / "oof_probabilities.csv", index=False)
cal = []
for (ds, m), g in P.groupby(["dataset", "model"]):
    p = g.p.clip(1e-4, 1 - 1e-4).values; lp = np.log(p / (1 - p)); yy = g.y.values
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], yy).coef_[0][0]
    # intercept (calibration-in-the-large): logistic model with logit(p) as offset
    b = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(lp + b))); b += (yy - mu).sum() / (mu * (1 - mu)).sum()
    cal.append(dict(dataset=ds, model=m, calibration_slope=round(slope, 2), calibration_intercept=round(b, 2),
                    mean_predicted=round(p.mean() * 100, 1), observed=round(yy.mean() * 100, 1)))
cal = pd.DataFrame(cal); cal.to_csv(R / "calibration.csv", index=False)

pd.set_option("display.width", 220)
for name, d in [("Comparisons (corrected t-test)", comp), ("AUC with 95% CI", aucs), ("Switzerland specificity (Wilson CI)", sw),
                ("Swarm comparison without Switzerland", sens), ("Calibration", cal)]:
    print(f"\n{name}:"); print(d.to_string(index=False))
