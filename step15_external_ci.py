"""
step15_external_ci.py — 95% bootstrap confidence intervals (2,000 patient-level resamples) and calibration
for the external validation of step2 (train on Cleveland, test on the other hospitals). Same models and seeds as step2.
Usage: python step15_external_ci.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import step2_leakage_experiment as s2

rng = np.random.default_rng(0); rows = []
Xc, yc = s2.load_uci("cleveland")
for ds in s2.HOSPITALS[1:]:
    Xo, yo = s2.load_uci(ds); miss = np.isnan(Xo).mean(0)
    for version, cols in [("all 13 features", np.arange(13)), ("available features", np.where(miss < 0.5)[0])]:
        f = s2.prep_fit(Xc[:, cols]); Xc_ = f(Xc[:, cols])
        mask = s2.select(*SMOTE(random_state=0).fit_resample(Xc_, yc), 0)
        for cn, clf in s2.classifiers().items():
            clf.fit(Xc_[:, mask], yc); Z = f(Xo[:, cols])[:, mask]
            pred = clf.predict(Z); prob = clf.predict_proba(Z)[:, 1]
            acc, auc = accuracy_score(yo, pred), roc_auc_score(yo, prob)
            ba, bu = [], []
            for _ in range(2000):
                b = rng.integers(0, len(yo), len(yo))
                ba.append(accuracy_score(yo[b], pred[b]))
                if len(set(yo[b])) > 1: bu.append(roc_auc_score(yo[b], prob[b]))
            p = np.clip(prob, 1e-4, 1 - 1e-4); lp = np.log(p / (1 - p))
            slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], yo).coef_[0][0]
            rows.append(dict(test=ds, features=version, clf=cn, acc=round(acc * 100, 1),
                             acc_ci=f"{np.percentile(ba, 2.5)*100:.1f}–{np.percentile(ba, 97.5)*100:.1f}",
                             auc=round(auc * 100, 1), auc_ci=f"{np.percentile(bu, 2.5)*100:.1f}–{np.percentile(bu, 97.5)*100:.1f}",
                             cal_slope=round(slope, 2), mean_pred=round(prob.mean() * 100, 1), prevalence=round(yo.mean() * 100, 1)))
out = pd.DataFrame(rows); out.to_csv("results/external_validation_ci.csv", index=False); print(out.to_string(index=False))
