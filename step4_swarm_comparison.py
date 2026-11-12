"""
step4_swarm_comparison.py — هل الخوارزمية "الأحسن" تبقى الأحسن بعد تصحيح الأخطاء؟
الخوارزميات: All features, PSO, GWO, PSOGWO, LGWO   |   المصنفات: SVM و Stacking
البيانات: 4 مستشفيات UCI (نظيفة) + Kaggle 1025 (فيها تكرار)
التقييم: 10-fold CV مكرر 3 مرات (30 fold)، وكل المعالجة داخل fold التدريب.
الإحصاء:
  - الطيات مش مستقلة، فنحسبوا متوسط كل تكرار (repeat) في كل مستشفى، ونطبقوا Friedman
    على هذي المتوسطات (4 مستشفيات × 3 تكرارات = 12 كتلة) — بيانات UCI بس.
  - الميزانية: نسجلوا عدد طلبات دالة اللياقة (calls) والتقييمات الفعلية (evals) والزمن لكل خوارزمية.

الاستعمال: python step4_swarm_comparison.py     (حوالي 25 دقيقة على Colab)
"""
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import friedmanchisquare, rankdata
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from fs_algorithms import ALGORITHMS, Fitness

D = Path("datasets"); OUT = Path("results"); OUT.mkdir(exist_ok=True)
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
METHODS = ["All features"] + list(ALGORITHMS)


def load(name):
    if name == "kaggle1025 (duplicated)":
        k = pd.read_csv(D / "heart1025_raw.csv")
        return k.drop(columns="target").values.astype(float), k.target.values
    d = pd.read_csv(D / f"uci_original/processed.{name}.data", header=None, names=COLS, na_values="?")
    d.loc[d.chol == 0, "chol"] = np.nan
    return d.drop(columns="num").values.astype(float), (d.num > 0).astype(int).values


def classifiers():
    return {"SVM": SVC(random_state=0),
            "Stacking": StackingClassifier(
                estimators=[("rf", RandomForestClassifier(n_estimators=100, random_state=0)),
                            ("xgb", XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss",
                                                  random_state=0, verbosity=0)),
                            ("svm", SVC(random_state=0)), ("knn", KNeighborsClassifier(7))],
                final_estimator=LogisticRegression(max_iter=1000), cv=5)}


def run_fold(ds, X, y, k, tr, te):
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr])
    sc = StandardScaler().fit(imp.transform(X[tr]))
    Xtr, Xte = sc.transform(imp.transform(X[tr])), sc.transform(imp.transform(X[te]))
    Xtr, ytr = SMOTE(random_state=k).fit_resample(Xtr, y[tr])
    rows = []
    for m in METHODS:
        calls = evals = 0; secs = 0.0
        if m == "All features":
            mask = np.ones(X.shape[1], bool)
        else:
            fit = Fitness(Xtr, ytr, seed=k); t0 = time.perf_counter()
            mask = ALGORITHMS[m](fit, X.shape[1], seed=k)[0]
            secs = time.perf_counter() - t0; calls, evals = fit.calls, fit.evals
            mask = mask if mask.any() else np.ones(X.shape[1], bool)
        for cn, clf in classifiers().items():
            p = clf.fit(Xtr[:, mask], ytr).predict(Xte[:, mask])
            rows.append(dict(dataset=ds, fold=k, repeat=k // 10, method=m, clf=cn, n_feat=int(mask.sum()),
                             acc=accuracy_score(y[te], p), f1=f1_score(y[te], p),
                             fitness_calls=calls, fitness_evals=evals, fs_seconds=round(secs, 2)))
    return rows


if __name__ == "__main__":
    rows = []
    for ds in ["cleveland", "hungarian", "switzerland", "va", "kaggle1025 (duplicated)"]:
        X, y = load(ds)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        out = Parallel(n_jobs=2)(delayed(run_fold)(ds, X, y, k, tr, te) for k, (tr, te) in enumerate(cv.split(X, y)))
        rows += [r for o in out for r in o]; print(f"{ds} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "swarm_folds.csv", index=False)

    # ---- Friedman على متوسط كل تكرار (UCI بس): كتلة = (مستشفى، تكرار) ----
    uci = df[df.dataset != "kaggle1025 (duplicated)"]
    fr = []
    for cn, g in uci.groupby("clf"):
        B = g.groupby(["dataset", "repeat", "method"]).acc.mean().unstack("method")[METHODS]
        ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, B.values).mean(0)   # 1 = الأحسن
        p = friedmanchisquare(*[B[m] for m in METHODS]).pvalue
        for m, r in zip(METHODS, ranks):
            fr.append(dict(clf=cn, method=m, mean_rank=round(r, 2), friedman_p=round(p, 4), n_blocks=len(B)))
    fr = pd.DataFrame(fr); fr.to_csv(OUT / "swarm_friedman.csv", index=False)

    summ = []
    for (ds, cn, m), h in df.groupby(["dataset", "clf", "method"]):
        f = fr[(fr.clf == cn) & (fr.method == m)]
        summ.append(dict(dataset=ds, clf=cn, method=m, acc=round(h.acc.mean() * 100, 2),
                         std=round(h.acc.std() * 100, 2), n_feat=round(h.n_feat.mean(), 1),
                         fitness_calls=round(h.fitness_calls.mean()), fitness_evals=round(h.fitness_evals.mean()),
                         fs_seconds=round(h.fs_seconds.mean(), 2),
                         mean_rank=f.mean_rank.iloc[0] if ds != "kaggle1025 (duplicated)" else "",
                         friedman_p=f.friedman_p.iloc[0] if ds != "kaggle1025 (duplicated)" else ""))
    summ = pd.DataFrame(summ); summ.to_csv(OUT / "swarm_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.to_string(index=False))
    print("\nFriedman (blocks = hospital x repeat, UCI only):"); print(fr.to_string(index=False))
