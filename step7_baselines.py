"""
step7_baselines.py — خطوط مرجعية + ضبط المعاملات + L5 (الضبط على الاختبار) + فحص KNN k=1

(أ) على 4 مستشفيات UCI، نفس طيات 5×10 CV، بدون اختيار خصائص وبدون SMOTE:
    - Logistic Regression (خط مرجعي سريري معروف)
    - SVM مضبوط بشكل صحيح: nested CV (الضبط داخل fold التدريب بـ 5-fold داخلي)
    - L5 "ضبط على الاختبار": نجربوا نفس الشبكة ونختاروا الأحسن على fold الاختبار نفسه ❌
      => الفرق بين L5 و nested = تضخم الضبط على الاختبار
(ب) فحص Almustafa: KNN بـ k=1 على ملف Kaggle 1025 بتكراره، وبعد حذف المكرر، وعلى UCI Cleveland

الاستعمال: python step7_baselines.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, brier_score_loss
from joblib import Parallel, delayed

D = Path("datasets"); OUT = Path("results"); OUT.mkdir(exist_ok=True)
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
HOSPITALS = ["cleveland", "hungarian", "switzerland", "va"]
GRID = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]}


def load_uci(name):
    d = pd.read_csv(D / f"uci_original/processed.{name}.data", header=None, names=COLS, na_values="?")
    d.loc[d.chol == 0, "chol"] = np.nan
    return d.drop(columns="num").values.astype(float), (d.num > 0).astype(int).values


def pipe(est):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), est)


def mets(y, pred, score, prob=None):
    return dict(acc=accuracy_score(y, pred), auc=roc_auc_score(y, score) if len(set(y)) > 1 else np.nan,
                sens=recall_score(y, pred), spec=recall_score(y, pred, pos_label=0),
                brier=brier_score_loss(y, prob) if prob is not None else np.nan)


def fold(ds, X, y, k, tr, te):
    rows = []
    lr = pipe(LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
    p = lr.predict_proba(X[te])[:, 1]
    rows.append(dict(dataset=ds, fold=k, model="Logistic regression", **mets(y[te], (p > .5).astype(int), p, p)))
    # nested: الضبط داخل التدريب بس ✅
    gs = GridSearchCV(pipe(SVC()), GRID, cv=StratifiedKFold(5, shuffle=True, random_state=k)).fit(X[tr], y[tr])
    s = gs.decision_function(X[te])
    rows.append(dict(dataset=ds, fold=k, model="SVM tuned (nested CV)", **mets(y[te], gs.predict(X[te]), s)))
    # L5: نختاروا الإعداد اللي يعطي أحسن دقة على الاختبار نفسه ❌
    best = None
    for C in GRID["svc__C"]:
        for g in GRID["svc__gamma"]:
            m = pipe(SVC(C=C, gamma=g)).fit(X[tr], y[tr]); pr = m.predict(X[te])
            a = accuracy_score(y[te], pr)
            if best is None or a > best[0]: best = (a, pr, m.decision_function(X[te]))
    rows.append(dict(dataset=ds, fold=k, model="SVM tuned on test fold (L5)", **mets(y[te], best[1], best[2])))
    return rows


def corrected(d, ratio=1 / 9):
    k = len(d); se = np.sqrt((1 / k + ratio) * d.var(ddof=1)); m = d.mean(); tc = t_dist.ppf(.975, k - 1)
    return m, m - tc * se, m + tc * se, 2 * t_dist.sf(abs(m / se), k - 1)


if __name__ == "__main__":
    rows = []
    for ds in HOSPITALS:
        X, y = load_uci(ds)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)   # نفس طيات step2
        out = Parallel(n_jobs=2)(delayed(fold)(ds, X, y, k, tr, te) for k, (tr, te) in enumerate(cv.split(X, y)))
        rows += [r for o in out for r in o]; print(ds, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "baselines_folds.csv", index=False)
    summ = []
    for (ds, m), g in df.groupby(["dataset", "model"]):
        r = dict(dataset=ds, model=m, **{c: round(g[c].mean() * (1 if c == "brier" else 100), 3 if c == "brier" else 2)
                                          for c in ["acc", "auc", "sens", "spec", "brier"]}, acc_sd=round(g.acc.std() * 100, 2))
        if m.endswith("(L5)"):
            base = df[(df.dataset == ds) & (df.model == "SVM tuned (nested CV)")].sort_values("fold").acc.values
            d_, lo, hi, p = corrected(g.sort_values("fold").acc.values - base)
            r.update(inflation_vs_nested=round(d_ * 100, 2), ci_low=round(lo * 100, 2), ci_high=round(hi * 100, 2), p_value=round(p, 4))
        summ.append(r)
    summ = pd.DataFrame(summ); summ.to_csv(OUT / "baselines_summary.csv", index=False)
    pd.set_option("display.width", 220); print(summ.to_string(index=False))

    # ---- (ب) فحص Almustafa: KNN k=1 ----
    k = pd.read_csv(D / "heart1025_raw.csv")
    Xk, yk = k.drop(columns="target").values.astype(float), k.target.values
    kd = k.drop_duplicates(); Xd, yd = kd.drop(columns="target").values.astype(float), kd.target.values
    Xc, yc = load_uci("cleveland")
    knn = []
    for name, X, y in [("Kaggle 1,025 (with duplicates)", Xk, yk), ("Kaggle, duplicates removed (302)", Xd, yd),
                       ("UCI Cleveland (303)", Xc, yc)]:
        accs = []
        for tr, te in RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42).split(X, y):
            accs.append(accuracy_score(y[te], pipe(KNeighborsClassifier(1)).fit(X[tr], y[tr]).predict(X[te])))
        knn.append(dict(data=name, rows=len(y), knn1_acc=round(np.mean(accs) * 100, 2), sd=round(np.std(accs) * 100, 2)))
    knn = pd.DataFrame(knn); knn.to_csv(OUT / "knn1_check.csv", index=False)
    print("\nKNN (k = 1), 5×10 CV:"); print(knn.to_string(index=False))
