"""
step8_exhaustive.py — هل المشكلة في المُحسِّن (PSO/GWO) ولا في دالة اللياقة نفسها؟

13 خاصية => 8,191 مجموعة جزئية بس، فنقدروا نجربوهم كلهم (exhaustive search).
في كل fold تدريب (بعد التعويض والتطبيع و SMOTE داخل التدريب):
  1) All features                : SVM بكل الخصائص
  2) Exhaustive (KNN fitness)     : أحسن مجموعة من الـ 8,191 حسب نفس دالة اللياقة (KNN، 3-fold داخلي)
  3) GWO (KNN fitness)            : زي step4
  4) GWO (SVM fitness)            : دالة اللياقة تستعمل نفس المصنف النهائي (SVM) => ما فيش عدم تطابق
والمصنف النهائي في الكل SVM. نسجلوا كمان الدقة الداخلية (inner) لأحسن مجموعة، باش نشوفوا الفرق
بين اللي "تشوفه" دالة اللياقة واللي يصير فعلاً على الاختبار.
الحساب السريع: KNN بمصفوفات مسافات محسوبة مسبقاً لكل خاصية (نفس الطيات الداخلية ونفس k=5).

الاستعمال: python step8_exhaustive.py      (3×10 CV على 4 مستشفيات)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import friedmanchisquare
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from fs_algorithms import gwo, Fitness

D = Path("datasets"); OUT = Path("results"); OUT.mkdir(exist_ok=True)
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
ALPHA = 0.99


def load_uci(name):
    d = pd.read_csv(D / f"uci_original/processed.{name}.data", header=None, names=COLS, na_values="?")
    d.loc[d.chol == 0, "chol"] = np.nan
    return d.drop(columns="num").values.astype(float), (d.num > 0).astype(int).values


def exhaustive(X, y, seed, k=5):
    """نفس دالة اللياقة في fs_algorithms (alpha*الخطأ + (1-alpha)*النسبة، KNN k=5، 3-fold) لكل الـ 8,191 مجموعة."""
    n, D_ = X.shape
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y))
    sq = np.stack([(X[:, None, j] - X[None, :, j]) ** 2 for j in range(D_)])   # D × n × n
    best = (np.inf, None, None)
    for code in range(1, 2 ** D_):
        mask = np.array([(code >> b) & 1 for b in range(D_)], bool)
        dist = sq[mask].sum(0)
        acc = 0.0
        for tr, te in folds:
            dd = dist[np.ix_(te, tr)]
            nn = np.argpartition(dd, k, axis=1)[:, :k]
            pred = (y[tr][nn].mean(1) > 0.5).astype(int)
            acc += (pred == y[te]).mean() / 3
        fit = ALPHA * (1 - acc) + (1 - ALPHA) * mask.sum() / D_
        if fit < best[0]: best = (fit, mask, acc)
    return best[1], best[2]


def run_fold(ds, X, y, k, tr, te):
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr])
    sc = StandardScaler().fit(imp.transform(X[tr]))
    Xtr, Xte = sc.transform(imp.transform(X[tr])), sc.transform(imp.transform(X[te]))
    Xtr, ytr = SMOTE(random_state=k).fit_resample(Xtr, y[tr])
    m_ex, inner_ex = exhaustive(Xtr, ytr, k)
    m_gk = gwo(Fitness(Xtr, ytr, seed=k), Xtr.shape[1], seed=k)[0]
    fs = Fitness(Xtr, ytr, seed=k, clf=SVC()); m_gs = gwo(fs, Xtr.shape[1], seed=k)[0]
    rows = []
    for name, mask, inner in [("All features", np.ones(13, bool), np.nan), ("Exhaustive (KNN fitness)", m_ex, inner_ex),
                              ("GWO (KNN fitness)", m_gk, np.nan), ("GWO (SVM fitness)", m_gs, np.nan)]:
        mask = mask if mask.any() else np.ones(13, bool)
        acc = accuracy_score(y[te], SVC().fit(Xtr[:, mask], ytr).predict(Xte[:, mask]))
        rows.append(dict(dataset=ds, fold=k, repeat=k // 10, method=name, n_feat=int(mask.sum()),
                         acc=acc, inner_acc=inner))
    return rows


if __name__ == "__main__":
    rows = []
    for ds in ["cleveland", "hungarian", "switzerland", "va"]:
        X, y = load_uci(ds)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)   # نفس طيات step4
        out = Parallel(n_jobs=2)(delayed(run_fold)(ds, X, y, k, tr, te) for k, (tr, te) in enumerate(cv.split(X, y)))
        rows += [r for o in out for r in o]; print(ds, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "exhaustive_folds.csv", index=False)
    M = ["All features", "Exhaustive (KNN fitness)", "GWO (KNN fitness)", "GWO (SVM fitness)"]
    B = df.groupby(["dataset", "repeat", "method"]).acc.mean().unstack("method")[M]
    p = friedmanchisquare(*[B[m] for m in M]).pvalue
    summ = df.groupby("method").agg(acc=("acc", "mean"), acc_sd=("acc", "std"), n_feat=("n_feat", "mean"),
                                    inner_acc=("inner_acc", "mean")).reindex(M)
    summ = (summ * [100, 100, 1, 100]).round(2); summ["friedman_p"] = round(p, 4)
    per_ds = (df.groupby(["dataset", "method"]).acc.mean().unstack("method")[M] * 100).round(2)
    summ.to_csv(OUT / "exhaustive_summary.csv"); per_ds.to_csv(OUT / "exhaustive_by_hospital.csv")
    pd.set_option("display.width", 200); print(summ.to_string()); print(per_ds.to_string())
