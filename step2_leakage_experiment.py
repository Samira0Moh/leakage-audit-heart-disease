"""
step2_leakage_experiment.py — كم يضخم كل نوع من التسريب (data leakage) الدقة؟
البيانات: 4 مستشفيات مستقلة من UCI (Cleveland, Hungarian, Switzerland, VA)
اختيار الخصائص: GWO  |  المصنفات: SVM و Stacking

الحالات:
  (كل حالة تختلف عن C0 في خطأ واحد بس، والباقي نفسه)
  C0  صحيح      : كل المعالجة (تعويض الناقص، التطبيع، SMOTE، GWO) داخل fold التدريب فقط
  L1  تطبيع قبل : التعويض والتطبيع على كل البيانات قبل التقسيم
  L2  GWO قبل   : اختيار الخصائص بـ GWO على كل البيانات قبل التقسيم
  L3  SMOTE قبل : موازنة الفئات بـ SMOTE على كل البيانات قبل التقسيم
  L4a تكرار مضبوط : Cleveland الأصلي نكرروا سجلاته عشوائياً لين يوصل 1025 صف (نفس حجم Kaggle)
                    والباقي كله ثابت => أثر التكرار بروحه
  L4b ملف Kaggle   : ملف Kaggle الحقيقي (1025) — يختلف في التكرار والترميز معاً، نحللوه بروحه

تصميم الحالات (كل حالة تغير خطوة وحدة عن C0):
  L2  : GWO على كل البيانات بنفس دالة اللياقة ونفس SMOTE زي C0، وبذرة مختلفة لكل fold (مش قناع واحد)
  L3  : SMOTE على كل البيانات، لكن طيات الاختبار هي نفسها طيات C0 (سجلات أصلية بس)،
        والتعويض والتطبيع داخل fold التدريب => الفرق الوحيد عن C0 هو SMOTE قبل التقسيم
  L4a/L4b: البيانات مختلفة عن C0 => مقارنة غير مزدوجة (unpaired) بالتباين المصحح
  مقاييس إضافية: AUC و sensitivity و specificity و Brier

الإحصاء:
  الطيات في الـ CV المكرر تتشارك بيانات التدريب => مش مستقلة => Wilcoxon العادي متفائل.
  نستعملوا corrected resampled t-test (Nadeau & Bengio 2003) + فترة ثقة 95% للفرق.
  Wilcoxon نخلوه كاختبار ثانوي بس.

+ تحقق خارجي: تدريب على Cleveland واختبار على كل مستشفى ثاني

الاستعمال: python step2_leakage_experiment.py      (5×10 CV، حوالي 25 دقيقة)
           python step2_leakage_experiment.py 1    (10-fold مرة وحدة، حوالي 5 دقايق)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import wilcoxon, t as t_dist
import sys
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, brier_score_loss, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from fs_algorithms import gwo, Fitness

D = Path("datasets"); OUT = Path("results"); OUT.mkdir(exist_ok=True)
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
HOSPITALS = ["cleveland", "hungarian", "switzerland", "va"]
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 5   # 5×10 CV (للتجربة السريعة: python step2_leakage_experiment.py 1)


def load_uci(name):
    d = pd.read_csv(D / f"uci_original/processed.{name}.data", header=None, names=COLS, na_values="?")
    d.loc[d.chol == 0, "chol"] = np.nan          # كوليسترول = 0 مستحيل طبياً => قيمة ناقصة
    X = d.drop(columns="num").values.astype(float)
    y = (d.num > 0).astype(int).values           # 1 = مريض (حسب التوثيق الأصلي)
    return X, y


def corrected_ttest(d, test_train_ratio=1 / 9):
    """Nadeau & Bengio: الفرق d بين حالتين في كل fold.
    التباين يتضخم بـ (1/k + n_test/n_train) لأن الطيات مش مستقلة. في 10-fold النسبة = 1/9."""
    k = len(d); m = d.mean(); v = d.var(ddof=1)
    se = np.sqrt((1 / k + test_train_ratio) * v)
    if se == 0: return m, m, m, np.nan
    tc = t_dist.ppf(0.975, k - 1)
    p = 2 * t_dist.sf(abs(m / se), k - 1)
    return m, m - tc * se, m + tc * se, p


def classifiers():
    return {
        "SVM": SVC(probability=True, random_state=0),
        "Stacking": StackingClassifier(
            estimators=[("rf", RandomForestClassifier(n_estimators=200, random_state=0)),
                        ("xgb", XGBClassifier(n_estimators=200, max_depth=3, eval_metric="logloss",
                                              random_state=0, verbosity=0)),
                        ("svm", SVC(random_state=0)), ("knn", KNeighborsClassifier(7))],
            final_estimator=LogisticRegression(max_iter=1000), cv=5),
    }


def prep_fit(X):                       # تعويض + تطبيع (يتعلم من البيانات اللي تعطيه لها)
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)  # Switzerland: chol ناقص بالكامل
    sc = StandardScaler().fit(imp.transform(X))
    return lambda Z: sc.transform(imp.transform(Z))


def select(X, y, seed, scoring="accuracy"):   # GWO لاختيار الخصائص
    mask, _, _ = gwo(Fitness(X, y, seed=seed, scoring=scoring), X.shape[1], seed=seed)
    return mask if mask.any() else np.ones(X.shape[1], bool)


def metrics(yte, pred, prob):
    return dict(acc=accuracy_score(yte, pred), f1=f1_score(yte, pred),
                auc=roc_auc_score(yte, prob) if len(set(yte)) > 1 else np.nan,
                sens=recall_score(yte, pred),
                spec=recall_score(yte, pred, pos_label=0) if (yte == 0).any() else np.nan,   # no healthy patient in fold => undefined
                brier=brier_score_loss(yte, prob))


def run_fold(ds, cond, X, y, tr, te, k, synth=None):
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    f = prep_fit(X) if cond.startswith("L1") else prep_fit(Xtr)     # L1 ❌ يشوف الاختبار
    if cond.startswith("L3"):
        # ❌ سجلات SMOTE الاصطناعية مبنية من كل المرضى (حتى مرضى الاختبار) وتنضاف للتدريب
        Xtr = np.vstack([Xtr, synth[0]]); ytr = np.concatenate([ytr, synth[1]])
        f = prep_fit(Xtr)
        Xtr, Xte = f(Xtr), f(Xte)
    else:
        Xtr, Xte = f(Xtr), f(Xte)
        Xtr, ytr = SMOTE(random_state=k).fit_resample(Xtr, ytr)      # ✅ SMOTE على التدريب بس
    if cond.startswith("L2"):
        # ❌ GWO يختار الخصائص من كل البيانات (نفس دالة اللياقة ونفس SMOTE زي C0، بذرة الـ fold)
        g = prep_fit(X); Xa, ya = SMOTE(random_state=k).fit_resample(g(X), y)
        mask = select(Xa, ya, k)
    else:
        mask = select(Xtr, ytr, k)
    rows = []
    for cn, clf in classifiers().items():
        clf.fit(Xtr[:, mask], ytr)
        pred = clf.predict(Xte[:, mask]); prob = clf.predict_proba(Xte[:, mask])[:, 1]
        rows.append(dict(dataset=ds, condition=cond, clf=cn, fold=k, n_feat=int(mask.sum()),
                         n_test=len(te), **metrics(yte, pred, prob)))
    return rows


def run_condition(ds, cond, X, y):
    synth = None
    if cond.startswith("L3"):
        # SMOTE على كل البيانات (بعد تعويض مؤقت لازم لـ SMOTE)، ونحتفظوا بالسجلات الاصطناعية بس
        Xi = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
        Xs, ys = SMOTE(random_state=0).fit_resample(Xi, y)
        synth = (Xs[len(y):], ys[len(y):])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=REPEATS, random_state=42)
    out = Parallel(n_jobs=2)(delayed(run_fold)(ds, cond, X, y, tr, te, k, synth)
                             for k, (tr, te) in enumerate(cv.split(X, y)))
    return [r for o in out for r in o]


def unpaired_corrected(a, b, test_train_ratio=1 / 9):
    """مقارنة غير مزدوجة (L4): كل حالة بتباينها المصحح (Nadeau-Bengio) والفرق بينهم."""
    k = len(a)
    se = np.sqrt((1 / k + test_train_ratio) * (a.var(ddof=1) + b.var(ddof=1)))
    m = a.mean() - b.mean(); tc = t_dist.ppf(0.975, k - 1)
    p = 2 * t_dist.sf(abs(m / se), k - 1) if se > 0 else np.nan
    return m, m - tc * se, m + tc * se, p


if __name__ == "__main__":
    rows = []
    conds = ["C0 correct", "L1 scaling before split", "L2 GWO before split", "L3 SMOTE before split"]
    ONLY = sys.argv[2] if len(sys.argv) > 2 else None    # مثال: python step2_leakage_experiment.py 5 L2
    if ONLY:
        old = pd.read_csv(OUT / "leakage_folds.csv")
        conds = [c for c in conds if c.startswith(ONLY)]
        rows = old[~old.condition.str.startswith(ONLY)].to_dict("records")
    for ds in HOSPITALS:
        X, y = load_uci(ds)
        print(f"{ds}: {X.shape}, disease rate = {y.mean():.2f}", flush=True)
        for c in conds:
            rows += run_condition(ds, c, X, y)
    if not ONLY or ONLY.startswith("L4"):
        # L4a: تكرار مضبوط — نفس بيانات Cleveland الأصلية، نختاروا 1025 صف بالإرجاع (with replacement)
        Xc, yc = load_uci("cleveland")
        idx = np.random.default_rng(0).choice(len(yc), size=1025, replace=True)
        print(f"L4a: 1025 rows, unique patients = {len(np.unique(idx))}", flush=True)
        rows += run_condition("cleveland", "L4a controlled duplication", Xc[idx], yc[idx])
        # L4b: ملف Kaggle الحقيقي (تكرار + ترميز مختلف) — نحللوه بروحه
        k = pd.read_csv(D / "heart1025_raw.csv")
        rows += run_condition("cleveland", "L4b real Kaggle file (1025)",
                              k.drop(columns="target").values.astype(float), k.target.values)
    df = pd.DataFrame(rows); df.to_csv(OUT / "leakage_folds.csv", index=False)

    # ---- الجدول: الفرق عن C0 + فترة ثقة 95% + corrected t-test (+ Wilcoxon ثانوي) ----
    # L1, L2, L3: نفس طيات الاختبار => مقارنة مزدوجة.  L4a/L4b: بيانات مختلفة => غير مزدوجة.
    summ = []
    for (ds, cn), g in df.groupby(["dataset", "clf"]):
        base = g[g.condition == "C0 correct"].sort_values("fold").acc.values
        for cond, h in g.groupby("condition"):
            a = h.sort_values("fold").acc.values
            r = dict(dataset=ds, clf=cn, condition=cond, acc=round(a.mean() * 100, 2),
                     std=round(a.std() * 100, 2), inflation=round((a.mean() - base.mean()) * 100, 2),
                     ci_low="", ci_high="", p_value="", wilcoxon_p="")
            r.update(auc=round(h.auc.mean() * 100, 2), sens=round(h.sens.mean() * 100, 2),
                     spec=round(h.spec.mean() * 100, 2), brier=round(h.brier.mean(), 3),
                     n_test=round(h.n_test.mean(), 1))
            if cond != "C0 correct":
                m, lo, hi, p = (unpaired_corrected(a, base) if cond.startswith("L4")
                                else corrected_ttest(a - base))
                w = (wilcoxon(a, base).pvalue if np.any(a != base) else np.nan) if not cond.startswith("L4") else np.nan
                r.update(ci_low=round(lo * 100, 2), ci_high=round(hi * 100, 2),
                         p_value=round(p, 4) if p == p else "", wilcoxon_p=round(w, 4) if w == w else "")
            summ.append(r)
    summ = pd.DataFrame(summ); summ.to_csv(OUT / "leakage_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))

    # ---- تحقق خارجي: تدريب على Cleveland ← اختبار على المستشفيات الأخرى ----
    # نسختين: (أ) كل الخصائص 13، (ب) الخصائص المتوفرة فعلاً في مستشفى الاختبار (ناقص < 50%)
    Xc, yc = load_uci("cleveland")
    ext = []
    for ds in HOSPITALS[1:]:
        Xo, yo = load_uci(ds)
        miss = np.isnan(Xo).mean(0)
        for version, cols in [("all 13 features", np.arange(13)), ("available features", np.where(miss < 0.5)[0])]:
            f = prep_fit(Xc[:, cols]); Xc_ = f(Xc[:, cols])
            mask = select(*SMOTE(random_state=0).fit_resample(Xc_, yc), 0)
            for cn, clf in classifiers().items():
                clf.fit(Xc_[:, mask], yc)
                Z = f(Xo[:, cols])[:, mask]
                pred = clf.predict(Z); prob = clf.predict_proba(Z)[:, 1]
                mt = metrics(yo, pred, prob)
                ext.append(dict(train="cleveland", test=ds, features=version, n_features_used=len(cols),
                                clf=cn, acc=round(mt["acc"] * 100, 2),
                                bal_acc=round(balanced_accuracy_score(yo, pred) * 100, 2),
                                auc=round(mt["auc"] * 100, 2), sens=round(mt["sens"] * 100, 2),
                                spec=round(mt["spec"] * 100, 2), f1=round(mt["f1"] * 100, 2),
                                brier=round(mt["brier"], 3), majority_acc=round(max(yo.mean(), 1 - yo.mean()) * 100, 1),
                                prevalence=round(yo.mean() * 100, 1),
                                missing_pct=round(np.isnan(Xo).mean() * 100, 1),
                                dropped=",".join(COLS[c] for c in range(13) if c not in cols)))
    ext = pd.DataFrame(ext); ext.to_csv(OUT / "external_validation.csv", index=False)
    print("\nExternal validation (train on Cleveland):"); print(ext.to_string(index=False))
