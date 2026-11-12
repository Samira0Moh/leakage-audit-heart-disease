"""
step3_reproduce_original.py — إعادة نتيجة الورقة الأصلية (Pratama & Putra: PSOGWO + SVM)
الورقة نشرت: SVM بدون اختيار خصائص = 91.22%  |  PSOGWO + SVM = 93.66%

البروتوكول (أ) "زي الورقة":   بيانات Kaggle 1025 بتكرارها، StandardScaler و PSOGWO على كل البيانات،
                               ثم تقسيم واحد 80/20. نكرره على 20 تقسيم عشوائي عشان نشوفوا التذبذب.
البروتوكول (ب) "الصحيح":      بيانات UCI Cleveland الأصلية (303 مريض)، و 10-fold CV مكرر 5 مرات،
                               وكل المعالجة و PSOGWO داخل fold التدريب فقط.

الاستعمال: python step3_reproduce_original.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from fs_algorithms import psogwo, Fitness

D = Path("datasets"); OUT = Path("results"); OUT.mkdir(exist_ok=True)
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
PAPER = {"SVM (all features)": 91.22, "PSOGWO + SVM": 93.66}


def fs_psogwo(X, y, seed):
    mask, _, _ = psogwo(Fitness(X, y, seed=seed, clf=SVC()), X.shape[1], seed=seed)
    return mask if mask.any() else np.ones(X.shape[1], bool)


# ---------------- (أ) بروتوكول الورقة ----------------
k = pd.read_csv(D / "heart1025_raw.csv")
Xk = StandardScaler().fit_transform(k.drop(columns="target").values.astype(float))   # ❌ على كل البيانات
yk = k.target.values
mask_k = fs_psogwo(Xk, yk, 0)                                                       # ❌ على كل البيانات

def paper_split(s):
    tr, te = train_test_split(np.arange(len(yk)), test_size=0.2, random_state=s, stratify=yk)
    a_all = accuracy_score(yk[te], SVC().fit(Xk[tr], yk[tr]).predict(Xk[te]))
    a_fs = accuracy_score(yk[te], SVC().fit(Xk[tr][:, mask_k], yk[tr]).predict(Xk[te][:, mask_k]))
    return a_all, a_fs

A = np.array([paper_split(s) for s in range(20)]) * 100

# ---------------- (ب) البروتوكول الصحيح ----------------
u = pd.read_csv(D / "uci_original/processed.cleveland.data", header=None, names=COLS, na_values="?")
Xu = u.drop(columns="num").values.astype(float); yu = (u.num > 0).astype(int).values

def correct_fold(k_, tr, te):
    imp = SimpleImputer(strategy="median").fit(Xu[tr]); sc = StandardScaler().fit(imp.transform(Xu[tr]))
    Xtr, Xte = sc.transform(imp.transform(Xu[tr])), sc.transform(imp.transform(Xu[te]))
    m = fs_psogwo(Xtr, yu[tr], k_)
    return (accuracy_score(yu[te], SVC().fit(Xtr, yu[tr]).predict(Xte)),
            accuracy_score(yu[te], SVC().fit(Xtr[:, m], yu[tr]).predict(Xte[:, m])), m.sum())

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
B = np.array(Parallel(n_jobs=2)(delayed(correct_fold)(i, tr, te) for i, (tr, te) in enumerate(cv.split(Xu, yu))))

rows = []
for i, name in enumerate(PAPER):
    rows.append(dict(method=name, paper_reported=PAPER[name],
                     paper_protocol=f"{A[:, i].mean():.2f} ± {A[:, i].std():.2f} (range {A[:, i].min():.1f}–{A[:, i].max():.1f})",
                     correct_protocol=f"{B[:, i].mean()*100:.2f} ± {B[:, i].std()*100:.2f}",
                     inflation=round(A[:, i].mean() - B[:, i].mean() * 100, 2)))
res = pd.DataFrame(rows); res.to_csv(OUT / "reproduce_original.csv", index=False)
pd.set_option("display.width", 220)
print(res.to_string(index=False))
print(f"\nالخصائص اللي اختارها PSOGWO (بروتوكول الورقة): {int(mask_k.sum())} من 13 "
      f"| متوسط عدد الخصائص (الصحيح): {B[:, 2].mean():.1f}")
