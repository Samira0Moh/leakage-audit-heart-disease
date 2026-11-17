"""
step3b_reproduce_intermediate.py — the intermediate condition for the reproduction (Section 2.5):
the leak-free protocol (5 x 10 CV, scaling and PSOGWO fitted inside the training folds)
applied to the duplicated 1,025-row Kaggle file. It separates the effect of the protocol from the effect of the data:
    published protocol on Kaggle  ->  leak-free protocol on Kaggle  ->  leak-free protocol on UCI Cleveland
Usage: python step3b_reproduce_intermediate.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from fs_algorithms import psogwo, Fitness

D = Path("datasets"); OUT = Path("results")
k = pd.read_csv(D / "heart1025_raw.csv")
X = k.drop(columns="target").values.astype(float); y = k.target.values

def fold(i, tr, te):
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    m, _, _ = psogwo(Fitness(Xtr, y[tr], seed=i, clf=SVC()), X.shape[1], seed=i)
    if not m.any(): m = np.ones(X.shape[1], bool)
    return (accuracy_score(y[te], SVC().fit(Xtr, y[tr]).predict(Xte)),
            accuracy_score(y[te], SVC().fit(Xtr[:, m], y[tr]).predict(Xte[:, m])), m.sum())

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
B = np.array(Parallel(n_jobs=2)(delayed(fold)(i, tr, te) for i, (tr, te) in enumerate(cv.split(X, y))))
res = pd.DataFrame([dict(method="SVM (all features)", leakfree_kaggle=f"{B[:,0].mean()*100:.2f} ± {B[:,0].std()*100:.2f}"),
                    dict(method="PSOGWO + SVM", leakfree_kaggle=f"{B[:,1].mean()*100:.2f} ± {B[:,1].std()*100:.2f}")])
res.to_csv(OUT / "reproduce_intermediate.csv", index=False)
print(res.to_string(index=False)); print("mean features:", B[:, 2].mean())
