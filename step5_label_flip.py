"""
step5_label_flip.py — شن يصير لما الباحث يصدق توثيق Kaggle (target=1 = مريض) والتصنيف في الحقيقة مقلوب؟
الدقة ما تتغيرش، لكن التفسير الطبي ينقلب.

نقارن:
  (أ) "اللي يفهمه الباحث": بيانات Kaggle، ويفترض إن target=1 مريض
  (ب) "الحقيقة": نفس المرضى بتصنيف UCI الأصلي (1 = مريض)
ونطلعوا: الدقة، والـ Recall لفئة "المريض"، واتجاه أثر كل خاصية (معاملات Logistic Regression).

الاستعمال: python step5_label_flip.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = Path("datasets"); OUT = Path("results"); (OUT / "figures").mkdir(parents=True, exist_ok=True)
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
KEY = ["age", "sex", "trestbps", "chol", "thalach", "oldpeak"]
NUM = ["age", "trestbps", "chol", "thalach", "oldpeak"]   # متغيرات رقمية ترميزها ما تغيرش

k = pd.read_csv(D / "cleveland.csv", encoding="utf-8-sig").drop_duplicates()
u = pd.read_csv(D / "uci_original/processed.cleveland.data", header=None, names=COLS, na_values="?")
u["disease"] = (u.num > 0).astype(int)
m = k.merge(u[KEY + ["disease"]], on=KEY)          # نفس المرضى، والتصنيفين جنب بعض
X = m[NUM + ["sex", "exang"]].values.astype(float)
names = NUM + ["sex", "exang"]

cv = StratifiedKFold(10, shuffle=True, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
rows, coefs = [], {}
for label, y in [("Researcher's belief (Kaggle target=1 'disease')", m.target.values),
                 ("Truth (UCI: 1 = disease)", m.disease.values)]:
    pred = cross_val_predict(model, X, y, cv=cv)
    rows.append(dict(view=label, accuracy=round(accuracy_score(y, pred) * 100, 2),
                     recall_disease=round(recall_score(y, pred) * 100, 2),
                     precision_disease=round(precision_score(y, pred) * 100, 2)))
    coefs[label] = model.fit(X, y)[-1].coef_[0]

res = pd.DataFrame(rows)
cf = pd.DataFrame(coefs, index=names).round(3)
cf["interpretation_flipped"] = np.sign(cf.iloc[:, 0]) != np.sign(cf.iloc[:, 1])
res.to_csv(OUT / "label_flip_metrics.csv", index=False); cf.to_csv(OUT / "label_flip_coefficients.csv")

# ---- ملاحظة: Recall "للمرضى" عند الباحث هو في الحقيقة Specificity (للسليمين) ----
true_y = m.disease.values
pred_k = cross_val_predict(model, X, m.target.values, cv=cv)        # نموذج الباحث
spec_true = recall_score(true_y, 1 - pred_k, pos_label=0)
pd.set_option("display.width", 200)
print(res.to_string(index=False))
print(f"\n'Recall للمرضى' اللي ينشره الباحث = {res.recall_disease[0]}%"
      f" وهو في الحقيقة Specificity للسليمين = {spec_true*100:.2f}%")
print("\nمعاملات Logistic Regression (موجب = يزيد احتمال المرض):")
print(cf.to_string())

# ---- الرسم ----
fig, ax = plt.subplots(figsize=(7, 3.6)); y_ = np.arange(len(names)); h = .38
ax.barh(y_ + h / 2, cf.iloc[:, 0], h, color="#E45756", label="Researcher's belief (Kaggle labels)")
ax.barh(y_ - h / 2, cf.iloc[:, 1], h, color="#4C78A8", label="Truth (original UCI labels)")
ax.axvline(0, c="k", lw=.8); ax.set_yticks(y_); ax.set_yticklabels(names)
ax.set_xlabel("Logistic regression coefficient  (> 0 = increases 'disease' risk)")
ax.set_title("Inverted labels reverse every clinical interpretation"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "figures/label_flip.png", dpi=180)
