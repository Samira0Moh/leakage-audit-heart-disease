"""
الخطوة 1: فحص البيانات + التكرار + خط الأساس (SVM)
الاستعمال:  python step1_check_data.py heart.csv
"""
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

path = sys.argv[1] if len(sys.argv) > 1 else "heart.csv"
df = pd.read_csv(path, encoding="utf-8-sig")

print("=" * 50)
print("عدد السجلات:", len(df), "| عدد الأعمدة:", df.shape[1])
print("عدد القيم الناقصة:", int(df.isna().sum().sum()))
print("توزيع الفئات (target):", df["target"].value_counts().to_dict())

# ---- فحص التكرار ----
n_dup = int(df.duplicated().sum())
df_clean = df.drop_duplicates().reset_index(drop=True)
print("عدد السجلات المكررة:", n_dup)
print("عدد السجلات الفريدة بعد الحذف:", len(df_clean))
print("=" * 50)

def evaluate(data, name):
    X, y = data.drop(columns="target"), data["target"]
    model = make_pipeline(StandardScaler(), SVC())
    # (أ) تقسيم واحد 80/20 مثل أغلب الأوراق
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    single = model.fit(Xtr, ytr).score(Xte, yte)
    # (ب) 10-fold cross-validation (تقييم أقوى)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"{name:<22} تقسيم واحد: {single:.4f} | 10-fold CV: {scores.mean():.4f} ± {scores.std():.4f}")

evaluate(df, "SVM - البيانات كما هي")
evaluate(df_clean, "SVM - بعد حذف التكرار")
