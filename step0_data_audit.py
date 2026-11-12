"""
step0_data_audit.py — تدقيق نسخ Kaggle مقابل البيانات الأصلية من UCI
يكشف أربع مشاكل:
  (1) السجلات المكررة
  (2) التصنيف المقلوب (target)
  (3) إعادة ترميز المتغيرات الفئوية بشكل غير متسق مع توثيق UCI
  (4) تداخل المرضى بين مجموعات بيانات يُفترض أنها مستقلة (Statlog ⊂ Cleveland)

الاستعمال: python step0_data_audit.py
"""
import pandas as pd
from pathlib import Path

D = Path("datasets"); OUT = Path("results"); OUT.mkdir(exist_ok=True)
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
KEY = ["age", "sex", "trestbps", "chol", "thalach", "oldpeak"]   # متغيرات رقمية لا يغيرها الترميز

# توثيق UCI الرسمي (ملف heart-disease.names)
UCI_MEANING = {
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
    "restecg": {0: "normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
}

uci = pd.read_csv(D / "uci_original/processed.cleveland.data", header=None, names=COLS, na_values="?")
uci["disease"] = (uci.num > 0).astype(int)          # 1 = مريض حسب UCI
report = []

for name, f in [("Kaggle 1025 (johnsmith88)", "heart1025_raw.csv"), ("Kaggle 303", "cleveland.csv")]:
    k = pd.read_csv(D / f, encoding="utf-8-sig")
    n_dup = int(k.duplicated().sum()); ku = k.drop_duplicates()
    m = ku.merge(uci, on=KEY, suffixes=("_k", "_u"))
    agree = (m.target == m.disease).mean()
    print(f"\n===== {name} =====")
    print(f"(1) السجلات: {len(k)} | المكرر: {n_dup} ({n_dup/len(k):.1%}) | الفريد: {len(ku)}")
    print(f"    الفريد اللي يطابق مرضى UCI Cleveland: {len(m)} من {len(ku)}")
    print(f"(2) تطابق target مع تشخيص UCI: {agree:.1%}  =>  {'مقلوب! target=1 يعني سليم' if agree < .05 else 'سليم'}")
    print("(3) إعادة الترميز (قيمة UCI ← قيمة Kaggle):")
    for c, meaning in UCI_MEANING.items():
        mp = m.groupby(m[c + "_u"].fillna(-1))[c + "_k"].agg(lambda s: s.mode()[0]).to_dict()
        txt = ", ".join(f"{meaning.get(int(u), 'MISSING(?)')}: {int(u) if u >= 0 else '?'}→{v}" for u, v in mp.items())
        print(f"    {c:8s} {txt}")
        report.append(dict(dataset=name, variable=c, mapping_uci_to_kaggle=txt))
    report.append(dict(dataset=name, variable="target", mapping_uci_to_kaggle=f"label agreement with UCI = {agree:.1%}"))
    report.append(dict(dataset=name, variable="duplicates", mapping_uci_to_kaggle=f"{n_dup}/{len(k)}"))

s = pd.read_csv(D / "statlog.csv")
ms = s.merge(uci, on=KEY)
print(f"\n(4) Statlog: {len(ms)} من {len(s)} مريض موجودين في UCI Cleveland  =>  مش مجموعة مستقلة")
report.append(dict(dataset="Statlog (270)", variable="overlap with Cleveland", mapping_uci_to_kaggle=f"{len(ms)}/{len(s)}"))

for f in ["hungarian", "switzerland", "va"]:
    d = pd.read_csv(D / f"uci_original/processed.{f}.data", header=None, names=COLS, na_values="?")
    print(f"    UCI {f:12s}: {d.shape[0]} مريض، تداخل مع Cleveland = {len(d.merge(uci, on=KEY))}")

pd.DataFrame(report).to_csv(OUT / "data_audit.csv", index=False)
print("\nتم الحفظ: results/data_audit.csv")
