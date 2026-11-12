"""
make_figures.py — الرسومات والجداول النهائية للورقة (يقرا ملفات results/)
  Fig 1  figures/fig1_leakage_inflation.png   : كم يضخم كل نوع تسريب الدقة (لكل مستشفى)
  Fig 2  figures/fig2_reproduction.png        : الورقة الأصلية vs بروتوكولها vs البروتوكول الصحيح
  Fig 3  figures/fig3_external_validation.png : داخل المستشفى (CV) vs مستشفى ثاني
  Fig 4  figures/label_flip.png               : (يطلعه step5)
  Fig 5  figures/fig5_swarm_ranks.png          : ترتيب خوارزميات السرب: بيانات مكررة vs نظيفة
  Table  paper_tables.md                        : كل الجداول جاهزة للنسخ في الورقة
الاستعمال: python make_figures.py
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = Path("results"); F = R / "figures"; F.mkdir(parents=True, exist_ok=True)
fmt_p = lambda p: "" if p == "" or pd.isna(p) else ("<0.001" if float(p) < 0.001 else f"{float(p):.3f}")
md = []

# ---------- Fig 1: leakage inflation ----------
L = pd.read_csv(R / "leakage_summary.csv")
conds = ["L1 scaling before split", "L2 GWO before split", "L3 SMOTE before split",
         "L4a controlled duplication", "L4b real Kaggle file (1025)"]
short = {"L1 scaling before split": "Scaling\nbefore split", "L2 GWO before split": "GWO feature\nselection before split",
         "L3 SMOTE before split": "SMOTE\nbefore split", "L4a controlled duplication": "Controlled\nduplication",
         "L4b real Kaggle file (1025)": "Real Kaggle\nfile (1,025)"}
hosp = ["cleveland", "hungarian", "switzerland", "va"]; col = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, clf in zip(axes, ["SVM", "Stacking"]):
    x = np.arange(len(conds)); w = .2
    for i, h in enumerate(hosp):
        g = L[(L.dataset == h) & (L.clf == clf)].set_index("condition").reindex(conds)
        b = ax.bar(x + (i - 1.5) * w, g.inflation, w, color=col[i], label=h.capitalize())
        lo = pd.to_numeric(g.ci_low, errors="coerce"); hi = pd.to_numeric(g.ci_high, errors="coerce")
        ax.errorbar(x + (i - 1.5) * w, g.inflation, yerr=[g.inflation - lo, hi - g.inflation],
                    fmt="none", ecolor="k", elinewidth=.6, capsize=1.5)
        for xi, (inf, p) in enumerate(zip(g.inflation, g.p_value)):
            if pd.notna(inf) and p != "" and pd.notna(p) and float(p) < 0.05:
                ax.text(xi + (i - 1.5) * w, inf + (0.4 if inf >= 0 else -1.2), "*", ha="center", fontsize=11)
    ax.axhline(0, c="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels([short[c] for c in conds], fontsize=8)
    ax.set_title(clf); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("Accuracy inflation vs. correct pipeline (pp)"); axes[0].legend(fontsize=8)
fig.suptitle("Effect of each leakage type (5×10-fold CV; bars = 95% CI; * = corrected t-test p < 0.05; duplication: Cleveland only)", fontsize=10)
plt.tight_layout(); plt.savefig(F / "fig1_leakage_inflation.png", dpi=200); plt.close()

t = L.copy(); t["p_value"] = t.p_value.map(fmt_p)
md += ["## Table 1. Leakage experiment (5×10-fold CV)", "",
       t.rename(columns={"acc": "Acc %", "std": "SD", "inflation": "Δ vs correct (pp)"}).to_markdown(index=False), ""]

# ---------- Fig 2: reproduction ----------
P = pd.read_csv(R / "reproduce_original.csv")
mean = lambda s: float(str(s).split("±")[0])
sd = lambda s: float(str(s).split("±")[1].split("(")[0])
fig, ax = plt.subplots(figsize=(7, 3.8)); x = np.arange(len(P)); w = .27
b1 = ax.bar(x - w, P.paper_reported, w, color="#999", label="Reported in original paper")
b2 = ax.bar(x, P.paper_protocol.map(mean), w, yerr=P.paper_protocol.map(sd), color="#E45756", label="Our run, paper's protocol")
b3 = ax.bar(x + w, P.correct_protocol.map(mean), w, yerr=P.correct_protocol.map(sd), color="#4C78A8", label="Our run, correct protocol")
for b in (b1, b2, b3): ax.bar_label(b, fmt="%.1f", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels(P.method); ax.set_ylim(60, 100); ax.set_ylabel("Accuracy (%)")
ax.legend(fontsize=8, loc="lower right"); ax.set_title("Reproducing Pratama & Putra (PSOGWO + SVM)")
plt.tight_layout(); plt.savefig(F / "fig2_reproduction.png", dpi=200); plt.close()
md += ["## Table 2. Reproduction of the original paper", "", P.to_markdown(index=False), ""]

# ---------- Fig 3: external validation ----------
E = pd.read_csv(R / "external_validation.csv")
if "features" in E.columns: E = E[E.features == "all 13 features"]
c0 = L[L.condition == "C0 correct"]
fig, ax = plt.subplots(figsize=(7, 3.8)); x = np.arange(3); w = .38
for j, clf in enumerate(["SVM", "Stacking"]):
    ext = E[E.clf == clf].set_index("test").reindex(hosp[1:]).acc
    b = ax.bar(x + (j - .5) * w, ext, w, color=["#4C78A8", "#E45756"][j], label=f"{clf}: trained on Cleveland")
    ax.bar_label(b, fmt="%.1f", fontsize=8)
cv_c = c0[(c0.dataset == "cleveland")].acc.mean()
ax.axhline(cv_c, ls="--", c="k", lw=1, label=f"Within-Cleveland CV accuracy ≈ {cv_c:.1f}%")
ax.set_xticks(x); ax.set_xticklabels([h.capitalize() for h in hosp[1:]]); ax.set_ylim(40, 100)
ax.set_ylabel("Accuracy on unseen hospital (%)"); ax.legend(fontsize=8); ax.set_title("External validation")
plt.tight_layout(); plt.savefig(F / "fig3_external_validation.png", dpi=200); plt.close()
md += ["## Table 3. External validation (train: Cleveland)", "", E.to_markdown(index=False), ""]

# ---------- Table: data audit + label flip ----------
md += ["## Table 4. Data audit (Kaggle vs original UCI)", "", pd.read_csv(R / "data_audit.csv").to_markdown(index=False), ""]
md += ["## Table 5. Label inversion — metrics", "", pd.read_csv(R / "label_flip_metrics.csv").to_markdown(index=False), "",
       "## Table 6. Label inversion — logistic regression coefficients", "",
       pd.read_csv(R / "label_flip_coefficients.csv").to_markdown(index=False), ""]

# ---------- Fig 5: swarm comparison ----------
if (R / "swarm_summary.csv").exists():
    S = pd.read_csv(R / "swarm_summary.csv")
    meth = ["All features", "PSO", "GWO", "PSOGWO", "LGWO"]
    dss = ["kaggle1025 (duplicated)", "cleveland", "hungarian", "switzerland", "va"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, clf in zip(axes, ["SVM", "Stacking"]):
        piv = S[S.clf == clf].pivot(index="dataset", columns="method", values="acc").reindex(dss)[meth]
        x = np.arange(len(dss)); w = .16
        for i, m in enumerate(meth):
            ax.bar(x + (i - 2) * w, piv[m], w, label=m,
                   color=["#999", "#4C78A8", "#F58518", "#54A24B", "#E45756"][i])
        fr_path = R / "swarm_friedman.csv"
        fp = pd.read_csv(fr_path).query("clf == @clf").friedman_p.iloc[0] if fr_path.exists() else ""
        ax.set_xticks(x); ax.set_xticklabels([d.split(' ')[0] for d in dss], fontsize=8)
        ax.set_ylim(60, 101); ax.set_title(f"{clf} (Friedman on UCI, p = {fmt_p(fp)})"); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("Accuracy (%) — 3×10-fold CV"); axes[0].legend(fontsize=7, ncol=5, loc="upper right")
    plt.tight_layout(); plt.savefig(F / "fig5_swarm_comparison.png", dpi=200); plt.close()
    md += ["## Table 7. Swarm feature selection comparison (3×10-fold CV)", "", S.to_markdown(index=False), "",
           "## Table 8. Friedman test (blocks = hospital × repeat, UCI only)", "",
           pd.read_csv(R / "swarm_friedman.csv").to_markdown(index=False) if (R / "swarm_friedman.csv").exists() else "(run step4 first)", ""]

(R / "paper_tables.md").write_text("\n".join(md), encoding="utf-8")
print("Figures:", sorted(p.name for p in F.glob("*.png")))
