"""
make_paper_figures.py — رسومات الورقة بعرض عمود واحد (3.4 inch) وتنفع للطباعة أبيض وأسود
  figures/paper_fig1_leakage.png      : الفرق عن المسار الصحيح + فترة الثقة 95% لكل نوع تسريب
  figures/paper_fig2_reproduction.png : إعادة الورقة المنشورة
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7,
                     "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6})
R = Path("results"); F = R / "figures"; F.mkdir(parents=True, exist_ok=True)
L = pd.read_csv(R / "leakage_summary.csv")
# L5 (الضبط على الاختبار) من step7: الفرق عن SVM nested، نحطوه كصف إضافي في لوحة SVM
if (R / "baselines_summary.csv").exists():
    Bs = pd.read_csv(R / "baselines_summary.csv")
    Bs = Bs[Bs.model == "SVM tuned on test fold (L5)"]
    L = pd.concat([L, pd.DataFrame(dict(dataset=Bs.dataset, clf="SVM", condition="L5 tuning on test fold",
                                        inflation=Bs.inflation_vs_nested, ci_low=Bs.ci_low, ci_high=Bs.ci_high))])
conds = ["L1 scaling before split", "L2 GWO before split", "L3 SMOTE before split",
         "L4a controlled duplication", "L4b real Kaggle file (1025)", "L5 tuning on test fold"]
lab = ["L1 scaling", "L2 GWO selection", "L3 SMOTE", "L4a duplication\n(controlled)", "L4b Kaggle file", "L5 test-set\ntuning*"]
hosp = ["cleveland", "hungarian", "switzerland", "va"]
style = {"cleveland": ("#2a78d6", "o"), "hungarian": ("#eb6834", "s"), "switzerland": ("#1baf7a", "^"), "va": ("#444444", "D")}
fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.2), sharex=True,
                         gridspec_kw=dict(height_ratios=[len(conds), len(conds) - 1]))
for ax, clf in zip(axes, ["SVM", "Stacking"]):
    cs = conds if clf == "SVM" else conds[:-1]          # L5 لـ SVM بس
    for i, h in enumerate(hosp):
        g = L[(L.dataset == h) & (L.clf == clf)].set_index("condition").reindex(cs)
        y = np.arange(len(cs)) + (i - 1.5) * 0.17
        lo = pd.to_numeric(g.ci_low, errors="coerce"); hi = pd.to_numeric(g.ci_high, errors="coerce")
        ok = g.inflation.notna()
        c, m = style[h]
        ax.hlines(y[ok], lo[ok], hi[ok], color=c, lw=.8)
        ax.plot(g.inflation[ok], y[ok], m, color=c, ms=3.2, mec="white", mew=.4,
                label=h.capitalize() if h != "va" else "Long Beach VA")
    ax.axvline(0, color="k", lw=.6)
    ax.set_yticks(range(len(cs))); ax.set_yticklabels(lab[:len(cs)]); ax.set_ylim(len(cs) - 0.4, -0.6)
    ax.set_title(clf, fontsize=7.5, pad=3); ax.grid(axis="x", color="#ddd", lw=.4)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
axes[1].set_xlabel("Gain over leak-free pipeline (pp, 95% CI)")
axes[1].legend(fontsize=6, frameon=False, loc="upper right", handletextpad=.2)
plt.tight_layout(h_pad=.6); plt.savefig(F / "paper_fig1_leakage.png", dpi=600); plt.close()

P = pd.read_csv(R / "reproduce_original.csv")
mean = lambda s: float(str(s).split("±")[0]); sd = lambda s: float(str(s).split("±")[1].split("(")[0])
fig, ax = plt.subplots(figsize=(3.4, 1.95))
x = np.arange(len(P)); w = .26
specs = [("paper_reported", "Reported by the authors", "#bbbbbb", ""), ("paper_protocol", "Our run, published protocol", "#666666", "////"),
         ("correct_protocol", "Our run, leak-free protocol", "#2a78d6", "")]
for j, (col, name, c, hatch) in enumerate(specs):
    vals = P[col] if col == "paper_reported" else P[col].map(mean)
    err = None if col == "paper_reported" else P[col].map(sd)
    b = ax.bar(x + (j - 1) * w, vals, w * .92, color=c, hatch=hatch, edgecolor="white", lw=.3, label=name,
               yerr=err, error_kw=dict(lw=.6, capsize=1.5))
    ax.bar_label(b, fmt="%.1f", fontsize=5.5, padding=1.5 if err is None else 6)
ax.set_xticks(x); ax.set_xticklabels(["SVM, all features", "PSOGWO + SVM"])
ax.set_ylim(60, 104); ax.set_ylabel("Accuracy (%)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.32), handlelength=1.2, columnspacing=.8)
plt.tight_layout(); plt.savefig(F / "paper_fig2_reproduction.png", dpi=600); plt.close()
print("ok")
