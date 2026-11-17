"""
make_paper_figures_ar.py — the two paper figures with Arabic labels (used in the Arabic version of the paper).
Requires matplotlib >= 3.11 (native Arabic text shaping).
Usage: python make_paper_figures_ar.py   ->  results/figures/paper_fig1_leakage_ar.png, paper_fig2_reproduction_ar.png
"""
import numpy as np, pandas as pd
from decimal import Decimal, ROUND_HALF_UP
def r1(v): return float(Decimal(str(v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))  # round half up, as in the tables
from pathlib import Path


import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def ar(s): return s   # matplotlib >= 3.11 shapes Arabic text natively
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5,
                     "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6})
R = Path("results"); F = R / "figures"; F.mkdir(exist_ok=True)
L = pd.read_csv(R / "leakage_summary.csv")
Bs = pd.read_csv(R / "baselines_summary.csv"); Bs = Bs[Bs.model == "SVM tuned on test fold (L5)"]
L = pd.concat([L, pd.DataFrame(dict(dataset=Bs.dataset, clf="SVM", condition="L5 tuning on test fold",
                                    inflation=Bs.inflation_vs_nested, ci_low=Bs.ci_low, ci_high=Bs.ci_high))])
conds = ["L1 scaling before split", "L2 GWO before split", "L3 SMOTE before split",
         "L4a controlled duplication", "L4b real Kaggle file (1025)", "L5 tuning on test fold"]
lab = [ar("L1 التقييس"), ar("L2 اختيار الخصائص"), ar("L3 SMOTE"), ar("L4a تكرار\n(مضبوط)"), ar("L4b ملف Kaggle"), ar("L5 الضبط على\nالاختبار*")]
hosp = ["cleveland", "hungarian", "switzerland", "va"]
names = {"cleveland": "كليفلاند", "hungarian": "المجر", "switzerland": "سويسرا", "va": "لونغ بيتش VA"}
style = {"cleveland": ("#2a78d6", "o"), "hungarian": ("#eb6834", "s"), "switzerland": ("#1baf7a", "^"), "va": ("#444444", "D")}
fig, axes = plt.subplots(2, 1, figsize=(3.6, 5.3), sharex=True, gridspec_kw=dict(height_ratios=[6, 5]))
for ax, clf, ttl in zip(axes, ["SVM", "Stacking"], ["SVM", ar("التكديس (Stacking)")]):
    cs = conds if clf == "SVM" else conds[:-1]
    for i, h in enumerate(hosp):
        g = L[(L.dataset == h) & (L.clf == clf)].set_index("condition").reindex(cs)
        y = np.arange(len(cs)) + (i - 1.5) * 0.17
        lo = pd.to_numeric(g.ci_low, errors="coerce"); hi = pd.to_numeric(g.ci_high, errors="coerce")
        ok = g.inflation.notna(); c, m = style[h]
        ax.hlines(y[ok], lo[ok], hi[ok], color=c, lw=.8)
        for yy, v in zip(y[ok], g.inflation[ok]):
            ax.text(1.02, yy, f"{r1(v):+.1f}", transform=ax.get_yaxis_transform(), color=c, fontsize=5.0, va="center", ha="left", clip_on=False)
        ax.plot(g.inflation[ok], y[ok], m, color=c, ms=3.2, mec="white", mew=.4, label=ar(names[h]))
    ax.axvline(0, color="k", lw=.6)
    ax.set_yticks(range(len(cs))); ax.set_yticklabels(lab[:len(cs)]); ax.set_ylim(len(cs) - 0.4, -0.6)
    ax.set_title(ttl, fontsize=7, pad=3); ax.grid(axis="x", color="#ddd", lw=.4)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
axes[1].set_xlabel(ar("الزيادة عن المسار الخالي من التسرّب (نقطة مئوية، فترة ثقة 95%)"))
axes[1].legend(fontsize=5.8, frameon=False, loc="upper right", handletextpad=.2)
plt.tight_layout(h_pad=.6); plt.savefig(F / "paper_fig1_leakage_ar.png", dpi=600); plt.close()

P = pd.read_csv(R / "reproduce_original.csv")
mean = lambda s: float(str(s).split("±")[0]); sd = lambda s: float(str(s).split("±")[1].split("(")[0])
fig, ax = plt.subplots(figsize=(3.6, 2.1))
x = np.arange(len(P)); w = .26
specs = [("paper_reported", "المُعلَن من المؤلفين", "#bbbbbb", ""), ("paper_protocol", "تشغيلنا، البروتوكول المنشور", "#666666", "////"),
         ("correct_protocol", "تشغيلنا، البروتوكول الخالي من التسرّب", "#2a78d6", "")]
for j, (col, name, c, hatch) in enumerate(specs):
    vals = P[col] if col == "paper_reported" else P[col].map(mean)
    err = None if col == "paper_reported" else P[col].map(sd)
    b = ax.bar(x + (j - 1) * w, vals, w * .92, color=c, hatch=hatch, edgecolor="white", lw=.3, label=ar(name),
               yerr=err, error_kw=dict(lw=.6, capsize=1.5))
    ax.bar_label(b, fmt="%.1f", fontsize=5.5, padding=1.5 if err is None else 6)
ax.set_xticks(x); ax.set_xticklabels([ar("SVM بكل الخصائص"), "PSOGWO + SVM"])
ax.set_ylim(60, 104); ax.set_ylabel(ar("الدقة (%)"))
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=5.8, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.34), handlelength=1.2, columnspacing=.8)
plt.tight_layout(); plt.savefig(F / "paper_fig2_reproduction_ar.png", dpi=600); plt.close()
print("ok")
