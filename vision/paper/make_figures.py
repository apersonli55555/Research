#!/usr/bin/env python3
"""Figures for the paper. Every number is copied from vision/RESULTS.md (run noted inline).

    python vision/paper/make_figures.py   ->  vision/paper/fig_flip.pdf, fig_budget.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# validated categorical palette (dataviz reference instance, slots 1-4): identity of the training source
COL = {"Staged": "#1baf7a", "Synthetic": "#eb6834", "Real": "#2a78d6", "Synthetic + real": "#eda100"}
MARK = {"Staged": "s", "Synthetic": "^", "Real": "o", "Synthetic + real": "D"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "legend.frameon": False})

# ---- Figure 1: ranking flip. CLIP+VideoMAE probes (runs 4, 5) and fine-tuned VideoMAE (run 7, 3 seeds)
probe = {  # (AUROC, CI low, CI high)
    "FallOrNot (OOPS!)": {"Staged": (0.759, 0.698, 0.814), "Synthetic": (0.814, 0.753, 0.869),
                          "Real": (0.899, 0.858, 0.935), "Synthetic + real": (0.872, 0.825, 0.914)},  # run 10, compilation-level
    "HMDB51 (external)": {"Staged": (0.899, 0.870, 0.926), "Synthetic": (0.917, 0.892, 0.943),
                          "Real": (0.888, 0.850, 0.925), "Synthetic + real": (0.901, 0.869, 0.932)},
}
finetuned = {  # (mean, s.d.) over 3 seeds
    "FallOrNot (OOPS!)": {"Synthetic": (0.838, 0.017), "Real": (0.902, 0.012), "Synthetic + real": (0.894, 0.005)},
    "HMDB51 (external)": {"Synthetic": (0.960, 0.006), "Real": (0.920, 0.026), "Synthetic + real": (0.954, 0.011)},
}
order = ["Staged", "Synthetic", "Real", "Synthetic + real"]
fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.2), sharey=True)
for ax, (title, d) in zip(axes, probe.items()):
    for i, src in enumerate(order):
        v, lo, hi = d[src]
        ax.errorbar(i - 0.12, v, yerr=[[v - lo], [hi - v]], fmt=MARK[src], color=COL[src], ms=4.5,
                    elinewidth=1, capsize=0, markeredgecolor="white", markeredgewidth=0.5)
        if src in finetuned[title]:
            m, sd = finetuned[title][src]
            ax.errorbar(i + 0.14, m, yerr=sd, fmt=MARK[src], color=COL[src], ms=4.5, mfc="white",
                        elinewidth=1, capsize=0, markeredgewidth=1)
    ax.set_title(title, fontsize=7.5, color=INK, pad=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Staged", "Synthetic", "Real", "Synth.\n+ real"], fontsize=6.2,
                                                        rotation=35, ha="right", rotation_mode="anchor")
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlim(-0.5, 3.5)
axes[0].set_ylabel("AUROC")
axes[0].set_ylim(0.68, 1.0)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker="o", color=MUTED, ls="", ms=4, label="frozen probe (95% CI)"),
                    Line2D([], [], marker="o", color=MUTED, mfc="white", ls="", ms=4, label="fine-tuned (mean ± s.d., 3 seeds)")],
           loc="lower center", ncol=2, fontsize=6.3, bbox_to_anchor=(0.53, -0.02), handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
fig.savefig(os.path.join(HERE, "fig_flip.pdf"))

# ---- Figure 2: equal budget on FallOrNot, CLIP+VideoMAE (run 4); mean ± s.d. over random draws
budget = {
    "Staged": [(4, .751, .020), (8, .745, .014), (15, .744, .009), (30, .752, .008), (60, .754, 0)],
    "Synthetic": [(4, .723, .025), (8, .724, .020), (15, .742, .011), (30, .768, .024), (60, .786, .024),
                  (125, .825, .018), (250, .818, .017), (500, .819, .004), (1000, .818, .024), (2000, .817, .006),
                  (2136, .812, 0)],
    "Real": [(4, .670, .130), (8, .772, .028), (15, .789, .027), (30, .836, .018), (60, .862, .013),
             (125, .896, .003), (131, .895, 0)],
}
fig, ax = plt.subplots(figsize=(3.35, 2.05))
for src in ("Real", "Synthetic", "Staged"):
    pts = budget[src]
    x, y, e = zip(*pts)
    ax.errorbar(x, y, yerr=e, color=COL[src], marker=MARK[src], ms=3.8, lw=1.6, elinewidth=0.8, capsize=0,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.annotate(src, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
ax.set_xscale("log")
ax.set_xlabel("training falls (plus an equal number of non-falls)")
ax.set_ylabel("AUROC on FallOrNot")
ax.set_xlim(3, 6000); ax.set_ylim(0.52, 0.92)
ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_budget.pdf"))

# ---- Figure 3: anchored probe (run 12); mean ± s.d. over 10 draws, CLIP+VideoMAE
adapt = {
    "FallOrNot": dict(syn=0.814, k=[4, 8, 15, 30, 60, 125],
                      anch=[(.813, .005), (.819, .012), (.834, .015), (.853, .014), (.875, .010), (.893, .005)],
                      real=[(.687, .124), (.756, .063), (.791, .022), (.826, .030), (.871, .010), (.892, .005)]),
    "OF-ItW test": dict(syn=0.873, k=[4, 8, 15, 30, 60, 125],
                        anch=[(.876, .005), (.877, .006), (.891, .012), (.907, .019), (.921, .008), (.935, .007)],
                        real=[(.794, .046), (.825, .024), (.847, .024), (.891, .014), (.908, .016), (.937, .007)]),
}
fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.1))
for ax, (title, d) in zip(axes, adapt.items()):
    ax.axhline(d["syn"], color=COL["Synthetic"], lw=1.2, ls=(0, (4, 2)))
    for key, src, mk in (("real", "Real", "o"), ("anch", "Synthetic + real", "D")):
        y, e = zip(*d[key])
        ax.errorbar(d["k"], y, yerr=e, color=COL[src], marker=mk, ms=3.6, lw=1.5, elinewidth=0.8, capsize=0,
                    markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xscale("log"); ax.set_xticks([4, 15, 60, 125]); ax.set_xticklabels(["4", "15", "60", "125"])
    ax.minorticks_off()
    ax.set_title(title, fontsize=7.5, color=INK, pad=3); ax.set_xlabel("real training falls")
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("AUROC")
fig.legend(handles=[Line2D([], [], marker="D", color=COL["Synthetic + real"], ms=3.6, lw=1.5, label="anchored"),
                    Line2D([], [], marker="o", color=COL["Real"], ms=3.6, lw=1.5, label="real only"),
                    Line2D([], [], color=COL["Synthetic"], lw=1.2, ls=(0, (4, 2)), label="synthetic only")],
           loc="lower center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.53, -0.02), handletextpad=0.4, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.8)
fig.savefig(os.path.join(HERE, "fig_adapt.pdf"))

# ---- Figure 4: zero-label adaptation. Left: size of the unlabelled pool (run 16; mean of 3 random subsets; full pool = run 15).
#      Right: falls vs other accidents on the hard subsets (run 17).
TEST_COL = {"FallOrNot": COL["Real"], "OF-ItW": COL["Staged"], "HMDB51": COL["Synthetic"]}
TEST_MARK = {"FallOrNot": "o", "OF-ItW": "s", "HMDB51": "^"}
pool = {"FallOrNot": (0.814, [.816, .864, .874, .890, .894, .894]), "OF-ItW": (0.872, [.866, .912, .918, .930, .937, .937]),
        "HMDB51": (0.917, [.924, .945, .944, .949, .950, .953])}
npool = [141, 353, 706, 1765, 3531, 7063]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw={"width_ratios": [1, 1.15]})
for t, (syn, ys) in pool.items():
    ax.plot(npool, ys, color=TEST_COL[t], marker=TEST_MARK[t], ms=3.8, lw=1.5, markeredgecolor="white", markeredgewidth=0.4)
    ax.plot([npool[0], npool[-1]], [syn, syn], color=TEST_COL[t], lw=0.9, ls=(0, (3, 2)))
    ax.annotate(t, (npool[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5, color=INK)
ax.set_xscale("log"); ax.set_xlim(110, 30000); ax.set_ylim(0.80, 0.97)
ax.set_xticks([141, 706, 3531, 7063]); ax.set_xticklabels(["141", "706", "3.5k", "7k"]); ax.minorticks_off()
ax.set_xlabel("accident videos used (no fall annotations)"); ax.set_ylabel("AUROC")
ax.set_title("no real-video fall annotations (dashed: synthetic only)", fontsize=7, color=INK, pad=3)
methods = [("Accident detector", .724, 0, .884, 0), ("Random start", .767, .021, .886, .029), ("Synthetic only", .772, 0, .880, 0),
           ("Self-trained", .835, 0, .920, 0), ("Hand labels", .860, 0, .777, 0)]
yy = range(len(methods))[::-1]
for (name, a_, sa, b_, sb), y in zip(methods, yy):
    c = COL["Synthetic + real"] if name == "Self-trained" else MUTED
    bx.plot([a_, b_], [y, y], color=GRID, lw=1.2, zorder=1)
    bx.errorbar(a_, y, xerr=sa or None, fmt="o", color=c, ms=4.5, elinewidth=0.8, capsize=0, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    bx.errorbar(b_, y, xerr=sb or None, fmt="^", color=c, ms=4.5, mfc="white", elinewidth=0.8, capsize=0, markeredgewidth=1, zorder=3)
bx.set_yticks(list(yy)); bx.set_yticklabels([m[0] for m in methods], fontsize=6.5, color=INK)
bx.set_xlim(0.69, 0.95); bx.set_xlabel("AUROC, falls vs. other accidents"); bx.xaxis.grid(True, color=GRID, lw=0.5); bx.set_axisbelow(True)
bx.legend(handles=[Line2D([], [], marker="o", color=MUTED, ls="", ms=4, label="FallOrNot hard"),
                   Line2D([], [], marker="^", color=MUTED, mfc="white", ls="", ms=4, label="HMDB51 hard")],
          loc="lower right", fontsize=6.2, handletextpad=0.2, borderaxespad=0.2)
for a_ in (ax, bx):
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(HERE, "fig_nolabel.pdf"))
print("wrote fig_flip.pdf, fig_budget.pdf, fig_adapt.pdf, fig_nolabel.pdf")
