#!/usr/bin/env python3
"""Paper figures, drawn only from analysis outputs (pilot_summary.json, full_summary.json).

  python ovd/make_figures.py --pilot OUT/pilot_summary.json [--full OUT/full_summary.json] --dest ovd/paper
"""
import argparse, json, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"gdino": "#1f5fa8", "yoloworld": "#d9822b", "owlv2": "#3a8f5c"}
NAME = {"gdino": "Grounding DINO", "yoloworld": "YOLO-World", "owlv2": "OWLv2"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def pilot_fig(P, dest):
    groups = ["k1", "rand5", "near5", "rand20", "near20", "all80"]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
    for m, r in P.items():
        x = range(len(groups))
        ax[0].plot(x, [r["mean_raw_score"].get(g, float("nan")) for g in groups], "o-", c=COL[m], label=NAME[m], ms=3)
        ax[1].plot(x, [r["flip_vs_k1"].get(g, float("nan")) for g in groups], "o-", c=COL[m], ms=3)
        ax[2].plot(x, [r["map50"].get(g, float("nan")) for g in groups], "o-", c=COL[m], ms=3)
    for a, t in zip(ax, ["mean raw target score", "detection flips vs. k=1", "mAP50 (deployed)"]):
        a.set_xticks(range(len(groups))); a.set_xticklabels(groups, rotation=40); a.set_title(t)
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(dest, "fig_pilot.pdf")); plt.close(fig)


def full_fig(F, dest):
    from matplotlib.lines import Line2D
    for ds in ("coco", "lvis"):
        rows = {k.split("/")[1]: v for k, v in F.items() if k.startswith(ds + "/") and k.split("/")[1] in COL}
        if not rows:
            continue
        fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
        ticks, labels = [], []
        for mi, (m, S) in enumerate(rows.items()):
            G = S["groups"]
            sizes = sorted((int(re.sub(r"\D", "", g)), g) for g in G if g.startswith("rand"))
            top = 80 if ds == "coco" else 1203
            xs = [1] + [k for k, _ in sizes] + [top]
            keys = ["k1"] + [g for _, g in sizes] + ["all"]
            ax[0].plot(xs, [G[g]["mean_abs_score_change"] for g in keys], "o-", c=COL[m], ms=3, label=NAME[m])
            ax[1].plot(xs, [G[g]["det_flip_rate"] for g in keys], "o-", c=COL[m], ms=3)
            for fam, mk in (("near", "^"), ("vis", "s"), ("far", "v")):
                fs = sorted((int(g[len(fam):]), g) for g in G if re.fullmatch(fam + r"\d+", g))
                ax[1].plot([k for k, _ in fs], [G[g]["det_flip_rate"] for _, g in fs], mk, c=COL[m], ms=4, mfc="none")
            vals = [G["k1"]["mAP50_deployed"], G["all"]["mAP50_deployed"], G.get("FIX", {}).get("mAP50_deployed", float("nan"))]
            xb = [mi * 4 + i for i in range(3)]
            ax[2].bar(xb, vals, color=COL[m], alpha=0.95, edgecolor="k", linewidth=0.3, hatch=None)
            for x, v in zip(xb, vals):
                ax[2].text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
            ticks += xb; labels += ["k=1", "full", "inv."]
        for a in ax[:2]:
            a.set_xscale("log"); a.set_xlabel("vocabulary size")
        ax[0].set_title("(a) mean |target score change|"); ax[1].set_title("(b) detection flips vs. k=1")
        ax[2].set_title("(c) mAP50"); ax[2].set_xticks(ticks); ax[2].set_xticklabels(labels, fontsize=6)
        ax[0].legend(frameon=False, fontsize=7)
        ax[1].legend(handles=[Line2D([], [], marker=mk, ls="", c="gray", mfc="none", label=l) for mk, l in
                              (("o", "random"), ("^", "similar"), ("s", "visual"), ("v", "unrelated"))], frameon=False, fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(dest, f"fig_{ds}.pdf")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot"); ap.add_argument("--full"); ap.add_argument("--dest", default="ovd/paper")
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    if a.pilot:
        pilot_fig(json.load(open(a.pilot)), a.dest)
    if a.full:
        full_fig(json.load(open(a.full)), a.dest)
    print("figures written to", a.dest)


if __name__ == "__main__":
    main()
