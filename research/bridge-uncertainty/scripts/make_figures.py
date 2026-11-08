"""Make publication figures and a markdown results summary from results/.

Usage:  python scripts/make_figures.py      (run after run_experiments.py)
Outputs: figures/*.pdf and *.png (300 dpi), results/summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kw51uq import data, models  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Palette: first three categorical slots validated for all-pairs CVD separation
# (dataviz reference palette). Extra baselines are gray with distinct markers.
C = {"Physics + GBM (proposed)": "#2a78d6", "GP": "#eb6834", "GBM (no physics)": "#1baf7a",
     "Cubic(T)": "#8a8984", "Linear(T)": "#b9b8b2", "Physics only": "#8fb4e3",
     "GP + linear kernel": "#eb6834", "Linear + GBM": "#1baf7a", "Physics(no cold) + GBM": "#8fb4e3"}
MK = {"Physics + GBM (proposed)": "o", "GP": "s", "GBM (no physics)": "^",
      "Cubic(T)": "D", "Linear(T)": "v", "Physics only": "o",
      "GP + linear kernel": "P", "Linear + GBM": "X", "Physics(no cold) + GBM": "d"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.5, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def label_panel(ax, s):
    ax.text(-0.12, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


BIN_ORDER = ["(0.0, 5.0]", "(5.0, 10.0]", "(10.0, 15.0]", "(15.0, 20.0]", "(20.0, inf]"]
BIN_LABEL = ["0–5", "5–10", "10–15", "15–20", ">20"]


def fig_overview(df, cfg, mode=6):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 3.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.4, 1], hspace=0.08))
    y = df[f"f{mode}"]
    a1.plot(y.index, y, ".", ms=1.2, color=MUTED, alpha=0.6, rasterized=True)
    a2.plot(df.index, df["T_steel"], lw=0.5, color=INK)
    spans = [("train", cfg["train"], "#2a78d6", 0.10), ("cal", cfg["cal"], "#1baf7a", 0.14),
             ("test", cfg["test"], "#eda100", 0.14), ("post-retrofit test", cfg["post"], "#e87ba4", 0.12)]
    for ax in (a1, a2):
        ax.axvspan(pd.Timestamp(cfg["retrofit"][0]), pd.Timestamp(cfg["retrofit"][1]),
                   color="#8a8984", alpha=0.18, lw=0)
        for _, (s, e), c, al in spans:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta("1D"), color=c, alpha=al, lw=0)
    ymax = np.nanquantile(y, 0.999)
    for name, (s, e), _, _ in spans:
        mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
        a1.text(mid, ymax, name, ha="center", va="bottom", fontsize=7, color=INK)
    mid = pd.Timestamp(cfg["retrofit"][0]) + (pd.Timestamp(cfg["retrofit"][1]) - pd.Timestamp(cfg["retrofit"][0])) / 2
    a1.text(mid, ymax, "retrofit (excluded)", ha="center", va="bottom", fontsize=7, color=MUTED)
    a1.set_ylim(np.nanquantile(y, 0.001), ymax * 1.004)
    a1.set_ylabel(f"Mode {mode} frequency [Hz]")
    a2.set_ylabel("Steel temp. [°C]")
    fig.align_ylabels()
    save(fig, "fig1_overview")


def fig_physics_fit(df, cfg, mode=6):
    d = df.dropna(subset=[f"f{mode}", "T_steel"])
    fit = d.loc[cfg["train"][0]:cfg["cal"][1]]
    te = d.loc[cfg["test"][0]:cfg["test"][1]]
    tr = d.loc[cfg["train"][0]:cfg["train"][1]]
    m = models.PhysicsOnly().fit(tr, tr[f"f{mode}"].to_numpy())
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(fit["T_steel"], fit[f"f{mode}"], ".", ms=2, color="#8a8984", alpha=0.5,
            label="train + calibration (Nov–Feb)", rasterized=True)
    ax.plot(te["T_steel"], te[f"f{mode}"], ".", ms=2, color="#eda100", alpha=0.5,
            label="test (Mar–May 14)", rasterized=True)
    Tg = np.linspace(-3, 27, 300)
    ax.plot(Tg, models.physics_curve(m.theta, Tg), color="#2a78d6", lw=2, label="physics model (train fit)")
    ax.axvline(tr["T_steel"].max(), color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("Steel temperature [°C]"); ax.set_ylabel(f"Mode {mode} frequency [Hz]")
    lo, hi = np.nanquantile(d[f"f{mode}"], [0.002, 0.998]); ax.set_ylim(lo, hi + 0.012)
    ax.text(tr["T_steel"].max() + 0.4, lo + 0.002, "highest training\ntemperature", fontsize=6.5,
            color=MUTED, va="bottom")
    leg = ax.legend(frameon=True, loc="upper right", markerscale=3, edgecolor="none", framealpha=0.95)
    leg.set_zorder(5)
    save(fig, "fig2_physics_fit")


def coverage_by_bin(cond, mods, modes=None):
    c = cond[(cond.T_bin != "ALL") & cond.model.isin(mods)]
    if modes is not None:
        c = c[c["mode"].isin(modes)]
    return c


def plot_cov_bins(ax, cond, mods, mode=None, alpha=0.1, show_ci=True):
    c = coverage_by_bin(cond, mods)
    x = np.arange(len(BIN_ORDER))
    offs = np.linspace(-0.2, 0.2, len(mods))
    for o, mname in zip(offs, mods):
        cm = c[c.model == mname]
        if mode is None:
            g = cm.groupby("T_bin")["coverage"]
            mu = g.mean().reindex(BIN_ORDER)
            lo, hi = g.min().reindex(BIN_ORDER), g.max().reindex(BIN_ORDER)
        else:
            cm = cm[cm["mode"] == mode].set_index("T_bin").reindex(BIN_ORDER)
            mu, lo, hi = cm["coverage"], cm["ci_lo"], cm["ci_hi"]
        ax.errorbar(x + o, mu, yerr=[mu - lo, hi - mu] if show_ci else None, fmt=MK[mname] + "-",
                    color=C[mname], ms=4.5, lw=1.2, capsize=0, elinewidth=1,
                    markeredgecolor="white", markeredgewidth=0.6, label=mname)
    ax.axhline(1 - alpha, color=INK, lw=0.9, ls="--", label=f"target {1 - alpha:.0%}")
    ax.set_xticks(x, BIN_LABEL)
    ax.set_xlabel("Steel temperature in test period [°C]")
    ax.set_ylabel("Empirical coverage of 90% interval")
    ax.set_ylim(-0.02, 1.05)


def fig_conditional_coverage(cond):
    mods = ["GP", "GBM (no physics)", "Physics + GBM (proposed)"]
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.4), sharey=True)
    for ax, mode in zip(axes, sorted(cond["mode"].unique())):
        plot_cov_bins(ax, cond, mods + ["Cubic(T)"], mode=mode)
        ax.set_title(f"Mode {mode}"); ax.set_xlabel("")
        if mode != 3:
            ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=45)
    fig.supxlabel("Steel temperature in test period [°C]   (error bars: 95% day-block bootstrap CI)",
                  fontsize=7.5, y=-0.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.1), fontsize=7)
    save(fig, "fig4_coverage_by_temperature_all_modes")


def fig_calibration(cov):
    mods = ["Linear(T)", "Cubic(T)", "GP", "GBM (no physics)", "Physics + GBM (proposed)"]
    c = cov[(cov.method == "split conformal") & cov.model.isin(mods)]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, split, title in zip(axes, ["random", "chronological"],
                                ["(a) Random hourly split (exchangeable)",
                                 "(b) Chronological split (unseen season)"]):
        s = c[c.split == split].groupby(["model", "alpha"])["coverage"].mean().reset_index()
        for mname in mods:
            sm = s[s.model == mname].sort_values("alpha")
            ax.plot(1 - sm.alpha, sm.coverage, MK[mname] + "-", color=C[mname], ms=4,
                    markeredgecolor="white", markeredgewidth=0.5, label=mname)
        ax.plot([0.45, 1], [0.45, 1], color=INK, lw=0.8, ls="--")
        ax.set_title(title, loc="left"); ax.set_xlabel("Nominal coverage 1 − α")
        ax.set_xlim(0.45, 1.0); ax.set_ylim(0.3, 1.0)
    axes[0].set_ylabel("Empirical coverage (mean of 5 modes)")
    axes[1].legend(frameon=False, loc="lower right", fontsize=6.5)
    save(fig, "fig3_calibration_curves")


def fig_detection(preds, det, mode=6):
    fig, ax = plt.subplots(figsize=(7, 2.4))
    for mname, col in [("GP", C["GP"]), ("Physics + GBM (proposed)", C["Physics + GBM (proposed)"])]:
        p = preds[(mode, mname)]
        for part in ("te", "post"):
            r = (p[part]["y"] - p[part]["yhat"]) / p["q"]
            dly = r.groupby(r.index.floor("D")).median()
            n = r.groupby(r.index.floor("D")).count()
            dly = dly[n >= 12]
            ax.plot(dly.index, dly, MK[mname], ms=3, color=col,
                    label=mname if part == "te" else None, markeredgecolor="white", markeredgewidth=0.4)
    ax.axhspan(-1, 1, color="#8a8984", alpha=0.15, lw=0)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.text(pd.Timestamp("2019-07-20"), 0, "no-alarm band |D| ≤ 1\n(90% conformal band)", fontsize=6.5,
            ha="center", va="center", color=MUTED)
    ax.set_ylabel("Daily median residual / q")
    ax.set_title(f"Mode {mode}: pre-retrofit test (Mar–May 2019) vs post-retrofit (Sep 2019–Jan 2020)",
                 loc="left")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig5_detection_mode6")


def fig_key(preds, cond, mode=6):
    """Single key figure for the one-page summary."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.35, 1]))
    sl = slice("2019-04-15", "2019-04-25")  # a warm spell in the test period
    for mname in ["GP", "Physics + GBM (proposed)"]:
        p = preds[(mode, mname)]["te"].loc[sl]; q = preds[(mode, mname)]["q"]
        a.fill_between(p.index, p.yhat - q, p.yhat + q, color=C[mname], alpha=0.22, lw=0,
                       label=f"{mname.replace(' (proposed)', '')} 90% band")
    p = preds[(mode, "Physics + GBM (proposed)")]["te"].loc[sl]
    a.plot(p.index, p.y, ".", ms=2.5, color=INK, label="measured frequency")
    a.set_ylabel(f"Mode {mode} frequency [Hz]")
    a.set_title("(a) Warm spell in test period, 15–25 Apr 2019", loc="left")
    a.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d %b"))
    a.legend(frameon=False, fontsize=6.3, loc="lower left")
    plot_cov_bins(b, cond, ["GP", "GBM (no physics)", "Physics + GBM (proposed)"], mode=None, show_ci=True)
    b.set_title("(b) Coverage vs temperature, 5 modes", loc="left")
    b.legend(frameon=False, fontsize=6.0, loc="lower left")
    b.text(0.0, -0.32, "markers = mean over modes; bars = min–max across modes", transform=b.transAxes,
           fontsize=6, color=MUTED)
    fig.tight_layout()
    save(fig, "key_figure")


def fig_robustness(rob):
    """Coverage on test hours whose temperature lies OUTSIDE the training range,
    for every fold; one row per model, one dot per (fold, mode)."""
    order = ["Linear(T)", "Cubic(T)", "GP", "GP + linear kernel", "GBM (no physics)",
             "Linear + GBM", "Physics only", "Physics(no cold) + GBM", "Physics + GBM (proposed)"]
    folds = list(dict.fromkeys(rob["fold"]))
    fmk = dict(zip(folds, ["o", "s", "^", "D", "v"]))
    r = rob[(rob.region == "outside") & (rob.n >= 24)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, mname in enumerate(order):
        rm = r[r.model == mname]
        col = C.get(mname, MUTED)
        for j, f in enumerate(folds):
            rf = rm[rm.fold == f]
            yy = np.full(len(rf), i) + (j - 2) * 0.09
            ax.scatter(rf.coverage, yy, marker=fmk[f], s=16, color=col, alpha=0.85,
                       edgecolors="white", linewidths=0.4)
        med = rm.coverage.median()
        ax.plot([med, med], [i - 0.32, i + 0.32], color=INK, lw=1.6)
    ax.axvline(0.9, color=INK, lw=0.9, ls="--")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("Coverage of 90% interval on test hours OUTSIDE the training temperature range")
    ax.set_xlim(-0.02, 1.02); ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=fmk[f], ls="", color=MUTED, ms=4.5) for f in folds]
    ax.legend(h, folds, frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="fold (marker)", title_fontsize=6.5)
    ax.text(0.905, len(order) - 0.45, "target", fontsize=6.5, color=INK)
    ax.set_title("Robustness: 5 folds × 5 modes (black tick = median; only cells with ≥24 outside-range hours)",
                 loc="left")
    save(fig, "fig6_robustness_outside_range")


ORDER9 = ["Linear(T)", "Cubic(T)", "GP", "GP + linear kernel", "GBM (no physics)",
          "Linear + GBM", "Physics only", "Physics(no cold) + GBM", "Physics + GBM (proposed)"]


def fig_rolling(cells):
    """Rolling-origin study: coverage on out-of-range hours, every window x mode cell."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, (direction, side, title) in zip(axes, [
            ("forward", "warm", "(a) Forward windows: test warmer than training"),
            ("backward", "cold", "(b) Backward windows: test colder than training")]):
        c = cells[(cells.direction == direction) & (cells.region == side) & (cells.n >= 24)]
        for i, mname in enumerate(ORDER9):
            cm = c[c.model == mname]
            jit = np.random.default_rng(i).uniform(-0.22, 0.22, len(cm))
            ax.scatter(cm.coverage, i + jit, s=7, color=C.get(mname, MUTED), alpha=0.55, lw=0)
            if len(cm):
                ax.plot([cm.coverage.median()] * 2, [i - 0.34, i + 0.34], color=INK, lw=1.6)
        ax.axvline(0.9, color=INK, lw=0.9, ls="--")
        ax.set_title(title, loc="left", fontsize=7.8)
        ax.set_xlim(-0.02, 1.02); ax.grid(axis="y", visible=False)
        n_cells = c.groupby("model").size().max() if len(c) else 0
        ax.set_xlabel(f"Coverage of 90% interval ({n_cells} window×mode cells)", fontsize=7)
    axes[0].set_yticks(range(len(ORDER9)), ORDER9)
    save(fig, "fig7_rolling_origin")


def fig_forest(pt):
    """Paired coverage difference (proposed - comparator) with 95% day-block bootstrap CI."""
    settings = [s for s in ["Rolling forward windows, warm-side out-of-range",
                            "Folds F2+F3, warm-side out-of-range",
                            "Rolling backward windows, cold-side out-of-range",
                            "Rolling (all windows), inside training range"] if s in set(pt.setting)]
    comps = [m for m in ORDER9 if m != "Physics + GBM (proposed)"]
    fig, axes = plt.subplots(1, len(settings), figsize=(1.9 * len(settings) + 1.6, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, settings):
        d = pt[pt.setting == st].set_index("comparator").reindex(comps)
        y = np.arange(len(comps))
        ax.errorbar(d.delta, y, xerr=[d.delta - d.ci_lo, d.ci_hi - d.delta], fmt="o", ms=4,
                    color="#2a78d6", ecolor="#2a78d6", elinewidth=1.2, capsize=0,
                    markeredgecolor="white", markeredgewidth=0.5)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_title(st.replace(", ", ",\n"), loc="left", fontsize=6.6)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(comps)), comps)
    fig.supxlabel("Coverage difference: proposed − comparator (95% CI, days resampled)  →  positive favours proposed",
                  fontsize=7, y=-0.04)
    save(fig, "fig8_paired_differences")


def summary_md(cov, cond, det, phys, aci):
    L = ["# Results summary (auto-generated by scripts/make_figures.py)\n"]
    c = cov[(cov.alpha == 0.1) & (cov.method == "split conformal")]
    for split in ["random", "chronological"]:
        t = c[c.split == split].pivot_table(index="model", columns="mode", values="coverage").round(3)
        t["mean"] = t.mean(1).round(3)
        L += [f"\n## Coverage of 90% split-conformal intervals, {split} split\n", t.to_markdown(), ""]
    t = c[c.split == "chronological"].pivot_table(index="model", columns="mode", values="interval_score")
    L += ["\n## Interval score (Hz, lower = better), chronological split, alpha=0.1\n",
          t.round(4).to_markdown(), ""]
    t = c[c.split == "chronological"].pivot_table(index="model", columns="mode", values="width_Hz")
    L += ["\n## Mean interval width (Hz), chronological split\n", t.round(4).to_markdown(), ""]
    hot = cond[cond.T_bin == "(20.0, inf]"].pivot_table(index="model", columns="mode", values="coverage")
    L += ["\n## Coverage when test temperature > 20 °C (never seen in training; ~68 h / few days per mode)\n",
          hot.round(2).to_markdown(), ""]
    L += ["\n## ACI vs split conformal (chronological, alpha=0.1)\n", aci.round(3).to_markdown(index=False), ""]
    L += ["\n## Retrofit detection (daily median |r|/q > 1)\n", det.round(3).to_markdown(index=False), ""]
    L += ["\n## Fitted physics parameters (train period only)\n",
          phys.drop_duplicates("mode").round(5).to_markdown(index=False), ""]
    rob = pd.read_csv(RES / "robustness_folds.csv")
    for region in ["all", "inside", "outside"]:
        t = rob[rob.region == region].pivot_table(index="model", columns="fold", values="coverage",
                                                    aggfunc="median")
        L += [f"\n## Robustness: median coverage over 5 modes, region = {region} training T range\n",
              t.round(3).to_markdown(), ""]
    t = rob[rob.region == "outside"].pivot_table(index="model", columns="fold", values="n", aggfunc="median")
    L += ["\n## Robustness: median number of outside-range test hours per mode\n", t.to_markdown(), ""]
    sub = pd.read_csv(RES / "robustness_subzero.csv")
    L += ["\n## Fold B1 (train Apr-May, test Nov-Feb): coverage on hours below 0 °C\n",
          sub.pivot_table(index="model", columns="mode", values="coverage_below_0C").round(2).to_markdown(),
          f"\n(n hours below 0 °C per mode: {sorted(sub.drop_duplicates('mode').n.tolist())}; "
          f"days: {sorted(sub.drop_duplicates('mode').n_days.tolist())})", ""]
    if (RES / "rolling_cells.csv").exists():
        rc = pd.read_csv(RES / "rolling_cells.csv")
        for direction, side in [("forward", "warm"), ("backward", "cold"), ("forward", "inside"), ("backward", "inside")]:
            c = rc[(rc.direction == direction) & (rc.region == side) & (rc.n >= 24)]
            t = c.groupby("model").coverage.agg(["count", "median", "min",
                                                 lambda x: (x >= 0.85).mean()]).round(3)
            t.columns = ["cells", "median", "worst", "share of cells >= 0.85"]
            L += [f"\n## Rolling-origin, {direction} windows, region = {side} (cells with >= 24 h)\n",
                  t.sort_values("median").to_markdown(), ""]
    if (RES / "paired_tests.csv").exists():
        pt = pd.read_csv(RES / "paired_tests.csv")
        L += ["\n## Paired coverage differences (proposed - comparator), day-block bootstrap\n",
              pt.round(3).to_markdown(index=False), ""]
    (RES / "summary.md").write_text("\n".join(L))


def main():
    cfg = json.load(open(RES / "config.json"))
    cov = pd.read_csv(RES / "coverage_all.csv")
    cond = pd.read_csv(RES / "coverage_by_temperature.csv")
    det = pd.read_csv(RES / "detection.csv")
    phys = pd.read_csv(RES / "physics_parameters.csv")
    aci = pd.read_csv(RES / "aci.csv")
    preds = pd.read_pickle(RES / "predictions.pkl")
    df = data.add_features(data.load(data.download(ROOT / "data")))
    fig_overview(df, cfg); fig_physics_fit(df, cfg); fig_calibration(cov)
    fig_conditional_coverage(cond); fig_detection(preds, det); fig_key(preds, cond)
    fig_robustness(pd.read_csv(RES / "robustness_folds.csv"))
    if (RES / "rolling_cells.csv").exists():
        fig_rolling(pd.read_csv(RES / "rolling_cells.csv"))
    if (RES / "paired_tests.csv").exists():
        fig_forest(pd.read_csv(RES / "paired_tests.csv"))
    summary_md(cov, cond, det, phys, aci)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
