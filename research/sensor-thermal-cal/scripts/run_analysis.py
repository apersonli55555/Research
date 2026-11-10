"""Run the full analysis on a folder of recordings.

  python scripts/run_analysis.py                 # real data in data/raw
  python scripts/run_analysis.py --simulated     # FAKE data in data/simulated (pipeline test only)

Real and fake data are never mixed, and outputs go to separate folders
(results/real, results/simulated). Everything made from fake data is stamped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from thermcal import evaluate, io  # noqa: E402

COL = {"Linear": "#b9b8b2", "Quadratic": "#eb6834", "Cubic": "#8a8984", "Quadratic + rate": "#2a78d6",
       "Boosted trees": "#1baf7a"}
STAMP = "SIMULATED DATA: pipeline test only, NOT a result"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulated", action="store_true")
    ap.add_argument("--t-lo", type=float, default=15.0)
    ap.add_argument("--t-hi", type=float, default=30.0)
    a = ap.parse_args()
    kind = "simulated" if a.simulated else "real"
    df = io.load_runs(ROOT / "data" / ("simulated" if a.simulated else "raw"))
    if df["simulated"].any() != a.simulated:
        sys.exit("Refusing to continue: the simulated flag of the files does not match the folder.")
    res_dir, fig_dir = ROOT / "results" / kind, ROOT / "figures" / kind
    res_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    stamp = STAMP if a.simulated else ""

    print(f"{len(df)} blocks, units {sorted(df.unit.unique())}, {df.run.nunique()} runs, "
          f"T from {df['T'].min():.1f} to {df['T'].max():.1f} degC")
    r = evaluate.evaluate(df, a.t_lo, a.t_hi)
    r.to_csv(res_dir / "per_fold.csv", index=False)

    # pooled over units, channels and folds (weighted by number of test blocks)
    def pool(g):
        w = g["n"]
        return pd.Series(dict(n_blocks=int(w.sum()), coverage_split=np.average(g.cov_split, weights=w),
                              coverage_weighted=np.average(g.cov_weighted_given_finite.fillna(0), weights=w * (1 - g.weighted_refused))
                              if (w * (1 - g.weighted_refused)).sum() > 0 else np.nan,
                              weighted_refused=np.average(g.weighted_refused, weights=w),
                              worst_cell_coverage=g.cov_split.min()))
    summ = r.groupby(["design", "region", "model"]).apply(pool, include_groups=False).reset_index()
    summ.to_csv(res_dir / "coverage_summary.csv", index=False)
    comp = r.groupby(["design", "channel", "region", "model"])[["raw_rms", "resid_rms"]].median().reset_index()
    comp.to_csv(res_dir / "compensation_summary.csv", index=False)
    head = f"# {stamp}\n\n" if stamp else ""
    (res_dir / "summary.md").write_text(
        head + f"Calibration range for the restricted design: {a.t_lo} to {a.t_hi} degC. Target coverage 90 %.\n\n"
        "## Coverage of 90 % intervals\n\n" + summ.round(3).to_markdown(index=False)
        + "\n\n## Error before and after compensation (median RMS over folds)\n\n" + comp.round(4).to_markdown(index=False) + "\n")

    # Figure 1: sensor error vs temperature, one panel per channel, first unit
    u0 = sorted(df.unit.unique())[0]; d = df[df.unit == u0]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    for ax, ch in zip(axes, io.CHANNELS):
        y = evaluate.remove_run_offset(d, ch)
        for run, g in d.groupby("run"):
            ax.plot(g["T"], y[g.index], ".", ms=1.5, alpha=.5)
        ax.axvspan(a.t_lo, a.t_hi, color="#8a8984", alpha=.15, lw=0)
        ax.set_title(f"{ch} [{io.UNITS_OF[ch]}]", fontsize=9); ax.set_xlabel("chip temperature [°C]")
    axes[0].set_ylabel(f"unit {u0}: error relative to 20–30 °C")
    fig.suptitle(stamp, color="#b00020", fontsize=9); fig.tight_layout()
    fig.savefig(fig_dir / "fig1_error_vs_temperature.png", dpi=200); plt.close(fig)

    # Figure 2: coverage by region, restricted vs full calibration
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    regs = ["cold", "inside", "warm"]
    for ax, design in zip(axes, ["restricted", "full"]):
        s = summ[summ.design == design]
        for k, (m, g) in enumerate(s.groupby("model")):
            g = g.set_index("region").reindex(regs)
            ax.plot(np.arange(3) + (k - 2) * 0.06, g.coverage_split, "o-", color=COL.get(m, "k"), ms=4, label=m)
        ax.axhline(0.9, color="k", ls="--", lw=.8); ax.set_xticks(range(3), ["colder", "inside", "warmer"])
        ax.set_title(f"{design} calibration" + (f" ({a.t_lo:g}–{a.t_hi:g} °C)" if design == "restricted" else " (all temperatures)"),
                     fontsize=9)
        ax.set_xlabel("test temperature relative to calibration range")
    axes[0].set_ylabel("coverage of 90 % interval"); axes[0].set_ylim(-0.02, 1.02); axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle(stamp, color="#b00020", fontsize=9); fig.tight_layout()
    fig.savefig(fig_dir / "fig2_coverage_by_region.png", dpi=200); plt.close(fig)
    print(summ[summ.design == "restricted"].round(3).to_string(index=False))
    print("outputs ->", res_dir, "and", fig_dir)


if __name__ == "__main__":
    main()
