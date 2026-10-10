"""Plot T2, SPE and the ridge score of one faulty run, each against its threshold in results/thresholds.csv."""

from __future__ import annotations

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from common import FAULT, FAULT_START, FIGURES, KEYS, MINUTES_PER_SAMPLE, RESULTS, RUN, SAMPLE

matplotlib.use("Agg")

PANELS = [("T2", "T2", "$T^2$"), ("SPE", "SPE", "SPE"), ("ridge", "score", "ridge score")]


def run(fault: int, sim_run: int) -> None:
    """Write figures/detectors_fault{fault}_run{sim_run}.png."""
    thresholds = dict(pl.read_csv(RESULTS / "thresholds.csv").iter_rows())
    pca = pl.read_parquet(RESULTS / "scores_pca.parquet")
    ridge = pl.read_parquet(RESULTS / "scores_ridge.parquet")
    g = (pca.join(ridge, on=KEYS, how="left")
         .filter((pl.col(FAULT) == fault) & (pl.col(RUN) == sim_run)).sort(SAMPLE))
    hours = g[SAMPLE].to_numpy() * MINUTES_PER_SAMPLE / 60
    onset = FAULT_START * MINUTES_PER_SAMPLE / 60

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 3.6), sharex=True)
    for ax, (detector, column, label) in zip(axes, PANELS):
        values = g[column].to_numpy()
        ax.plot(hours, values, color="#2a6fb0", lw=1.0)
        ax.set_ylim(0, 1.3 * max(values[~np.isnan(values)].max(), thresholds[detector]))
        ax.axhline(thresholds[detector], color="#555555", ls="--", lw=1,
                   label=f"threshold {thresholds[detector]:.1f}")
        ax.axvline(onset, color="#999999", ls=":", lw=1)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    axes[0].text(onset, axes[0].get_ylim()[1], " fault on", ha="left", va="top", fontsize=8, color="#333333")
    axes[0].set_title(f"IDV({fault}), run {sim_run}", fontsize=10)
    axes[-1].set_xlabel("time (h)")
    fig.tight_layout()

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"detectors_fault{fault}_run{sim_run}.png", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault", type=int, default=4)
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()
    run(args.fault, args.run)
