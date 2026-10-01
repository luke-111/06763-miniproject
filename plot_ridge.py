"""Plot the ridge score of one faulty run against the 99th-percentile validation threshold."""

from __future__ import annotations

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from common import FAULT, FAULT_START, FIGURES, MINUTES_PER_SAMPLE, RESULTS, RUN, SAMPLE, VALIDATION

matplotlib.use("Agg")


def threshold(scores: pl.DataFrame) -> float:
    """Return numpy.quantile(score, 0.99) over the validation runs."""
    val = scores.filter((pl.col(FAULT) == 0) & pl.col(RUN).is_between(*VALIDATION))
    return float(np.quantile(val["score"].to_numpy(), 0.99))


def run(fault: int, sim_run: int) -> None:
    """Write figures/ridge_fault{fault}_run{sim_run}.png."""
    scores = pl.read_parquet(RESULTS / "scores_ridge.parquet")
    limit = threshold(scores)
    g = scores.filter((pl.col(FAULT) == fault) & (pl.col(RUN) == sim_run)).sort(SAMPLE)
    hours = g[SAMPLE].to_numpy() * MINUTES_PER_SAMPLE / 60

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.plot(hours, g["score"].to_numpy(), color="#2a6fb0", lw=1.2, label="score")
    ax.axhline(limit, color="#555555", ls="--", lw=1, label=f"threshold {limit:.1f}")
    ax.axvline(FAULT_START * MINUTES_PER_SAMPLE / 60, color="#999999", ls=":", lw=1)
    ax.text(FAULT_START * MINUTES_PER_SAMPLE / 60, ax.get_ylim()[1], " fault on", ha="left", va="top",
            fontsize=8, color="#333333")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("ridge score")
    ax.set_title(f"Ridge forecast-residual score, IDV({fault}), run {sim_run}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"ridge_fault{fault}_run{sim_run}.png", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault", type=int, default=4)
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()
    run(args.fault, args.run)
