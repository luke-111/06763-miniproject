"""Plot the channels that drive the SPE and ridge alarms for a few faults, as shares of the mean statistic."""

from __future__ import annotations

import argparse

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

from common import FIGURES
from diagnosis import TOP, mean_table

matplotlib.use("Agg")

# Downs and Vogel (1993); streams in brackets.
DESCRIPTION = {
    "xmeas_1": "A feed (1)", "xmeas_2": "D feed (2)", "xmeas_3": "E feed (3)", "xmeas_4": "A and C feed (4)",
    "xmeas_5": "recycle flow (8)", "xmeas_6": "reactor feed rate (6)", "xmeas_7": "reactor pressure",
    "xmeas_8": "reactor level", "xmeas_9": "reactor temperature", "xmeas_10": "purge rate (9)",
    "xmeas_11": "separator temperature", "xmeas_12": "separator level", "xmeas_13": "separator pressure",
    "xmeas_14": "separator underflow (10)", "xmeas_15": "stripper level", "xmeas_16": "stripper pressure",
    "xmeas_17": "stripper underflow (11)", "xmeas_18": "stripper temperature", "xmeas_19": "stripper steam flow",
    "xmeas_20": "compressor work", "xmeas_21": "reactor cooling water outlet temp.",
    "xmeas_22": "separator cooling water outlet temp.",
    **{f"xmeas_{23 + i}": f"reactor feed {c} (6)" for i, c in enumerate("ABCDEF")},
    **{f"xmeas_{29 + i}": f"purge gas {c} (9)" for i, c in enumerate("ABCDEFGH")},
    **{f"xmeas_{37 + i}": f"product {c} (11)" for i, c in enumerate("DEFGH")},
    "xmv_1": "D feed flow (2)", "xmv_2": "E feed flow (3)", "xmv_3": "A feed flow (1)",
    "xmv_4": "A and C feed flow (4)", "xmv_5": "compressor recycle valve", "xmv_6": "purge valve (9)",
    "xmv_7": "separator pot liquid flow (10)", "xmv_8": "stripper liquid product flow (11)",
    "xmv_9": "stripper steam valve", "xmv_10": "reactor cooling water flow", "xmv_11": "condenser cooling water flow",
}

# Chiang, Russell and Braatz (2000), Table 1.
FAULT_NAME = {
    1: "A/C feed ratio, B composition constant (4), step", 2: "B composition, A/C ratio constant (4), step",
    3: "D feed temperature (2), step", 4: "reactor cooling water inlet temperature, step",
    5: "condenser cooling water inlet temperature, step", 6: "A feed loss (1), step",
    7: "C header pressure loss (4), step", 8: "A, B, C feed composition (4), random",
    9: "D feed temperature (2), random", 10: "C feed temperature (4), random",
    11: "reactor cooling water inlet temperature, random", 12: "condenser cooling water inlet temperature, random",
    13: "reaction kinetics, slow drift", 14: "reactor cooling water valve, sticking",
    15: "condenser cooling water valve, sticking",
}

BAR = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def shares(means: pl.DataFrame) -> pl.DataFrame:
    """Add each channel's share of the mean statistic, the sum of the mean contributions of its fault and detector."""
    return means.with_columns(share=pl.col("contribution") / pl.col("contribution").sum().over("fault", "detector"))


def panel(ax: plt.Axes, g: pl.DataFrame, detector: str, right: float) -> None:
    """Draw the top channels of one fault and detector as horizontal bars, largest on top."""
    ax.set_xlim(0, right)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="x", labelsize=6.5, colors=MUTED, length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if g.is_empty():
        ax.set_yticks([])
        ax.set_title(f"{detector}: no alarm sample after sample 20", fontsize=7, color=MUTED, loc="right", pad=3)
        return
    g = g.sort("share", descending=True).head(TOP)
    y = list(range(g.height))[::-1]
    ax.barh(y, g["share"].to_numpy(), height=0.62, color=BAR)
    for yi, share in zip(y, g["share"]):
        ax.text(share + 0.012, yi, f"{share:.0%}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, [f"{c}  {DESCRIPTION[c]}" for c in g["channel"]], fontsize=6.5, color=INK)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_title(f"{detector} · {g['samples'][0]:,} alarm samples", fontsize=7, color=MUTED, loc="right", pad=3)


def run(faults: list[int]) -> None:
    """Write figures/contributions.png: the top channels of each fault, SPE on the left and ridge on the right."""
    means = shares(mean_table()).filter(pl.col("fault").is_in(faults))
    top = means.sort("share", descending=True).group_by("fault", "detector").head(TOP)
    right = min(1.0, (top["share"].max() or 0.0) + 0.12)

    height = 0.6 + 1.55 * len(faults)
    fig, axes = plt.subplots(len(faults), 2, figsize=(6.5, height), sharex=True, squeeze=False, layout="constrained")
    fig.get_layout_engine().set(hspace=0.2, rect=(0, 0, 1, 1 - 0.22 / height))
    for row, fault in zip(axes, faults):
        for ax, detector in zip(row, ("SPE", "ridge")):
            panel(ax, top.filter((pl.col("fault") == fault) & (pl.col("detector") == detector)), detector, right)
        title = row[0].annotate(f"IDV({fault}): {FAULT_NAME.get(fault, 'unknown')}", xy=(0.0, 1.0),
                                xycoords=("figure fraction", "axes fraction"), xytext=(2, 16),
                                textcoords="offset points", fontsize=8, color=INK, weight="bold")
        title.set_in_layout(False)
    fig.supxlabel("share of the mean statistic over the alarm samples after sample 20", fontsize=6.5, color=MUTED)

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "contributions.png", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--faults", type=int, nargs="+", default=[4, 6, 10])
    args = parser.parse_args()
    run(args.faults)
