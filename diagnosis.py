"""Diagnosis: the channels that drive the SPE and ridge alarms, per fault.

SPE and the ridge score are sums of squares over the 52 channels, so each splits into one
contribution per channel. For each fault and detector, the contributions are averaged over every
sample in alarm after sample 20, pooled over the 20 runs, and the five largest are kept.

The alarms are the evaluation's: evaluate.py's threshold and three-in-a-row rule, applied to the
scores in results/, so the samples averaged here are the ones counted in detection.csv.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from common import CHANNELS, FAULT, FAULT_START, FAULTS, KEYS, RESULTS, RUN, SAMPLE, Standardizer, load
from evaluate import alarm_flags, validation_threshold
from forecast_monitor import ForecastMonitor, make_table
from pca_monitor import fit_pca, spe_contributions

TOP = 5
SCORES = {"SPE": ("scores_pca.parquet", "SPE"), "ridge": ("scores_ridge.parquet", "score")}


def channel_terms(train: pl.DataFrame, faulty: pl.DataFrame) -> dict[str, tuple[pl.DataFrame, np.ndarray]]:
    """Return, for SPE and ridge, the keys of the faulty rows and each channel's term of the statistic.

    SPE: (z_j - (P P^T z)_j)^2 from the PCA monitor. Ridge: the squared standardized residual of
    channel j from the forecast monitor. Each row of terms sums to that row's SPE or ridge score.
    """
    std = Standardizer(train)
    z_train, z = std.transform(train), std.transform(faulty)
    table = make_table(z)
    return {
        "SPE": (z.select(KEYS), spe_contributions(z, fit_pca(z_train))),
        "ridge": (table.select(KEYS), ForecastMonitor().fit(make_table(z_train)).contributions(table)),
    }


def alarm_samples(detector: str, keys: pl.DataFrame, terms: np.ndarray) -> np.ndarray:
    """Return which faulty rows are in alarm after sample 20, by the evaluation's rule on the scores in results/.

    Raises if the channel terms do not add up to those scores, i.e. if results/ is out of date.
    """
    name, column = SCORES[detector]
    scores = pl.read_parquet(RESULTS / name)
    threshold = validation_threshold(scores, column)
    faulty = scores.filter(pl.col(FAULT) > 0).sort(KEYS)
    if not faulty.select(KEYS).equals(keys):
        raise ValueError(f"results/{name} does not hold the faulty rows the monitor scores")
    score = faulty[column].to_numpy()
    gap = float(np.max(np.abs(terms.sum(axis=1) - score) / score))
    if gap > 1e-6:
        raise ValueError(f"the channel terms do not add up to results/{name} (relative gap {gap:.1e})")
    print(f"{detector}: threshold {threshold!r}; terms add up to results/{name} to a relative {gap:.1e}")

    flags = np.concatenate([alarm_flags(g[column].to_numpy(), threshold)
                            for _, g in faulty.group_by([FAULT, RUN], maintain_order=True)])
    return flags & (faulty[SAMPLE].to_numpy() > FAULT_START)


def mean_contributions(detector: str, keys: pl.DataFrame, terms: np.ndarray) -> pl.DataFrame:
    """Average each channel's term over the alarm samples after sample 20 of each fault, pooled over its runs.

    Faults with no such sample are left out, since their average is undefined.
    """
    keep = alarm_samples(detector, keys, terms)
    fault = keys[FAULT].to_numpy()

    frames = []
    for f in range(FAULTS[0], FAULTS[1] + 1):
        m = keep & (fault == f)
        if m.any():
            frames.append(pl.DataFrame({"channel": CHANNELS, "contribution": terms[m].mean(axis=0)})
                          .with_columns(fault=pl.lit(f), detector=pl.lit(detector),
                                        samples=pl.lit(int(m.sum()))))
    return pl.concat(frames).select("fault", "detector", "channel", "contribution", "samples")


def mean_table() -> pl.DataFrame:
    """Return the mean contribution of every channel, per fault and detector."""
    train, scored = load()
    terms = channel_terms(train, scored.filter(pl.col(FAULT) > 0))
    return pl.concat([mean_contributions(detector, keys, t) for detector, (keys, t) in terms.items()])


def top_channels(means: pl.DataFrame) -> pl.DataFrame:
    """Return the TOP channels with the largest mean contribution per fault and detector, ranked from 1."""
    return (means.sort(["fault", "detector", "contribution"], descending=[False, False, True])
            .group_by(["fault", "detector"], maintain_order=True).head(TOP)
            .with_columns(rank=pl.int_range(1, pl.len() + 1).over("fault", "detector"))
            .select("fault", "detector", "rank", "channel", "contribution"))


def run() -> pl.DataFrame:
    """Write results/contributions.csv; print the thresholds and the leading channel per fault and detector."""
    means = mean_table()
    top = top_channels(means)
    RESULTS.mkdir(exist_ok=True)
    top.write_csv(RESULTS / "contributions.csv")

    lead = (top.filter(pl.col("rank") == 1)
            .join(means.select("fault", "detector", "samples").unique(), on=["fault", "detector"])
            .sort("fault", "detector"))
    with pl.Config(tbl_rows=-1):
        print(lead.select("fault", "detector", "samples", "channel", "contribution"))
    have = set(zip(lead["fault"].to_list(), lead["detector"].to_list()))
    empty = [f"{f}/{d}" for f in range(FAULTS[0], FAULTS[1] + 1) for d in ("SPE", "ridge") if (f, d) not in have]
    print(f"{top.height} rows written; no alarm sample after sample {FAULT_START}: {', '.join(empty) or 'none'}")
    return top


if __name__ == "__main__":
    run()
