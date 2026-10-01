"""Paths, data splits and the standardization shared by the PCA and forecast monitors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FREE = DATA / "tep_fault_free_training.parquet"
FAULTY = DATA / "tep_faulty_training_runs01-20.parquet"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FAULT = "faultNumber"
RUN = "simulationRun"
SAMPLE = "sample"
KEYS = [FAULT, RUN, SAMPLE]
CHANNELS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]

TRAIN = (1, 300)
VALIDATION = (301, 400)
TEST = (401, 500)
FAULTS = (1, 20)
FAULT_RUNS = (1, 20)
FAULT_START = 20
MINUTES_PER_SAMPLE = 3


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return training runs 1 to 300 and the validation, test and faulty rows, both sorted by fault, run and sample."""
    free = pl.read_parquet(FREE, columns=KEYS + CHANNELS)
    faulty = pl.read_parquet(FAULTY, columns=KEYS + CHANNELS)
    train = free.filter(pl.col(RUN).is_between(*TRAIN)).sort(KEYS)
    scored = pl.concat([
        free.filter(pl.col(RUN).is_between(VALIDATION[0], TEST[1])),
        faulty.filter(pl.col(FAULT).is_between(*FAULTS) & pl.col(RUN).is_between(*FAULT_RUNS)),
    ]).sort(KEYS)
    return train, scored


class Standardizer:
    """Per-channel mean and standard deviation (ddof=1) from the training runs."""

    def __init__(self, train: pl.DataFrame):
        x = train.select(CHANNELS).to_numpy()
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=1)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return the keys and the 52 standardized channels."""
        z = (frame.select(CHANNELS).to_numpy() - self.mean) / self.std
        return frame.select(KEYS).with_columns(
            [pl.Series(c, z[:, j]) for j, c in enumerate(CHANNELS)])


def split_of(frame: pl.DataFrame) -> pl.DataFrame:
    """Add a split column: validation, test or faulty."""
    return frame.with_columns(
        split=pl.when(pl.col(FAULT) > 0).then(pl.lit("faulty"))
        .when(pl.col(RUN).is_between(*VALIDATION)).then(pl.lit("validation"))
        .otherwise(pl.lit("test")))
