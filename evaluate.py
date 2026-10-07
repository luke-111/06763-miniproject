"""Evaluate the PCA and ridge fault monitors."""

from __future__ import annotations

import numpy as np
import polars as pl

from common import KEYS, RESULTS


VALIDATION_RUNS = (301, 400)
TEST_RUNS = (401, 500)

FAULT_ONSET_SAMPLE = 20
MINUTES_PER_SAMPLE = 3
CONSECUTIVE_EXCEEDANCES = 3

DETECTORS = ("T2", "SPE", "ridge")
FAULTS = range(1, 21)


def alarm_flags(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return alarms produced by three consecutive threshold exceedances."""

    exceedances = scores > threshold

    consecutive = np.convolve(
        exceedances.astype(int),
        np.ones(CONSECUTIVE_EXCEEDANCES, dtype=int),
        mode="valid",
    )

    alarms = consecutive >= CONSECUTIVE_EXCEEDANCES

    return np.concatenate(
        [
            np.zeros(CONSECUTIVE_EXCEEDANCES - 1, dtype=bool),
            alarms,
        ]
    )


def validation_threshold(
    scores: pl.DataFrame,
    score_column: str,
) -> float:
    """Calculate the validation-set 99th percentile."""

    validation_scores = (
        scores.filter(
            (pl.col("faultNumber") == 0)
            & pl.col("simulationRun").is_between(
                *VALIDATION_RUNS
            )
        )
        .get_column(score_column)
        .to_numpy()
    )

    if validation_scores.size == 0:
        raise ValueError(
            f"No validation scores found for {score_column}."
        )

    return float(np.quantile(validation_scores, 0.99))


def detector_metrics(
    scores: pl.DataFrame,
    score_column: str,
    detector: str,
    threshold: float,
) -> list[dict[str, object]]:
    """Calculate false alarms, detection rates and detection delays."""

    per_run: list[tuple[int, float, float | None]] = []

    grouped = (
        scores.select(KEYS + [score_column])
        .sort(KEYS)
        .group_by(
            ["faultNumber", "simulationRun"],
            maintain_order=True,
        )
    )

    for (fault, run), run_scores in grouped:
        values = run_scores.get_column(score_column).to_numpy()
        samples = run_scores.get_column("sample").to_numpy()

        alarms = alarm_flags(values, threshold)

        if fault == 0:
            if TEST_RUNS[0] <= run <= TEST_RUNS[1]:
                per_run.append(
                    (
                        0,
                        float(alarms.mean()),
                        None,
                    )
                )

            continue

        post_fault = samples > FAULT_ONSET_SAMPLE
        post_fault_alarms = alarms & post_fault

        detection_rate = float(
            alarms[post_fault].mean()
        )

        alarm_positions = np.flatnonzero(
            post_fault_alarms
        )

        if alarm_positions.size:
            first_alarm_sample = samples[
                alarm_positions[0]
            ]

            delay = float(
                (
                    first_alarm_sample
                    - FAULT_ONSET_SAMPLE
                )
                * MINUTES_PER_SAMPLE
            )
        else:
            delay = None

        per_run.append(
            (
                int(fault),
                detection_rate,
                delay,
            )
        )

    rows: list[dict[str, object]] = []

    false_alarm_runs = [
        row for row in per_run if row[0] == 0
    ]

    if len(false_alarm_runs) != 100:
        raise ValueError(
            f"{detector}: expected 100 test runs, "
            f"found {len(false_alarm_runs)}."
        )

    rows.append(
        {
            "fault": 0,
            "detector": detector,
            "detection_rate": float(
                np.mean(
                    [
                        row[1]
                        for row in false_alarm_runs
                    ]
                )
            ),
            "median_delay_min": None,
            "runs_missed": None,
        }
    )

    for fault in FAULTS:
        fault_runs = [
            row for row in per_run
            if row[0] == fault
        ]

        if len(fault_runs) != 20:
            raise ValueError(
                f"{detector}, fault {fault}: "
                f"expected 20 runs, "
                f"found {len(fault_runs)}."
            )

        detected_delays = [
            row[2]
            for row in fault_runs
            if row[2] is not None
        ]

        rows.append(
            {
                "fault": fault,
                "detector": detector,
                "detection_rate": float(
                    np.mean(
                        [
                            row[1]
                            for row in fault_runs
                        ]
                    )
                ),
                "median_delay_min": (
                    float(
                        np.median(
                            detected_delays
                        )
                    )
                    if detected_delays
                    else None
                ),
                "runs_missed": sum(
                    row[2] is None
                    for row in fault_runs
                ),
            }
        )

    return rows


def run() -> None:
    """Generate thresholds.csv and detection.csv."""

    pca_path = RESULTS / "scores_pca.parquet"
    ridge_path = RESULTS / "scores_ridge.parquet"

    pca_scores = pl.read_parquet(pca_path)
    ridge_scores = pl.read_parquet(ridge_path)

    thresholds = {
        "T2": validation_threshold(
            pca_scores,
            "T2",
        ),
        "SPE": validation_threshold(
            pca_scores,
            "SPE",
        ),
        "ridge": validation_threshold(
            ridge_scores,
            "score",
        ),
    }

    threshold_table = pl.DataFrame(
        {
            "detector": list(thresholds),
            "threshold": list(thresholds.values()),
        }
    )

    metric_rows: list[dict[str, object]] = []

    metric_rows.extend(
        detector_metrics(
            pca_scores,
            "T2",
            "T2",
            thresholds["T2"],
        )
    )

    metric_rows.extend(
        detector_metrics(
            pca_scores,
            "SPE",
            "SPE",
            thresholds["SPE"],
        )
    )

    metric_rows.extend(
        detector_metrics(
            ridge_scores,
            "score",
            "ridge",
            thresholds["ridge"],
        )
    )

    detection_table = pl.DataFrame(
        metric_rows,
        schema={
            "fault": pl.Int64,
            "detector": pl.String,
            "detection_rate": pl.Float64,
            "median_delay_min": pl.Float64,
            "runs_missed": pl.Int64,
        },
    ).sort(["fault", "detector"])

    if detection_table.height != 63:
        raise ValueError(
            "Expected 63 detection rows, "
            f"found {detection_table.height}."
        )

    duplicate_rows = (
        detection_table.height
        - detection_table.unique(
            subset=["fault", "detector"]
        ).height
    )

    if duplicate_rows:
        raise ValueError(
            "Duplicate fault-detector rows found."
        )

    RESULTS.mkdir(exist_ok=True)

    threshold_path = RESULTS / "thresholds.csv"
    detection_path = RESULTS / "detection.csv"

    threshold_table.write_csv(threshold_path)
    detection_table.write_csv(detection_path)

    print("THRESHOLDS")
    print(threshold_table)

    print("\nFALSE-ALARM RATES")
    print(
        detection_table.filter(
            pl.col("fault") == 0
        )
    )

    print("\nEXPECTED DIFFICULT FAULTS")
    print(
        detection_table.filter(
            pl.col("fault").is_in([3, 9, 15])
        )
    )

    print(
        f"\nSaved thresholds to {threshold_path}"
    )
    print(
        f"Saved detection results to {detection_path}"
    )


if __name__ == "__main__":
    run()