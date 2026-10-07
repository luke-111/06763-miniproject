"""PCA process monitor using Hotelling's T-squared and SPE statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from common import (
    CHANNELS,
    KEYS,
    RESULTS,
    Standardizer,
    load,
)

VARIANCE_TARGET = 0.90
EXPECTED_SCORE_ROWS = 300_000


@dataclass(frozen=True)
class PCAModel:
    """Parameters required to apply the fitted PCA monitor."""

    all_eigenvalues: np.ndarray
    retained_eigenvalues: np.ndarray
    loadings: np.ndarray
    n_components: int
    cumulative_explained_variance: float


def fit_pca(
    standardized_train: pl.DataFrame,
    variance_target: float = VARIANCE_TARGET,
) -> PCAModel:
    """Fit PCA to standardized fault-free training observations."""

    z_train = standardized_train.select(CHANNELS).to_numpy()

    # np.cov uses sample normalization (N - 1) by default.
    covariance = np.cov(z_train, rowvar=False)

    # eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    explained_ratio = eigenvalues / eigenvalues.sum()
    cumulative_variance = np.cumsum(explained_ratio)

    n_components = int(
        np.searchsorted(
            cumulative_variance,
            variance_target,
            side="left",
        )
        + 1
    )

    retained_eigenvalues = eigenvalues[:n_components]
    loadings = eigenvectors[:, :n_components]

    if np.any(retained_eigenvalues <= 0):
        raise ValueError("All retained eigenvalues must be positive.")

    return PCAModel(
        all_eigenvalues=eigenvalues,
        retained_eigenvalues=retained_eigenvalues,
        loadings=loadings,
        n_components=n_components,
        cumulative_explained_variance=float(
            cumulative_variance[n_components - 1]
        ),
    )


def statistics(
    standardized_frame: pl.DataFrame,
    model: PCAModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute T-squared and SPE for every standardized observation."""

    z = standardized_frame.select(CHANNELS).to_numpy()

    component_scores = z @ model.loadings

    t_squared = np.sum(
        component_scores**2 / model.retained_eigenvalues,
        axis=1,
    )

    reconstruction = component_scores @ model.loadings.T
    residual = z - reconstruction
    spe = np.sum(residual**2, axis=1)

    return t_squared, spe


def spe_contributions(
    standardized_frame: pl.DataFrame,
    model: PCAModel,
) -> np.ndarray:
    """Return the per-channel squared PCA residual contributions."""

    z = standardized_frame.select(CHANNELS).to_numpy()
    component_scores = z @ model.loadings
    residual = z - component_scores @ model.loadings.T

    return residual**2


def score(
    standardized_frame: pl.DataFrame,
    model: PCAModel,
) -> pl.DataFrame:
    """Return identifiers, T-squared and SPE for every row."""

    t_squared, spe = statistics(
        standardized_frame,
        model,
    )

    return standardized_frame.select(KEYS).with_columns(
        pl.Series("T2", t_squared),
        pl.Series("SPE", spe),
    )


def validate_scores(scores: pl.DataFrame) -> None:
    """Check the required output structure before writing it."""

    expected_columns = [
        "faultNumber",
        "simulationRun",
        "sample",
        "T2",
        "SPE",
    ]

    if scores.columns != expected_columns:
        raise ValueError(
            f"Unexpected output columns: {scores.columns}"
        )

    if scores.height != EXPECTED_SCORE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SCORE_ROWS:,} rows, "
            f"found {scores.height:,}."
        )

    unique_rows = scores.unique(subset=KEYS).height

    if unique_rows != scores.height:
        raise ValueError("Duplicate identifier rows found.")

    for column in ("T2", "SPE"):
        values = scores[column].to_numpy()

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite values found in {column}."
            )

        if np.any(values < 0):
            raise ValueError(
                f"Negative values found in {column}."
            )


def run() -> pl.DataFrame:
    """Fit the PCA monitor and write results/scores_pca.parquet."""

    train, scored = load()

    standardizer = Standardizer(train)
    standardized_train = standardizer.transform(train)
    standardized_scored = standardizer.transform(scored)

    model = fit_pca(standardized_train)

    explained_ratio = (
        model.all_eigenvalues
        / model.all_eigenvalues.sum()
    )
    cumulative = np.cumsum(explained_ratio)

    print(f"Retained PCA components: {model.n_components}")
    print(
        "Cumulative explained variance: "
        f"{model.cumulative_explained_variance:.6f}"
    )

    if model.n_components > 1:
        print(
            "Variance with k - 1 components: "
            f"{cumulative[model.n_components - 2]:.6f}"
        )

    scores = score(standardized_scored, model)
    validate_scores(scores)

    RESULTS.mkdir(exist_ok=True)
    output_path = RESULTS / "scores_pca.parquet"
    scores.write_parquet(output_path)

    print(f"{scores.height:,} PCA rows scored")
    print(f"Saved PCA scores to {output_path}")

    return scores


if __name__ == "__main__":
    run()