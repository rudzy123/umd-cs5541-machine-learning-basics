"""Deterministic synthetic datasets.

Nothing in this module reads a file or contacts the network. Every array is
drawn from ``numpy.random.Generator``, and every dataset says that it is
synthetic in ``source`` and in ``description``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from umd_ml.validation import (
    require_finite_float,
    require_nonnegative_float,
    require_positive_int,
    require_seed,
)

DatasetKind = Literal["regression", "classification"]

SYNTHETIC_BANNER = (
    "Synthetic dataset generated locally by a deterministic NumPy Generator. "
    "These values are not measurements of real people or events."
)


@dataclass(frozen=True, repr=False)
class SyntheticDataset:
    """A locally generated table of features and targets.

    ``X`` has shape ``(n_samples, n_features)``. ``y`` has shape ``(n_samples,)``.
    The object is frozen, but the arrays themselves remain mutable NumPy arrays;
    copy them before an experiment that must not change the source table.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    description: str
    seed: int
    kind: DatasetKind
    source: str
    noise: float

    def __post_init__(self) -> None:
        if self.source != "synthetic":
            raise ValueError("Datasets in this package are synthetic and must be labeled as such.")
        if "synthetic" not in self.description.lower():
            raise ValueError("Dataset descriptions must state that the data is synthetic.")
        if self.kind not in ("regression", "classification"):
            raise ValueError("kind must be 'regression' or 'classification'.")
        if self.X.ndim != 2:
            raise ValueError(f"X must have shape (n_samples, n_features); got ndim={self.X.ndim}.")
        if self.y.shape != (self.X.shape[0],):
            raise ValueError("y must have one label per row of X.")
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature_names must contain one name per column of X.")
        if (
            not np.isfinite(self.X).all()
            or not np.isfinite(np.asarray(self.y, dtype=np.float64)).all()
        ):
            raise ValueError("Synthetic features and labels must be finite.")

    def __repr__(self) -> str:
        return (
            f"SyntheticDataset(kind={self.kind!r}, source='synthetic', "
            f"shape={self.X.shape}, seed={self.seed})"
        )

    def to_frame(self) -> pd.DataFrame:
        """Return features and the target as a table.

        The synthetic label is stored on ``DataFrame.attrs`` rather than as a
        column, so it cannot be mistaken for a feature.
        """
        frame = pd.DataFrame(np.array(self.X, copy=True), columns=list(self.feature_names))
        target_name = "target" if self.kind == "regression" else "label"
        frame[target_name] = np.array(self.y, copy=True)
        frame.attrs["source"] = self.source
        frame.attrs["description"] = self.description
        frame.attrs["seed"] = self.seed
        return frame

    def label_counts(self) -> pd.Series:
        """Count classification labels. Regression targets are continuous, so they are rejected."""
        if self.kind != "classification":
            raise ValueError("label_counts is for classification datasets.")
        return pd.Series(self.y, name="label").value_counts().sort_index()


def _feature_names(n_features: int) -> tuple[str, ...]:
    return tuple(f"x{index}" for index in range(1, n_features + 1))


def make_regression(
    n_samples: int = 100,
    n_features: int = 1,
    *,
    coefficients: np.ndarray | None = None,
    intercept: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> SyntheticDataset:
    """Draw a linear regression table from a local Generator.

    Generation order for a fixed seed is: feature matrix, then coefficients
    when they are not supplied, then Gaussian noise when ``noise > 0``.
    With ``noise=0`` the rows satisfy ``y = X @ coefficients + intercept``
    exactly.
    """
    n_samples = require_positive_int(n_samples, name="n_samples")
    n_features = require_positive_int(n_features, name="n_features")
    intercept = require_finite_float(intercept, name="intercept")
    noise = require_nonnegative_float(noise, name="noise")
    seed = require_seed(seed)

    rng = np.random.default_rng(seed)
    features = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    if coefficients is None:
        slopes = rng.normal(loc=0.0, scale=1.0, size=n_features)
    else:
        slopes = np.asarray(coefficients, dtype=np.float64)
        if slopes.shape != (n_features,):
            raise ValueError(f"coefficients must have shape ({n_features},); got {slopes.shape}.")
        if not np.isfinite(slopes).all():
            raise ValueError("coefficients must be finite.")
    target = features @ slopes + intercept
    if noise > 0.0:
        target = target + rng.normal(loc=0.0, scale=noise, size=n_samples)

    description = (
        f"{SYNTHETIC_BANNER} Linear regression table with {n_samples} rows, "
        f"{n_features} features, intercept {intercept}, noise scale {noise}, and seed {seed}."
    )
    return SyntheticDataset(
        X=features,
        y=target,
        feature_names=_feature_names(n_features),
        description=description,
        seed=seed,
        kind="regression",
        source="synthetic",
        noise=noise,
    )


def make_sine_regression(
    n_samples: int = 80,
    *,
    noise: float = 0.0,
    seed: int = 0,
) -> SyntheticDataset:
    """Draw a one-column synthetic table whose noise-free target is ``sin(x1)``."""
    n_samples = require_positive_int(n_samples, name="n_samples")
    noise = require_nonnegative_float(noise, name="noise")
    seed = require_seed(seed)
    rng = np.random.default_rng(seed)
    features = rng.uniform(-3.0, 3.0, size=(n_samples, 1))
    target = np.sin(features[:, 0])
    if noise > 0.0:
        target = target + rng.normal(loc=0.0, scale=noise, size=n_samples)
    description = (
        f"{SYNTHETIC_BANNER} Sine regression table with {n_samples} rows, "
        f"noise scale {noise}, and seed {seed}. The noise-free target is sin(x1)."
    )
    return SyntheticDataset(
        X=features,
        y=target,
        feature_names=("x1",),
        description=description,
        seed=seed,
        kind="regression",
        source="synthetic",
        noise=noise,
    )


def make_classification(
    n_samples: int = 100,
    n_features: int = 2,
    *,
    n_classes: int = 2,
    separation: float = 2.0,
    noise: float = 0.5,
    seed: int = 0,
) -> SyntheticDataset:
    """Draw class-conditional Gaussian blobs and shuffle the rows.

    Class ``c`` is centered at ``((c - (n_classes - 1) / 2) * separation, ...)``.
    When a second feature exists, the centers alternate slightly in that
    coordinate so a two-dimensional plot is not a single line of blobs.
    Labels are the integers ``0, ..., n_classes - 1``.
    """
    n_samples = require_positive_int(n_samples, name="n_samples")
    n_features = require_positive_int(n_features, name="n_features")
    n_classes = require_positive_int(n_classes, name="n_classes")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")
    if n_classes > n_samples:
        raise ValueError("n_classes cannot exceed n_samples.")
    separation = require_finite_float(separation, name="separation")
    if separation <= 0.0:
        raise ValueError("separation must be greater than 0.")
    noise = require_nonnegative_float(noise, name="noise")
    seed = require_seed(seed)

    rng = np.random.default_rng(seed)
    base, extra = divmod(n_samples, n_classes)
    counts = [base + (1 if index < extra else 0) for index in range(n_classes)]
    centers = np.zeros((n_classes, n_features), dtype=np.float64)
    for class_index in range(n_classes):
        centers[class_index, 0] = (class_index - (n_classes - 1) / 2.0) * separation
        if n_features > 1:
            centers[class_index, 1] = ((-1.0) ** class_index) * separation * 0.15

    blocks = []
    labels = []
    for class_index, count in enumerate(counts):
        scale = 1.0 if noise == 0.0 else noise
        # noise == 0 places every row on its center. A zero scale would make
        # Generator.normal reject the draw on some NumPy versions, so the
        # center is repeated explicitly.
        if noise == 0.0:
            block = np.repeat(centers[class_index][None, :], count, axis=0)
        else:
            block = rng.normal(loc=centers[class_index], scale=scale, size=(count, n_features))
        blocks.append(block)
        labels.append(np.full(count, class_index, dtype=np.int64))

    features = np.vstack(blocks)
    target = np.concatenate(labels)
    order = rng.permutation(n_samples)
    description = (
        f"{SYNTHETIC_BANNER} Classification blobs with {n_samples} rows, "
        f"{n_features} features, {n_classes} classes, separation {separation}, "
        f"noise scale {noise}, and seed {seed}."
    )
    return SyntheticDataset(
        X=features[order],
        y=target[order],
        feature_names=_feature_names(n_features),
        description=description,
        seed=seed,
        kind="classification",
        source="synthetic",
        noise=noise,
    )
