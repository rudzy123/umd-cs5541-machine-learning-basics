"""Feature standardization that cannot see future rows unless the caller cheats.

``StandardScaler`` learns the mean and scale in ``fit`` and only applies them
in ``transform``. ``standardize_splits`` fits on the training matrix and then
transforms every later matrix with those frozen numbers. Fitting on a stack of
training and test rows is leakage; this module does not offer that shortcut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from umd_ml.validation import FloatMatrix, require_finite_matrix

_VARIANCE_FLOOR = 1e-12


class StandardScaler:
    """Subtract the training mean and divide by the training standard deviation.

    The standard deviation is the population statistic (divide by ``n``, not
    ``n - 1``), matching the usual machine-learning convention. A column whose
    training standard deviation is below ``1e-12`` is left at scale 1 after
    centering, so a constant feature becomes a column of zeros instead of
    producing infinities.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        """Learn column statistics from ``X`` only."""
        array = require_finite_matrix(X, name="X")
        mean = np.mean(array, axis=0)
        variance = np.mean((array - mean) ** 2, axis=0)
        scale = np.sqrt(variance)
        self.mean_ = mean
        self.scale_ = np.where(scale < _VARIANCE_FLOOR, 1.0, scale)
        self.n_features_in_ = int(array.shape[1])
        return self

    def transform(self, X: np.ndarray) -> FloatMatrix:
        """Apply the statistics learned in ``fit`` without updating them."""
        mean, scale, n_features = self._fitted_statistics()
        array = require_finite_matrix(X, name="X")
        if array.shape[1] != n_features:
            raise ValueError(
                f"X has {array.shape[1]} features, but the scaler was fit on {n_features}."
            )
        return (array - mean) / scale

    def fit_transform(self, X: np.ndarray) -> FloatMatrix:
        """Fit on ``X`` and return the transformed training matrix."""
        return self.fit(X).transform(X)

    def _fitted_statistics(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self.mean_ is None or self.scale_ is None or self.n_features_in_ is None:
            raise RuntimeError("Fit the scaler before calling transform.")
        return self.mean_, self.scale_, self.n_features_in_


def standardize_splits(
    X_train: np.ndarray,
    *others: np.ndarray,
) -> tuple[StandardScaler, tuple[FloatMatrix, ...]]:
    """Fit a scaler on ``X_train`` and transform the train matrix and ``others``.

    Pass validation and test matrices as ``others``. Do not concatenate them
    into ``X_train`` first: that would let test rows change the mean and scale.
    """
    scaler = StandardScaler().fit(X_train)
    transformed = tuple(scaler.transform(part) for part in (X_train, *others))
    return scaler, transformed


def standardize_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize named columns, fitting the statistics on ``train`` only.

    The inputs are copied. Columns that are not listed in ``feature_columns``
    are left unchanged, which is the right behavior for a label column sitting
    in the same table.
    """
    if not feature_columns:
        raise ValueError("feature_columns must name at least one column.")
    missing = [
        column
        for column in feature_columns
        if column not in train.columns or column not in test.columns
    ]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}.")

    scaler = StandardScaler().fit(train.loc[:, feature_columns].to_numpy(dtype=np.float64))
    train_out = train.copy()
    test_out = test.copy()
    train_out.loc[:, feature_columns] = scaler.transform(
        train.loc[:, feature_columns].to_numpy(dtype=np.float64)
    )
    test_out.loc[:, feature_columns] = scaler.transform(
        test.loc[:, feature_columns].to_numpy(dtype=np.float64)
    )
    return train_out, test_out, scaler
