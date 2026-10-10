"""K-nearest-neighbors classification with a deterministic tie policy.

For a query, rank training rows by Euclidean distance. ``argsort(..., kind="mergesort")``
is stable, so equal distances keep the smaller training-row index. The ``k``
closest labels then vote. ``np.unique`` sorts labels, and the winner is the
first label that attains the highest count, which is the smallest label.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from umd_ml.validation import require_finite_matrix, require_labels, require_positive_int


def euclidean_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between rows of ``a`` and rows of ``b``.

    The result has shape ``(a.shape[0], b.shape[0])``.
    """
    left = require_finite_matrix(a, name="a")
    right = require_finite_matrix(b, name="b")
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"a has {left.shape[1]} features and b has {right.shape[1]}; they must match."
        )
    return np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)


def majority_vote(labels: np.ndarray) -> np.generic:
    """Return the smallest label among those with the highest count."""
    values, counts = np.unique(labels, return_counts=True)
    if values.size == 0:
        raise ValueError("majority_vote requires at least one label.")
    winners = values[counts == counts.max()]
    return winners[0]


class KNeighborsClassifier:
    """Store the training table and classify by a uniform majority vote.

    ``k`` is fixed at construction. It must be at least 1 and, once ``fit``
    sees the training table, no larger than the number of training rows.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = require_positive_int(k, name="k")
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Remember a copy of the training features and labels."""
        features = require_finite_matrix(X, name="X")
        labels = require_labels(y, n_samples=features.shape[0])
        if self.k > features.shape[0]:
            raise ValueError(
                f"k={self.k} cannot exceed the number of training samples ({features.shape[0]})."
            )
        self._X = np.array(features, copy=True)
        self._y = labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Label each row by the majority class among its ``k`` nearest neighbors."""
        train_x, train_y = self._fitted()
        queries = require_finite_matrix(X, name="X")
        if queries.shape[1] != train_x.shape[1]:
            raise ValueError(
                f"X has {queries.shape[1]} features, but the classifier was fit on "
                f"{train_x.shape[1]}."
            )
        distances = euclidean_distances(queries, train_x)
        neighbor_index = np.argsort(distances, axis=1, kind="mergesort")[:, : self.k]
        neighbor_labels = train_y[neighbor_index]
        predictions = np.empty(queries.shape[0], dtype=train_y.dtype)
        for row in range(queries.shape[0]):
            predictions[row] = majority_vote(neighbor_labels[row])
        return predictions

    def _fitted(self) -> tuple[np.ndarray, np.ndarray]:
        if self._X is None or self._y is None:
            raise RuntimeError("Fit the classifier before calling predict.")
        return self._X, self._y

    def __repr__(self) -> str:
        return f"KNeighborsClassifier(k={self.k}, fitted={self._X is not None})"
