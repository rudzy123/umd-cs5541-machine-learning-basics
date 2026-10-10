"""Scores and the train/test split.

Metrics compare a prediction with a label. They do not look back at the
training procedure. The split is here because a metric only answers an
evaluation question when the rows were held out of fitting.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from umd_ml.validation import require_finite_matrix, require_seed

Average = Literal["binary", "macro"]


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average of ``(y_true - y_pred) ** 2``."""
    truth, prediction = _paired_vectors(y_true, y_pred, dtype=np.float64)
    return float(np.mean((truth - prediction) ** 2))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """One minus the residual sum of squares divided by the total sum of squares.

    When the targets have zero variance, the denominator vanishes. The score is
    then 1 if the predictions match the target and 0 otherwise.
    """
    truth, prediction = _paired_vectors(y_true, y_pred, dtype=np.float64)
    residual = float(np.sum((truth - prediction) ** 2))
    centered = float(np.sum((truth - np.mean(truth)) ** 2))
    if centered == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return 1.0 - residual / centered


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of rows whose predicted label equals the true label."""
    truth, prediction = _paired_vectors(y_true, y_pred)
    return float(np.mean(truth == prediction))


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: Average = "binary",
    positive_label: object = 1,
) -> float:
    """True positives divided by predicted positives.

    ``average='binary'`` scores ``positive_label`` and treats every other label
    as negative. ``average='macro'`` computes that score for each label that
    appears in ``y_true`` or ``y_pred``, then takes the unweighted mean.
    A class with no predicted positives scores 0 rather than dividing by zero.
    """
    return _average_class_score(
        y_true,
        y_pred,
        _precision_one,
        average=average,
        positive_label=positive_label,
    )


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: Average = "binary",
    positive_label: object = 1,
) -> float:
    """True positives divided by actual positives. See ``precision`` for averaging."""
    return _average_class_score(
        y_true,
        y_pred,
        _recall_one,
        average=average,
        positive_label=positive_label,
    )


def f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: Average = "binary",
    positive_label: object = 1,
) -> float:
    """Harmonic mean of precision and recall. See ``precision`` for averaging."""
    return _average_class_score(
        y_true,
        y_pred,
        _f1_one,
        average=average,
        positive_label=positive_label,
    )


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Count pairs of true and predicted labels.

    Rows are true labels and columns are predicted labels. When ``labels`` is
    omitted, the axis order is the sorted unique labels present in either
    vector. When ``labels`` is provided, that order is used, and every observed
    label must be included.
    """
    truth, prediction = _paired_vectors(y_true, y_pred)
    if labels is None:
        axis_labels = np.unique(np.concatenate([truth, prediction]))
    else:
        axis_labels = np.asarray(labels)
        if axis_labels.ndim != 1 or axis_labels.size == 0:
            raise ValueError("labels must be a non-empty one-dimensional array.")
        observed = np.concatenate([truth, prediction])
        if not np.isin(observed, axis_labels).all():
            raise ValueError("Every label in y_true and y_pred must be included in labels.")
    lookup = {label: index for index, label in enumerate(axis_labels.tolist())}
    matrix = np.zeros((axis_labels.size, axis_labels.size), dtype=np.int64)
    for true_label, predicted_label in zip(truth.tolist(), prediction.tolist(), strict=True):
        matrix[lookup[true_label], lookup[predicted_label]] += 1
    return matrix


def classification_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-label precision, recall, F1, and support as a table."""
    truth, prediction = _paired_vectors(y_true, y_pred)
    if labels is None:
        axis_labels = np.unique(np.concatenate([truth, prediction]))
    else:
        axis_labels = np.asarray(labels)
    rows: list[dict[str, object]] = []
    for label in axis_labels.tolist():
        rows.append(
            {
                "label": label,
                "precision": precision(truth, prediction, average="binary", positive_label=label),
                "recall": recall(truth, prediction, average="binary", positive_label=label),
                "f1": f1(truth, prediction, average="binary", positive_label=label),
                "support": int(np.sum(truth == label)),
            }
        )
    return pd.DataFrame(rows)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 0,
    stratify: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle rows with a Generator and hold out a test portion.

    ``test_size`` is a fraction strictly between 0 and 1. The test count is
    ``round(n * test_size)``, then clipped so neither side is empty.

    ``stratify``, when given, must be one group label per row. The same
    fraction is applied inside each group that has at least two rows. A group
    with a single row stays in the training side. Using the target as
    ``stratify`` balances classes; it is not feature leakage, because it does
    not estimate a transformation from test features. It does mean you have
    looked at the test labels while designing the split, which is acceptable
    for evaluation design and is not a license to fit a preprocessor on the
    test features.
    """
    features = require_finite_matrix(X, name="X")
    target = _as_split_target(y, n_samples=features.shape[0])
    fraction = _test_fraction(test_size)
    if features.shape[0] < 2:
        raise ValueError("A train/test split needs at least two samples.")
    rng = np.random.default_rng(require_seed(seed))

    if stratify is None:
        train_index, test_index = _partition(features.shape[0], fraction, rng)
    else:
        groups = np.asarray(stratify)
        if groups.ndim != 1 or groups.shape[0] != features.shape[0]:
            raise ValueError("stratify must have one entry per sample.")
        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for label in np.unique(groups):
            member = np.flatnonzero(groups == label)
            if member.size < 2:
                train_parts.append(member)
                continue
            local_train, local_test = _partition(int(member.size), fraction, rng)
            train_parts.append(member[local_train])
            test_parts.append(member[local_test])
        if not test_parts:
            raise ValueError("stratify could not place any sample in the test set.")
        train_index = np.concatenate(train_parts)
        test_index = np.concatenate(test_parts)
        rng.shuffle(train_index)
        rng.shuffle(test_index)

    return (
        features[train_index],
        features[test_index],
        target[train_index],
        target[test_index],
    )


def _paired_vectors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    dtype: np.dtype[np.generic] | type[np.generic] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    truth = _as_vector(y_true, name="y_true", dtype=dtype)
    prediction = _as_vector(y_pred, name="y_pred", dtype=dtype)
    if truth.shape != prediction.shape:
        raise ValueError(
            "y_true and y_pred must have the same length; "
            f"got {truth.shape[0]} and {prediction.shape[0]}."
        )
    if truth.size == 0:
        raise ValueError("Cannot score an empty prediction.")
    return truth, prediction


def _as_vector(
    values: np.ndarray,
    *,
    name: str,
    dtype: np.dtype[np.generic] | type[np.generic] | None,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    return array


def _as_split_target(values: np.ndarray, *, n_samples: int) -> np.ndarray:
    array = _as_vector(values, name="y", dtype=None)
    if array.shape[0] != n_samples:
        raise ValueError(
            f"y length must match the number of samples ({n_samples}); got {array.shape[0]}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("y must be numeric.")
    if not np.isfinite(array.astype(np.float64)).all():
        raise ValueError("y must contain only finite values.")
    return array


def _test_fraction(test_size: float) -> float:
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise TypeError("test_size must be a float strictly between 0 and 1.")
    fraction = float(test_size)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1.")
    return fraction


def _partition(
    n: int,
    test_size: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n)
    n_test = int(round(n * test_size))
    n_test = min(max(n_test, 1), n - 1)
    return order[n_test:], order[:n_test]


def _average_class_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_one,
    *,
    average: Average,
    positive_label: object,
) -> float:
    if average not in ("binary", "macro"):
        raise ValueError("average must be 'binary' or 'macro'.")
    truth, prediction = _paired_vectors(y_true, y_pred)
    if average == "binary":
        return score_one(truth, prediction, positive_label)
    labels = np.unique(np.concatenate([truth, prediction]))
    scores = [score_one(truth, prediction, label) for label in labels.tolist()]
    return float(np.mean(scores))


def _counts(y_true: np.ndarray, y_pred: np.ndarray, positive: object) -> tuple[int, int, int]:
    actual = y_true == positive
    predicted = y_pred == positive
    true_positive = int(np.sum(actual & predicted))
    false_positive = int(np.sum(~actual & predicted))
    false_negative = int(np.sum(actual & ~predicted))
    return true_positive, false_positive, false_negative


def _divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _precision_one(y_true: np.ndarray, y_pred: np.ndarray, positive: object) -> float:
    true_positive, false_positive, _false_negative = _counts(y_true, y_pred, positive)
    return _divide(true_positive, true_positive + false_positive)


def _recall_one(y_true: np.ndarray, y_pred: np.ndarray, positive: object) -> float:
    true_positive, _false_positive, false_negative = _counts(y_true, y_pred, positive)
    return _divide(true_positive, true_positive + false_negative)


def _f1_one(y_true: np.ndarray, y_pred: np.ndarray, positive: object) -> float:
    precision_value = _precision_one(y_true, y_pred, positive)
    recall_value = _recall_one(y_true, y_pred, positive)
    total = precision_value + recall_value
    if total == 0.0:
        return 0.0
    return 2.0 * precision_value * recall_value / total
