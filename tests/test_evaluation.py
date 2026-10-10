"""Metric arithmetic and the seeded train/test split."""

from __future__ import annotations

import numpy as np
import pytest

from umd_ml.evaluation import (
    accuracy,
    classification_scores,
    confusion_matrix,
    f1,
    mean_squared_error,
    precision,
    r_squared,
    recall,
    train_test_split,
)


def test_classification_metrics_match_a_hand_count() -> None:
    truth = np.array([0, 0, 1, 1])
    prediction = np.array([0, 1, 1, 1])
    np.testing.assert_array_equal(confusion_matrix(truth, prediction), np.array([[1, 1], [0, 2]]))
    assert accuracy(truth, prediction) == pytest.approx(0.75)
    assert precision(truth, prediction) == pytest.approx(2.0 / 3.0)
    assert recall(truth, prediction) == pytest.approx(1.0)
    assert f1(truth, prediction) == pytest.approx(0.8)
    assert precision(truth, prediction, average="macro") == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert recall(truth, prediction, average="macro") == pytest.approx((0.5 + 1.0) / 2.0)


def test_zero_division_scores_are_zero() -> None:
    truth = np.array([0, 0, 0])
    prediction = np.array([0, 0, 0])
    assert precision(truth, prediction, positive_label=1) == 0.0
    assert recall(truth, prediction, positive_label=1) == 0.0
    assert f1(truth, prediction, positive_label=1) == 0.0


def test_regression_scores() -> None:
    truth = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error(truth, np.array([1.0, 2.0, 4.0])) == pytest.approx(1.0 / 3.0)
    assert r_squared(truth, truth) == pytest.approx(1.0)
    assert r_squared(truth, np.array([2.0, 2.0, 2.0])) == pytest.approx(0.0)
    assert r_squared(truth, np.array([1.0, 2.0, 4.0])) == pytest.approx(0.5)
    assert r_squared(np.array([2.0, 2.0]), np.array([2.0, 2.0])) == 1.0


def test_confusion_matrix_respects_label_order() -> None:
    truth = np.array([0, 1])
    prediction = np.array([0, 1])
    matrix = confusion_matrix(truth, prediction, labels=np.array([1, 0]))
    np.testing.assert_array_equal(matrix, np.array([[1, 0], [0, 1]]))
    with pytest.raises(ValueError, match="labels"):
        confusion_matrix(truth, prediction, labels=np.array([0]))


def test_classification_score_table_has_one_row_per_label() -> None:
    frame = classification_scores(np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert list(frame["label"]) == [0, 1]
    assert list(frame.columns) == ["label", "precision", "recall", "f1", "support"]


def test_metric_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        accuracy(np.array([0, 1]), np.array([0]))
    with pytest.raises(ValueError, match="empty"):
        mean_squared_error(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="average"):
        precision(np.array([0, 1]), np.array([0, 1]), average="weighted")  # type: ignore[arg-type]


def test_split_is_a_deterministic_partition() -> None:
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    target = np.arange(10, dtype=np.float64)
    first = train_test_split(features, target, test_size=0.3, seed=1)
    second = train_test_split(features, target, test_size=0.3, seed=1)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    x_train, x_test, y_train, y_test = first
    assert x_test.shape[0] == 3
    assert x_train.shape[0] == 7
    combined = np.sort(np.concatenate([x_train[:, 0], x_test[:, 0]]))
    np.testing.assert_array_equal(combined, np.sort(features[:, 0]))
    np.testing.assert_array_equal(
        np.sort(np.concatenate([y_train, y_test])),
        np.sort(target),
    )

    other = train_test_split(features, target, test_size=0.3, seed=2)
    assert not np.array_equal(other[1], x_test)


def test_stratified_split_keeps_class_counts() -> None:
    features = np.arange(16, dtype=np.float64).reshape(8, 2)
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    _x_train, _x_test, _y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        seed=0,
        stratify=target,
    )
    assert int(np.sum(y_test == 0)) == 1
    assert int(np.sum(y_test == 1)) == 1

    rare = np.array([0, 0, 0, 0, 1])
    rare_features = np.arange(10, dtype=np.float64).reshape(5, 2)
    _x_train, _x_test, y_train, y_test = train_test_split(
        rare_features,
        rare,
        test_size=0.25,
        seed=0,
        stratify=rare,
    )
    assert 1 in y_train
    assert 1 not in y_test


def test_split_rejects_bad_fractions() -> None:
    features = np.ones((4, 1))
    target = np.arange(4, dtype=float)
    with pytest.raises(ValueError):
        train_test_split(features, target, test_size=0.0)
    with pytest.raises(ValueError):
        train_test_split(features, target, test_size=1.0)
    with pytest.raises(ValueError, match="stratify"):
        train_test_split(features, target, stratify=np.array([0, 1]))
