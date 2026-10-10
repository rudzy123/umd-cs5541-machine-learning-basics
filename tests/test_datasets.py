"""Synthetic generators stay local, labeled, and seeded."""

from __future__ import annotations

import numpy as np
import pytest

from umd_ml.datasets import make_classification, make_regression, make_sine_regression


def test_regression_is_labeled_synthetic_and_obeys_its_equation() -> None:
    coefficients = np.array([1.5, -2.0])
    data = make_regression(
        25,
        2,
        coefficients=coefficients,
        intercept=0.25,
        noise=0.0,
        seed=4,
    )
    assert data.source == "synthetic"
    assert "synthetic" in data.description.lower()
    assert "real people" in data.description
    np.testing.assert_allclose(data.y, data.X @ coefficients + 0.25)
    assert data.to_frame().attrs["source"] == "synthetic"
    assert list(data.to_frame().columns) == ["x1", "x2", "target"]


def test_generators_repeat_for_a_fixed_seed_and_change_otherwise() -> None:
    first = make_regression(15, 2, noise=0.1, seed=7)
    second = make_regression(15, 2, noise=0.1, seed=7)
    third = make_regression(15, 2, noise=0.1, seed=8)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    assert not np.array_equal(first.X, third.X)

    blobs = make_classification(20, 2, seed=3)
    repeat = make_classification(20, 2, seed=3)
    np.testing.assert_array_equal(blobs.X, repeat.X)
    np.testing.assert_array_equal(blobs.y, repeat.y)
    assert blobs.source == "synthetic"
    assert "synthetic" in blobs.description.lower()


def test_classification_labels_and_counts() -> None:
    data = make_classification(11, 2, n_classes=3, seed=1)
    assert set(np.unique(data.y).tolist()) == {0, 1, 2}
    counts = data.label_counts()
    assert int(counts.sum()) == 11
    with pytest.raises(ValueError, match="classification"):
        make_regression(4, 1, seed=0).label_counts()


def test_sine_table_matches_the_generating_curve() -> None:
    data = make_sine_regression(30, noise=0.0, seed=2)
    np.testing.assert_allclose(data.y, np.sin(data.X[:, 0]))
    assert data.source == "synthetic"


def test_bad_generator_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        make_classification(4, 2, n_classes=1)
    with pytest.raises(ValueError):
        make_regression(5, 2, coefficients=np.array([1.0]))
    with pytest.raises(ValueError):
        make_regression(5, 1, noise=-0.1)
