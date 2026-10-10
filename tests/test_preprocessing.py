"""Standardization learns training statistics and leaves the test rows out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from umd_ml.preprocessing import StandardScaler, standardize_frames, standardize_splits


def test_scaler_uses_training_mean_and_population_scale() -> None:
    train = np.array([[1.0], [3.0]])
    test = np.array([[7.0], [9.0]])
    scaler, (train_transformed, test_transformed) = standardize_splits(train, test)
    assert scaler.mean_ is not None
    assert scaler.mean_[0] == pytest.approx(2.0)
    assert scaler.scale_ is not None
    assert scaler.scale_[0] == pytest.approx(1.0)
    np.testing.assert_allclose(train_transformed, np.array([[-1.0], [1.0]]))
    np.testing.assert_allclose(test_transformed, np.array([[5.0], [7.0]]))

    leaked = StandardScaler().fit(np.vstack([train, test]))
    assert leaked.mean_ is not None
    assert leaked.mean_[0] == pytest.approx(5.0)
    assert scaler.mean_[0] != pytest.approx(float(leaked.mean_[0]))


def test_constant_column_is_centered_without_a_divide_by_zero() -> None:
    transformed = StandardScaler().fit_transform(np.array([[5.0, 1.0], [5.0, 3.0]]))
    np.testing.assert_allclose(transformed[:, 0], np.array([0.0, 0.0]))
    assert np.all(np.isfinite(transformed))


def test_transform_before_fit_and_width_mismatches_fail() -> None:
    scaler = StandardScaler()
    with pytest.raises(RuntimeError, match="Fit the scaler"):
        scaler.transform(np.ones((2, 1)))
    scaler.fit(np.ones((2, 2)))
    with pytest.raises(ValueError, match="features"):
        scaler.transform(np.ones((2, 3)))


def test_frames_are_transformed_from_training_statistics_only() -> None:
    train = pd.DataFrame({"x1": [1.0, 3.0], "label": [0, 1]})
    test = pd.DataFrame({"x1": [7.0, 9.0], "label": [0, 1]})
    train_snapshot = train.copy()
    test_snapshot = test.copy()
    train_out, test_out, scaler = standardize_frames(train, test, ["x1"])
    pd.testing.assert_frame_equal(train, train_snapshot)
    pd.testing.assert_frame_equal(test, test_snapshot)
    assert scaler.mean_ is not None
    assert scaler.mean_[0] == pytest.approx(2.0)
    np.testing.assert_allclose(train_out["x1"].to_numpy(), np.array([-1.0, 1.0]))
    np.testing.assert_allclose(test_out["x1"].to_numpy(), np.array([5.0, 7.0]))
    assert train_out["label"].tolist() == [0, 1]


def test_frame_helper_rejects_unknown_columns() -> None:
    train = pd.DataFrame({"x1": [1.0, 2.0]})
    test = pd.DataFrame({"x2": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Missing"):
        standardize_frames(train, test, ["x1"])
