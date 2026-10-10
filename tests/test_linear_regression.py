"""Known coefficients, descent, and shape checks for linear regression."""

from __future__ import annotations

import numpy as np
import pytest

from umd_ml.linear_regression import LinearRegression, mse_loss_and_gradients


def test_gradient_at_zero_matches_the_hand_derivative() -> None:
    features = np.array([[1.0], [3.0]])
    target = np.array([1.0, 1.0])
    loss, grad_coef, grad_intercept = mse_loss_and_gradients(
        features,
        target,
        np.array([0.0]),
        0.0,
    )
    assert loss == pytest.approx(1.0)
    assert grad_coef[0] == pytest.approx(-4.0)
    assert grad_intercept == pytest.approx(-2.0)


def test_one_gradient_step_applies_that_derivative() -> None:
    features = np.array([[1.0], [3.0]])
    target = np.array([1.0, 1.0])
    model = LinearRegression(learning_rate=0.1, n_iterations=1).fit(features, target)
    assert model.coef_ is not None
    assert model.coef_[0] == 0.0 - 0.1 * (-4.0)
    assert model.intercept_ == 0.0 - 0.1 * (-2.0)
    assert model.loss_history_[0] == pytest.approx(1.0)
    assert model.loss_history_[1] < model.loss_history_[0]
    assert model.n_iterations_ == 1
    assert model.loss_history_.shape == (2,)


def test_simple_regression_recovers_known_line() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    target = 2.0 * features[:, 0] + 3.0
    model = LinearRegression(learning_rate=0.05, n_iterations=4000).fit(features, target)
    assert model.coef_ is not None
    np.testing.assert_allclose(model.coef_, np.array([2.0]), atol=1e-6)
    assert model.intercept_ == pytest.approx(3.0, abs=1e-6)
    np.testing.assert_allclose(model.predict(features), target, atol=1e-6)


def test_multivariable_regression_recovers_known_coefficients() -> None:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(60, 3))
    coef = np.array([1.5, -2.0, 0.5])
    intercept = 0.25
    target = features @ coef + intercept
    model = LinearRegression(learning_rate=0.05, n_iterations=4000).fit(features, target)
    assert model.coef_ is not None
    np.testing.assert_allclose(model.coef_, coef, atol=1e-5)
    assert model.intercept_ == pytest.approx(intercept, abs=1e-5)


def test_gradient_is_zero_at_the_noise_free_solution() -> None:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(30, 3))
    coef = np.array([1.5, -2.0, 0.75])
    intercept = -0.3
    target = features @ coef + intercept
    loss, grad_coef, grad_intercept = mse_loss_and_gradients(features, target, coef, intercept)
    assert loss == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(grad_coef, np.zeros(3), atol=1e-8)
    assert grad_intercept == pytest.approx(0.0, abs=1e-8)


def test_loss_decreases_monotonically_for_a_moderate_learning_rate() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    target = 2.0 * features[:, 0] + 3.0
    model = LinearRegression(learning_rate=0.05, n_iterations=200).fit(features, target)
    history = model.loss_history_
    assert history[-1] < history[0]
    assert np.all(np.diff(history) <= 1e-8)


def test_fit_without_intercept_stays_through_the_origin() -> None:
    features = np.array([[1.0], [2.0], [3.0]])
    target = np.array([7.0, 9.0, 11.0])
    model = LinearRegression(learning_rate=0.05, n_iterations=2000, fit_intercept=False)
    model.fit(features, target)
    assert model.intercept_ == 0.0
    assert model.coef_ is not None
    assert model.coef_[0] == pytest.approx(58.0 / 14.0, abs=1e-4)


def test_l2_penalty_shrinks_coefficients() -> None:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(40, 2))
    target = features @ np.array([3.0, -1.0]) + 0.5
    plain = LinearRegression(learning_rate=0.1, n_iterations=2000, l2=0.0).fit(features, target)
    regularized = LinearRegression(learning_rate=0.1, n_iterations=2000, l2=0.5).fit(
        features,
        target,
    )
    assert plain.coef_ is not None and regularized.coef_ is not None
    assert np.linalg.norm(regularized.coef_) < np.linalg.norm(plain.coef_)


def test_repeated_fits_match_exactly() -> None:
    features = np.array([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5], [-1.0, 0.25]])
    target = np.array([1.0, 0.0, 2.0, -0.5])
    first = LinearRegression(learning_rate=0.1, n_iterations=25).fit(features, target)
    second = LinearRegression(learning_rate=0.1, n_iterations=25).fit(features, target)
    assert first.coef_ is not None and second.coef_ is not None
    np.testing.assert_array_equal(first.coef_, second.coef_)
    assert first.intercept_ == second.intercept_
    np.testing.assert_array_equal(first.loss_history_, second.loss_history_)


def test_input_shapes_are_rejected() -> None:
    model = LinearRegression(n_iterations=5)
    with pytest.raises(RuntimeError, match="Fit the model"):
        model.predict(np.ones((2, 1)))
    with pytest.raises(RuntimeError, match="Fit the model"):
        _ = model.loss_history_
    with pytest.raises(ValueError, match="n_samples"):
        model.fit(np.ones((2, 2, 1)), np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        model.fit(np.array([[1.0], [np.nan]]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="match"):
        model.fit(np.ones((3, 1)), np.ones(2))

    model.fit(np.ones((4, 2)), np.ones(4))
    with pytest.raises(ValueError, match="features"):
        model.predict(np.ones((3, 3)))


def test_invalid_hyperparameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        LinearRegression(learning_rate=0.0)
    with pytest.raises(TypeError):
        LinearRegression(n_iterations=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LinearRegression(l2=-0.1)
