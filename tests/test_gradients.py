"""Finite-difference checks for selected linear and network parameters."""

from __future__ import annotations

import numpy as np

from umd_ml.gradients import (
    finite_difference_gradient,
    finite_difference_scalar,
    relative_gradient_error,
)
from umd_ml.linear_regression import LinearRegression, mse_loss_and_gradients
from umd_ml.neural_network import NeuralNetwork


def _assert_close(analytic: np.ndarray, numeric: np.ndarray, *, tolerance: float = 1e-5) -> None:
    error = relative_gradient_error(analytic, numeric)
    assert error < tolerance, error


def test_linear_regression_gradients_match_finite_differences() -> None:
    features = np.array([[1.0], [3.0], [2.0], [0.5]])
    target = np.array([1.0, 1.5, 0.0, 0.25])
    model = LinearRegression(learning_rate=0.05, n_iterations=3).fit(features, target)
    assert model.coef_ is not None
    original_coef = model.coef_.copy()
    original_intercept = model.intercept_

    def loss() -> float:
        assert model.coef_ is not None
        value, _, _ = mse_loss_and_gradients(features, target, model.coef_, model.intercept_)
        return value

    numeric_coef = finite_difference_gradient(model.coef_, loss)
    analytic_coef, analytic_intercept = model.compute_gradient(features, target)
    _assert_close(analytic_coef, numeric_coef)
    numeric_intercept = finite_difference_scalar(
        lambda: model.intercept_,
        lambda value: setattr(model, "intercept_", value),
        loss,
    )
    assert (
        relative_gradient_error(np.array([analytic_intercept]), np.array([numeric_intercept]))
        < 1e-5
    )
    np.testing.assert_array_equal(model.coef_, original_coef)
    assert model.intercept_ == original_intercept


def test_relu_network_parameter_gradients_match_finite_differences() -> None:
    network = NeuralNetwork(n_inputs=1, n_hidden=2, hidden_activation="relu", seed=0)
    network.W1 = np.array([[1.0, -2.0]], dtype=np.float64)
    network.b1 = np.array([0.0, 1.0], dtype=np.float64)
    network.W2 = np.array([[0.5], [4.0]], dtype=np.float64)
    network.b2 = np.array([-1.0], dtype=np.float64)
    features = np.array([[2.0], [1.0], [-1.0]])
    target = np.array([1.0, 0.0, 0.5])
    original = network.parameters()
    _, analytic = network.loss_and_gradients(features, target)

    def loss() -> float:
        value, _grads = network.loss_and_gradients(features, target)
        return value

    for name in ("W1", "b1", "W2", "b2"):
        numeric = finite_difference_gradient(getattr(network, name), loss)
        _assert_close(analytic[name], numeric)
    restored = network.parameters()
    for name, values in original.items():
        np.testing.assert_array_equal(restored[name], values)


def test_classification_output_layer_matches_finite_differences() -> None:
    network = NeuralNetwork(
        n_inputs=2,
        n_hidden=3,
        hidden_activation="sigmoid",
        task="classification",
        seed=1,
    )
    features = np.array([[0.2, -0.4], [1.0, 0.3], [-0.6, 0.8], [0.0, 0.1]])
    target = np.array([0.0, 1.0, 1.0, 0.0])
    _, analytic = network.loss_and_gradients(features, target)

    def loss() -> float:
        value, _grads = network.loss_and_gradients(features, target)
        return value

    for name in ("W2", "b2"):
        numeric = finite_difference_gradient(getattr(network, name), loss)
        _assert_close(analytic[name], numeric, tolerance=1e-5)
