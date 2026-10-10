"""Forward pass, backprop, descent, and determinism for the small network."""

from __future__ import annotations

import numpy as np
import pytest

from umd_ml.datasets import make_classification
from umd_ml.evaluation import accuracy
from umd_ml.neural_network import NeuralNetwork


def _hand_network() -> NeuralNetwork:
    network = NeuralNetwork(n_inputs=1, n_hidden=2, hidden_activation="relu", seed=0)
    network.W1 = np.array([[1.0, -2.0]], dtype=np.float64)
    network.b1 = np.array([0.0, 1.0], dtype=np.float64)
    network.W2 = np.array([[0.5], [4.0]], dtype=np.float64)
    network.b2 = np.array([-1.0], dtype=np.float64)
    return network


def test_hand_worked_forward_and_backprop() -> None:
    # x = 2, z1 = [2, -3], ReLU -> [2, 0], linear output = 0.
    # y = 1, so the regression delta at the output is 2 * (0 - 1) = -2.
    network = _hand_network()
    features = np.array([[2.0]])
    assert network.forward(features)[0] == pytest.approx(0.0)
    loss, grads = network.loss_and_gradients(features, np.array([1.0]))
    assert loss == pytest.approx(1.0)
    np.testing.assert_allclose(grads["W1"], np.array([[-2.0, 0.0]]))
    np.testing.assert_allclose(grads["b1"], np.array([-1.0, 0.0]))
    np.testing.assert_allclose(grads["W2"], np.array([[-4.0], [0.0]]))
    np.testing.assert_allclose(grads["b2"], np.array([-2.0]))


def test_classification_threshold_is_deterministic_at_one_half() -> None:
    network = NeuralNetwork(
        n_inputs=1,
        n_hidden=2,
        hidden_activation="relu",
        task="classification",
        seed=0,
    )
    network.W1 = np.zeros((1, 2))
    network.b1 = np.zeros(2)
    network.W2 = np.zeros((2, 1))
    network.b2 = np.array([0.0])
    assert int(network.predict(np.array([[0.0]]))[0]) == 1
    network.b2 = np.array([-1.0])
    assert int(network.predict(np.array([[0.0]]))[0]) == 0


def test_training_loss_decreases_for_relu_and_sigmoid() -> None:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(40, 2))
    target = 0.5 * features[:, 0] - 0.25 * features[:, 1]
    for activation in ("relu", "sigmoid"):
        network = NeuralNetwork(
            n_inputs=2,
            n_hidden=8,
            hidden_activation=activation,  # type: ignore[arg-type]
            learning_rate=0.05,
            n_iterations=300,
            seed=0,
        )
        network.fit(features, target)
        history = network.loss_history_
        assert history[-1] < history[0], activation
        assert np.all(np.isfinite(history))


def test_binary_network_learns_separated_synthetic_blobs() -> None:
    data = make_classification(80, 2, separation=2.5, noise=0.3, seed=0)
    network = NeuralNetwork(
        n_inputs=2,
        n_hidden=8,
        hidden_activation="relu",
        task="classification",
        learning_rate=0.1,
        n_iterations=400,
        seed=0,
    )
    network.fit(data.X, data.y)
    assert network.loss_history_[-1] < network.loss_history_[0]
    assert accuracy(data.y, network.predict(data.X)) > 0.9


def test_fit_is_deterministic_for_a_fixed_seed() -> None:
    features = np.array([[0.0, 1.0], [1.0, -1.0], [0.5, 0.25], [-0.5, 0.5]])
    target = np.array([0.1, -0.2, 0.0, 0.4])
    first = NeuralNetwork(n_inputs=2, n_hidden=4, n_iterations=20, seed=7).fit(features, target)
    second = NeuralNetwork(n_inputs=2, n_hidden=4, n_iterations=20, seed=7).fit(features, target)
    np.testing.assert_array_equal(first.W1, second.W1)
    np.testing.assert_array_equal(first.W2, second.W2)
    np.testing.assert_array_equal(first.loss_history_, second.loss_history_)
    first.fit(features, target)
    np.testing.assert_array_equal(first.W1, second.W1)

    other = NeuralNetwork(n_inputs=2, n_hidden=4, seed=8)
    same = NeuralNetwork(n_inputs=2, n_hidden=4, seed=7)
    assert not np.array_equal(other.W1, same.W1)


def test_input_shapes_and_targets_are_rejected() -> None:
    network = NeuralNetwork(n_inputs=3, n_hidden=4, seed=0)
    with pytest.raises(RuntimeError, match="Fit the model"):
        _ = network.loss_history_
    with pytest.raises(ValueError, match="features"):
        network.fit(np.ones((5, 2)), np.ones(5))
    with pytest.raises(ValueError, match="match"):
        network.fit(np.ones((5, 3)), np.ones(4))
    classifier = NeuralNetwork(n_inputs=1, n_hidden=2, task="classification", seed=0)
    with pytest.raises(ValueError, match="0 or 1"):
        classifier.fit(np.ones((3, 1)), np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="relu"):
        NeuralNetwork(n_inputs=1, n_hidden=2, hidden_activation="tanh")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NeuralNetwork(n_inputs=1, n_hidden=0)
