"""A fully connected network with one hidden layer.

Architecture, for one row:

    z1 = x @ W1 + b1
    a1 = relu(z1) or sigmoid(z1)
    z2 = a1 @ W2 + b2
    output = z2                  for regression (mean squared error)
    output = sigmoid(z2)         for binary classification (cross-entropy)

``fit`` restarts from a Generator seeded by ``seed`` and then runs full-batch
gradient descent. The same seed and the same row order reproduce the same
weights. Biases start at zero. Hidden weights use He scale for ReLU and
Xavier scale for sigmoid.

Binary targets must be 0 or 1. A probability of exactly 0.5 is classified as
class 1. This network does not implement softmax or more than one hidden layer.
"""

from __future__ import annotations

from typing import Literal, Self

import numpy as np

from umd_ml.validation import (
    FloatMatrix,
    FloatVector,
    ensure_finite_loss,
    require_finite_matrix,
    require_learning_rate,
    require_positive_int,
    require_seed,
    require_target,
    require_tol,
)

Activation = Literal["relu", "sigmoid"]
Task = Literal["regression", "classification"]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    positive = values >= 0.0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


class NeuralNetwork:
    """One-hidden-layer network trained by batch gradient descent."""

    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        *,
        hidden_activation: Activation = "relu",
        learning_rate: float = 0.05,
        n_iterations: int = 400,
        seed: int = 0,
        task: Task = "regression",
        tol: float | None = None,
    ) -> None:
        self.n_inputs = require_positive_int(n_inputs, name="n_inputs")
        self.n_hidden = require_positive_int(n_hidden, name="n_hidden")
        if hidden_activation not in ("relu", "sigmoid"):
            raise ValueError("hidden_activation must be 'relu' or 'sigmoid'.")
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'.")
        self.hidden_activation: Activation = hidden_activation
        self.task: Task = task
        self.learning_rate = require_learning_rate(learning_rate)
        self.n_iterations = require_positive_int(n_iterations, name="n_iterations")
        self.seed = require_seed(seed)
        self.tol = require_tol(tol)
        self._loss_history: FloatVector | None = None
        self.n_iterations_: int | None = None
        self._initialize()

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Restart from the seeded initialization and descend on the full batch."""
        features, target = self._prepare(X, y)
        self._initialize()
        history: list[float] = []
        loss, grads = self._loss_and_gradients(features, target)
        ensure_finite_loss(loss)
        history.append(loss)
        updates = 0
        for _ in range(self.n_iterations):
            self._apply(grads)
            updates += 1
            loss, grads = self._loss_and_gradients(features, target)
            ensure_finite_loss(loss)
            history.append(loss)
            if self.tol is not None and abs(history[-2] - history[-1]) < self.tol:
                break
        self._loss_history = np.asarray(history, dtype=np.float64)
        self.n_iterations_ = updates
        return self

    def forward(self, X: np.ndarray) -> FloatVector:
        """Network output for each row, before a classification threshold."""
        features = self._check_features(X)
        return self._forward(features)["output"].reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Regression output, or class labels using a 0.5 threshold."""
        values = self.forward(X)
        if self.task == "classification":
            return (values >= 0.5).astype(np.int64)
        return values

    def loss_and_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, dict[str, FloatMatrix]]:
        """Loss and analytic parameter gradients at the current weights."""
        features, target = self._prepare(X, y)
        return self._loss_and_gradients(features, target)

    def parameters(self) -> dict[str, FloatMatrix]:
        """Copies of ``W1``, ``b1``, ``W2``, and ``b2``."""
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    @property
    def n_parameters(self) -> int:
        """Number of scalar weights and biases."""
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    @property
    def loss_history_(self) -> FloatVector:
        """Loss before the first update, then the loss after each update."""
        if self._loss_history is None:
            raise RuntimeError("Fit the model before reading the loss history.")
        return self._loss_history.copy()

    def _initialize(self) -> None:
        rng = np.random.default_rng(self.seed)
        if self.hidden_activation == "relu":
            hidden_scale = np.sqrt(2.0 / self.n_inputs)
        else:
            hidden_scale = np.sqrt(1.0 / self.n_inputs)
        self.W1 = rng.normal(0.0, hidden_scale, size=(self.n_inputs, self.n_hidden))
        self.b1 = np.zeros(self.n_hidden, dtype=np.float64)
        output_scale = np.sqrt(1.0 / self.n_hidden)
        self.W2 = rng.normal(0.0, output_scale, size=(self.n_hidden, 1))
        self.b2 = np.zeros(1, dtype=np.float64)

    def _prepare(self, X: np.ndarray, y: np.ndarray) -> tuple[FloatMatrix, FloatMatrix]:
        features = self._check_features(X)
        target = require_target(y, n_samples=features.shape[0])
        if self.task == "classification" and not np.isin(target, [0.0, 1.0]).all():
            raise ValueError("Binary classification targets must be 0 or 1.")
        return features, target.reshape(-1, 1)

    def _check_features(self, X: np.ndarray) -> FloatMatrix:
        features = require_finite_matrix(X, name="X")
        if features.shape[1] != self.n_inputs:
            raise ValueError(
                f"X has {features.shape[1]} features, but the network expects {self.n_inputs}."
            )
        return features

    def _forward(self, features: FloatMatrix) -> dict[str, FloatMatrix]:
        z1 = features @ self.W1 + self.b1
        if self.hidden_activation == "relu":
            hidden = np.maximum(z1, 0.0)
        else:
            hidden = _sigmoid(z1)
        z2 = hidden @ self.W2 + self.b2
        output = _sigmoid(z2) if self.task == "classification" else z2
        return {"z1": z1, "a1": hidden, "z2": z2, "output": output}

    def _loss_and_gradients(
        self,
        features: FloatMatrix,
        target: FloatMatrix,
    ) -> tuple[float, dict[str, FloatMatrix]]:
        cache = self._forward(features)
        output = cache["output"]
        n_samples = features.shape[0]
        if self.task == "classification":
            clipped = np.clip(output, 1e-12, 1.0 - 1e-12)
            loss = float(
                -np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))
            )
            delta2 = (output - target) / n_samples
        else:
            error = output - target
            loss = float(np.mean(error**2))
            delta2 = (2.0 / n_samples) * error

        dW2 = cache["a1"].T @ delta2
        db2 = np.sum(delta2, axis=0)
        da1 = delta2 @ self.W2.T
        if self.hidden_activation == "relu":
            dz1 = da1 * (cache["z1"] > 0.0)
        else:
            hidden = cache["a1"]
            dz1 = da1 * hidden * (1.0 - hidden)
        grads = {
            "W1": features.T @ dz1,
            "b1": np.sum(dz1, axis=0),
            "W2": dW2,
            "b2": db2,
        }
        return loss, grads

    def _apply(self, grads: dict[str, FloatMatrix]) -> None:
        step = self.learning_rate
        self.W1 -= step * grads["W1"]
        self.b1 -= step * grads["b1"]
        self.W2 -= step * grads["W2"]
        self.b2 -= step * grads["b2"]

    def __repr__(self) -> str:
        return (
            "NeuralNetwork("
            f"n_inputs={self.n_inputs}, n_hidden={self.n_hidden}, "
            f"hidden_activation={self.hidden_activation!r}, task={self.task!r}, "
            f"seed={self.seed})"
        )
