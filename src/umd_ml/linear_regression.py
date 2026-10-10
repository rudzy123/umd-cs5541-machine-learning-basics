"""Simple and multivariable linear regression fit by batch gradient descent.

The model predicts ``y_hat = X @ coef + intercept``. Simple regression is the
one-column case of the same update. The training objective is

    loss = mean((y_hat - y) ** 2) + l2 * sum(coef ** 2)

The intercept is not penalized. Differentiating the mean of squares produces
the factor of 2 in the gradient. Writing the loss as ``1 / (2 n)`` times the
sum of squares would cancel that 2 and only rescale a learning rate.

``loss_history_`` is the convergence history. Index 0 is the loss at the zero
vector, before any update. Each later entry is the loss after that update.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from umd_ml.evaluation import mean_squared_error
from umd_ml.validation import (
    FloatMatrix,
    FloatVector,
    ensure_finite_loss,
    require_finite_matrix,
    require_learning_rate,
    require_nonnegative_float,
    require_positive_int,
    require_target,
    require_tol,
)


def mse_loss_and_gradients(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    *,
    l2: float = 0.0,
    fit_intercept: bool = True,
) -> tuple[float, FloatVector, float]:
    """Return ``(loss, gradient_coef, gradient_intercept)`` at the given parameters.

    ``gradient_intercept`` is 0 when ``fit_intercept`` is false. The penalty
    gradient ``2 * l2 * coef`` is added only to the slope.
    """
    predictions = X @ coef + intercept
    error = predictions - y
    n_samples = y.shape[0]
    grad_coef = (2.0 / n_samples) * (X.T @ error) + 2.0 * l2 * coef
    grad_intercept = (2.0 / n_samples) * float(np.sum(error)) if fit_intercept else 0.0
    loss = float(np.mean(error**2) + l2 * np.dot(coef, coef))
    return loss, grad_coef, grad_intercept


class LinearRegression:
    """Ordinary least squares approximated by full-batch gradient descent.

    Parameters start at zero, so a fit is deterministic without a seed. Call
    ``fit`` again to restart from zero and repeat the same trajectory.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        l2: float = 0.0,
        tol: float | None = None,
    ) -> None:
        self.learning_rate = require_learning_rate(learning_rate)
        self.n_iterations = require_positive_int(n_iterations, name="n_iterations")
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a bool.")
        self.fit_intercept = fit_intercept
        self.l2 = require_nonnegative_float(l2, name="l2")
        self.tol = require_tol(tol)
        self.coef_: FloatVector | None = None
        self.intercept_: float = 0.0
        self._loss_history: FloatVector | None = None
        self.n_iterations_: int | None = None
        self._n_features: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Minimize the training objective and store the convergence history."""
        features = require_finite_matrix(X, name="X")
        target = require_target(y, n_samples=features.shape[0])
        self.coef_ = np.zeros(features.shape[1], dtype=np.float64)
        self.intercept_ = 0.0
        self._n_features = int(features.shape[1])

        history: list[float] = []
        loss, grad_coef, grad_intercept = self._objective(features, target)
        ensure_finite_loss(loss)
        history.append(loss)
        updates = 0
        for _ in range(self.n_iterations):
            self.coef_ = self.coef_ - self.learning_rate * grad_coef
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * grad_intercept
            updates += 1
            loss, grad_coef, grad_intercept = self._objective(features, target)
            ensure_finite_loss(loss)
            history.append(loss)
            if self.tol is not None and abs(history[-2] - history[-1]) < self.tol:
                break

        self._loss_history = np.asarray(history, dtype=np.float64)
        self.n_iterations_ = updates
        return self

    def predict(self, X: np.ndarray) -> FloatVector:
        """Return ``X @ coef + intercept`` for a fitted model."""
        coef, intercept, n_features = self._fitted_parameters()
        features = require_finite_matrix(X, name="X")
        if features.shape[1] != n_features:
            raise ValueError(
                f"X has {features.shape[1]} features, but the model was fit on {n_features}."
            )
        return features @ coef + intercept

    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error between a target and a prediction."""
        return mean_squared_error(y_true, y_pred)

    def compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple[FloatVector, float]:
        """Gradient of the training objective at the current parameters."""
        features = require_finite_matrix(X, name="X")
        target = require_target(y, n_samples=features.shape[0])
        coef, intercept, n_features = self._fitted_parameters()
        if features.shape[1] != n_features:
            raise ValueError(
                f"X has {features.shape[1]} features, but the model was fit on {n_features}."
            )
        _loss, grad_coef, grad_intercept = mse_loss_and_gradients(
            features,
            target,
            coef,
            intercept,
            l2=self.l2,
            fit_intercept=self.fit_intercept,
        )
        return grad_coef, grad_intercept

    @property
    def loss_history_(self) -> FloatVector:
        """Convergence history. Available after ``fit``."""
        if self._loss_history is None:
            raise RuntimeError("Fit the model before reading the loss history.")
        return self._loss_history.copy()

    def _objective(
        self,
        features: FloatMatrix,
        target: FloatVector,
    ) -> tuple[float, FloatVector, float]:
        if self.coef_ is None:
            raise RuntimeError("Fit the model before evaluating the objective.")
        return mse_loss_and_gradients(
            features,
            target,
            self.coef_,
            self.intercept_,
            l2=self.l2,
            fit_intercept=self.fit_intercept,
        )

    def _fitted_parameters(self) -> tuple[FloatVector, float, int]:
        if self.coef_ is None or self._n_features is None:
            raise RuntimeError("Fit the model before using its parameters.")
        return self.coef_, self.intercept_, self._n_features

    def __repr__(self) -> str:
        return (
            "LinearRegression("
            f"learning_rate={self.learning_rate}, n_iterations={self.n_iterations}, "
            f"fit_intercept={self.fit_intercept}, l2={self.l2}, fitted={self.coef_ is not None})"
        )
