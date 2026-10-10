"""Short training run used by continuous integration.

The smoke test fits each educational model on a small synthetic table and
refuses to pass when the loss does not fall or the classifier misses an easy
separated problem. It does not download data and it does not write figures.
"""

from __future__ import annotations

import numpy as np

from umd_ml.datasets import make_classification, make_regression
from umd_ml.evaluation import accuracy, train_test_split
from umd_ml.knn import KNeighborsClassifier
from umd_ml.linear_regression import LinearRegression
from umd_ml.neural_network import NeuralNetwork
from umd_ml.preprocessing import standardize_splits


def main() -> None:
    """Train linear regression, KNN, and a small network on synthetic rows."""
    _check_linear_regression()
    _check_knn()
    _check_network()
    print("Training smoke test passed on synthetic data.")


def _check_linear_regression() -> None:
    data = make_regression(
        40,
        1,
        coefficients=np.array([2.0]),
        intercept=1.0,
        noise=0.0,
        seed=0,
    )
    model = LinearRegression(learning_rate=0.1, n_iterations=400).fit(data.X, data.y)
    if not model.loss_history_[-1] < model.loss_history_[0]:
        raise RuntimeError("Linear regression loss did not decrease on synthetic data.")
    if model.coef_ is None or abs(float(model.coef_[0]) - 2.0) > 1e-3:
        raise RuntimeError(f"Linear regression did not recover slope 2; got {model.coef_}.")
    if abs(model.intercept_ - 1.0) > 1e-3:
        raise RuntimeError(
            f"Linear regression did not recover intercept 1; got {model.intercept_}."
        )
    print(
        f"synthetic linear regression final loss={model.loss_history_[-1]:.6g} "
        f"slope={float(model.coef_[0]):.6g}"
    )


def _check_knn() -> None:
    data = make_classification(80, 2, separation=3.0, noise=0.3, seed=0)
    x_train, x_test, y_train, y_test = train_test_split(
        data.X,
        data.y,
        test_size=0.25,
        seed=0,
        stratify=data.y,
    )
    model = KNeighborsClassifier(k=3).fit(x_train, y_train)
    score = accuracy(y_test, model.predict(x_test))
    if score < 0.9:
        raise RuntimeError(f"KNN held-out accuracy on synthetic blobs was {score:.3f}.")
    print(f"synthetic knn held-out accuracy={score:.3f}")


def _check_network() -> None:
    data = make_regression(
        50,
        2,
        coefficients=np.array([0.5, -0.25]),
        intercept=0.0,
        noise=0.0,
        seed=0,
    )
    _scaler, (x_train,) = standardize_splits(data.X)
    model = NeuralNetwork(
        n_inputs=2,
        n_hidden=8,
        hidden_activation="relu",
        learning_rate=0.05,
        n_iterations=250,
        seed=0,
    )
    model.fit(x_train, data.y)
    initial = float(model.loss_history_[0])
    final = float(model.loss_history_[-1])
    if not final < initial:
        raise RuntimeError(f"Network loss did not decrease: initial={initial}, final={final}.")
    print(f"synthetic network training loss {initial:.6g} -> {final:.6g}")


if __name__ == "__main__":
    main()
