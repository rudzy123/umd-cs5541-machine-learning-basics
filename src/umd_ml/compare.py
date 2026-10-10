"""Compare the educational estimators with scikit-learn on synthetic rows.

The linear comparison isolates the optimizer. Both models minimize the same
unpenalized mean squared error; gradient descent only approaches the
least-squares solution that scikit-learn computes directly. The neighbor
comparison uses one shared train/test split and records tie cases separately
so a match is not confused with an identical tie policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier

from umd_ml.datasets import make_classification, make_regression
from umd_ml.evaluation import accuracy, mean_squared_error, train_test_split
from umd_ml.knn import KNeighborsClassifier, euclidean_distances, majority_vote
from umd_ml.linear_regression import LinearRegression
from umd_ml.validation import require_seed

_LEARNING_RATE = 0.1
_ITERATION_SCHEDULE = (5, 2000)
_GENERATING_COEF = np.array([1.5, -2.0, 0.75])
_GENERATING_INTERCEPT = -0.3

DIFFERENCE_EXPLANATION = """\
### Linear regression

The educational model minimizes mean squared error with batch gradient descent,
starting from the zero vector. scikit-learn's `LinearRegression` minimizes the
same unpenalized objective by solving the normal equations. With an intercept,
scikit-learn centers the training columns before that solve. On a full-rank
design the two procedures share one minimum.

A short run stops before the minimum, so coefficients still differ. A long run
closes the gap. A residual at convergence is floating-point rounding from two
algorithms, including the order in which BLAS accumulates sums. It is not a
different definition of a line.

This comparison uses noise-free synthetic rows, so the generating coefficients
are also the least-squares solution. The gap versus the generator and the gap
versus scikit-learn are two views of the same approach to that solution.

### K-nearest neighbors

Both classifiers use Euclidean distance and a uniform vote. Predictions match
when the k nearest labels have a unique majority and the k-th distance is
unique. Two edge cases are deterministic here and only possibly different in
scikit-learn.

Equal distances: this repository keeps the earlier training row. `argsort` with
a stable mergesort preserves the original index order. scikit-learn's neighbor
search (`brute`, `kd_tree`, or `ball_tree`) can return equal-distance neighbors
in another order and change the vote.

Equal votes: this repository selects the smallest label. scikit-learn resolves
that tie through its class list and its mode implementation. The tie table
records the prediction each implementation returned; it does not assume a
scikit-learn version's winner.
"""


@dataclass(frozen=True)
class ComparisonResult:
    """Measured gaps between the educational models and scikit-learn."""

    metrics: pd.DataFrame
    optimization_gaps: pd.DataFrame
    tie_examples: pd.DataFrame
    narrative: str
    seed: int
    knn_mismatches: int
    knn_test_size: int


def compare_with_sklearn(seed: int = 0) -> ComparisonResult:
    """Fit both implementations on the same locally generated rows."""
    seed = require_seed(seed)
    gaps = _optimization_gaps(seed)
    knn_frame, mismatches, n_test = _knn_comparison(seed)
    ties = _tie_examples()
    metrics = pd.concat([_regression_metric_rows(gaps), knn_frame], ignore_index=True)
    narrative = _narrative(seed, gaps, mismatches, n_test)
    return ComparisonResult(
        metrics=metrics,
        optimization_gaps=gaps,
        tie_examples=ties,
        narrative=narrative,
        seed=seed,
        knn_mismatches=mismatches,
        knn_test_size=n_test,
    )


def render_comparison(result: ComparisonResult) -> str:
    """Markdown report for one comparison run."""
    sections = [
        "# Educational models and scikit-learn",
        "",
        "All rows in this report are synthetic. They were generated locally from a NumPy "
        "Generator.",
        "No dataset was downloaded.",
        "",
        result.narrative,
        "",
        "## Regression optimization gaps",
        "",
        _markdown_table(result.optimization_gaps),
        "",
        "## Held-out metrics",
        "",
        _markdown_table(result.metrics),
        "",
        "## Neighbor tie examples",
        "",
        _markdown_table(result.tie_examples),
        "",
        "## Why small differences show up",
        "",
        DIFFERENCE_EXPLANATION.strip(),
        "",
    ]
    return "\n".join(sections)


def _optimization_gaps(seed: int) -> pd.DataFrame:
    data = make_regression(
        120,
        3,
        coefficients=_GENERATING_COEF,
        intercept=_GENERATING_INTERCEPT,
        noise=0.0,
        seed=seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        data.X,
        data.y,
        test_size=0.25,
        seed=seed,
    )
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(x_train, y_train)
    sklearn_intercept = float(sklearn_model.intercept_)
    rows: list[dict[str, float | int]] = []
    for iterations in _ITERATION_SCHEDULE:
        model = LinearRegression(learning_rate=_LEARNING_RATE, n_iterations=iterations)
        model.fit(x_train, y_train)
        if model.coef_ is None:
            raise RuntimeError("Linear regression did not store coefficients.")
        rows.append(
            {
                "iterations": iterations,
                "gap_versus_sklearn": _parameter_gap(
                    model.coef_,
                    model.intercept_,
                    np.asarray(sklearn_model.coef_, dtype=np.float64),
                    sklearn_intercept,
                ),
                "gap_versus_generator": _parameter_gap(
                    model.coef_,
                    model.intercept_,
                    _GENERATING_COEF,
                    _GENERATING_INTERCEPT,
                ),
                "educational_test_mse": mean_squared_error(y_test, model.predict(x_test)),
                "sklearn_test_mse": mean_squared_error(y_test, sklearn_model.predict(x_test)),
            }
        )
    return pd.DataFrame(rows)


def _regression_metric_rows(gaps: pd.DataFrame) -> pd.DataFrame:
    longest = gaps.iloc[-1]
    return pd.DataFrame(
        [
            {
                "model": "linear_regression",
                "implementation": "educational",
                "split": "test",
                "metric": "mean_squared_error",
                "value": float(longest["educational_test_mse"]),
            },
            {
                "model": "linear_regression",
                "implementation": "sklearn",
                "split": "test",
                "metric": "mean_squared_error",
                "value": float(longest["sklearn_test_mse"]),
            },
            {
                "model": "linear_regression",
                "implementation": "educational",
                "split": "parameters",
                "metric": "max_abs_gap_versus_sklearn",
                "value": float(longest["gap_versus_sklearn"]),
            },
        ]
    )


def _knn_comparison(seed: int) -> tuple[pd.DataFrame, int, int]:
    data = make_classification(
        160,
        2,
        n_classes=2,
        separation=3.0,
        noise=0.35,
        seed=seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        data.X,
        data.y,
        test_size=0.25,
        seed=seed,
        stratify=data.y,
    )
    educational = KNeighborsClassifier(k=5).fit(x_train, y_train)
    sklearn_model = SklearnKNeighborsClassifier(n_neighbors=5, algorithm="brute")
    sklearn_model.fit(x_train, y_train)
    educational_pred = educational.predict(x_test)
    sklearn_pred = np.asarray(sklearn_model.predict(x_test))
    mismatches = int(np.sum(educational_pred != sklearn_pred))
    frame = pd.DataFrame(
        [
            {
                "model": "knn",
                "implementation": "educational",
                "split": "test",
                "metric": "accuracy",
                "value": accuracy(y_test, educational_pred),
            },
            {
                "model": "knn",
                "implementation": "sklearn",
                "split": "test",
                "metric": "accuracy",
                "value": accuracy(y_test, sklearn_pred),
            },
            {
                "model": "knn",
                "implementation": "comparison",
                "split": "test",
                "metric": "mismatched_predictions",
                "value": float(mismatches),
            },
        ]
    )
    return frame, mismatches, int(y_test.shape[0])


def _tie_examples() -> pd.DataFrame:
    query = np.array([[0.0, 0.0]], dtype=np.float64)
    cases = (
        (
            "equal_votes",
            np.array([[0.0, 1.0], [0.0, -1.0], [5.0, 5.0]], dtype=np.float64),
            np.array([0, 1, 1], dtype=np.int64),
            2,
            "Two neighbors at distance 1 have labels 0 and 1. The smallest label wins.",
        ),
        (
            "equal_distances",
            np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64),
            np.array([1, 0], dtype=np.int64),
            1,
            "Both training rows are distance 1. Stable order keeps training index 0.",
        ),
    )
    rows: list[dict[str, object]] = []
    for name, features, labels, k, note in cases:
        educational = KNeighborsClassifier(k=k).fit(features, labels)
        sklearn_model = SklearnKNeighborsClassifier(n_neighbors=k, algorithm="brute")
        sklearn_model.fit(features, labels)
        order = np.argsort(euclidean_distances(query, features)[0], kind="mergesort")[:k]
        educational_prediction = int(educational.predict(query)[0])
        vote = int(majority_vote(labels[order]))
        if educational_prediction != vote:
            raise RuntimeError("Educational KNN disagreed with its own tie policy.")
        rows.append(
            {
                "case": name,
                "neighbor_labels_in_order": ",".join(str(int(label)) for label in labels[order]),
                "educational_prediction": educational_prediction,
                "sklearn_prediction": int(sklearn_model.predict(query)[0]),
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def _narrative(seed: int, gaps: pd.DataFrame, mismatches: int, n_test: int) -> str:
    short = gaps.iloc[0]
    long = gaps.iloc[-1]
    return (
        f"Seed {seed}. On noise-free synthetic regression data, batch gradient descent "
        f"after {int(short['iterations'])} steps differed from scikit-learn's least-squares "
        f"coefficients by at most {float(short['gap_versus_sklearn']):.3e}. After "
        f"{int(long['iterations'])} steps that gap was {float(long['gap_versus_sklearn']):.3e}. "
        f"The KNN comparison on a separate synthetic classification set mismatched on "
        f"{mismatches} of {n_test} held-out rows."
    )


def _parameter_gap(
    coef_a: np.ndarray,
    intercept_a: float,
    coef_b: np.ndarray,
    intercept_b: float,
) -> float:
    coefficient_gap = float(np.max(np.abs(coef_a - coef_b)))
    return max(coefficient_gap, abs(intercept_a - intercept_b))


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{float(value):.6g}")
    header = "| " + " | ".join(str(column) for column in display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])
