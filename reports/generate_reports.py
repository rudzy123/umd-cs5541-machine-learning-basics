"""Save small synthetic-data figures and the scikit-learn comparison.

The figures are an illustration of a fit, a training-loss curve, a
two-dimensional decision boundary, and a held-out confusion matrix. They are
not a substitute for the tests. Regenerate them with:

    python reports/generate_reports.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="umd-ml-mpl-"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from umd_ml.compare import compare_with_sklearn, render_comparison
from umd_ml.datasets import make_classification, make_regression, make_sine_regression
from umd_ml.evaluation import confusion_matrix, train_test_split
from umd_ml.knn import KNeighborsClassifier
from umd_ml.linear_regression import LinearRegression
from umd_ml.neural_network import NeuralNetwork

REPORT_SEED = 5541
MAX_FIGURE_BYTES = 1_000_000
_CLASS_COLORS = {0: "#2171b5", 1: "#d94801"}


def main() -> None:
    """Fit the educational models and write the report files."""
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
        }
    )
    report_dir = Path(__file__).resolve().parent
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    regression_data, regression_model, sklearn_regression = _fit_regression()
    network = _fit_network()
    classifier, x_train, y_train, x_test, y_test = _fit_classifier()

    figures = [
        _plot_regression(regression_data, regression_model, sklearn_regression, figure_dir),
        _plot_losses(regression_model.loss_history_, network.loss_history_, figure_dir),
        _plot_boundary(classifier, x_train, y_train, x_test, y_test, figure_dir),
        _plot_confusion(y_test, classifier.predict(x_test), figure_dir),
    ]

    comparison = compare_with_sklearn(seed=REPORT_SEED)
    _expect_gap_story(comparison.optimization_gaps)
    markdown = render_comparison(comparison)
    markdown += _figure_section(figures)
    (report_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    comparison.metrics.to_csv(report_dir / "comparison_metrics.csv", index=False)
    comparison.optimization_gaps.to_csv(report_dir / "optimization_gaps.csv", index=False)
    comparison.tie_examples.to_csv(report_dir / "tie_examples.csv", index=False)
    print(f"Wrote reports to {report_dir}")


def _fit_regression():
    data = make_regression(
        60,
        1,
        coefficients=np.array([1.5]),
        intercept=-0.5,
        noise=0.35,
        seed=REPORT_SEED,
    )
    _require_synthetic(data.source, data.description)
    model = LinearRegression(learning_rate=0.1, n_iterations=800).fit(data.X, data.y)
    _require_decrease(model.loss_history_, "linear regression")
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(data.X, data.y)
    return data, model, sklearn_model


def _fit_network() -> NeuralNetwork:
    data = make_sine_regression(80, noise=0.05, seed=REPORT_SEED)
    _require_synthetic(data.source, data.description)
    network = NeuralNetwork(
        n_inputs=1,
        n_hidden=16,
        hidden_activation="relu",
        learning_rate=0.01,
        n_iterations=800,
        seed=REPORT_SEED,
    )
    network.fit(data.X, data.y)
    _require_decrease(network.loss_history_, "neural network")
    return network


def _fit_classifier():
    data = make_classification(160, 2, separation=2.2, noise=0.7, seed=REPORT_SEED)
    _require_synthetic(data.source, data.description)
    if data.X.shape[1] != 2:
        raise ValueError(
            "Decision boundaries in this report require two-dimensional synthetic data."
        )
    x_train, x_test, y_train, y_test = train_test_split(
        data.X,
        data.y,
        test_size=0.25,
        seed=REPORT_SEED,
        stratify=data.y,
    )
    model = KNeighborsClassifier(k=7).fit(x_train, y_train)
    return model, x_train, y_train, x_test, y_test


def _plot_regression(data, model, sklearn_model, figure_dir: Path) -> Path:
    grid = np.linspace(float(data.X.min()) - 0.3, float(data.X.max()) + 0.3, 200).reshape(-1, 1)
    figure, axis = plt.subplots(figsize=(6.6, 4.4))
    axis.scatter(data.X[:, 0], data.y, c="#4c4c4c", s=28, label="synthetic observations", zorder=3)
    axis.plot(
        grid[:, 0],
        model.predict(grid),
        color="#2171b5",
        lw=2.2,
        label="educational gradient descent",
    )
    axis.plot(
        grid[:, 0],
        sklearn_model.predict(grid),
        color="#d94801",
        lw=1.6,
        ls="--",
        label="scikit-learn least squares",
    )
    axis.plot(
        grid[:, 0],
        1.5 * grid[:, 0] - 0.5,
        color="#238b45",
        lw=1.3,
        ls=":",
        label="synthetic generating line",
    )
    axis.set_xlabel("x1")
    axis.set_ylabel("target")
    axis.set_title("Synthetic regression fit (training rows only)")
    axis.legend(fontsize=8)
    return _save(figure, figure_dir / "regression_fit.png")


def _plot_losses(
    regression_history: np.ndarray, network_history: np.ndarray, figure_dir: Path
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].plot(regression_history, color="#2171b5")
    axes[0].set_title("Linear regression")
    axes[0].set_xlabel("Update (0 is the zero initialization)")
    axes[0].set_ylabel("Training objective")
    axes[1].plot(network_history, color="#d94801")
    axes[1].set_title("One-hidden-layer network")
    axes[1].set_xlabel("Update (0 is the seeded initialization)")
    axes[1].set_ylabel("Training mean squared error")
    figure.suptitle("Training loss on synthetic data — not a held-out score", fontsize=12)
    figure.tight_layout()
    return _save(figure, figure_dir / "training_loss.png")


def _plot_boundary(model, x_train, y_train, x_test, y_test, figure_dir: Path) -> Path:
    if x_train.shape[1] != 2:
        raise ValueError(
            "Decision boundaries in this report require two-dimensional synthetic data."
        )
    stacked = np.vstack([x_train, x_test])
    x_min, x_max = _span(stacked[:, 0])
    y_min, y_max = _span(stacked[:, 1])
    resolution = 180
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    regions = model.predict(grid).reshape(xx.shape)
    figure, axis = plt.subplots(figsize=(6.6, 5.0))
    axis.contourf(xx, yy, regions, levels=[-0.1, 0.5, 1.1], colors=["#c6dbef", "#fdd0a2"])
    _scatter_split(axis, x_train, y_train, marker="o", prefix="train")
    _scatter_split(axis, x_test, y_test, marker="x", prefix="test")
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_title("Synthetic 2D KNN boundary (fit on training rows)")
    axis.legend(fontsize=8, loc="best")
    axis.grid(False)
    return _save(figure, figure_dir / "decision_boundary.png")


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, figure_dir: Path) -> Path:
    labels = np.array([0, 1])
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["0", "1"])
    axis.set_yticks([0, 1], ["0", "1"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Synthetic held-out confusion matrix")
    axis.grid(False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column, row, str(int(matrix[row, column])), ha="center", va="center", color="black"
            )
    figure.colorbar(image, ax=axis, label="Count")
    return _save(figure, figure_dir / "confusion_matrix.png")


def _scatter_split(
    axis, features: np.ndarray, labels: np.ndarray, *, marker: str, prefix: str
) -> None:
    for label, color in _CLASS_COLORS.items():
        mask = labels == label
        axis.scatter(
            features[mask, 0],
            features[mask, 1],
            c=color,
            marker=marker,
            s=36,
            linewidths=1.2,
            label=f"{prefix} class {label}",
            zorder=3,
        )


def _span(values: np.ndarray, pad: float = 0.4) -> tuple[float, float]:
    low = float(np.min(values)) - pad
    high = float(np.max(values)) + pad
    if high <= low:
        high = low + 1.0
    return low, high


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    size = path.stat().st_size
    if size > MAX_FIGURE_BYTES:
        raise RuntimeError(f"{path.name} is {size} bytes. Report images must stay small.")
    if size < 1_000:
        raise RuntimeError(f"{path.name} looks empty ({size} bytes).")
    return path


def _figure_section(paths: list[Path]) -> str:
    lines = [
        "",
        "## Figures",
        "",
        "Regenerate with `python reports/generate_reports.py`. Seed 5541. Synthetic data only.",
        "",
    ]
    for path in paths:
        lines.append(f"- `figures/{path.name}` ({path.stat().st_size} bytes)")
    lines.append("")
    return "\n".join(lines)


def _expect_gap_story(gaps) -> None:
    short_gap = float(gaps.iloc[0]["gap_versus_sklearn"])
    long_gap = float(gaps.iloc[-1]["gap_versus_sklearn"])
    if not short_gap > long_gap:
        raise RuntimeError(
            "A short gradient run should sit farther from least squares than a long run."
        )
    if long_gap >= 1e-3:
        raise RuntimeError(f"Long gradient run did not approach scikit-learn (gap={long_gap}).")


def _require_decrease(history: np.ndarray, name: str) -> None:
    if not history[-1] < history[0]:
        raise RuntimeError(f"{name} training loss did not decrease.")
    if float(np.max(history)) > float(history[0]) * 1.05:
        raise RuntimeError(
            f"{name} training loss spiked above its initial value. Lower the learning rate."
        )


def _require_synthetic(source: str, description: str) -> None:
    if source != "synthetic" or "synthetic" not in description.lower():
        raise RuntimeError("Report data must be labeled synthetic.")


if __name__ == "__main__":
    main()
