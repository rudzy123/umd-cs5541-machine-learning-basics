"""Educational implementations of foundational machine learning algorithms.

The estimators are NumPy. scikit-learn is used only for comparison. Datasets
produced by this package are synthetic and are labeled as such.
"""

from umd_ml.compare import ComparisonResult, compare_with_sklearn, render_comparison
from umd_ml.datasets import (
    SYNTHETIC_BANNER,
    SyntheticDataset,
    make_classification,
    make_regression,
    make_sine_regression,
)
from umd_ml.evaluation import (
    accuracy,
    classification_scores,
    confusion_matrix,
    f1,
    mean_squared_error,
    precision,
    r_squared,
    recall,
    train_test_split,
)
from umd_ml.gradients import (
    finite_difference_gradient,
    finite_difference_scalar,
    relative_gradient_error,
)
from umd_ml.knn import KNeighborsClassifier, euclidean_distances, majority_vote
from umd_ml.linear_regression import LinearRegression, mse_loss_and_gradients
from umd_ml.neural_network import NeuralNetwork
from umd_ml.preprocessing import StandardScaler, standardize_frames, standardize_splits

__version__ = "0.1.0"

__all__ = [
    "SYNTHETIC_BANNER",
    "ComparisonResult",
    "KNeighborsClassifier",
    "LinearRegression",
    "NeuralNetwork",
    "StandardScaler",
    "SyntheticDataset",
    "accuracy",
    "classification_scores",
    "compare_with_sklearn",
    "confusion_matrix",
    "euclidean_distances",
    "f1",
    "finite_difference_gradient",
    "finite_difference_scalar",
    "majority_vote",
    "make_classification",
    "make_regression",
    "make_sine_regression",
    "mean_squared_error",
    "mse_loss_and_gradients",
    "precision",
    "r_squared",
    "recall",
    "relative_gradient_error",
    "render_comparison",
    "standardize_frames",
    "standardize_splits",
    "train_test_split",
]
