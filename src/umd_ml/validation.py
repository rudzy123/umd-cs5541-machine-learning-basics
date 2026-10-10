"""Input checks shared by the educational estimators.

The checks are strict on purpose. A shape or dtype mistake should stop at the
boundary of the algorithm, with a message that says what was expected.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatMatrix = npt.NDArray[np.float64]
FloatVector = npt.NDArray[np.float64]


def require_seed(seed: int) -> int:
    """Return a non-negative integer seed."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    return seed


def require_positive_int(value: int, *, name: str) -> int:
    """Return ``value`` when it is an integer greater than or equal to 1."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def require_learning_rate(value: float) -> float:
    """Return a finite learning rate strictly greater than zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("learning_rate must be a real number.")
    rate = float(value)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be a finite number greater than 0.")
    return rate


def require_nonnegative_float(value: float, *, name: str) -> float:
    """Return a finite float that is greater than or equal to zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite number greater than or equal to 0.")
    return number


def require_finite_float(value: float, *, name: str) -> float:
    """Return a finite float. Booleans are rejected because they are integers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def require_tol(tol: float | None) -> float | None:
    """Return an optional non-negative convergence tolerance."""
    if tol is None:
        return None
    return require_nonnegative_float(tol, name="tol")


def require_finite_matrix(values: np.ndarray, *, name: str = "X") -> FloatMatrix:
    """Coerce ``values`` to a finite float matrix with shape ``(n_samples, n_features)``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_features); got ndim={array.ndim}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            f"{name} must contain at least one row and one column; got shape {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def require_target(values: np.ndarray, *, n_samples: int, name: str = "y") -> FloatVector:
    """Coerce a regression target to a finite 1D vector aligned with ``X``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional target vector; got shape {array.shape}."
        )
    if array.shape[0] != n_samples:
        raise ValueError(
            f"{name} length must match the number of samples ({n_samples}); got {array.shape[0]}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def require_labels(values: np.ndarray, *, n_samples: int, name: str = "y") -> np.ndarray:
    """Coerce numeric class labels to a 1D vector aligned with ``X``."""
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = np.reshape(array, -1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional label vector; got shape {array.shape}.")
    if array.shape[0] != n_samples:
        raise ValueError(
            f"{name} length must match the number of samples ({n_samples}); got {array.shape[0]}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} labels must be numeric.")
    if not np.isfinite(array.astype(np.float64)).all():
        raise ValueError(f"{name} labels must be finite.")
    return np.array(array, copy=True)


def ensure_finite_loss(loss: float) -> None:
    """Stop training when the objective can no longer be used for an update."""
    if not np.isfinite(loss):
        raise FloatingPointError(
            "Training loss became non-finite. Lower the learning rate, "
            "reduce the penalty strength, or standardize the features."
        )
