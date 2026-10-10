"""Finite-difference checks for hand-derived gradients.

A two-sided difference estimates one partial derivative:

    (L(theta + eps) - L(theta - eps)) / (2 * eps)

The helpers mutate a parameter array while they probe it, restore every entry
before returning, and expect ``loss_fn`` only to read the parameters.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def relative_gradient_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """Scale-free distance between two gradient arrays.

    The denominator is the sum of the two norms, so a pair of exact zeros has
    a tiny error from the stabilizer rather than a division by zero.
    """
    analytic_array = np.asarray(analytic, dtype=np.float64)
    numeric_array = np.asarray(numeric, dtype=np.float64)
    difference = np.linalg.norm(analytic_array - numeric_array)
    scale = np.linalg.norm(analytic_array) + np.linalg.norm(numeric_array)
    return float(difference / (scale + 1e-12))


def finite_difference_gradient(
    parameters: np.ndarray,
    loss_fn: Callable[[], float],
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Partial derivative of ``loss_fn`` with respect to every entry of ``parameters``."""
    if not np.issubdtype(parameters.dtype, np.floating):
        raise TypeError("finite-difference checks require a floating-point parameter array.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    gradient = np.zeros(parameters.shape, dtype=np.float64)
    for index in np.ndindex(parameters.shape):
        original = parameters[index].item()
        try:
            parameters[index] = original + eps
            loss_plus = float(loss_fn())
            parameters[index] = original - eps
            loss_minus = float(loss_fn())
        finally:
            parameters[index] = original
        gradient[index] = (loss_plus - loss_minus) / (2.0 * eps)
    return gradient


def finite_difference_scalar(
    get_value: Callable[[], float],
    set_value: Callable[[float], None],
    loss_fn: Callable[[], float],
    *,
    eps: float = 1e-6,
) -> float:
    """Partial derivative of ``loss_fn`` with respect to one Python float."""
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    original = float(get_value())
    try:
        set_value(original + eps)
        loss_plus = float(loss_fn())
        set_value(original - eps)
        loss_minus = float(loss_fn())
    finally:
        set_value(original)
    return (loss_plus - loss_minus) / (2.0 * eps)
