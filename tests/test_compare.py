"""The scikit-learn comparison stays close after enough gradient steps."""

from __future__ import annotations

from umd_ml.compare import compare_with_sklearn, render_comparison


def test_long_gradient_run_approaches_sklearn_and_short_run_does_not() -> None:
    result = compare_with_sklearn(seed=0)
    gaps = result.optimization_gaps
    short_gap = float(gaps.iloc[0]["gap_versus_sklearn"])
    long_gap = float(gaps.iloc[-1]["gap_versus_sklearn"])
    assert short_gap > long_gap
    assert long_gap < 1e-3
    assert float(gaps.iloc[-1]["educational_test_mse"]) < 1e-6
    assert result.knn_mismatches == 0
    assert result.knn_test_size > 0

    rendered = render_comparison(result)
    assert "synthetic" in rendered.lower()
    assert "gradient descent" in rendered.lower()
    assert "tie" in rendered.lower()
    assert "equal_votes" in rendered
    assert "equal_distances" in rendered
