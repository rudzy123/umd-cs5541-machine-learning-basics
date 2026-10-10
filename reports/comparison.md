# Educational models and scikit-learn

All rows in this report are synthetic. They were generated locally from a NumPy Generator.
No dataset was downloaded.

Seed 5541. On noise-free synthetic regression data, batch gradient descent after 5 steps differed from scikit-learn's least-squares coefficients by at most 7.086e-01. After 2000 steps that gap was 8.882e-16. The KNN comparison on a separate synthetic classification set mismatched on 0 of 40 held-out rows.

## Regression optimization gaps

| iterations | gap_versus_sklearn | gap_versus_generator | educational_test_mse | sklearn_test_mse |
| --- | --- | --- | --- | --- |
| 5 | 0.708613 | 0.708613 | 1.70135 | 2.17923e-30 |
| 2000 | 8.88178e-16 | 8.88178e-16 | 2.09623e-30 | 2.17923e-30 |

## Held-out metrics

| model | implementation | split | metric | value |
| --- | --- | --- | --- | --- |
| linear_regression | educational | test | mean_squared_error | 2.09623e-30 |
| linear_regression | sklearn | test | mean_squared_error | 2.17923e-30 |
| linear_regression | educational | parameters | max_abs_gap_versus_sklearn | 8.88178e-16 |
| knn | educational | test | accuracy | 1 |
| knn | sklearn | test | accuracy | 1 |
| knn | comparison | test | mismatched_predictions | 0 |

## Neighbor tie examples

| case | neighbor_labels_in_order | educational_prediction | sklearn_prediction | note |
| --- | --- | --- | --- | --- |
| equal_votes | 0,1 | 0 | 0 | Two neighbors at distance 1 have labels 0 and 1. The smallest label wins. |
| equal_distances | 1 | 1 | 1 | Both training rows are distance 1. Stable order keeps training index 0. |

## Why small differences show up

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

## Figures

Regenerate with `python reports/generate_reports.py`. Seed 5541. Synthetic data only.

- `figures/regression_fit.png` (50607 bytes)
- `figures/training_loss.png` (53718 bytes)
- `figures/decision_boundary.png` (44169 bytes)
- `figures/confusion_matrix.png` (22771 bytes)
