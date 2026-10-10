# Evaluation

A metric is an answer to a specific question about predictions and labels. It does not know how the predictions were produced. That ignorance is useful only when the surrounding procedure was honest.

## Split before you fit

`train_test_split` returns four arrays: training features, test features, training labels, test labels. Fit every estimator and every scaler on the training pair. Call `predict` or `transform` on the test features. Score with the test labels.

The split is a seeded permutation. Same seed, same arrays, including row order. Full-batch gradient descent is mathematically invariant to row order and only approximately invariant in floating point, so the seed fixes the order too.

`test_size` is a fraction in `(0, 1)`. The test count is `round(n * test_size)`, clipped into `1, ..., n - 1`.

Stratified splitting uses group labels, usually `y`, and applies that fraction inside each group of size at least 2. A singleton group stays in training. This looks at labels while building the partition. It does not look at test features while estimating a mean, a weight, or a neighbor table. Those are different acts. Balancing a split is acceptable. Fitting on the test features is not.

If you try several values of `k` or several learning rates, do it against a validation split cut from the training rows. The test split stays out of that loop. This repository does not ship a model-selection loop. A single test accuracy in a report is one sealed look, not the best of twenty.

## Leakage checklist

Before trusting a test number, answer these for every statistic in the pipeline:

1. Which rows were visible when the statistic was computed?
2. Would the number change if the test rows were replaced?
3. Is the test score then presented as evidence about unseen rows?

For this codebase the concrete rule is: `StandardScaler.fit` sees `X_train` and nothing else. `standardize_splits` and `standardize_frames` are the paths that make that easy. Concatenating first is the path that cheats.

A second, quieter leak is choosing features, a network width, or `k` because of the test score, then reporting that same score. The tests in this repository are assertions about known answers, not a leaderboard. The report figures label training curves as training curves.

## Metrics and the questions they answer

### Mean squared error

Average squared distance between prediction and target. It is the linear model's training objective and the regression network's training objective, so a test MSE compares the thing training optimized. Units are the square of the target's units. Outliers move it hard, because a residual of 3 contributes 9.

### R-squared

One minus residual sum of squares over total sum of squares, where the total is measured against the mean of the targets being scored. It is unitless. A value near 1 means the residuals are small relative to the spread of `y`. A value near 0 means the predictions do about as well as guessing that mean. A negative value means they do worse. It does not by itself say the model is linear, causal, or useful.

When `y` is constant the denominator is 0. The function returns 1 if the predictions match that constant and 0 otherwise, and the cheat sheet records that choice because the usual ratio is undefined.

### Accuracy

Fraction of rows with the correct label. Appropriate when classes are balanced and every mistake has a similar cost. Misleading when one class dominates: a constant predictor can score 0.99 and still be useless. The synthetic blobs in the reports are generated with nearly equal class counts so that accuracy is readable. That balance is a property of the generator.

### Precision, recall, and F1

Precision is `tp / (tp + fp)`: of the rows you called positive, how many were. Recall is `tp / (tp + fn)`: of the rows that were positive, how many you found. F1 is their harmonic mean, so a zero on either side pulls the score to zero.

The binary versions take `positive_label` (default 1) and treat every other label as negative. Macro averaging repeats that for every label that appears in `y_true` or `y_pred` and averages the results with equal weight per label, not per row. A label you never predict gets precision 0 here, rather than an exception. That is a reporting choice. It is harsher than dropping the label, and it is documented so a 0 is not a surprise.

Which one you emphasize is a cost decision. If a false positive is expensive, read precision. If a missed positive is expensive, read recall. F1 is a compromise, not a moral default.

### Confusion matrix

`confusion_matrix` returns integer counts. Rows are true labels, columns are predicted labels. The default axis order is the sorted unique labels. Pass `labels` to force an order, including a class that happened to be absent from one side; every label that does appear must be listed.

Read the diagonal as agreements and the off-diagonal as the actual mistakes. A single accuracy number cannot tell you that every error is class 0 predicted as class 1. The report figure is this matrix on the held-out synthetic test rows.

`classification_scores` turns the same counts into a pandas table with precision, recall, F1, and support per label. Support is the number of true rows for that label.

## Training loss is not a test metric

`loss_history_` records the objective on the rows inside `fit`. It should fall under the conditions the tests name: a moderate learning rate, features that are not wildly scaled, and a model that can represent the target. A falling curve means the optimizer is moving. It does not mean the model will predict new rows well. The report title on the loss figure says it is not a held-out score.

The regression fit figure is also drawn on the rows that were fit. The generating line is plotted so noise is visible. Held-out linear numbers live in `reports/comparison.md`, which the report script regenerates.

## KNN geometry versus KNN score

The decision-boundary figure is a grid of predictions from a classifier fit on the training rows only. Test rows are drawn as crosses so you can see that they did not move the boundary. The confusion matrix is those crosses, counted. A boundary picture without a count still hides how many crosses landed on the wrong side.

## Comparing two implementations

Use the same rows, the same split, and the same `k`. For linear regression, compare coefficients as well as test MSE. A small test-MSE gap can hide a coefficient gap, and the reverse, on noise-free data both should shrink together because the generating line is the least-squares solution.

Do not compare this repository's `l2` penalty with scikit-learn's `Ridge` without re-deriving the scaling. The objectives are relatives, not copies. The comparison module does not make that claim.
