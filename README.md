# UMD CS 5541 — Machine learning basics

Educational implementations of linear regression, k-nearest neighbors, and a one-hidden-layer network. The updates are written in NumPy so the arithmetic stays visible. scikit-learn is a comparison point, not the place where the algorithms are defined. Every dataset in this repository is **synthetic**. It is drawn locally from a NumPy Generator. Tests do not download data.

Python 3.12.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Quality gates, in the order continuous integration runs them:

```bash
ruff check .
ruff format --check .
pytest
python -m umd_ml.smoke
python reports/generate_reports.py
```

The notebook `notebooks/01_foundations.ipynb` is a walkthrough of the same code. The tests are the executable check.

## Layout

| Path | Role |
| --- | --- |
| `src/umd_ml/` | Estimators, metrics, leakage-safe standardization, synthetic data |
| `tests/` | Known coefficients, descent, neighbors, metrics, shapes, seeds, gradient checks |
| `reports/generate_reports.py` | Fit plot, training-loss curve, 2D decision boundary, confusion matrix |
| `docs/` | Cheat sheet, evaluation notes, responsible-ML notes, learning log |
| `.github/workflows/ci.yml` | Lint, tests, smoke training, report generation |

## Supervised learning

Supervised learning fits a rule that maps features to a label, using examples where the label is already known. The fit is judged on rows the rule did not see. This repository stays inside that setting: every model has an `X` and a `y`, and the interesting question is what the training procedure is allowed to look at.

Unsupervised learning, which looks for structure without labels, is outside this repository.

## Regression versus classification

Regression predicts a number. The linear model and the network's regression task both do this, and the training loss is mean squared error. A good prediction is close to the target on the number line.

Classification predicts a category. K-nearest neighbors does this for any numeric labels. The network's classification task is binary: targets must be 0 or 1, the output is a probability, and the predicted class is 1 when that probability is at least 0.5.

The same features can be used for either task. The loss, the output, and the metric are what change. Accuracy on a regression target, or mean squared error on arbitrary class codes, answers a question the model was not trained to answer.

## Features and labels

A row is one example. Columns of `X` are features: the quantities the model may use as input. `y` is the label, also called the target. In the synthetic tables here, feature names are `x1`, `x2`, and so on. Regression frames call the label `target`. Classification frames call it `label`.

`SyntheticDataset.source` is always `"synthetic"`, and `description` says the rows are not measurements of real people or events. That sentence is part of the data contract, not a caption you add later.

## Dataset splitting

`train_test_split` shuffles row indices with `numpy.random.Generator` and holds out a fraction of them. The fraction must sit strictly between 0 and 1. The test count is `round(n * test_size)`, then clipped so neither side is empty.

Fit on the training rows. Score on the test rows. The test score is one estimate of behavior on new rows from the same generator. It is not a license to keep changing the model until that number looks good: once the test set is used for decisions, it becomes another training set. Tuning `k`, the learning rate, or the width belongs on a validation split, or in cross-validation, and the test set stays sealed. This repository implements the split and not a search loop, so the reports do not pretend a single test number was free.

`stratify=y` applies the same fraction inside each class that has at least two rows. A class with one row stays in training. Using labels to balance the split is evaluation design. It is not the same mistake as fitting a preprocessor on test features.

## Leakage

Leakage is any path that lets information from the rows you will score on influence the model that is scored. The usual classroom version is a scaler.

`StandardScaler.fit` records the mean and the population standard deviation of the matrix it is given. `transform` applies those frozen numbers and does not update them. `standardize_splits(X_train, X_valid, X_test)` fits on `X_train` only. A constant training column is centered and then left at scale 1, so the transform does not divide by zero.

Fitting the scaler on `vstack([X_train, X_test])` is leakage even when the later arithmetic looks tidy. Test rows have moved the mean. The test score then flatters the pipeline. The same rule covers imputation, feature selection, and any statistic you would be unwilling to recompute without the test set. The preprocessor tests use a four-number example where the honest training mean is 2 and the leaked mean of all four numbers is 5.

## Bias and variance

Bias is error from a model that cannot represent the pattern, even with unlimited data of the kind you are fitting. A straight line has high bias on a sine wave. Variance is error from a model that chases the particular sample it was shown. K-nearest neighbors with `k=1` has low bias and high variance: it can isolate a single noisy row. A large `k` smooths that noise and can smear two classes together.

A linear model on data that really is linear can have low bias. Its variance grows when the sample is small, the features are collinear, or the fit is unregularized and high-dimensional. The network can reduce bias relative to a line, because the hidden layer can bend, and it can raise variance, because it has more parameters than a line.

You cannot see variance in the training loss alone. Training loss falls when the model memorizes. The test split, or a second synthetic draw from the same generator, is what shows whether the fit moved with the sample.

## Overfitting

Overfitting is a good training score and a worse score on rows that were held out. Concrete versions in this codebase:

- KNN with `k=1` copies the training label of the nearest row. On the training set its accuracy is perfect whenever duplicate feature rows do not carry conflicting labels. That number is not a test result.
- A network trained for many steps on a small table can drive training loss down by fitting noise.
- Gradient descent on a linear model is less dramatic, because the model class is small, but a short run can still look different from the least-squares fit. That gap is under-optimization, not overfitting. The comparison report separates the two.

The response this repository actually implements is a held-out split, an optional L2 penalty on linear slopes, and a `k` you choose on purpose. It does not implement early stopping on a validation set, dropout, or a search over `k`.

## Regularization

The linear model can add `l2 * sum(coef ** 2)` to the mean squared error. The intercept is not penalized: shifting all predictions should not be treated as a complex slope. The extra gradient is `2 * l2 * coef`, so the slope is pulled toward zero while the data gradient pulls it toward the least-squares value. Larger `l2` means a smaller coefficient vector and a worse fit on the training points, traded for less sensitivity to those points.

scikit-learn's `Ridge` minimizes a related but not identical objective (it does not divide the squared error by `n` in the same place). This repository does not compare the penalty against `Ridge`. The unpenalized comparison against `LinearRegression` is the one that should agree.

Weight decay in the network would add the same kind of term to `W1` and `W2`. It is described here and not implemented, so the backprop derivation stays the plain loss.

## Gradient descent

Start at a parameter vector. Compute the gradient of the training loss. Step in the opposite direction by `learning_rate` times that gradient. Repeat.

For mean squared error `mean((y_hat - y) ** 2)`, the derivative with respect to a prediction brings a factor of 2. The code keeps that factor. Dropping it, or writing the loss as `1 / (2 n)` times the sum of squares, only rescales the learning rate. It does not define a different line.

Linear regression starts at the zero vector, so the path does not need a seed. The network restarts from a Generator seeded by `seed` every time `fit` is called, then runs the same full-batch updates. Both store a loss history whose first entry is the loss before any update.

Full-batch means every training row is in every gradient. There is no minibatch and no epoch shuffle. For a fixed training matrix the linear path is deterministic. The network path is deterministic for a fixed seed and a fixed row order. A learning rate that is too large makes the loss diverge; the trainers raise `FloatingPointError` when the loss stops being finite.

The loss can fail to fall when the learning rate is too large, the features have wild scales, or a ReLU network is initialized so that every hidden unit is off. The tests use moderate rates and, where it matters, features on a modest scale. "The loss decreased" is a statement about those conditions, not a promise about every hyperparameter.

## Neural-network layers

The network has three stages and one hidden layer:

1. The input stage is the feature row. It has no weights of its own.
2. The hidden layer computes `z1 = x @ W1 + b1`, then applies ReLU or sigmoid.
3. The output layer computes `z2 = a1 @ W2 + b2`. Regression stops there. Binary classification passes `z2` through a sigmoid.

ReLU is `max(z, 0)`. Its derivative is 1 on positive pre-activations and 0 elsewhere, including exactly zero. Sigmoid is the logistic function, with the stable branch that avoids `exp` of a large positive number. He scaling initializes ReLU weights. Xavier scaling initializes sigmoid weights. Biases start at zero.

Backpropagation applies the chain rule from the output loss back to `W1` and `b1`. For regression the output delta is `(2 / n) * (y_hat - y)`. For binary cross-entropy with a sigmoid output, the delta simplifies to `(y_hat - y) / n`. The hidden delta multiplies by the activation derivative. The cheat sheet writes the shapes next to the code in `src/umd_ml/neural_network.py`.

This is not a deep network. There is no softmax, no convolution, and no second hidden layer. Multi-class labels belong to KNN in this repository.

## Metric selection

Use a metric that matches the decision.

| Question | Metric |
| --- | --- |
| How far are numeric predictions from the targets, in the same units the loss uses? | Mean squared error |
| What fraction of target variance does the fit capture, on a unitless scale? | R-squared |
| On balanced classes, how often is the label right? | Accuracy |
| When a positive prediction is made, how often is it right? | Precision |
| Of the actual positives, how many were found? | Recall |
| One number that punishes a collapse of either precision or recall | F1 |
| Where do the mistakes sit? | Confusion matrix |

Accuracy on a 99-to-1 problem can look excellent for a model that never predicts the rare class. These synthetic blobs are balanced, so accuracy is an honest summary here and would not be on a skewed problem. Precision, recall, and F1 default to the positive label `1`. `average="macro"` scores each observed label and takes the unweighted mean. A class with no predicted positives scores 0 instead of dividing by zero. The confusion matrix uses rows as the true label and columns as the predicted label.

R-squared is 1 when the predictions match the targets. It is 0 when the predictions do no better than predicting the mean of the targets being scored. When the targets have zero variance, the usual denominator vanishes. The implementation returns 1 if the predictions match and 0 otherwise.

Mean squared error is the training objective for the linear model and for the regression network. A lower test MSE is the comparison that lines up with training. R-squared removes the units, so targets with different spreads are easier to compare.

## Limitations of synthetic data

The generators draw Gaussians, a line plus Gaussian noise, or a sine plus Gaussian noise. Class blobs are round, similarly scaled, and labeled by the integer of the blob that produced them. That is enough to check an update rule. It is not a population.

Synthetic accuracy does not transfer to a medical, credit, hiring, or grading decision. It cannot reveal label bias, missingness, or a feature that only exists because of an earlier policy. Two well-separated blobs make KNN look strong; a sine wave makes a line look weak. Both outcomes were chosen when the generator was written. The responsible-ML note in `docs/responsible-ml.md` is the longer version of this limit. The short version: do not quote a number from this repository as evidence about people.

## Reproducibility

- Linear regression initializes at zero and uses full-batch updates, so two fits on the same array match.
- The network and the dataset generators take an integer `seed` and use `numpy.random.default_rng`. `fit` on the network re-draws that initialization before training, so a second `fit` repeats the first.
- KNN has no randomness. Distance ties keep the smaller training index. Vote ties keep the smallest label.
- Arrays are float64. Classification labels stay integers.
- No training step reads a URL or a data file.
- In-process repeats are the determinism the tests require. A different BLAS can still change the last bits of a long matrix product across machines. Assertions on recovered coefficients use a tolerance for that reason. Assertions that two calls match use exact equality inside one process.
- Report figures use seed `5541`. The unit comparison uses seed `0`.
- CI pins the interpreter to Python 3.12 and sets `MPLBACKEND=Agg` so the report script does not need a display.

## Comparison with scikit-learn

`compare_with_sklearn` fits both linear implementations on one noise-free synthetic design, and both neighbor classifiers on one synthetic blob problem.

The educational line uses gradient descent. scikit-learn's `LinearRegression` solves the least-squares problem. After a handful of steps the coefficient gap is visible. After a long run it falls below `1e-3` on the designs used here, and the remaining difference is numerical. The report writes both gaps. A wider gap is not a bug in the definition of squared error unless the long run fails to close it.

KNN predictions match on the separated test set used in the comparison, where neighbors do not tie. They are allowed to differ when distances tie or votes tie. The report includes a two-row tie table computed from both implementations, so the policy is observed rather than assumed. Details and the formulas live in `docs/ml-cheat-sheet.md` and `docs/evaluation.md`.

## What this repository leaves out

Minibatch stochastic gradient descent, softmax, more than one hidden layer, cross-validation search, and any downloaded dataset. Adding them before the four update rules above are boringly correct usually hides the bug in a library call.

## License

MIT. See `LICENSE`.
