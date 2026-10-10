# Machine-learning cheat sheet

Notation used throughout this repository. `n` is the number of rows, `d` is the number of features, and a row vector `x` is one example. Matrices are stored with one example per row.

## Supervised fit

Training chooses parameters by looking at `(X_train, y_train)`. A test score uses those frozen parameters on `(X_test, y_test)`. Anything that estimates a statistic — a mean, a scale, a selected feature — is a parameter for this purpose.

## Linear regression

Prediction:

```
y_hat = X @ coef + intercept
```

Objective, with optional penalty `l2 >= 0`. The intercept is not inside the penalty.

```
loss = mean((y_hat - y) ** 2) + l2 * sum(coef ** 2)
```

Gradients:

```
error = y_hat - y
grad_coef = (2 / n) * X.T @ error + 2 * l2 * coef
grad_intercept = (2 / n) * sum(error)     # or 0 when fit_intercept is false
```

Update, from a zero start:

```
coef <- coef - learning_rate * grad_coef
intercept <- intercept - learning_rate * grad_intercept
```

`loss_history_[0]` is the loss at zero. Later entries are the loss after each update. Simple regression is this update with `d = 1`.

The factor 2 is the derivative of a square. An objective written as `1 / (2 n)` times the sum of squares cancels it and only changes which learning rate you type.

## Standardization

Fit on training rows only:

```
mean = mean(X_train, axis=0)
scale = sqrt(mean((X_train - mean) ** 2, axis=0))
```

The square uses `/ n`, not `/ (n - 1)`. Columns with `scale < 1e-12` are given scale 1 after centering.

```
X_any_transformed = (X_any - mean) / scale
```

## K-nearest neighbors

Euclidean distance between a query `q` and a training row `x`:

```
distance = sqrt(sum((q - x) ** 2))
```

Rank training rows with a stable sort. Equal distances keep the smaller training index. Take the first `k` labels. The predicted label is the one with the highest count; if several share that count, take the smallest label.

## Metrics

Mean squared error:

```
MSE = mean((y - y_hat) ** 2)
```

R-squared, when the targets are not constant:

```
R2 = 1 - sum((y - y_hat) ** 2) / sum((y - mean(y)) ** 2)
```

Accuracy:

```
accuracy = mean(y == y_hat)
```

For one positive label, write `tp`, `fp`, `fn` for true positives, false positives, and false negatives.

```
precision = tp / (tp + fp)     # 0 when the denominator is 0
recall    = tp / (tp + fn)     # 0 when the denominator is 0
F1        = 2 * precision * recall / (precision + recall)    # 0 when both are 0
```

Macro averaging computes the score for each observed label and takes the unweighted mean. The confusion matrix has `matrix[true, predicted] = count`.

## Network

```
z1 = X @ W1 + b1          # W1 is (d, h), b1 is (h,)
a1 = relu(z1) or sigmoid(z1)
z2 = a1 @ W2 + b2         # W2 is (h, 1), b2 is (1,)
```

ReLU and its derivative:

```
relu(z) = max(z, 0)
relu'(z) = 1 if z > 0 else 0
```

Sigmoid and its derivative:

```
sigmoid(z) = 1 / (1 + exp(-z))
sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
```

Regression output is `z2`. Loss is mean squared error. With `error = y_hat - y` stored as a column:

```
delta2 = (2 / n) * error
```

Classification output is `sigmoid(z2)`. Loss is mean binary cross-entropy. The chain rule through the sigmoid simplifies:

```
delta2 = (y_hat - y) / n
```

Shared backward step:

```
dW2 = a1.T @ delta2
db2 = sum(delta2, axis=0)
da1 = delta2 @ W2.T
dz1 = da1 * activation'(z1)
dW1 = X.T @ dz1
db1 = sum(dz1, axis=0)
```

Initialization: biases are 0. Hidden weights are normal with scale `sqrt(2 / d)` for ReLU and `sqrt(1 / d)` for sigmoid. Output weights use scale `sqrt(1 / h)`. One Generator consumes those draws in that order, from `seed`.

A probability of 0.5 is class 1.

## Finite-difference check

For a single parameter `theta` and a small `eps`:

```
(L(theta + eps) - L(theta - eps)) / (2 * eps)
```

Compare that estimate with the analytic gradient using

```
||analytic - numeric|| / (||analytic|| + ||numeric||)
```

The tests do this for the linear slope and intercept, and for the network weights, at parameters where a ReLU kink is not sitting on a pre-activation.

## What a short gradient run is allowed to get wrong

scikit-learn's `LinearRegression` solves the normal equations. Five gradient steps are not that solution. Two thousand steps on a well-scaled synthetic design should land within `1e-3` in the coefficients. If a long run does not, the gradient or the learning rate is wrong. If a short run already matches to many decimals, the story is still the same and the problem was easy.
