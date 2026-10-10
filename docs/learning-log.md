# Learning log

Course context: UMD CS 5541, machine learning basics. This log is the design record for the repository. It is not a diary of invented debugging sessions. When you run the reports, add a dated note with the numbers you actually got.

## 2026-10-10 — Foundations in NumPy

### Questions worth answering before opening scikit-learn

- What is the model output, in arithmetic, for one row?
- Which quantity is differentiated, and with respect to which parameter?
- Which rows are allowed to influence the mean, the scale, the neighbors, and the weights?
- Which score matches that training objective, and which rows is it computed on?
- What would a perfect score on this synthetic generator fail to prove?

### What the code is responsible for

- Linear regression, one feature or many, predicts `X @ coef + intercept`. Training is full-batch gradient descent from zero. The history of the objective is `loss_history_`. An optional `l2` penalty pulls slopes toward zero and leaves the intercept alone.
- KNN ranks rows by Euclidean distance, takes `k` of them, and votes. Equal distances keep the earlier training index. Equal votes keep the smallest label.
- The network has one hidden layer, ReLU or sigmoid, then a linear output or a sigmoid output. Regression uses mean squared error. Binary classification uses cross-entropy and accepts only labels 0 and 1. `fit` restarts from `seed` so a repeated call is a repeated experiment.
- Evaluation covers the split, accuracy, precision, recall, F1, the confusion matrix, mean squared error, and R-squared.
- Standardization fits on the training matrix and applies that transform afterward. The package does not offer a helper that fits on a stacked train/test matrix.

### Observations to record when you generate reports

Run `python reports/generate_reports.py` and read `reports/comparison.md`.

- Record the coefficient gap after 5 gradient steps and after 2000. The short gap should be larger. The long gap should be small. If the long gap is not small, the gradient or the step size is a better suspect than "scikit-learn uses a different line."
- Record whether the KNN mismatch count on the separated blobs is zero. Then read the tie table. A zero mismatch count does not mean the tie policy is the same as scikit-learn's. It means this particular test set did not exercise a tie.
- The regression figure overlays three lines: gradient descent, least squares, and the synthetic generating line. On a long run the first two should be hard to tell apart. The observations should not lie on the generating line, because that figure adds noise on purpose. The figure is a training fit. Do not caption it as test performance.
- The loss figure has two panels, linear and network, both on rows that were used for fitting. A decrease means the update ran. It is not an evaluation.

### Misreadings this layout is meant to block

- Treating a library call as the definition of the model. The definition is the loss and the update. scikit-learn's least-squares solve is a different algorithm for the same unpenalized linear objective.
- Fitting a scaler on every row you intend to score.
- Reading training accuracy of 1-nearest neighbors as competence.
- Quoting a synthetic accuracy as a result about people. The datasets say they are synthetic in `source` and in `description`. The responsible-use note says what that excludes.
- Confusing a vote tie with a probability of 0.5. KNN breaks ties by the smallest label. The network maps probability 0.5 to class 1. Those are separate deterministic rules.

### Checks already encoded as tests

Known slopes on a line and on a three-column design. One explicit gradient step from zero, matched to the hand derivative. Monotone decrease at a moderate learning rate. Neighbor labels, both tie rules, and a 3-4-5 distance. Metric counts on a four-row confusion matrix. Shape and finiteness checks. Two calls with one seed match inside the process. Analytic gradients for selected parameters sit close to a two-sided finite difference, including a fully worked ReLU backward pass.

### Follow-ups, deliberately not in this repository

Minibatch stochastic gradient descent, a second hidden layer, softmax, a validation loop for choosing `k`, and any real dataset. Each of those is easier to add once a wrong gradient is obvious on a problem with a known answer.
