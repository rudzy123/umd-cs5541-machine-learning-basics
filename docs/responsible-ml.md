# Responsible use of this codebase

These models are coursework. They exist so the update rules can be read, tested, and compared with a library. They are not a decision system for people.

## Synthetic data is not a population

`make_regression`, `make_sine_regression`, and `make_classification` draw arrays from a Generator on this machine. The descriptions say so, and `source` is `"synthetic"`. The blobs are round and similarly scaled. The labels are the index of the blob. Noise is Gaussian because that was the easy distribution to write, not because a measurement process produced it.

A high accuracy on those blobs means the classifier recovered a gap the generator inserted. It does not mean the same code would rank applicants, read a scan, or grade a student fairly. Fairness questions need a real labeling process: who is missing, who is mislabeled, which feature is a proxy for a protected attribute, and what happens to people when the prediction is wrong. None of that structure is in a pair of Gaussians. Reporting a synthetic accuracy as if it were evidence about people is a false claim, even when the arithmetic is correct.

KNN stores its training rows and looks them up at prediction time. If those rows were people, the model would be a copy of their feature vectors. That is one reason the training tables here are generated rather than collected. Do not point this loader at a file of personal data and treat the result as a class project that happens to be deployable.

## Leakage makes an honest metric lie

A test score is an ethical claim as well as a statistical one: "this is how the procedure behaves on rows it did not use." Standardization fit on the combined train and test matrices uses the test rows. So does picking `k` because that choice won on the test set, then quoting the winning score. The number can be calculated correctly and still answer the wrong question.

`standardize_splits` is the supported path because the easy path should be the one that fits on training rows only. The four-number example in the tests is there so the leaked mean (5) and the training mean (2) cannot be confused.

## Metrics hide the rows you do not name

Accuracy treats every mistake as equal. Precision and recall do not. A confusion matrix shows which way the mistakes go. Macro averages give a rare label the same weight as a common one; a micro average, which this repository does not implement, would not. Choosing accuracy because it is larger is not a neutral reporting step.

The synthetic class counts are nearly equal by construction. An accuracy computed here does not teach you how the same function would behave on a 99-to-1 problem. Say which metric you computed, which label was positive, and which rows were held out.

## Overfitting and overconfidence

A training loss that falls only says the optimizer ran. KNN with `k=1` can score perfectly on its training table and still be a lookup of noise. Quoting that training score as performance is the same kind of overclaim as quoting a leaked test score.

A single split is one partition. Another seed would move the test rows. This repository does not average over splits. A report number is the result for the stated seed, not a confidence interval.

## Reproducibility is part of the claim

Someone else should be able to run the tests and the report script and see the same qualitative result: known lines recovered, loss falling under the stated rates, neighbor ties broken the way the docstring says, and a long gradient run approaching scikit-learn. Seeds, float64, local generation, and the tie policy are what make that check possible.

Bit-level equality across machines is a weaker promise. Matrix products can depend on BLAS. The tests that recover a known coefficient use a tolerance. The tests that call the same function twice in one process ask for exact equality. Do not tighten the first kind of test until it fails on a different CPU and then "fix" the algorithm.

No test, smoke run, or report downloads a dataset. Adding a fetch "just for a figure" would break that contract and would start the leakage and licensing questions this repository is built to avoid.

## What not to do with the fitted objects

Do not use a fitted `LinearRegression`, `KNeighborsClassifier`, or `NeuralNetwork` from this package to make a decision that affects a person's health, money, freedom, education, or work. The implementations are short on purpose. They omit calibration, monitoring, appeal, and any dataset that would be required even to ask whether a model was fit for a real task.

If a later project uses real data, the obligations change: consent and privacy, a pre-registered evaluation split, a metric tied to the actual cost of errors, and a clear statement of who is allowed to act on a prediction. None of those steps are satisfied by passing this repository's tests.
