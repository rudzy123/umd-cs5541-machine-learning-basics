"""Distances, votes, ties, and shape checks for k-nearest neighbors."""

from __future__ import annotations

import numpy as np
import pytest

from umd_ml.knn import KNeighborsClassifier, euclidean_distances, majority_vote


def test_euclidean_distance_of_a_three_four_five_triangle() -> None:
    distances = euclidean_distances(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]))
    assert distances.shape == (1, 1)
    assert distances[0, 0] == pytest.approx(5.0)


def test_knn_labels_obvious_neighbors() -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    model = KNeighborsClassifier(k=1).fit(features, labels)
    queries = np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.2]])
    np.testing.assert_array_equal(model.predict(queries), np.array([0, 1, 0]))

    model_k3 = KNeighborsClassifier(k=3).fit(features, labels)
    np.testing.assert_array_equal(model_k3.predict(queries), np.array([0, 1, 0]))


def test_equal_votes_choose_the_smallest_label() -> None:
    features = np.array([[0.0, 1.0], [0.0, -1.0], [5.0, 5.0]])
    labels = np.array([0, 1, 1])
    prediction = KNeighborsClassifier(k=2).fit(features, labels).predict(np.array([[0.0, 0.0]]))
    assert int(prediction[0]) == 0
    assert int(majority_vote(np.array([1, 0, 0, 1]))) == 0


def test_equal_distances_keep_the_earlier_training_row() -> None:
    features = np.array([[1.0, 0.0], [-1.0, 0.0]])
    labels = np.array([1, 0])
    prediction = KNeighborsClassifier(k=1).fit(features, labels).predict(np.array([[0.0, 0.0]]))
    assert int(prediction[0]) == 1


def test_predictions_do_not_depend_on_call_order() -> None:
    features = np.array([[0.0, 0.0], [1.0, 1.0], [0.2, -0.2], [0.9, 1.2]])
    labels = np.array([0, 1, 0, 1])
    first = KNeighborsClassifier(k=3).fit(features, labels)
    second = KNeighborsClassifier(k=3).fit(features, labels)
    queries = np.array([[0.1, 0.1], [0.8, 0.7]])
    np.testing.assert_array_equal(first.predict(queries), second.predict(queries))


def test_input_shapes_and_k_are_rejected() -> None:
    with pytest.raises(ValueError):
        KNeighborsClassifier(k=0)
    with pytest.raises(TypeError):
        KNeighborsClassifier(k=True)  # type: ignore[arg-type]
    model = KNeighborsClassifier(k=3)
    with pytest.raises(RuntimeError, match="Fit the classifier"):
        model.predict(np.ones((1, 2)))
    with pytest.raises(ValueError, match="cannot exceed"):
        model.fit(np.ones((2, 2)), np.array([0, 1]))
    with pytest.raises(ValueError, match="n_samples"):
        KNeighborsClassifier(k=1).fit(np.ones(4), np.ones(4))
    fitted = KNeighborsClassifier(k=1).fit(np.ones((2, 2)), np.array([0, 1]))
    with pytest.raises(ValueError, match="features"):
        fitted.predict(np.ones((2, 3)))


def test_mismatched_feature_counts_in_the_distance_helper() -> None:
    with pytest.raises(ValueError, match="features"):
        euclidean_distances(np.ones((2, 2)), np.ones((2, 1)))
