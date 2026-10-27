from dataclasses import dataclass

import numpy as np

from ._validation import positive_integer, prediction_data, training_data


@dataclass
class Node:
    probabilities: np.ndarray
    samples: int
    impurity: float
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


class DecisionTreeClassifier:
    """Числовое CART-дерево с критерием Gini, без pruning и случайного выбора признаков."""

    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X, y = training_data(X, y)
        if self.max_depth is not None:
            positive_integer(self.max_depth, "max_depth", minimum=0)
        positive_integer(self.min_samples_split, "min_samples_split", minimum=2)
        positive_integer(self.min_samples_leaf, "min_samples_leaf")
        self.classes_, labels = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.zeros(X.shape[1])
        self.depth_ = 0
        self.n_leaves_ = 0
        self.root_ = self._grow(X, labels, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    @staticmethod
    def _gini(counts):
        probabilities = counts / counts.sum()
        return 1.0 - np.sum(probabilities ** 2)

    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=len(self.classes_))
        node = Node(counts / len(y), len(y), float(self._gini(counts)))
        self.depth_ = max(self.depth_, depth)
        if (node.impurity <= 1e-12 or len(y) < self.min_samples_split
                or len(y) < 2 * self.min_samples_leaf
                or (self.max_depth is not None and depth >= self.max_depth)):
            self.n_leaves_ += 1
            return node
        best_gain, best_feature, best_threshold = -np.inf, None, None
        for feature in range(X.shape[1]):
            order = np.argsort(X[:, feature], kind="stable")
            values, sorted_y = X[order, feature], y[order]
            left_counts = np.cumsum(np.eye(len(self.classes_))[sorted_y], axis=0)[:-1]
            right_counts = counts - left_counts
            left_size = np.arange(1, len(y))
            right_size = len(y) - left_size
            left_gini = 1.0 - ((left_counts / left_size[:, None]) ** 2).sum(axis=1)
            right_gini = 1.0 - ((right_counts / right_size[:, None]) ** 2).sum(axis=1)
            gain = node.impurity - (left_size * left_gini + right_size * right_gini) / len(y)
            valid = ((values[:-1] < values[1:]) & (left_size >= self.min_samples_leaf)
                     & (right_size >= self.min_samples_leaf))
            gain[~valid] = -np.inf
            position = int(np.argmax(gain))
            if gain[position] > best_gain + 1e-12:
                best_gain = float(gain[position])
                best_feature = feature
                # Полусумма не переполняется для больших значений одного знака.
                midpoint = values[position] / 2 + values[position + 1] / 2
                # Округление не должно включить правое значение в левую ветку.
                best_threshold = float(max(values[position], min(midpoint, np.nextafter(values[position + 1], -np.inf))))
        if best_feature is None:
            self.n_leaves_ += 1
            return node
        mask = X[:, best_feature] <= best_threshold
        node.feature, node.threshold = best_feature, best_threshold
        self.feature_importances_[best_feature] += len(y) * max(0.0, best_gain)
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def predict_proba(self, X):
        X = prediction_data(self, X)
        probabilities = np.empty((len(X), len(self.classes_)))
        for i, row in enumerate(X):
            node = self.root_
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            probabilities[i] = node.probabilities
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]
