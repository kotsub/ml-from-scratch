import numpy as np

from ._validation import positive_integer, prediction_data, training_data


class KNNClassifier:
    """Евклидово расстояние и равные голоса k ближайших соседей.

    При равном расстоянии первым идёт более ранний обучающий объект.
    При равном числе голосов выбирается первый класс в classes_ (порядок np.unique).
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X, y = training_data(X, y)
        positive_integer(self.n_neighbors, "n_neighbors")
        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors не может превышать число обучающих объектов")
        self.X_ = X.copy()
        self.classes_, self.y_encoded_ = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        X = prediction_data(self, X)
        result = np.zeros((len(X), len(self.classes_)))
        # Одна строка за раз: не создаём тензор размером test x train x features.
        for i, row in enumerate(X):
            distances = np.sum((self.X_ - row) ** 2, axis=1)
            neighbors = np.argsort(distances, kind="stable")[:self.n_neighbors]
            votes = np.bincount(self.y_encoded_[neighbors], minlength=len(self.classes_))
            result[i] = votes / self.n_neighbors
        return result

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]
