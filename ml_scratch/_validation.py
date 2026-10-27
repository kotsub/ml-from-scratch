import numpy as np


def as_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X должна быть непустой двумерной матрицей")
    if not np.isfinite(X).all():
        raise ValueError("X содержит NaN или бесконечность; обработай пропуски заранее")
    return X


def training_data(X, y, numeric_target=False):
    X = as_matrix(X)
    y = np.asarray(y, dtype=float if numeric_target else None)
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y должна быть одномерной и иметь столько же строк, сколько X")
    if numeric_target and not np.isfinite(y).all():
        raise ValueError("y содержит NaN или бесконечность")
    if not numeric_target:
        for label in y:
            if label is None or (isinstance(label, (float, np.floating)) and not np.isfinite(label)):
                raise ValueError("Метки классов не должны содержать пропуски или бесконечность")
    return X, y


def prediction_data(model, X):
    if not hasattr(model, "n_features_in_"):
        raise RuntimeError("Сначала вызови fit(X, y)")
    X = as_matrix(X)
    if X.shape[1] != model.n_features_in_:
        raise ValueError("Число признаков отличается от обучающей выборки")
    return X


def positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} должен быть целым числом >= {minimum}")
