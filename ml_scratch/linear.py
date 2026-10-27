import numpy as np

from ._validation import positive_integer, prediction_data, training_data


class LinearRegressionGD:
    """Полный градиентный спуск для MSE / 2 + l2 * ||w||² / 2.

    Свободный член не штрафуется. Перед fit желательно стандартизовать признаки.
    loss_history_ хранит значение цели до обучения и после каждого обновления.
    """

    def __init__(self, learning_rate=0.05, max_iter=10000, l2=0.0, tol=1e-8):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.tol = tol

    def fit(self, X, y):
        X, y = training_data(X, y, numeric_target=True)
        positive_integer(self.max_iter, "max_iter")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate должен быть положительным конечным числом")
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 должен быть неотрицательным конечным числом")
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol должен быть неотрицательным конечным числом")
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self.loss_history_ = []
        self.n_iter_ = 0
        try:
            with np.errstate(over="raise", invalid="raise"):
                for step in range(self.max_iter + 1):
                    residual = X @ self.coef_ + self.intercept_ - y
                    loss = np.mean(residual ** 2) / 2 + self.l2 * np.dot(self.coef_, self.coef_) / 2
                    self.loss_history_.append(float(loss))
                    grad_w = X.T @ residual / len(X) + self.l2 * self.coef_
                    grad_b = residual.mean()
                    if step == self.max_iter or max(np.max(np.abs(grad_w)), abs(grad_b)) <= self.tol:
                        break
                    self.coef_ -= self.learning_rate * grad_w
                    self.intercept_ -= self.learning_rate * grad_b
                    self.n_iter_ += 1
        except FloatingPointError as exc:
            raise ValueError("Градиентный спуск расходится: уменьши learning_rate и масштабируй признаки") from exc
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = prediction_data(self, X)
        return X @ self.coef_ + self.intercept_
