import numpy as np
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as SklearnTree

from ml_scratch import DecisionTreeClassifier, KNNClassifier, LinearRegressionGD


def test_knn_matches_sklearn_on_iris():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    own = KNNClassifier(5).fit(X_train, y_train)
    reference = KNeighborsClassifier(5).fit(X_train, y_train)
    np.testing.assert_allclose(own.predict_proba(X_test), reference.predict_proba(X_test))
    np.testing.assert_array_equal(own.predict(X_test), reference.predict(X_test))


def test_knn_ties_and_string_labels():
    model = KNNClassifier(2).fit([[0], [2]], ["b", "a"])
    assert model.predict([[1]]).tolist() == ["a"]
    np.testing.assert_allclose(model.predict_proba([[1]]), [[0.5, 0.5]])
    assert KNNClassifier(1).fit([[0], [2]], ["b", "a"]).predict([[1]]).tolist() == ["b"]


def test_knn_owns_training_data():
    X = np.array([[0.], [10.]])
    model = KNNClassifier(1).fit(X, [0, 1])
    X[:] = 100
    assert model.predict([[0]]).item() == 0


@pytest.mark.parametrize("k", [0, -1, 3, 1.5, True])
def test_knn_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        KNNClassifier(k).fit([[0], [1]], [0, 1])


def test_linear_finds_known_solution_and_decreases_loss():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(100, 3))
    weights = np.array([2.0, -3.0, 0.5])
    y = X @ weights + 7
    model = LinearRegressionGD(tol=1e-10).fit(X, y)
    np.testing.assert_allclose(model.coef_, weights, atol=1e-8)
    assert model.intercept_ == pytest.approx(7, abs=1e-8)
    assert np.all(np.diff(model.loss_history_) <= 1e-10)
    assert len(model.loss_history_) == model.n_iter_ + 1


@pytest.mark.parametrize("l2", [0.0, 0.1, 1.0])
def test_linear_matches_analytical_solution(l2):
    X, y = make_regression(n_samples=150, n_features=4, noise=1, random_state=42)
    X = StandardScaler().fit_transform(X)
    own = LinearRegressionGD(l2=l2, tol=1e-10).fit(X, y)
    # sklearn Ridge: sum squared errors + alpha * ||w||².
    reference = (Ridge(alpha=len(X) * l2) if l2 else LinearRegression()).fit(X, y)
    np.testing.assert_allclose(own.predict(X), reference.predict(X), atol=1e-7)


def test_l2_does_not_penalize_intercept():
    model = LinearRegressionGD(l2=10, learning_rate=0.01).fit(np.zeros((20, 2)), np.full(20, 12.))
    np.testing.assert_allclose(model.coef_, 0)
    assert model.intercept_ == pytest.approx(12, abs=1e-6)


@pytest.mark.parametrize("params", [{"learning_rate": 0}, {"l2": -1}, {"max_iter": 0}, {"tol": -1}])
def test_linear_rejects_invalid_parameters(params):
    with pytest.raises(ValueError):
        LinearRegressionGD(**params).fit([[0], [1]], [0, 1])


def test_tree_simple_split_matches_reference():
    X, y = [[0], [1], [2], [3]], [0, 0, 1, 1]
    own = DecisionTreeClassifier(max_depth=1).fit(X, y)
    reference = SklearnTree(max_depth=1, random_state=42).fit(X, y)
    assert own.root_.threshold == pytest.approx(1.5)
    assert own.depth_ == 1 and own.n_leaves_ == 2
    np.testing.assert_allclose(own.predict_proba(X), reference.predict_proba(X))
    np.testing.assert_allclose(own.feature_importances_, [1])


def test_tree_multiclass_and_leaf_constraints():
    X, y = load_iris(return_X_y=True)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=6).fit(X, y)
    assert tree.depth_ <= 3
    assert (tree.predict(X) == y).mean() > 0.90
    np.testing.assert_allclose(tree.predict_proba(X).sum(axis=1), 1)
    def check(node):
        if node.feature is None:
            assert node.samples >= 6
        else:
            check(node.left)
            check(node.right)
    check(tree.root_)


def test_tree_constant_features_and_single_class():
    tree = DecisionTreeClassifier().fit(np.ones((4, 2)), [0, 1, 0, 1])
    assert tree.depth_ == 0
    np.testing.assert_allclose(tree.predict_proba([[1, 1]]), [[0.5, 0.5]])
    np.testing.assert_allclose(tree.feature_importances_, [0, 0])
    tree.fit([[0], [1]], ["only", "only"])
    assert tree.predict([[100]]).item() == "only"
    assert tree.n_leaves_ == 1


def test_tree_can_learn_xor():
    X, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    # Нулевой выигрыш на первом уровне разрешён: следующий уровень разделит XOR.
    assert tree.depth_ == 2
    np.testing.assert_array_equal(tree.predict(X), y)


@pytest.mark.parametrize("params", [{"max_depth": -1}, {"min_samples_leaf": 0}, {"min_samples_split": 1}])
def test_tree_rejects_invalid_parameters(params):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**params).fit([[0], [1]], [0, 1])


@pytest.mark.parametrize("factory", [KNNClassifier, DecisionTreeClassifier, LinearRegressionGD])
def test_fit_before_predict_and_shape_validation(factory):
    with pytest.raises(RuntimeError, match="fit"):
        factory().predict([[1]])
    model = factory().fit([[0], [1], [2], [3], [4]], [0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="Число признаков"):
        model.predict([[1, 2]])
    for bad_X in [[[np.nan]], [[np.inf]], [], [1, 2]]:
        with pytest.raises(ValueError):
            model.predict(bad_X)
    with pytest.raises(ValueError, match="y"):
        factory().fit([[0], [1]], [0])


@pytest.mark.parametrize("factory", [KNNClassifier, DecisionTreeClassifier, LinearRegressionGD])
def test_missing_target_is_rejected(factory):
    with pytest.raises(ValueError):
        factory().fit([[0], [1]], [0, np.nan])
