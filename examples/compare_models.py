"""python -m examples.compare_models — метрики, CV-эксперименты и графики."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as SklearnTree

from ml_scratch import DecisionTreeClassifier, KNNClassifier, LinearRegressionGD

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def measure_classification(name, model, X_train, X_test, y_train, y_test):
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    prediction = model.predict(X_test)
    predict_seconds = perf_counter() - start
    return {"model": name, "accuracy": accuracy_score(y_test, prediction),
            "f1_macro": f1_score(y_test, prediction, average="macro"),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds}


def classification_cv(X, y):
    """Гиперпараметры исследуются только на train; scaler обучается внутри фолда."""
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    rows = []
    for algorithm, values in [("knn", [1, 3, 5, 9]), ("tree", [1, 2, 4, 8])]:
        for value in values:
            scores = []
            for train, validation in folds:
                X_fit, X_val = X[train], X[validation]
                if algorithm == "knn":
                    scaler = StandardScaler().fit(X_fit)
                    X_fit, X_val = scaler.transform(X_fit), scaler.transform(X_val)
                    model = KNNClassifier(value)
                else:
                    model = DecisionTreeClassifier(max_depth=value, min_samples_leaf=5)
                model.fit(X_fit, y[train])
                scores.append(accuracy_score(y[validation], model.predict(X_val)))
            rows.append({"algorithm": algorithm, "value": value,
                         "mean_cv_accuracy": float(np.mean(scores)), "std_cv_accuracy": float(np.std(scores))})
    return rows


def regularization_cv(X, y):
    rows = []
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(X))
    for l2 in [0.0, 0.01, 0.1, 1.0, 10.0]:
        scores, norms = [], []
        for train, validation in folds:
            scaler = StandardScaler().fit(X[train])
            model = LinearRegressionGD(l2=l2, learning_rate=0.02, max_iter=12000).fit(
                scaler.transform(X[train]), y[train])
            prediction = model.predict(scaler.transform(X[validation]))
            scores.append(float(np.sqrt(mean_squared_error(y[validation], prediction))))
            norms.append(float(np.linalg.norm(model.coef_)))
        rows.append({"l2": l2, "mean_cv_rmse": float(np.mean(scores)),
                     "std_cv_rmse": float(np.std(scores)), "mean_weight_norm": float(np.mean(norms))})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv_classification = classification_cv(X_train, y_train)
    best_k = max((row for row in cv_classification if row["algorithm"] == "knn"),
                 key=lambda row: row["mean_cv_accuracy"])["value"]
    best_depth = max((row for row in cv_classification if row["algorithm"] == "tree"),
                     key=lambda row: row["mean_cv_accuracy"])["value"]
    scaler = StandardScaler().fit(X_train)
    scaled_train, scaled_test = scaler.transform(X_train), scaler.transform(X_test)
    own_knn, sklearn_knn = KNNClassifier(best_k), KNeighborsClassifier(best_k)
    own_tree = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=5)
    sklearn_tree = SklearnTree(max_depth=best_depth, min_samples_leaf=5, random_state=SEED)
    classification = []
    for name, model, train, test in [
        ("NumPy kNN", own_knn, scaled_train, scaled_test),
        ("sklearn kNN", sklearn_knn, scaled_train, scaled_test),
        ("NumPy tree", own_tree, X_train, X_test),
        ("sklearn tree", sklearn_tree, X_train, X_test),
    ]:
        classification.append(measure_classification(name, model, train, test, y_train, y_test))
    knn_agreement = float(np.mean(own_knn.predict(scaled_test) == sklearn_knn.predict(scaled_test)))
    tree_agreement = float(np.mean(own_tree.predict(X_test) == sklearn_tree.predict(X_test)))

    X_reg, y_reg = load_diabetes(return_X_y=True)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=SEED)
    cv_regularization = regularization_cv(Xr_train, yr_train)
    reg_scaler = StandardScaler().fit(Xr_train)
    Xr_train, Xr_test = reg_scaler.transform(Xr_train), reg_scaler.transform(Xr_test)
    gd = LinearRegressionGD(learning_rate=0.05, max_iter=20000).fit(Xr_train, yr_train)
    reference = LinearRegression().fit(Xr_train, yr_train)
    regression = []
    for name, model in [("NumPy GD", gd), ("sklearn OLS", reference)]:
        prediction = model.predict(Xr_test)
        regression.append({"model": name, "rmse": float(np.sqrt(mean_squared_error(yr_test, prediction))),
                           "r2": r2_score(yr_test, prediction)})
    result = {
        "seed": SEED, "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn", "matplotlib"]},
        "classification_dataset": "sklearn breast_cancer (569 rows, 30 numeric features)",
        "regression_dataset": "sklearn diabetes (442 rows, 10 numeric features)",
        "classification_split": {"train": len(y_train), "test": len(y_test), "stratified": True},
        "regression_split": {"train": len(yr_train), "test": len(yr_test)},
        "selected_on_train_cv": {"knn_neighbors": best_k, "tree_max_depth": best_depth, "tree_min_samples_leaf": 5},
        "classification": classification, "knn_prediction_agreement": knn_agreement,
        "tree_prediction_agreement": tree_agreement, "regression": regression,
        "gd_iterations": gd.n_iter_, "gd_final_loss": gd.loss_history_[-1],
        "classification_cv": cv_classification, "regularization_cv": cv_regularization,
        "note": "GD vs OLS uses l2=0 fixed in advance; the L2 curve is a separate train-only CV experiment.",
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(gd.loss_history_)
    axes[0].set(xlabel="GD updates", ylabel="MSE / 2", title="Train loss")
    axes[0].set_yscale("log")
    for algorithm, label in [("knn", "kNN: k"), ("tree", "Tree: max_depth")]:
        rows = [row for row in cv_classification if row["algorithm"] == algorithm]
        axes[1].plot([r["value"] for r in rows], [r["mean_cv_accuracy"] for r in rows], "o-", label=label)
    axes[1].set(xlabel="Hyperparameter value", ylabel="Mean CV accuracy", title="Train-only classification CV")
    axes[1].legend(fontsize=8)
    axes[2].plot([r["l2"] for r in cv_regularization], [r["mean_cv_rmse"] for r in cv_regularization], "o-")
    axes[2].set_xscale("symlog", linthresh=0.01)
    axes[2].set(xlabel="L2 strength", ylabel="Mean CV RMSE", title="Train-only regression CV")
    fig.tight_layout()
    fig.savefig(args.output / "experiments.png", dpi=150)
    plt.close(fig)
    lines = ["# Измеренные результаты", "", "Команда: `python -m examples.compare_models`.", "",
             "Классификация: breast_cancer, 80/20 stratified split, seed=42. "
             "k и глубина выбраны на 5-fold CV обучающей части.", "",
             f"Выбраны k={best_k}, max_depth={best_depth}, min_samples_leaf=5.", "",
             "| Модель | Test accuracy | Test F1 macro |", "|---|---:|---:|"]
    for row in classification:
        lines.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['f1_macro']:.4f} |")
    lines += ["", f"Совпадение test-предсказаний: kNN — {knn_agreement:.1%}, дерево — {tree_agreement:.1%}.",
              "Разные правила разрешения равных разбиений и float32 в sklearn могут давать разные деревья.", "",
              "Регрессия: diabetes, 80/20 split, seed=42. Оба варианта без регуляризации; "
              "StandardScaler обучен только на train.", "", "| Модель | Test RMSE | Test R² |", "|---|---:|---:|"]
    for row in regression:
        lines.append(f"| {row['model']} | {row['rmse']:.4f} | {row['r2']:.4f} |")
    lines += ["", "## Что показывает эксперимент", "",
              "kNN должен совпадать с библиотекой при одинаковых соседях; масштабирование существенно "
              "для евклидова расстояния. Дерево ищет осевые пороги по Gini и умеет ограничивать "
              "глубину и размер листа. GD приближает решение МНК постепенно, поэтому результат "
              "зависит от числа итераций и learning rate.", "",
              "Отдельная кривая L2 получена на CV внутри train. При большой регуляризации "
              "коэффициенты сжимаются, но качество может ухудшиться из-за недообучения. "
              "Для сравнения GD с МНК заранее использован l2=0.", "",
              "Один split не характеризует все данные; train CV std не является доверительным интервалом. "
              "Время в JSON — одиночное измерение на текущем CPU, не строгий бенчмарк.", "",
              "![Эксперименты](experiments.png)", "",
              "Точные значения, версии и все CV-точки: [metrics.json](metrics.json).", ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"classification": classification, "regression": regression,
                      "selected": result["selected_on_train_cv"]}, indent=2))
    return result


if __name__ == "__main__":
    main()
