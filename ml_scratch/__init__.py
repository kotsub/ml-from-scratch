"""Небольшие реализации классических алгоритмов только на NumPy."""

from .knn import KNNClassifier
from .linear import LinearRegressionGD
from .tree import DecisionTreeClassifier

__all__ = ["KNNClassifier", "LinearRegressionGD", "DecisionTreeClassifier"]
