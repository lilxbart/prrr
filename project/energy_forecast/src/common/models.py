"""Собственные реализации моделей машинного обучения для задачи регрессии.

Используются только numpy – чтобы проект можно было запустить без
внешних ML-библиотек. Интерфейс совпадает со стилем scikit-learn
(методы fit / predict), что упрощает миграцию на LightGBM/sklearn
в production-контейнере.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Линейные модели
# ---------------------------------------------------------------------------

class LinearRegression:
    """Линейная регрессия методом наименьших квадратов (closed-form)."""

    def __init__(self):
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X1 = np.hstack([np.ones((X.shape[0], 1)), X])
        # решение через lstsq – устойчиво к плохо обусловленным X
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


class Ridge:
    """Гребневая регрессия (L2) через нормальные уравнения."""

    def __init__(self, alpha: float = 1.0, standardize: bool = True):
        self.alpha = alpha
        self.standardize = standardize
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.standardize:
            self._mu = X.mean(axis=0)
            self._sigma = X.std(axis=0)
            self._sigma[self._sigma == 0] = 1.0
            Xs = (X - self._mu) / self._sigma
        else:
            Xs = X
        y_mean = y.mean()
        yc = y - y_mean
        n_features = Xs.shape[1]
        A = Xs.T @ Xs + self.alpha * np.eye(n_features)
        b = Xs.T @ yc
        coef = np.linalg.solve(A, b)
        self.coef_ = coef
        self.intercept_ = float(y_mean)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.standardize and self._mu is not None:
            Xs = (X - self._mu) / self._sigma
        else:
            Xs = X
        return Xs @ self.coef_ + self.intercept_


# ---------------------------------------------------------------------------
# Простое решающее дерево (для регрессии)
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    feature: int = -1
    threshold: float = 0.0
    value: float = 0.0
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class DecisionTreeRegressor:
    """Быстрое numpy-решающее дерево регрессии.

    Для ускорения используется бинаризация признаков по квантилям
    (max_bins). Критерий – минимизация дисперсии (эквивалент MSE).
    """

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 20,
                 max_bins: int = 64, feature_subsample: float = 1.0,
                 random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.feature_subsample = feature_subsample
        self.rng = np.random.default_rng(random_state)
        self.root: Optional[_Node] = None
        self.n_features_: int = 0
        self._bin_edges: list[np.ndarray] = []

    # --- подготовка бинов ---
    def _prepare_bins(self, X: np.ndarray) -> None:
        self._bin_edges = []
        q = np.linspace(0, 1, self.max_bins + 1)[1:-1]
        for j in range(X.shape[1]):
            col = X[:, j]
            edges = np.unique(np.quantile(col, q))
            self._bin_edges.append(edges)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
        n, d = X.shape
        n_try = max(1, int(self.feature_subsample * d))
        feats = self.rng.choice(d, size=n_try, replace=False) if n_try < d else np.arange(d)

        best_gain = -np.inf
        best_feat = -1
        best_thr = 0.0
        y_sum = y.sum()
        y_sq_sum = (y * y).sum()
        parent_var = y_sq_sum - (y_sum * y_sum) / n

        for j in feats:
            edges = self._bin_edges[j]
            if edges.size == 0:
                continue
            col = X[:, j]
            order = np.argsort(col)
            col_s = col[order]
            y_s = y[order]
            csum = np.cumsum(y_s)
            csq = np.cumsum(y_s * y_s)
            idxs = np.searchsorted(col_s, edges, side="right")
            for k, thr in zip(idxs, edges):
                left_n = int(k)
                right_n = n - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                left_sum = csum[left_n - 1]
                right_sum = y_sum - left_sum
                left_sq = csq[left_n - 1]
                right_sq = y_sq_sum - left_sq
                left_var = left_sq - (left_sum * left_sum) / left_n
                right_var = right_sq - (right_sum * right_sum) / right_n
                gain = parent_var - (left_var + right_var)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(j)
                    best_thr = float(thr)
        return best_feat, best_thr, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node(value=float(y.mean()))
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return node
        feat, thr, gain = self._best_split(X, y)
        if feat < 0 or gain <= 1e-9:
            return node
        mask = X[:, feat] <= thr
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return node
        node.feature = feat
        node.threshold = thr
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        node.value = 0.0
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]
        self._prepare_bins(X)
        self.root = self._build(X, y, 0)
        return self

    def _predict_one(self, row: np.ndarray) -> float:
        node = self.root
        while node is not None and node.left is not None:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value if node is not None else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(r) for r in X], dtype=float)


# ---------------------------------------------------------------------------
# Градиентный бустинг на собственных деревьях
# ---------------------------------------------------------------------------

class GradientBoostingRegressor:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 5, min_samples_leaf: int = 30,
                 subsample: float = 1.0, max_bins: int = 64,
                 feature_subsample: float = 1.0,
                 random_state: Optional[int] = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_bins = max_bins
        self.feature_subsample = feature_subsample
        self.random_state = random_state
        self.trees: list[DecisionTreeRegressor] = []
        self.init_: float = 0.0
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.init_ = float(y.mean())
        pred = np.full_like(y, self.init_)
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        importances = np.zeros(X.shape[1])
        for i in range(self.n_estimators):
            residual = y - pred
            if self.subsample < 1.0:
                idx = rng.choice(n, size=int(self.subsample * n), replace=False)
                Xs, rs = X[idx], residual[idx]
            else:
                Xs, rs = X, residual
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_bins=self.max_bins,
                feature_subsample=self.feature_subsample,
                random_state=(self.random_state or 0) + i,
            )
            tree.fit(Xs, rs)
            update = tree.predict(X)
            pred += self.learning_rate * update
            self.trees.append(tree)
            # feature importance: суммарное снижение дисперсии
            self._accumulate_importance(tree.root, importances)
        self.feature_importances_ = importances / max(importances.sum(), 1e-9)
        return self

    def _accumulate_importance(self, node, acc):
        if node is None or node.left is None:
            return
        acc[node.feature] += 1.0
        self._accumulate_importance(node.left, acc)
        self._accumulate_importance(node.right, acc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.init_)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred
