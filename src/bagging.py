"""Bagging implemented from scratch on top of sklearn's DecisionTreeClassifier (only the base learner is sklearn)."""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from .common import N_CLASSES


def _fit_one(X, y, idx, tree_params, rs):
    t = DecisionTreeClassifier(**tree_params, random_state=rs)
    t.fit(X[idx], y[idx])
    return t


class BaggingFromScratch:
    """Bootstrap aggregation with majority (plurality) voting.

    * Each tree sees a bootstrap sample of size n (= training set size) drawn with replacement.
    * Prediction = class with the most votes; ties go to the smallest class index (np.argmax).
    * Trees are stored in order, so the first B trees form the ensemble of size B (nested ensembles).
    """

    def __init__(self, tree_params: dict, n_estimators: int, random_state: int, n_classes: int = N_CLASSES,
                 n_jobs: int = 1):
        self.tree_params = dict(tree_params)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_classes = n_classes
        self.n_jobs = n_jobs

    def fit(self, X, y):
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        self.boot_idx_ = [rng.integers(0, n, size=n) for _ in range(self.n_estimators)]
        self.tree_seeds_ = [int(rng.integers(0, 2**31 - 1)) for _ in range(self.n_estimators)]
        self.n_train_ = n
        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(X, y, idx, self.tree_params, rs) for idx, rs in zip(self.boot_idx_, self.tree_seeds_)
        )
        return self

    # ------------------------------------------------------------------ predictions
    def tree_predictions(self, X, B: int | None = None) -> np.ndarray:
        B = B or self.n_estimators
        return np.stack([t.predict(X) for t in self.trees_[:B]])  # (B, n)

    @staticmethod
    def votes_from_preds(P: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
        n = P.shape[1]
        V = np.zeros((n, n_classes), dtype=np.int64)
        for row in P:
            V[np.arange(n), row] += 1
        return V  # (n, K) vote counts

    def vote_counts(self, X, B=None):
        return self.votes_from_preds(self.tree_predictions(X, B), self.n_classes)

    def predict(self, X, B=None):
        return self.vote_counts(X, B).argmax(1)

    def vote_fraction(self, X, B=None):
        V = self.vote_counts(X, B)
        return V / V.sum(1, keepdims=True)

    # ------------------------------------------------------------------ diagnostics
    def distinct_counts(self, B=None):
        B = B or self.n_estimators
        return np.array([len(np.unique(i)) for i in self.boot_idx_[:B]])

    def oob_accuracy(self, Xtr, ytr, B=None, P_train=None):
        """OOB accuracy: each training point is predicted only by the trees whose bootstrap sample excluded it.
        Points that were in-bag for every tree (no OOB vote) are excluded; their count is returned too."""
        B = B or self.n_estimators
        if P_train is None:
            P_train = self.tree_predictions(Xtr, B)
        n = len(Xtr)
        V = np.zeros((n, self.n_classes), dtype=np.int64)
        for b in range(B):
            oob = np.ones(n, bool)
            oob[self.boot_idx_[b]] = False
            V[np.flatnonzero(oob), P_train[b, oob]] += 1
        has = V.sum(1) > 0
        if not has.any():
            return float("nan"), 0
        return float(np.mean(V[has].argmax(1) == ytr[has])), int(has.sum())


def pairwise_disagreement(P: np.ndarray) -> float:
    """Mean over all tree pairs of the fraction of samples on which the two trees predict different classes."""
    B = P.shape[0]
    if B < 2:
        return float("nan")
    tot, cnt = 0.0, 0
    for i in range(B - 1):
        d = (P[i + 1:] != P[i]).mean(1)
        tot += d.sum()
        cnt += len(d)
    return tot / cnt


def mean_vote_margin(V: np.ndarray) -> float:
    """Winning vote fraction minus runner-up vote fraction, averaged over samples."""
    F = V / V.sum(1, keepdims=True)
    s = np.sort(F, axis=1)
    return float(np.mean(s[:, -1] - s[:, -2]))
