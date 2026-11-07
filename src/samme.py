"""Multiclass SAMME AdaBoost implemented from scratch (sklearn DecisionTreeClassifier is only the weak learner).

    alpha_t = log((1 - e_t) / e_t) + log(K - 1)
    S_k(x)  = sum_t alpha_t * 1[h_t(x) = k]
    q_k(x)  = S_k(x) / sum_t alpha_t          (normalised weighted vote, used as confidence)

Weak learners that are not better than random multiclass guessing (e_t >= 1 - 1/K, i.e. alpha_t <= 0) are
handled explicitly: the learner is discarded (never enters the vote), the event is recorded, and boosting stops,
because the sample weights would not change and the next learner would be identical.
A perfect learner (e_t = 0) gets e_t clipped to 1e-10, is kept, and boosting stops (all weights would vanish).
"""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .common import N_CLASSES

EPS = 1e-10


class SAMME:
    def __init__(self, max_depth: int, n_rounds: int, random_state: int, n_classes: int = N_CLASSES,
                 keep_weight_history: bool = True):
        self.max_depth = max_depth
        self.n_rounds = n_rounds
        self.random_state = random_state
        self.K = n_classes
        self.keep_weight_history = keep_weight_history

    def fit(self, X, y, round_logger=None):
        n, K = len(X), self.K
        w = np.full(n, 1.0 / n)
        rng = np.random.default_rng(self.random_state)
        self.learners_, self.alphas_ = [], []
        self.hist_ = {"err": [], "alpha": [], "max_w": [], "top10_w": [], "status": []}
        self.w_hist_ = [w.copy()] if self.keep_weight_history else None  # weights BEFORE each round + final
        self.miss_hist_ = []
        self.stop_reason_ = "completed all rounds"
        for t in range(self.n_rounds):
            h = DecisionTreeClassifier(max_depth=self.max_depth, random_state=int(rng.integers(0, 2**31 - 1)))
            h.fit(X, y, sample_weight=w)
            pred = h.predict(X)
            miss = (pred != y).astype(float)
            err = float(np.dot(w, miss) / w.sum())
            if err >= 1.0 - 1.0 / K:  # not better than random multiclass prediction
                self.hist_["status"].append("rejected: e_t >= 1-1/K")
                self.stop_reason_ = f"round {t + 1}: weak learner error {err:.4f} >= 1-1/K = {1 - 1 / K:.4f}; discarded, stopped"
                if round_logger:
                    round_logger(t, w, pred, miss, err, None, None)
                break
            perfect = err <= 0.0
            e = max(err, EPS)
            alpha = float(np.log((1 - e) / e) + np.log(K - 1))
            w_new = w * np.exp(alpha * miss)
            w_new /= w_new.sum()
            if round_logger:
                round_logger(t, w, pred, miss, err, alpha, w_new)
            self.learners_.append(h)
            self.alphas_.append(alpha)
            self.miss_hist_.append(miss.astype(bool))
            self.hist_["err"].append(err)
            self.hist_["alpha"].append(alpha)
            w = w_new
            self.hist_["max_w"].append(float(w.max()))
            self.hist_["top10_w"].append(float(np.sort(w)[-10:].sum()))
            self.hist_["status"].append("perfect: stopped" if perfect else "ok")
            if self.keep_weight_history:
                self.w_hist_.append(w.copy())
            if perfect:
                self.stop_reason_ = f"round {t + 1}: perfect weak learner (e_t=0, clipped); stopped"
                break
        if not self.learners_:
            raise RuntimeError("SAMME: the very first weak learner was no better than random; nothing to vote with.")
        self.alphas_ = np.array(self.alphas_)
        self.miss_hist_ = np.array(self.miss_hist_)
        self.final_w_ = w
        return self

    @property
    def n_learners(self):
        return len(self.learners_)

    def learner_predictions(self, X, T=None):
        T = min(T or self.n_learners, self.n_learners)
        return np.stack([h.predict(X) for h in self.learners_[:T]])  # (T, n)

    def scores(self, X, T=None, P=None):
        """Normalised weighted vote q(x) = S(x) / sum alpha, using the first T learners."""
        T = min(T or self.n_learners, self.n_learners)
        if P is None:
            P = self.learner_predictions(X, T)
        S = np.zeros((X.shape[0], self.K))
        n = X.shape[0]
        for t in range(T):
            S[np.arange(n), P[t]] += self.alphas_[t]
        return S / self.alphas_[:T].sum()

    def predict(self, X, T=None):
        return self.scores(X, T).argmax(1)
