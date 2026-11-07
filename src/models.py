"""Deterministic constructors for the frozen models, with an on-disk cache (results/models, git-ignored).

Every model is refit from the saved split and a seed derived from the roll-number seed, so deleting the
cache and rerunning gives identical models.
"""
from __future__ import annotations

import joblib
from sklearn.tree import DecisionTreeClassifier

from .bagging import BaggingFromScratch
from .common import MODEL_DIR, child_seed, tree_params_from_selected
from .samme import SAMME

# fixed stream ids for child seeds (documented in README)
SEED_TREE, SEED_BAG, SEED_BOOST, SEED_Q3 = 1, 2, 3, 4


def _cached(name, builder):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    p = MODEL_DIR / f"{name}.joblib"
    if p.exists():
        return joblib.load(p)
    m = builder()
    joblib.dump(m, p, compress=3)
    return m


def fit_tree(params, X, y, seed):
    return DecisionTreeClassifier(**params, random_state=child_seed(seed, SEED_TREE)).fit(X, y)


def frozen_tree(cfg, D, sel):
    p = tree_params_from_selected(sel)
    return _cached("frozen_tree", lambda: fit_tree(p, D["Xtr"], D["ytr"], cfg["seed"]))


def bagging_max(cfg, D, sel, B_max, n_jobs=-1):
    """Bagging model with B_max trees; its first B trees are the size-B ensemble."""
    p = tree_params_from_selected(sel)
    return _cached(
        f"bagging_{B_max}",
        lambda: BaggingFromScratch(p, B_max, child_seed(cfg["seed"], SEED_BAG), n_jobs=n_jobs).fit(D["Xtr"], D["ytr"]),
    )


def boosting_full(cfg, D, depth, rounds):
    return _cached(
        f"samme_d{depth}_T{rounds}",
        lambda: SAMME(depth, rounds, child_seed(cfg["seed"], SEED_BOOST, depth)).fit(D["Xtr"], D["ytr"]),
    )
