"""Sanity checks for the from-scratch implementations.  Run:  python -m tests.test_samme_and_bagging"""
import numpy as np

from src.bagging import BaggingFromScratch, mean_vote_margin, pairwise_disagreement
from src.samme import SAMME


def test_alpha_and_update():
    X = np.arange(1, 10, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    m = SAMME(1, 1, 0, n_classes=3).fit(X, y)
    assert np.isclose(m.hist_["err"][0], 1 / 3)
    assert np.isclose(m.alphas_[0], np.log(2) + np.log(2))
    w = m.w_hist_[1]
    assert np.isclose(w.sum(), 1) and np.isclose(w.max() / w.min(), 4)  # misclassified weights x exp(alpha)=4
    print("alpha / weight update OK")


def test_not_better_than_random_is_rejected():
    # constant feature -> the stump cannot split -> predicts one class -> e = 2/3 = 1 - 1/K -> must be rejected
    X = np.zeros((9, 1))
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    try:
        SAMME(1, 5, 0, n_classes=3).fit(X, y)
    except RuntimeError as e:
        print("first learner rejected as expected:", e)
    else:
        raise AssertionError("learner with e_t >= 1-1/K was not rejected")


def test_perfect_learner_stops():
    X = np.arange(1, 10, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    m = SAMME(3, 10, 0, n_classes=3).fit(X, y)
    assert m.n_learners == 1 and "perfect" in m.stop_reason_
    print("perfect learner handled:", m.stop_reason_)


def test_bagging_helpers():
    P = np.array([[0, 1, 2], [0, 1, 1], [0, 2, 2]])
    assert np.isclose(pairwise_disagreement(P), np.mean([1 / 3, 2 / 3, 1 / 3]))
    V = BaggingFromScratch.votes_from_preds(P, 3)
    assert V.tolist() == [[3, 0, 0], [0, 2, 1], [0, 1, 2]]
    assert np.isclose(mean_vote_margin(V), np.mean([1, 1 / 3, 1 / 3]))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    b = BaggingFromScratch({"max_depth": None, "min_samples_leaf": 1}, 300, 0, n_classes=2).fit(X, y)
    assert abs(b.distinct_counts().mean() / 200 - (1 - (1 - 1 / 200) ** 200)) < 0.01
    print("bagging helpers OK; distinct fraction", b.distinct_counts().mean() / 200)


if __name__ == "__main__":
    test_alpha_and_update()
    test_not_better_than_random_is_rejected()
    test_perfect_learner_stops()
    test_bagging_helpers()
    print("all tests passed")
