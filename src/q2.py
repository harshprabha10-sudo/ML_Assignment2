"""Q2: multiclass SAMME AdaBoost from scratch.

Run from the repository root (after Q1):  python -m src.q2
"""
from __future__ import annotations

import time

import numpy as np
from joblib import Parallel, delayed

from .bagging import BaggingFromScratch
from .common import (TAB_DIR, acc, child_seed, get_logger, load_config, load_everything, load_selected, log_env,
                     save_csv, save_json, tree_params_from_selected, update_selected)
from .models import SEED_BAG, SEED_BOOST, bagging_max, boosting_full, frozen_tree
from .samme import SAMME


# --------------------------------------------------------------------------- helpers
def prefix_scores(model: SAMME, X, Ts):
    """Return {T: q(x)} for each prefix length T, computing learner predictions once."""
    P = model.learner_predictions(X)
    n = X.shape[0]
    S = np.zeros((n, model.K))
    out, Tset = {}, set(min(T, model.n_learners) for T in Ts)
    for t in range(model.n_learners):
        S[np.arange(n), P[t]] += model.alphas_[t]
        if t + 1 in Tset:
            out[t + 1] = S / model.alphas_[:t + 1].sum()
    return {T: out[min(T, model.n_learners)] for T in Ts}


def mask_region_features(cfg):
    a, b = cfg["mask_rows_cols"]
    s = cfg["image_size"]
    m = np.zeros((s, s), bool)
    m[a:b + 1, a:b + 1] = True
    return m.ravel()


def path_uses_mask(tree, x, maskfeat):
    """True if the decision path of sample x in `tree` tests at least one pixel inside the masked block."""
    node_idx = tree.decision_path(x.reshape(1, -1)).indices
    f = tree.tree_.feature[node_idx]
    f = f[f >= 0]
    return bool(maskfeat[f].any())


def split_fraction_in_mask(tree, maskfeat):
    f = tree.tree_.feature
    f = f[f >= 0]
    return float(maskfeat[f].mean()) if len(f) else 0.0


# --------------------------------------------------------------------------- (a) toy example
def toy_example(cfg, L):
    L.info("=== Q2(a) Toy 3-class SAMME example (9 samples, 1 feature, depth-1 stump) ===")
    X = np.arange(1, 10, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    K = 3
    log = []

    def round_logger(t, w, pred, miss, err, alpha, w_new):
        L.info(f"\n-- Round {t + 1} --")
        L.info(" i  x  y  w_before  h(x)  miss  w*exp(alpha*miss)  w_after(normalised)")
        unnorm = None if alpha is None else w * np.exp(alpha * miss)
        for i in range(len(y)):
            L.info(f"{i:2d} {X[i, 0]:2.0f} {y[i]:2d}  {w[i]:.5f}   {pred[i]:3d}   {int(miss[i]):3d}   "
                   f"{'-' if unnorm is None else f'{unnorm[i]:.5f}':>10}        "
                   f"{'-' if w_new is None else f'{w_new[i]:.5f}'}")
        L.info(f"weighted error e_t = sum(w*miss)/sum(w) = {err:.5f}   (random-guess threshold 1-1/K = {1 - 1 / K:.4f})")
        if alpha is None:
            L.info("learner rejected (not better than random)")
        else:
            L.info(f"alpha_t = log((1-e)/e) + log(K-1) = log({(1 - err) / err:.4f}) + log({K - 1}) = {alpha:.5f}")
            L.info(f"sum of unnormalised weights = {unnorm.sum():.5f}; weights after normalisation sum to {w_new.sum():.5f}")
        log.append({"round": t + 1, "w_before": w, "pred": pred, "miss": miss, "err": err, "alpha": alpha,
                    "w_unnorm": unnorm, "w_after": w_new})

    m = SAMME(cfg["q2"]["toy_max_depth"], 2, child_seed(cfg["seed"], 99), n_classes=K).fit(X, y, round_logger)
    q = m.scores(X)
    L.info(f"\nAfter {m.n_learners} rounds, normalised vote q(x) and prediction:")
    for i in range(len(y)):
        L.info(f"  x={X[i, 0]:.0f} y={y[i]} q={np.round(q[i], 3)} pred={q[i].argmax()}")
    save_json({"X": X.ravel(), "y": y, "rounds": log, "final_q": q}, TAB_DIR / "q2a_toy_samme.json")
    L.info("The face-data experiments below use exactly this SAMME class and update rule (src/samme.py).")


# --------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    L = get_logger("q2")
    log_env(L)
    D = load_everything(cfg, L)
    sel = load_selected()
    seed, q = cfg["seed"], cfg["q2"]
    t0 = time.time()
    toy_example(cfg, L)

    # ------------------------------------------------------------------ (b) grid
    L.info("\n=== Q2(b) SAMME on Olivetti: depth x rounds grid (selection on clean validation accuracy) ===")
    depths, Ts = q["weak_depths"], q["rounds_grid"]
    Tmax = max(Ts)
    models = dict(zip(depths, Parallel(n_jobs=min(len(depths), 4) if cfg["n_jobs"] != 1 else 1)(
        delayed(boosting_full)(cfg, D, d, Tmax) for d in depths)))
    rows = []
    for d in depths:
        m = models[d]
        L.info(f"depth {d}: {m.n_learners} learners kept ({m.stop_reason_})")
        sc = {k: prefix_scores(m, D[x], Ts) for k, x in [("tr", "Xtr"), ("va", "Xva"), ("te", "Xte")]}
        for T in Ts:
            r = {"depth": d, "rounds": T, "learners_used": min(T, m.n_learners),
                 "train_acc": acc(D["ytr"], sc["tr"][T].argmax(1)),
                 "val_acc": acc(D["yva"], sc["va"][T].argmax(1)),
                 "test_acc": acc(D["yte"], sc["te"][T].argmax(1)),
                 "mean_err_first_T": float(np.mean(m.hist_["err"][:T]))}
            rows.append(r)
            L.info(f"  depth={d} T={T:>3}: train={r['train_acc']:.4f} val={r['val_acc']:.4f} test={r['test_acc']:.4f}")
    save_csv(rows, TAB_DIR / "q2b_boosting_grid.csv")
    best = sorted(rows, key=lambda r: (-r["val_acc"], r["rounds"], r["depth"]))[0]
    bd, bT = best["depth"], best["rounds"]
    update_selected("boosting", {"max_depth": bd, "rounds": bT,
                                 "rule": "max clean validation accuracy; ties -> fewer rounds, then shallower"})
    sel = load_selected()
    L.info(f"Selected boosting: weak-learner depth {bd}, {bT} rounds (val {best['val_acc']:.4f}, test {best['test_acc']:.4f})")
    hist = {d: {k: v for k, v in models[d].hist_.items()} for d in depths}
    save_json({"selected": {"depth": bd, "rounds": bT}, "history": hist}, TAB_DIR / "q2b_histories.json")
    bm = models[bd]
    h = bm.hist_
    L.info("Selected model, per-round: t, weighted error, alpha, max sample weight, weight of 10 heaviest")
    for t in range(min(bT, bm.n_learners)):
        if t < 5 or (t + 1) % 10 == 0:
            L.info(f"  t={t + 1:3d} err={h['err'][t]:.4f} alpha={h['alpha'][t]:.4f} "
                   f"max_w={h['max_w'][t]:.4f} top10_w={h['top10_w'][t]:.4f}")

    # ------------------------------------------------------------------ (c) boosting vs bagging
    L.info("\n=== Q2(c) Boosting vs bagging with 25 / 50 / 100 learners (fresh fits, single core, timed) ===")
    tparams = tree_params_from_selected(sel)
    crow = []
    for n in q["compare_sizes"]:
        ts = time.perf_counter()
        bag = BaggingFromScratch(tparams, n, child_seed(seed, SEED_BAG), n_jobs=1).fit(D["Xtr"], D["ytr"])
        tbag = time.perf_counter() - ts
        ts = time.perf_counter()
        bst = SAMME(bd, n, child_seed(seed, SEED_BOOST, bd), keep_weight_history=False).fit(D["Xtr"], D["ytr"])
        tbst = time.perf_counter() - ts
        for name, mdl, tt in [("bagging", bag, tbag), ("boosting", bst, tbst)]:
            r = {"model": name, "n_learners": n, "train_acc": acc(D["ytr"], mdl.predict(D["Xtr"])),
                 "val_acc": acc(D["yva"], mdl.predict(D["Xva"])), "test_acc": acc(D["yte"], mdl.predict(D["Xte"])),
                 "fit_time_s": tt}
            crow.append(r)
            L.info(f"{name:8s} n={n:>3}: train={r['train_acc']:.4f} val={r['val_acc']:.4f} "
                   f"test={r['test_acc']:.4f} fit time={tt:.2f}s")
    save_csv(crow, TAB_DIR / "q2c_bagging_vs_boosting.csv")

    # heavy training face
    W = np.array(bm.w_hist_[:bT + 1])  # (T+1, n): weight before round 1 ... after round T
    T_ = W.shape[0] - 1
    i = int(W[1:].max(0).argmax())
    Ptr = bm.learner_predictions(D["Xtr"][i:i + 1], T_)[:, 0]
    miss = bm.miss_hist_[:T_, i]
    ytrue = int(D["ytr"][i])
    pred_counts = {int(c): int((Ptr == c).sum()) for c in np.unique(Ptr)}
    # which classes do the weak learners cover at all? (a depth-d tree has at most 2^d leaves -> <= 2^d classes)
    n_classes_out = [len(np.unique(hh.predict(D["Xtr"]))) for hh in bm.learners_[:T_]]
    face = {"train_pos": i, "dataset_index": int(D["train_idx"][i]), "true": ytrue,
            "weight_history": W[:, i], "uniform_weight": 1 / len(D["ytr"]), "max_weight": float(W[1:, i].max()),
            "rank_of_final_weight": int((W[-1] > W[-1, i]).sum()) + 1,
            "misclassified_by_learner": miss.astype(int), "learner_pred": Ptr, "alpha": bm.alphas_[:T_],
            "frac_rounds_misclassified": float(miss.mean()), "pred_counts": pred_counts,
            "final_q_true": float(bm.scores(D["Xtr"][i:i + 1], T_)[0, ytrue]),
            "final_pred": int(bm.predict(D["Xtr"][i:i + 1], T_)[0]),
            "mean_classes_output_per_learner": float(np.mean(n_classes_out))}
    save_json(face, TAB_DIR / "q2c_heavy_face.json")
    L.info(f"\nHeaviest training face: train#{i} (dataset idx {face['dataset_index']}), identity {ytrue}; "
           f"max weight {face['max_weight']:.4f} = {face['max_weight'] * len(D['ytr']):.1f}x uniform 1/n.")
    L.info(f"  misclassified by {int(miss.sum())}/{T_} weak learners; each learner outputs on average "
           f"{face['mean_classes_output_per_learner']:.1f} of 40 classes on the training set.")
    L.info("  round: weight-before, h_t(x), miss, alpha")
    for t in range(T_):
        if t < 15 or miss[t] == 0 or (t + 1) % 10 == 0:
            L.info(f"   t={t + 1:3d} w={W[t, i]:.5f} pred={Ptr[t]:2d} miss={int(miss[t])} alpha={bm.alphas_[t]:.3f}")
    L.info(f"  learner predictions for this face (class: count): {pred_counts}")
    L.info(f"  final ensemble prediction {face['final_pred']} with q_true={face['final_q_true']:.3f}")

    # ------------------------------------------------------------------ (d) clean vs masked for all three models
    L.info("\n=== Q2(d) Frozen models on clean and masked test sets ===")
    tree = frozen_tree(cfg, D, sel)
    Bsel = sel["bagging"]["B"]
    bag = bagging_max(cfg, D, sel, max(cfg["q1"]["bagging_B_grid"]), n_jobs=cfg["n_jobs"])
    yte = D["yte"]
    maskfeat = mask_region_features(cfg)
    res, store = {}, {}
    for name, Xs in [("clean", D["Xte"]), ("masked", D["Xte_mask"])]:
        tp, tc = tree.predict(Xs), tree.predict_proba(Xs).max(1)
        Pb = bag.tree_predictions(Xs, Bsel)
        Vb = BaggingFromScratch.votes_from_preds(Pb) / Bsel
        Pq = bm.learner_predictions(Xs, bT)
        Q = bm.scores(Xs, bT, P=Pq)
        store[name] = {"tp": tp, "Pb": Pb, "Vb": Vb, "Pq": Pq, "Q": Q}
        res[name] = {"tree_acc": acc(yte, tp), "tree_mean_conf": float(tc.mean()),
                     "bag_acc": acc(yte, Vb.argmax(1)), "bag_mean_win_vote": float(Vb.max(1).mean()),
                     "boost_acc": acc(yte, Q.argmax(1)), "boost_mean_win_q": float(Q.max(1).mean())}
        L.info(f"[{name}] tree acc={res[name]['tree_acc']:.4f} (mean max prob {res[name]['tree_mean_conf']:.3f}) | "
               f"bagging B={Bsel} acc={res[name]['bag_acc']:.4f} (mean winning vote {res[name]['bag_mean_win_vote']:.3f}) | "
               f"boosting acc={res[name]['boost_acc']:.4f} (mean winning q {res[name]['boost_mean_win_q']:.3f})")
    # structural use of the masked block
    bag_split_frac = float(np.mean([split_fraction_in_mask(t, maskfeat) for t in bag.trees_[:Bsel]]))
    bst_split_frac = float(np.average([split_fraction_in_mask(hh, maskfeat) for hh in bm.learners_[:bT]],
                                      weights=bm.alphas_[:bT]))
    res["fraction_of_split_nodes_in_mask"] = {"bagging_trees": bag_split_frac, "boosting_alpha_weighted": bst_split_frac,
                                              "mask_area_fraction": float(maskfeat.mean())}
    L.info(f"Fraction of split nodes testing masked pixels: bagging trees {bag_split_frac:.3f}, "
           f"boosting (alpha-weighted) {bst_split_frac:.3f}; masked area = {maskfeat.mean():.4f} of the image")

    c, m_ = store["clean"], store["masked"]
    bag_ok_c, bag_ok_m = c["Vb"].argmax(1) == yte, m_["Vb"].argmax(1) == yte
    bst_ok_c, bst_ok_m = c["Q"].argmax(1) == yte, m_["Q"].argmax(1) == yte
    cand = [(j, "bagging survives masking, boosting breaks") for j in np.flatnonzero(bag_ok_c & bag_ok_m & bst_ok_c & ~bst_ok_m)]
    cand += [(j, "boosting survives masking, bagging breaks") for j in np.flatnonzero(bag_ok_c & ~bag_ok_m & bst_ok_c & bst_ok_m)]
    if not cand:
        cand = [(j, "different response") for j in np.flatnonzero((bag_ok_m != bst_ok_m))]
    L.info(f"Faces where bagging and boosting respond differently to masking: {len(cand)} candidates "
           f"({sum(1 for _, k in cand if k.startswith('bagging'))} bagging-survives, "
           f"{sum(1 for _, k in cand if k.startswith('boosting'))} boosting-survives)")
    face_rows = []
    for j, kind in cand:
        yt = int(yte[j])
        xc, xm = D["Xte"][j], D["Xte_mask"][j]
        bag_flip = float(np.mean(c["Pb"][:, j] != m_["Pb"][:, j]))
        bst_flip_alpha = float(np.sum(bm.alphas_[:bT] * (c["Pq"][:, j] != m_["Pq"][:, j])) / bm.alphas_[:bT].sum())
        bag_path_mask = float(np.mean([path_uses_mask(t, xc, maskfeat) for t in bag.trees_[:Bsel]]))
        bst_path_mask = float(np.average([path_uses_mask(hh, xc, maskfeat) for hh in bm.learners_[:bT]],
                                         weights=bm.alphas_[:bT]))
        face_rows.append({
            "test_pos": int(j), "dataset_index": int(D["test_idx"][j]), "true": yt, "kind": kind,
            "bag_vote_true_clean": float(c["Vb"][j, yt]), "bag_vote_true_masked": float(m_["Vb"][j, yt]),
            "bag_pred_masked": int(m_["Vb"][j].argmax()), "bag_win_vote_masked": float(m_["Vb"][j].max()),
            "bag_distinct_preds_clean": int(len(np.unique(c["Pb"][:, j]))),
            "bag_distinct_preds_masked": int(len(np.unique(m_["Pb"][:, j]))),
            "bag_frac_trees_changed": bag_flip, "bag_frac_paths_through_mask": bag_path_mask,
            "boost_q_true_clean": float(c["Q"][j, yt]), "boost_q_true_masked": float(m_["Q"][j, yt]),
            "boost_pred_masked": int(m_["Q"][j].argmax()), "boost_win_q_masked": float(m_["Q"][j].max()),
            "boost_alpha_frac_changed": bst_flip_alpha, "boost_alpha_frac_paths_through_mask": bst_path_mask,
            "boost_train_weight_on_true_identity": float(W[-1][D["ytr"] == yt].sum()),
        })
    # most clear-cut example first: largest gap between the two models' true-class score on masked input
    face_rows.sort(key=lambda r: -abs((r["bag_vote_true_masked"] - r["bag_vote_true_clean"]) -
                                      (r["boost_q_true_masked"] - r["boost_q_true_clean"])))
    res["different_response_faces"] = face_rows
    save_json(res, TAB_DIR / "q2d_three_models.json")
    if face_rows:
        save_csv(face_rows, TAB_DIR / "q2d_different_response_faces.csv")
        shown = set()
        for r in face_rows:
            if r["kind"] in shown:
                continue
            shown.add(r["kind"])
            L.info(f"\nExample face: test#{r['test_pos']} (dataset idx {r['dataset_index']}), identity {r['true']} — {r['kind']}")
            for k, v in r.items():
                L.info(f"   {k}: {v}")
    L.info(f"\nQ2 total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
