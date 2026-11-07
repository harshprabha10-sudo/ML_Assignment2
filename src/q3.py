"""Q3: bias and variance as measured quantities.

30 training sets are drawn by sampling WITH replacement, within each identity, six faces from that identity's
original six training faces. The selected single tree, final bagged model and selected boosting model are refit
on each (no re-tuning). The clean test set (and, for completeness, the masked test set) is fixed.

Class vectors p_r(x):  single tree -> one-hot of its prediction;  bagging -> vote fractions;
                       boosting -> normalised weighted vote q(x).
    Bias^2   = 1/N  sum_i ||pbar(x_i) - e_i||^2
    Variance = 1/NR sum_{i,r} ||p_r(x_i) - pbar(x_i)||^2

Run from the repository root (after Q1 and Q2):  python -m src.q3
"""
from __future__ import annotations

import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from .bagging import BaggingFromScratch, pairwise_disagreement
from .common import (N_CLASSES, TAB_DIR, acc, child_seed, depth_name, get_logger, load_config, load_everything,
                     load_selected, load_json, log_env, save_csv, save_json)
from .models import SEED_Q3
from .samme import SAMME


def one_hot(pred, K=N_CLASSES):
    P = np.zeros((len(pred), K))
    P[np.arange(len(pred)), pred] = 1.0
    return P


def bias_variance(Pr: np.ndarray, y: np.ndarray):
    """Pr: (R, N, K) class vectors. Returns bias^2, variance and per-repetition accuracies."""
    pbar = Pr.mean(0)
    E = one_hot(y, Pr.shape[2])
    bias2 = float(np.mean(np.sum((pbar - E) ** 2, axis=1)))
    var = float(np.mean(np.sum((Pr - pbar[None]) ** 2, axis=2)))
    accs = (Pr.argmax(2) == y[None]).mean(1)
    return bias2, var, accs


def resample_within_identity(ytr: np.ndarray, rng) -> np.ndarray:
    idx = []
    for c in np.unique(ytr):
        pos = np.flatnonzero(ytr == c)
        idx.extend(rng.choice(pos, size=len(pos), replace=True))
    return np.array(idx)


def leaf_occupancy(tree, X):
    _, cnt = np.unique(tree.apply(X), return_counts=True)
    return float(cnt.mean()), float(np.mean(cnt == 1)), int(tree.get_n_leaves())


def one_repetition(r, D, cfg, sel, depths, n_bag_depth):
    seed = cfg["seed"]
    rng = np.random.default_rng(child_seed(seed, SEED_Q3, r))
    idx = resample_within_identity(D["ytr"], rng)
    X, y = D["Xtr"][idx], D["ytr"][idx]
    msl = sel["single_tree"]["min_samples_leaf"]
    sel_depth = sel["single_tree"]["max_depth"]
    Bsel = sel["bagging"]["B"]
    bd, bT = sel["boosting"]["max_depth"], sel["boosting"]["rounds"]
    sets = {"clean": D["Xte"], "masked": D["Xte_mask"]}
    out = {"n_distinct_train": int(len(np.unique(idx))), "vec": {}, "stats": {}}

    for d in depths:
        tp = {"max_depth": d, "min_samples_leaf": msl}
        t = DecisionTreeClassifier(**tp, random_state=child_seed(seed, SEED_Q3, r, 1)).fit(X, y)
        nb = max(n_bag_depth, Bsel) if d == sel_depth else n_bag_depth
        bag = BaggingFromScratch(tp, nb, child_seed(seed, SEED_Q3, r, 2), n_jobs=1).fit(X, y)
        mean_per_leaf, singleton, n_leaves = leaf_occupancy(t, X)
        bag_leaves = [leaf_occupancy(bt, X[bi]) for bt, bi in zip(bag.trees_[:n_bag_depth], bag.boot_idx_)]
        st = {"tree_n_leaves": n_leaves, "tree_mean_per_leaf": mean_per_leaf, "tree_singleton_frac": singleton,
              "bagtree_mean_n_leaves": float(np.mean([b[2] for b in bag_leaves])),
              "bagtree_mean_per_leaf": float(np.mean([b[0] for b in bag_leaves])),
              "bagtree_singleton_frac": float(np.mean([b[1] for b in bag_leaves]))}
        for sname, Xs in sets.items():
            P = bag.tree_predictions(Xs)
            out["vec"][(f"tree_d{depth_name(d)}", sname)] = one_hot(t.predict(Xs))
            out["vec"][(f"bag{n_bag_depth}_d{depth_name(d)}", sname)] = \
                BaggingFromScratch.votes_from_preds(P[:n_bag_depth]) / n_bag_depth
            if sname == "clean":
                st["bag_disagreement_test"] = pairwise_disagreement(P[:n_bag_depth])
            if d == sel_depth:
                out["vec"][("bagging_selected", sname)] = BaggingFromScratch.votes_from_preds(P[:Bsel]) / Bsel
        out["stats"][depth_name(d)] = st

    bst = SAMME(bd, bT, child_seed(seed, SEED_Q3, r, 3), keep_weight_history=False).fit(X, y)
    for sname, Xs in sets.items():
        out["vec"][("boosting_selected", sname)] = bst.scores(Xs)
        out["vec"][("tree_selected", sname)] = out["vec"][(f"tree_d{depth_name(sel_depth)}", sname)]
    return out


def main():
    cfg = load_config()
    L = get_logger("q3")
    log_env(L)
    D = load_everything(cfg, L)
    sel = load_selected()
    q = cfg["q3"]
    R, nbd = q["n_repetitions"], q["bagged_trees_per_depth"]
    sel_depth = sel["single_tree"]["max_depth"]
    depths = list(q["depths"])
    if sel_depth not in depths:
        depths.append(sel_depth)
    depths = sorted(set(depths), key=lambda d: 10**9 if d is None else d)
    L.info(f"Frozen settings (no re-tuning): tree {sel['single_tree']}, bagging B={sel['bagging']['B']}, "
           f"boosting {sel['boosting']}")
    L.info(f"Depths for Q3(b,c): {[depth_name(d) for d in depths]} (selected depth = {depth_name(sel_depth)}); "
           f"{nbd}-tree bagged ensemble at each depth; R = {R} resampled training sets.")
    if sel_depth is None:
        L.info("Note: the selected tree is fully grown (max_depth=None), so no deeper tree exists; "
               "shallow and intermediate depths provide the contrast.")
    t0 = time.time()
    reps = Parallel(n_jobs=cfg["n_jobs"], verbose=5)(
        delayed(one_repetition)(r, D, cfg, sel, depths, nbd) for r in range(R))
    L.info(f"Fitted all repetitions in {time.time() - t0:.1f}s")
    L.info(f"Distinct training faces per resampled set: mean {np.mean([o['n_distinct_train'] for o in reps]):.1f} of 240")

    yte = D["yte"]
    keys = sorted(reps[0]["vec"].keys())
    summary = {}
    for k in keys:
        Pr = np.stack([o["vec"][k] for o in reps])
        b2, v, accs = bias_variance(Pr, yte)
        summary[f"{k[0]}|{k[1]}"] = {"model": k[0], "set": k[1], "bias2": b2, "variance": v,
                                      "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
                                      "accs": accs}
    save_json(summary, TAB_DIR / "q3_bias_variance_all.json")

    # ------------------------------------------------------------------ (a)
    L.info("\n=== Q3(a) Selected models, fixed clean test set ===")
    rows_a = []
    for m in ("tree_selected", "bagging_selected", "boosting_selected"):
        s = summary[f"{m}|clean"]
        rows_a.append({k: s[k] for k in ("model", "bias2", "variance", "acc_mean", "acc_std")})
        L.info(f"{m:18s} bias^2={s['bias2']:.4f} variance={s['variance']:.4f} "
               f"test acc over {R} refits = {s['acc_mean']:.4f} ± {s['acc_std']:.4f}")
    save_csv(rows_a, TAB_DIR / "q3a_selected_models.csv")

    # ------------------------------------------------------------------ (b) + (c)
    L.info(f"\n=== Q3(b) Depth sweep: single tree vs {nbd}-tree bagging (clean test) ===")
    rows_b = []
    for d in depths:
        dn = depth_name(d)
        st = {k: float(np.mean([o["stats"][dn][k] for o in reps])) for k in reps[0]["stats"][dn]}
        t, b = summary[f"tree_d{dn}|clean"], summary[f"bag{nbd}_d{dn}|clean"]
        r = {"depth": dn, "is_selected": d == sel_depth,
             "tree_bias2": t["bias2"], "tree_var": t["variance"], "tree_acc": t["acc_mean"], "tree_acc_std": t["acc_std"],
             "bag_bias2": b["bias2"], "bag_var": b["variance"], "bag_acc": b["acc_mean"], "bag_acc_std": b["acc_std"],
             "var_removed": t["variance"] - b["variance"],
             "var_removed_pct": 100 * (t["variance"] - b["variance"]) / t["variance"] if t["variance"] > 0 else 0.0,
             "acc_gain": b["acc_mean"] - t["acc_mean"], **st}
        rows_b.append(r)
        L.info(f"depth {dn:>4}{' (selected)' if d == sel_depth else '':11s}: tree bias^2={r['tree_bias2']:.4f} "
               f"var={r['tree_var']:.4f} acc={r['tree_acc']:.4f} | bag bias^2={r['bag_bias2']:.4f} var={r['bag_var']:.4f} "
               f"acc={r['bag_acc']:.4f} | variance removed {r['var_removed']:.4f} ({r['var_removed_pct']:.1f}%) "
               f"| acc gain {r['acc_gain']:+.4f}")
    save_csv(rows_b, TAB_DIR / "q3b_depth_sweep.csv")

    L.info("\n=== Q3(c) Is the deepest tree where bagging helps most? ===")
    for r in rows_b:
        L.info(f"depth {r['depth']:>4}: tree leaves {r['tree_n_leaves']:.1f}, mean train samples/leaf "
               f"{r['tree_mean_per_leaf']:.2f}, singleton-leaf frac {r['tree_singleton_frac']:.3f}; "
               f"bagged-tree leaves {r['bagtree_mean_n_leaves']:.1f}, samples/leaf {r['bagtree_mean_per_leaf']:.2f}; "
               f"pairwise disagreement {r['bag_disagreement_test']:.3f}; var removed {r['var_removed']:.4f}; "
               f"acc gain {r['acc_gain']:+.4f}")
    best = max(rows_b, key=lambda r: r["acc_gain"])
    deepest = rows_b[-1]
    L.info(f"Largest bagging accuracy gain at depth {best['depth']} ({best['acc_gain']:+.4f}); "
           f"deepest depth {deepest['depth']} gain {deepest['acc_gain']:+.4f}. "
           f"Deepest == best: {best['depth'] == deepest['depth']}")
    gains = [r["acc_gain"] for r in rows_b]
    order = sorted(range(len(gains)), key=lambda i: -gains[i])
    if len(order) > 1:
        L.info(f"Gap between best and second-best depth: {gains[order[0]] - gains[order[1]]:+.4f} "
               f"(std of single-tree acc at best depth {best['tree_acc_std']:.4f}, over {R} refits)")

    # ------------------------------------------------------------------ (d) final table
    L.info("\n=== Q3(d) Final table ===")
    q2d = load_json(TAB_DIR / "q2d_three_models.json")
    final = []
    for label, m, ka in [("Single tree", "tree_selected", "tree_acc"), ("Bagging", "bagging_selected", "bag_acc"),
                         ("Boosting (SAMME)", "boosting_selected", "boost_acc")]:
        c, mk = summary[f"{m}|clean"], summary[f"{m}|masked"]
        row = {"model": label,
               "clean_test_acc": q2d["clean"][ka], "masked_test_acc": q2d["masked"][ka],
               "masking_drop": q2d["clean"][ka] - q2d["masked"][ka],
               "bias2": c["bias2"], "variance": c["variance"],
               "refit_clean_acc_mean": c["acc_mean"], "refit_clean_acc_std": c["acc_std"],
               "refit_masked_acc_mean": mk["acc_mean"], "refit_masked_acc_std": mk["acc_std"],
               "masked_bias2": mk["bias2"], "masked_variance": mk["variance"]}
        final.append(row)
    save_csv(final, TAB_DIR / "final_table.csv")
    lines = ["| Model | Clean test acc | Masked test acc | Drop | Bias² | Variance | Refit clean acc (mean ± sd) |",
             "|---|---|---|---|---|---|---|"]
    for r in final:
        lines.append(f"| {r['model']} | {100 * r['clean_test_acc']:.2f}% | {100 * r['masked_test_acc']:.2f}% | "
                     f"{100 * r['masking_drop']:.2f} pp | {r['bias2']:.4f} | {r['variance']:.4f} | "
                     f"{100 * r['refit_clean_acc_mean']:.2f} ± {100 * r['refit_clean_acc_std']:.2f}% |")
    (TAB_DIR / "final_table.md").write_text("\n".join(lines) + "\n")
    for ln in lines:
        L.info(ln)
    stab = min(final, key=lambda r: r["refit_clean_acc_std"])
    lowv = min(final, key=lambda r: r["variance"])
    sens = max(final, key=lambda r: r["masking_drop"])
    besta = max(final, key=lambda r: r["clean_test_acc"])
    L.info(f"Lowest refit accuracy s.d.: {stab['model']}; lowest measured variance: {lowv['model']}; "
           f"largest masking drop: {sens['model']}; best clean test acc: {besta['model']}. "
           f"Lowest-variance model has best accuracy: {lowv['model'] == besta['model']}")
    L.info(f"\nQ3 total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
