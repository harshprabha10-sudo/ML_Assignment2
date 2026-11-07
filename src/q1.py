"""Q1: decision-tree baseline and bagging from scratch.

Run from the repository root:  python -m src.q1
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .bagging import BaggingFromScratch, mean_vote_margin, pairwise_disagreement
from .common import (TAB_DIR, acc, child_seed, depth_name, get_logger, load_config, load_everything, log_env,
                     save_csv, save_json, update_selected, load_selected)
from .models import SEED_TREE, bagging_max, frozen_tree


def leaf_stats(tree, Xtr):
    leaves = tree.apply(Xtr)
    _, counts = np.unique(leaves, return_counts=True)
    n_leaves = tree.get_n_leaves()
    assert n_leaves == len(counts)
    return {
        "n_leaves": int(n_leaves),
        "mean_samples_per_leaf": float(counts.mean()),
        "frac_singleton_leaves": float(np.mean(counts == 1)),
        "actual_depth": int(tree.get_depth()),
    }


def depth_key(d):
    return 10**9 if d is None else d


def main():
    cfg = load_config()
    L = get_logger("q1")
    log_env(L)
    D = load_everything(cfg, L)
    seed, q = cfg["seed"], cfg["q1"]
    t0 = time.time()

    # ------------------------------------------------------------------ (a) single tree grid
    L.info("\n=== Q1(a) Decision-tree grid (selection on validation accuracy only) ===")
    rows = []
    for d in q["max_depth_grid"]:
        for m in q["min_samples_leaf_grid"]:
            t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=m, random_state=child_seed(seed, SEED_TREE))
            t.fit(D["Xtr"], D["ytr"])
            r = {"max_depth": depth_name(d), "min_samples_leaf": m,
                 "train_acc": acc(D["ytr"], t.predict(D["Xtr"])),
                 "val_acc": acc(D["yva"], t.predict(D["Xva"])),
                 "test_acc": acc(D["yte"], t.predict(D["Xte"])),
                 **leaf_stats(t, D["Xtr"])}
            rows.append(r)
            L.info(f"depth={r['max_depth']:>4} msl={m}  train={r['train_acc']:.4f} val={r['val_acc']:.4f} "
                   f"test={r['test_acc']:.4f} leaves={r['n_leaves']:3d} mean/leaf={r['mean_samples_per_leaf']:.2f} "
                   f"singleton={r['frac_singleton_leaves']:.3f} realdepth={r['actual_depth']}")
    save_csv(rows, TAB_DIR / "q1a_tree_grid.csv")

    # selection: highest validation accuracy; ties -> shallower depth, then larger min_samples_leaf (simpler tree)
    def parse(dn):
        return None if dn == "None" else int(dn)
    best = sorted(rows, key=lambda r: (-r["val_acc"], depth_key(parse(r["max_depth"])), -r["min_samples_leaf"]))[0]
    sel_tree = {"max_depth": parse(best["max_depth"]), "min_samples_leaf": best["min_samples_leaf"],
                "rule": "max validation accuracy; ties -> shallower depth, then larger min_samples_leaf"}
    update_selected("single_tree", sel_tree)
    sel = load_selected()
    L.info(f"\nFrozen tree setting: max_depth={best['max_depth']}, min_samples_leaf={best['min_samples_leaf']}")
    L.info(f"  train={best['train_acc']:.4f} val={best['val_acc']:.4f} test={best['test_acc']:.4f} "
           f"leaves={best['n_leaves']} mean samples/leaf={best['mean_samples_per_leaf']:.3f} "
           f"fraction singleton leaves={best['frac_singleton_leaves']:.3f}")
    L.info("Depth trend (at the frozen min_samples_leaf):")
    for r in rows:
        if r["min_samples_leaf"] == best["min_samples_leaf"]:
            L.info(f"  depth {r['max_depth']:>4}: train {r['train_acc']:.3f}, val {r['val_acc']:.3f}, "
                   f"leaves {r['n_leaves']}, mean/leaf {r['mean_samples_per_leaf']:.2f}, "
                   f"singleton {r['frac_singleton_leaves']:.2f}")

    tree = frozen_tree(cfg, D, sel)

    # ------------------------------------------------------------------ (b) bagging
    L.info("\n=== Q1(b) Bagging from scratch (bootstrap size = n_train, frozen tree setting) ===")
    Bs = q["bagging_B_grid"]
    Bmax = max(Bs)
    tb = time.time()
    bag = bagging_max(cfg, D, sel, Bmax, n_jobs=cfg["n_jobs"])
    L.info(f"Fitted {Bmax} bootstrap trees in {time.time() - tb:.1f}s (ensemble of size B = first B trees).")
    n = len(D["Xtr"])
    theory = n * (1 - (1 - 1 / n) ** n)
    L.info(f"Theoretical expected distinct examples per bootstrap: n(1-(1-1/n)^n) = {theory:.2f} "
           f"({theory / n:.4f} of n; limit 1-1/e = {1 - np.exp(-1):.4f})")

    P = {k: bag.tree_predictions(D[x]) for k, x in [("tr", "Xtr"), ("va", "Xva"), ("te", "Xte"), ("tm", "Xte_mask")]}
    brows = []
    for B in Bs:
        Vtr, Vva, Vte = (BaggingFromScratch.votes_from_preds(P[k][:B]) for k in ("tr", "va", "te"))
        dc = bag.distinct_counts(B)
        oob, n_oob = bag.oob_accuracy(D["Xtr"], D["ytr"], B, P_train=P["tr"][:B])
        r = {"B": B,
             "train_acc": acc(D["ytr"], Vtr.argmax(1)), "val_acc": acc(D["yva"], Vva.argmax(1)),
             "test_acc": acc(D["yte"], Vte.argmax(1)),
             "mean_distinct": float(dc.mean()), "std_distinct": float(dc.std()), "theory_distinct": theory,
             "oob_acc": oob, "n_oob_points": n_oob,
             "disagreement_val": pairwise_disagreement(P["va"][:B]),
             "vote_margin_val": mean_vote_margin(Vva)}
        brows.append(r)
        L.info(f"B={B:>3}: train={r['train_acc']:.4f} val={r['val_acc']:.4f} test={r['test_acc']:.4f} "
               f"distinct={r['mean_distinct']:.2f}±{r['std_distinct']:.2f} (theory {theory:.2f})")

    # ------------------------------------------------------------------ (c) OOB, disagreement, margin
    L.info("\n=== Q1(c) OOB accuracy, pairwise disagreement and vote margin (validation set) ===")
    for r in brows:
        L.info(f"B={r['B']:>3}: OOB acc={r['oob_acc'] if r['oob_acc'] is not None else float('nan'):.4f} "
               f"(on {r['n_oob_points']}/{n} pts)  disagreement={r['disagreement_val']:.4f}  "
               f"vote margin={r['vote_margin_val']:.4f}  val acc={r['val_acc']:.4f}")
    save_csv(brows, TAB_DIR / "q1bc_bagging_vs_B.csv")
    bestB = sorted(brows, key=lambda r: (-r["val_acc"], r["B"]))[0]["B"]
    update_selected("bagging", {"B": bestB, "rule": "max validation accuracy; ties -> smaller B",
                                "tree": sel["single_tree"]})
    sel = load_selected()
    L.info(f"Selected ensemble size B = {bestB}")

    # ------------------------------------------------------------------ (d) clean vs masked
    L.info("\n=== Q1(d) Frozen tree vs final bagged model on clean and masked test sets ===")
    yte = D["yte"]
    out = {}
    per_sample = []
    for name, Xs, kP in [("clean", D["Xte"], "te"), ("masked", D["Xte_mask"], "tm")]:
        tp = tree.predict(Xs)
        tconf = tree.predict_proba(Xs).max(1)
        V = BaggingFromScratch.votes_from_preds(P[kP][:bestB])
        bp = V.argmax(1)
        bconf = V.max(1) / bestB
        out[name] = {"tree_acc": acc(yte, tp), "tree_mean_conf": float(tconf.mean()),
                     "bag_acc": acc(yte, bp), "bag_mean_conf": float(bconf.mean()),
                     "tree_conf_correct": float(tconf[tp == yte].mean()) if (tp == yte).any() else None,
                     "tree_conf_wrong": float(tconf[tp != yte].mean()) if (tp != yte).any() else None,
                     "bag_conf_correct": float(bconf[bp == yte].mean()) if (bp == yte).any() else None,
                     "bag_conf_wrong": float(bconf[bp != yte].mean()) if (bp != yte).any() else None}
        L.info(f"[{name}] tree acc={out[name]['tree_acc']:.4f} mean conf (max leaf prob)={out[name]['tree_mean_conf']:.4f} | "
               f"bagging acc={out[name]['bag_acc']:.4f} mean conf (winning vote fraction)={out[name]['bag_mean_conf']:.4f}")
        L.info(f"         conf when correct/wrong: tree {out[name]['tree_conf_correct']}/{out[name]['tree_conf_wrong']}  "
               f"bag {out[name]['bag_conf_correct']}/{out[name]['bag_conf_wrong']}")
        for i in range(len(yte)):
            top = np.argsort(-V[i])[:3]
            per_sample.append({"set": name, "test_pos": i, "dataset_index": int(D["test_idx"][i]), "true": int(yte[i]),
                               "tree_pred": int(tp[i]), "tree_conf": float(tconf[i]),
                               "bag_pred": int(bp[i]), "bag_votes_true": int(V[i, yte[i]]),
                               "bag_top3": ";".join(f"{c}:{V[i, c]}" for c in top)})
    save_json(out, TAB_DIR / "q1d_clean_vs_masked.json")
    save_csv(per_sample, TAB_DIR / "q1d_per_sample.csv")

    def pick(setname, cond, k=2):
        rows_ = [r for r in per_sample if r["set"] == setname and cond(r)]
        # most informative first: largest vote share on the predicted class
        rows_.sort(key=lambda r: -int(r["bag_top3"].split(";")[0].split(":")[1]))
        return rows_[:k]

    cases = {}
    for setname in ("clean", "masked"):
        cases[f"{setname}_bag_corrects_tree"] = pick(setname, lambda r: r["tree_pred"] != r["true"] and r["bag_pred"] == r["true"])
        cases[f"{setname}_both_wrong"] = pick(setname, lambda r: r["tree_pred"] != r["true"] and r["bag_pred"] != r["true"])
    save_json(cases, TAB_DIR / "q1d_cases.json")
    for k, v in cases.items():
        L.info(f"\n{k}:")
        for r in v:
            L.info(f"  test#{r['test_pos']} (dataset idx {r['dataset_index']}) true={r['true']} tree_pred={r['tree_pred']} "
                   f"(p={r['tree_conf']:.2f}) bag_pred={r['bag_pred']} votes[true]={r['bag_votes_true']}/{bestB} "
                   f"top3 votes={r['bag_top3']}")
    n_fix = sum(1 for r in per_sample if r["set"] == "clean" and r["tree_pred"] != r["true"] and r["bag_pred"] == r["true"])
    n_brk = sum(1 for r in per_sample if r["set"] == "clean" and r["tree_pred"] == r["true"] and r["bag_pred"] != r["true"])
    L.info(f"\nClean test: bagging fixes {n_fix} tree errors and breaks {n_brk} tree-correct faces.")
    L.info(f"\nQ1 total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
