"""Builds every report figure from the saved tables in results/tables (no model fitting except a cached reload).

Run from the repository root (after q1, q2, q3):  python -m src.make_figures
"""
from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import FIG_DIR, TAB_DIR, load_config, load_everything, load_json, load_selected

# Fixed colour per entity (validated categorical slots 1-4); markers/labels give a non-colour cue too.
C = {"tree": "#2a78d6", "bag": "#eb6834", "boost": "#1baf7a", "aux": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False,
    "savefig.dpi": 300, "axes.axisbelow": True, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def rcsv(name):
    with open(TAB_DIR / name) as f:
        return list(csv.DictReader(f))


def fnum(x):
    return float("nan") if x in ("", None) else float(x)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)


def q1a(sel):
    rows = rcsv("q1a_tree_grid.csv")
    depths = ["2", "4", "8", "16", "None"]
    msl = sel["single_tree"]["min_samples_leaf"]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.3))
    sub = [r for r in rows if int(r["min_samples_leaf"]) == msl]
    x = np.arange(len(depths))
    for k, lab, col, mk in [("train_acc", "train", C["tree"], "o"), ("val_acc", "validation", C["bag"], "s"),
                            ("test_acc", "test", C["boost"], "^")]:
        ax[0].plot(x, [fnum(r[k]) for r in sub], marker=mk, color=col, label=lab)
    ax[0].set_xticks(x, depths)
    ax[0].set_xlabel("max_depth")
    ax[0].set_ylabel("accuracy")
    ax[0].set_title(f"Single tree, min_samples_leaf={msl}")
    ax[0].legend()
    for m, col, mk in [(1, C["tree"], "o"), (2, C["bag"], "s"), (4, C["boost"], "^")]:
        s = [r for r in rows if int(r["min_samples_leaf"]) == m]
        ax[1].plot(x, [fnum(r["n_leaves"]) for r in s], marker=mk, color=col, label=f"leaves, msl={m}")
    ax[1].set_xticks(x, depths)
    ax[1].set_xlabel("max_depth")
    ax[1].set_ylabel("number of leaves")
    ax[1].set_title("Tree size vs depth (240 training faces)")
    ax[1].legend()
    save(fig, "q1a_tree_depth")


def q1c(sel):
    rows = rcsv("q1bc_bagging_vs_B.csv")
    B = [int(r["B"]) for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(6.8, 4.0))
    ax = ax.ravel()
    for k, lab, col, mk in [("val_acc", "validation", C["bag"], "s"), ("test_acc", "test", C["boost"], "^"),
                            ("oob_acc", "out-of-bag", C["tree"], "o")]:
        ax[0].plot(B, [fnum(r[k]) for r in rows], marker=mk, color=col, label=lab)
    ax[0].set_title("Accuracy")
    ax[0].legend(loc="lower right")
    ax[1].plot(B, [fnum(r["disagreement_val"]) for r in rows], marker="o", color=C["tree"])
    ax[1].set_title("Pairwise disagreement (val)")
    ax[2].plot(B, [fnum(r["vote_margin_val"]) for r in rows], marker="o", color=C["tree"])
    ax[2].set_title("Mean vote margin (val)")
    ax[3].plot(B, [fnum(r["mean_distinct"]) for r in rows], marker="o", color=C["tree"], label="measured")
    ax[3].axhline(fnum(rows[0]["theory_distinct"]), color=MUTED, ls="--", lw=1, label="n(1-(1-1/n)^n)")
    ax[3].set_title("Distinct examples per bootstrap sample")
    ax[3].legend(loc="upper right")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("B (trees)")
        a.axvline(sel["bagging"]["B"], color=GRID, lw=3, zorder=0)
    fig.tight_layout()
    save(fig, "q1c_bagging_vs_B")


def faces_panel(D, cases, fname, title_fn, n_cols=4):
    fig, ax = plt.subplots(1, n_cols, figsize=(1.6 * n_cols, 1.95))
    for a, (img, title) in zip(np.atleast_1d(ax), cases):
        a.imshow(img, cmap="gray", vmin=0, vmax=1)
        a.set_title(title, fontsize=6.5)
        a.axis("off")
    save(fig, fname)


def q1d(D, cfg):
    from .common import mask_center
    cases = load_json(TAB_DIR / "q1d_cases.json")
    panels = []
    for key, lab in [("clean_bag_corrects_tree", "bag fixes tree"), ("clean_both_wrong", "both wrong")]:
        for r in cases[key][:2]:
            img = D["Xte"][r["test_pos"]].reshape(64, 64)
            top = r["bag_top3"].split(";")[0]
            panels.append((img, f"{lab}\ntrue {r['true']} | tree {r['tree_pred']}\nbag {r['bag_pred']} "
                                f"(votes {top.split(':')[1]}, true {r['bag_votes_true']})"))
    faces_panel(D, panels, "q1d_example_faces", None)
    # masked versions of the same 4 faces
    pm = [(mask_center(D["Xte"][[r["test_pos"]]], cfg).reshape(64, 64), f"masked test#{r['test_pos']}")
          for key in ("clean_bag_corrects_tree", "clean_both_wrong") for r in cases[key][:2]]
    faces_panel(D, pm, "q1d_example_faces_masked", None)


def q2b():
    h = load_json(TAB_DIR / "q2b_histories.json")
    s = h["selected"]
    hh = h["history"][str(s["depth"])]
    t = np.arange(1, len(hh["err"]) + 1)
    fig, ax = plt.subplots(2, 2, figsize=(6.8, 4.0))
    ax = ax.ravel()
    for a, k, title in [(ax[0], "err", "Weighted training error e_t"), (ax[1], "alpha", "Learner weight alpha_t"),
                        (ax[2], "max_w", "Max sample weight"), (ax[3], "top10_w", "Weight of 10 heaviest faces")]:
        a.plot(t, hh[k], color=C["boost"])
        a.axvline(s["rounds"], color=MUTED, ls="--", lw=0.8)
        a.set_title(title)
        a.set_xlabel("round t")
    ax[0].axhline(1 - 1 / 40, color=C["aux"], ls=":", lw=1.2, label="1-1/K")
    ax[0].legend(loc="lower right")
    ax[2].axhline(1 / 240, color=C["aux"], ls=":", lw=1.2, label="uniform 1/n")
    ax[2].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "q2b_boosting_training_curves")
    # grid heatmap of validation accuracy
    rows = rcsv("q2b_boosting_grid.csv")
    depths = sorted({int(r["depth"]) for r in rows})
    Ts = sorted({int(r["rounds"]) for r in rows})
    M = np.array([[fnum(next(r["val_acc"] for r in rows if int(r["depth"]) == d and int(r["rounds"]) == T))
                   for T in Ts] for d in depths])
    fig, a = plt.subplots(figsize=(2.8, 2.0))
    im = a.imshow(M, cmap="Blues", vmin=0, vmax=max(0.6, M.max()), aspect="auto")
    for i in range(len(depths)):
        for j in range(len(Ts)):
            a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if M[i, j] > 0.35 else INK)
    a.set_xticks(range(len(Ts)), Ts)
    a.set_yticks(range(len(depths)), depths)
    a.set_xlabel("rounds")
    a.set_ylabel("weak-learner depth")
    a.set_title("SAMME validation accuracy")
    a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.05)
    save(fig, "q2b_boosting_grid")


def q2c(D):
    rows = rcsv("q2c_bagging_vs_boosting.csv")
    ns = sorted({int(r["n_learners"]) for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.0))
    w = 0.36
    x = np.arange(len(ns))
    for off, m, col, lab in [(-w / 2, "bagging", C["bag"], "bagging"), (w / 2, "boosting", C["boost"], "boosting")]:
        ta = [fnum(next(r["test_acc"] for r in rows if r["model"] == m and int(r["n_learners"]) == n)) for n in ns]
        tt = [fnum(next(r["fit_time_s"] for r in rows if r["model"] == m and int(r["n_learners"]) == n)) for n in ns]
        b0 = ax[0].bar(x + off, ta, w * 0.94, color=col, label=lab)
        b1 = ax[1].bar(x + off, tt, w * 0.94, color=col, label=lab)
        ax[0].bar_label(b0, fmt="%.2f", fontsize=6, color=INK)
        ax[1].bar_label(b1, fmt="%.0fs", fontsize=6, color=INK)
    for a, t in [(ax[0], "Clean test accuracy"), (ax[1], "Training time (1 core)")]:
        a.set_xticks(x, ns)
        a.set_xlabel("number of learners")
        a.set_title(t)
        a.grid(axis="x", visible=False)
    ax[0].set_ylim(0, 1.12)
    ax[0].legend(loc="upper center", ncol=2)
    ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.12)
    fig.tight_layout()
    save(fig, "q2c_bagging_vs_boosting")

    f = load_json(TAB_DIR / "q2c_heavy_face.json")
    W = np.array(f["weight_history"])
    miss = np.array(f["misclassified_by_learner"])
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 1.95), gridspec_kw={"width_ratios": [1, 3.2], "wspace": 0.35})
    ax[0].imshow(D["Xtr"][f["train_pos"]].reshape(64, 64), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title(f"train face, identity {f['true']}", fontsize=7)
    ax[0].axis("off")
    t = np.arange(len(miss))
    ax[1].plot(np.arange(len(W)), W, color=C["boost"], label="weight before round t+1")
    ax[1].scatter(t[miss == 1], W[:-1][miss == 1], marker="x", color=C["bag"], s=14, label="learner t misclassifies")
    ax[1].scatter(t[miss == 0], W[:-1][miss == 0], marker="o", color=C["tree"], s=12, label="learner t correct")
    ax[1].axhline(f["uniform_weight"], color=MUTED, ls=":", lw=1)
    ax[1].set_xlabel("round")
    ax[1].set_ylabel("sample weight")
    ax[1].set_title("Boosting weight history of the heaviest training face")
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6)
    save(fig, "q2c_heavy_face")


def q2d(D, cfg):
    from .common import mask_center
    res = load_json(TAB_DIR / "q2d_three_models.json")
    fr = res.get("different_response_faces") or []
    fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.0), gridspec_kw={"width_ratios": [1, 1, 2.2]})
    names = ["tree", "bagging", "boosting"]
    keys = [("tree_acc", "tree"), ("bag_acc", "bag"), ("boost_acc", "boost")]
    if fr:
        r = fr[0]
        j = r["test_pos"]
        ax[0].imshow(D["Xte"][j].reshape(64, 64), cmap="gray", vmin=0, vmax=1)
        ax[1].imshow(mask_center(D["Xte"][[j]], cfg).reshape(64, 64), cmap="gray", vmin=0, vmax=1)
        ax[0].set_title(f"clean, identity {r['true']}", fontsize=7)
        ax[1].set_title("masked", fontsize=7)
    for a in ax[:2]:
        a.axis("off")
    x = np.arange(3)
    w = 0.36
    for off, s, hatch in [(-w / 2, "clean", None), (w / 2, "masked", "///")]:
        vals = [res[s][k] for k, _ in keys]
        bars = ax[2].bar(x + off, vals, w * 0.94, color=[C[c] for _, c in keys], hatch=hatch,
                         edgecolor="white", linewidth=0.5, label=s)
        ax[2].bar_label(bars, fmt="%.2f", fontsize=6, color=INK)
    ax[2].set_xticks(x, names)
    ax[2].set_title("Test accuracy: clean (solid) vs masked (hatched)")
    ax[2].grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax[2].legend(handles=[Patch(facecolor=MUTED, label="clean"),
                          Patch(facecolor=MUTED, hatch="///", edgecolor="white", label="masked")],
                 fontsize=6, loc="upper right")
    ax[2].set_ylim(0, 1.0)
    save(fig, "q2d_clean_vs_masked")


def q3():
    rows = rcsv("q3b_depth_sweep.csv")
    d = [r["depth"] for r in rows]
    x = np.arange(len(d))
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
    ax[0].plot(x, [fnum(r["tree_bias2"]) for r in rows], marker="o", color=C["tree"], label="single tree")
    ax[0].plot(x, [fnum(r["bag_bias2"]) for r in rows], marker="s", color=C["bag"], label="bagging (100)")
    ax[0].set_title("Bias$^2$")
    ax[1].plot(x, [fnum(r["tree_var"]) for r in rows], marker="o", color=C["tree"], label="single tree")
    ax[1].plot(x, [fnum(r["bag_var"]) for r in rows], marker="s", color=C["bag"], label="bagging (100)")
    ax[1].set_title("Variance")
    ax[2].plot(x, [fnum(r["acc_gain"]) for r in rows], marker="o", color=C["aux"], label="acc gain (bag - tree)")
    ax[2].plot(x, [fnum(r["var_removed"]) for r in rows], marker="^", color=C["boost"], label="variance removed")
    ax[2].set_title("What bagging buys")
    for a in ax:
        a.set_xticks(x, d)
        a.set_xlabel("max_depth")
        a.legend(fontsize=6)
    save(fig, "q3b_depth_sweep")
    f = rcsv("final_table.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))
    cols = [C["tree"], C["bag"], C["boost"]]
    labs = ["tree", "bagging", "boosting"]
    for a, k, t in [(ax[0], "bias2", "Bias$^2$ (clean test)"), (ax[1], "variance", "Variance (clean test)"),
                    (ax[2], "masking_drop", "Accuracy drop when masked")]:
        bars = a.bar(labs, [fnum(r[k]) for r in f], 0.6, color=cols)
        a.bar_label(bars, fmt="%.3f", fontsize=6, color=INK)
        a.set_title(t)
        a.grid(axis="x", visible=False)
    save(fig, "q3d_final_summary")


def main():
    cfg = load_config()
    sel = load_selected()
    D = load_everything(cfg)
    q1a(sel)
    q1c(sel)
    q1d(D, cfg)
    q2b()
    q2c(D)
    q2d(D, cfg)
    q3()
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
