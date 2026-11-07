"""Shared utilities: config, data loading, the saved split, masking, logging, I/O."""
from __future__ import annotations

import json
import logging
import os
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.json"
SELECTED_PATH = ROOT / "configs" / "selected.json"
SPLIT_DIR = ROOT / "split"
RESULTS = ROOT / "results"
LOG_DIR = RESULTS / "logs"
TAB_DIR = RESULTS / "tables"
FIG_DIR = RESULTS / "figures"
MODEL_DIR = RESULTS / "models"  # cached fitted models (git-ignored, regenerated deterministically)

N_CLASSES = 40


# --------------------------------------------------------------------------- config
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_selected() -> dict:
    if not SELECTED_PATH.exists():
        return {}
    with open(SELECTED_PATH) as f:
        return json.load(f)


def update_selected(key: str, value: dict) -> None:
    sel = load_selected()
    sel[key] = value
    save_json(sel, SELECTED_PATH)


def child_seed(seed: int, *keys: int) -> int:
    """Deterministic, independent integer seed derived from the roll-number seed."""
    ss = np.random.SeedSequence([seed, *keys])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


# --------------------------------------------------------------------------- logging
def get_logger(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(LOG_DIR / f"{name}.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def log_env(logger: logging.Logger) -> None:
    import joblib
    import matplotlib
    import sklearn

    logger.info(
        f"python {platform.python_version()} | numpy {np.__version__} | scikit-learn {sklearn.__version__} "
        f"| matplotlib {matplotlib.__version__} | joblib {joblib.__version__} | platform {platform.platform()}"
    )


# --------------------------------------------------------------------------- I/O
def _to_py(o):
    if isinstance(o, dict):
        return {str(k): _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return _to_py(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return None if np.isnan(v) else v
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_py(obj), f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_csv(rows: list[dict], path) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: _to_py(v) for k, v in r.items()})


def depth_name(d) -> str:
    return "None" if d is None else str(d)


# --------------------------------------------------------------------------- data
def load_faces(cfg: dict):
    """Olivetti faces via scikit-learn. X: (400, 4096) raw flattened pixels, y: (400,), images: (400, 64, 64)."""
    from sklearn.datasets import fetch_olivetti_faces

    data_home = cfg.get("data_home") or os.environ.get("SCIKIT_LEARN_DATA")
    d = fetch_olivetti_faces(data_home=data_home, shuffle=False)
    X = d.data.astype(np.float32)
    y = d.target.astype(np.int64)
    return X, y, d.images


def make_split(y: np.ndarray, seed: int, n_train=6, n_val=2, n_test=2):
    """Within each identity, randomly split its 10 images into 6/2/2 using default_rng(seed).

    Identities are processed in order 0..39 with ONE generator, so the split is fully determined by the seed.
    """
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        assert len(idx) == n_train + n_val + n_test
        perm = rng.permutation(idx)
        tr.extend(perm[:n_train])
        va.extend(perm[n_train:n_train + n_val])
        te.extend(perm[n_train + n_val:])
    return np.array(tr), np.array(va), np.array(te)


def get_split(cfg: dict, y: np.ndarray, logger=None):
    """Load the saved split if present (and verify it), otherwise create and save it."""
    path = SPLIT_DIR / "split.npz"
    s = cfg["split_per_identity"]
    fresh = make_split(y, cfg["seed"], s["train"], s["val"], s["test"])
    if path.exists():
        z = np.load(path)
        saved = (z["train_idx"], z["val_idx"], z["test_idx"])
        if int(z["seed"]) != cfg["seed"]:
            raise RuntimeError("split/split.npz was made with a different seed. Delete split/ to regenerate.")
        for a, b in zip(saved, fresh):
            if not np.array_equal(a, b):
                raise RuntimeError("Saved split does not match the split regenerated from the seed.")
        if logger:
            logger.info(f"Loaded saved split from {path.relative_to(ROOT)} (verified against seed {cfg['seed']}).")
        return saved
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_idx=fresh[0], val_idx=fresh[1], test_idx=fresh[2], seed=cfg["seed"])
    save_json(
        {
            "seed": cfg["seed"],
            "rule": "numpy.random.default_rng(seed); for identity c=0..39: perm=rng.permutation(indices of c); "
            "first 6 -> train, next 2 -> val, last 2 -> test",
            "train_idx": fresh[0], "val_idx": fresh[1], "test_idx": fresh[2],
        },
        SPLIT_DIR / "split.json",
    )
    if logger:
        logger.info(f"Created and saved split to {path.relative_to(ROOT)} and split/split.json.")
    return fresh


def mask_center(X: np.ndarray, cfg: dict) -> np.ndarray:
    """Zero the centred 16x16 block (rows and columns 24..39 inclusive) of each flattened 64x64 image."""
    a, b = cfg["mask_rows_cols"]
    s = cfg["image_size"]
    Xm = X.reshape(-1, s, s).copy()
    Xm[:, a:b + 1, a:b + 1] = 0.0
    return Xm.reshape(len(X), -1)


def load_everything(cfg: dict, logger=None):
    X, y, images = load_faces(cfg)
    tr, va, te = get_split(cfg, y, logger)
    D = {
        "X": X, "y": y, "images": images,
        "train_idx": tr, "val_idx": va, "test_idx": te,
        "Xtr": X[tr], "ytr": y[tr], "Xva": X[va], "yva": y[va], "Xte": X[te], "yte": y[te],
    }
    D["Xte_mask"] = mask_center(D["Xte"], cfg)
    if logger:
        logger.info(
            f"Data: X {X.shape}, train {len(tr)}, val {len(va)}, test {len(te)} (+ masked test {len(te)}); "
            f"seed {cfg['seed']}; raw flattened pixels, no dimensionality reduction."
        )
    return D


def acc(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def tree_params_from_selected(sel: dict) -> dict:
    t = sel["single_tree"]
    return {"max_depth": t["max_depth"], "min_samples_leaf": t["min_samples_leaf"]}
