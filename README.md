# Machine Learning Assignment 2: Bagging, Boosting and Bias–Variance on Face Recognition

**Name:** Harsh Prabha Bhaina  **Roll number:** M26ECM503  **Institute:** IIT Jodhpur  **Seed:** `26503` (all numeric characters of the roll number M26ECM503, in order)

**Repository:** https://github.com/harshprabha10-sudo/ML_Assignment2

**Final commit hash:** on page 1 of the report (a commit cannot contain its own hash).

Decision trees, bagging (from scratch) and multiclass SAMME AdaBoost (from scratch) on the Olivetti faces
(40 identities × 10 images, 64×64). Inputs are raw flattened pixels (4096 features), with no PCA or other
dimensionality reduction. `sklearn.tree.DecisionTreeClassifier` is used only as the base learner.
Bootstrap sampling, voting, OOB, SAMME weights and α, and the bias/variance estimates are all written by hand
in `src/`.

## Reproduce everything

```bash
# 0. get the code
git clone https://github.com/harshprabha10-sudo/ML_Assignment2.git
cd ML_Assignment2

# 1. environment (either one)
python -m pip install -r requirements.txt
#   or: conda env create -f environment.yml && conda activate ml-assignment2

# 2. full pipeline: Q1 -> Q2 -> Q3 -> figures (~1 h on 2 CPU cores; Q3 is ~50 min)
bash scripts/run_all.sh

# or one question at a time (run in order; each reads configs/selected.json written by the previous one)
python -m src.q1
python -m src.q2
python -m src.q3
python -m src.make_figures

# 3. unit checks for the from-scratch code (SAMME update, rejection of worse-than-random learners, bagging helpers)
python -m tests.test_samme_and_bagging
```

To run in Google Colab, open `run_in_colab.ipynb` and choose *Runtime → Run all* (it clones this repository).

**Data download.** The code calls `sklearn.datasets.fetch_olivetti_faces()`, which downloads the data on the
first run. If your network blocks the download, put a cached copy in a folder and set
`export SCIKIT_LEARN_DATA=/path/to/folder` (or set `"data_home"` in `configs/config.json`).

## Repository layout

| Path | Contents |
|---|---|
| `configs/config.json` | Seed, mask coordinates, every grid searched (input, edited by hand) |
| `configs/selected.json` | Frozen choices written by the code: tree setting (Q1a), ensemble size B (Q1c), boosting depth/rounds (Q2b) |
| `split/split.npz`, `split/split.json` | The saved train/val/test indices (6/2/2 per identity; 240/80/80). Every script reloads it and checks it against the seed |
| `src/common.py` | Data loading, split creation/verification, 16×16 centre mask (rows and cols 24–39), logging |
| `src/bagging.py` | `BaggingFromScratch`: bootstrap, majority vote, OOB accuracy, pairwise disagreement, vote margin |
| `src/samme.py` | `SAMME`: multiclass AdaBoost, α_t = log((1−e_t)/e_t) + log(K−1), q_k(x) = S_k(x)/Σα_t |
| `src/models.py` | Deterministic constructors and on-disk cache for the frozen models |
| `src/q1.py`, `src/q2.py`, `src/q3.py` | One script per question; each prints and logs every number the report uses |
| `src/make_figures.py` | Figure script: rebuilds every figure from `results/tables` |
| `results/logs/` | Final logs: `q1.log`, `q2.log`, `q3.log`, `run_all.log` (full console output, with library versions) |
| `results/tables/` | All numbers as CSV/JSON, plus `final_table.md` (Q3d) |
| `results/figures/` | All figures (PNG at 300 dpi and vector PDF) |
| `tests/` | Sanity checks for the from-scratch implementations |
| `requirements.txt`, `environment.yml` | Exact package versions used |

## Design decisions (so every number can be traced)

* **Split.** The script creates `rng = numpy.random.default_rng(26503)` once. For identity c = 0…39 in order, it
  draws `rng.permutation` of that identity's 10 indices: the first 6 go to train, the next 2 to validation and the
  last 2 to test. The **masked test set** is the clean test set with `img[24:40, 24:40] = 0`. Models are trained
  on clean images only.
* **Other randomness** (tree tie-breaking, bootstrap draws, Q3 resampling) uses seeds derived from the same seed
  through `numpy.random.SeedSequence([26503, stream, ...])` (`common.child_seed`), so one number controls
  everything.
* **Q1(a) selection:** highest validation accuracy. Ties go to the shallower depth, then the larger
  `min_samples_leaf` (the simpler tree).
* **Bagging:** each bootstrap sample has n = 240 draws with replacement. Prediction is a plurality vote, with ties
  going to the lowest class index. Ensembles are nested: the size-B model is the first B of 200 trees, so the curves
  over B are not confounded by fresh randomness. OOB accuracy is computed only on points that are out-of-bag for
  at least one tree (the count is logged). Disagreement and vote margin are measured on the validation set.
  B is selected by highest validation accuracy, with ties going to the smaller B.
* **Confidence:** for the tree, the max leaf class probability. For bagging, the winning vote fraction. For
  boosting, the max of the normalised weighted vote q(x).
* **SAMME weak learners that are no better than random** (e_t ≥ 1 − 1/K) are discarded. They never enter the
  vote, the event is logged and boosting stops (the weights would not change, so the next learner would be
  identical). A perfect learner has e_t clipped to 1e-10 and stops the run. See `tests/`, which forces both cases.
* **Q2(b) selection:** highest clean validation accuracy. Ties go to fewer rounds, then the shallower depth. The
  models for 10/25/50 rounds are prefixes of the 100-round run (SAMME is sequential, so the first T rounds are
  identical).
* **Q2(c)** fits every model fresh on one core so the training times are comparable. Its bagging ensembles
  therefore use different bootstrap draws from the nested Q1 ensembles, and the accuracies differ slightly.
* **Q3:** 30 training sets. For each identity, 6 faces are drawn with replacement from its 6 training faces. The
  tree, bagging and boosting settings are frozen (no re-tuning). The class vector p_r(x) is the one-hot
  prediction for the single tree, the vote-fraction vector for bagging, and q(x) for boosting.
  Bias² = (1/N) Σ_i ||p̄(x_i) − e_i||² and Variance = (1/NR) Σ_{i,r} ||p_r(x_i) − p̄(x_i)||², on the fixed clean
  test set (the masked set is also logged). Because the selected tree is fully grown (`max_depth=None`), no deeper
  tree exists, so Q3(b,c) sweeps depths 2, 8, 16 and None with a 100-tree bagged ensemble at each.
  Note when comparing models: boosting's q(x) is spread over many classes (its max is about 0.13), so its vectors
  are short. That is why its measured variance is small even though its refit accuracy varies a lot. Read the
  variance together with the s.d. of accuracy across refits.

## Results (seed 26503, produced by `bash scripts/run_all.sh`)

**Frozen choices** (`configs/selected.json`): tree `max_depth=None, min_samples_leaf=4`; bagging **B = 50**; SAMME weak-learner depth **4**, **50 rounds**.

**Q1(a) frozen tree:** train 99.17%, val 61.25%, test 56.25%; 40 leaves, 6.00 training faces per leaf, singleton-leaf fraction 0.000.

**Q1(b, c) bagging vs B** (disagreement and margin on validation):

| B | train | val | test | distinct/bootstrap (theory 151.89) | OOB | disagreement | vote margin |
|---|---|---|---|---|---|---|---|
| 1 | 59.17% | 46.25% | 37.50% | 153.00 | 16.09% | — | 1.000 |
| 5 | 93.75% | 61.25% | 55.00% | 151.80 | 31.96% | 0.786 | 0.245 |
| 10 | 99.58% | 65.00% | 70.00% | 153.90 | 38.72% | 0.766 | 0.285 |
| 25 | 100.00% | 78.75% | 77.50% | 152.16 | 52.08% | 0.780 | 0.274 |
| 50 | 100.00% | 85.00% | 78.75% | 151.74 | 61.25% | 0.792 | 0.264 |
| 100 | 100.00% | 82.50% | 81.25% | 151.90 | 71.25% | 0.786 | 0.273 |
| 200 | 100.00% | 82.50% | 83.75% | 151.66 | 72.92% | 0.788 | 0.273 |

**Q1(d), Q2(d): frozen models** (confidence = max leaf probability / winning vote fraction / max q):

| Model | clean acc | clean conf | masked acc | masked conf |
|---|---|---|---|---|
| Single tree | 56.25% | 0.990 | 21.25% | 1.000 |
| Bagging (B=50) | 78.75% | 0.398 | 53.75% | 0.251 |
| SAMME (depth 4, 50 rounds) | 53.75% | 0.129 | 42.50% | 0.124 |

**Q3(d) final table** (bias², variance over 30 within-identity resampled training sets, fixed clean test set):

| Model | Clean test acc | Masked test acc | Drop | Bias² | Variance | Refit clean acc (mean ± sd) |
|---|---|---|---|---|---|---|
| Single tree | 56.25% | 21.25% | 35.00 pp | 0.3935 | 0.6332 | 48.67 ± 5.03% |
| Bagging | 78.75% | 53.75% | 25.00 pp | 0.5150 | 0.0334 | 73.58 ± 3.98% |
| Boosting (SAMME) | 53.75% | 42.50% | 11.25 pp | 0.8751 | 0.0210 | 33.87 ± 7.06% |

**Q3(b, c) depth sweep:** bagging removes 84% (depth 2) to 96% (depth None) of the single-tree variance. The largest accuracy gain is at **depth 16 (+35.7 pp)**, not at the deepest tree (+27.7 pp). See `results/logs/q3.log`.

Full details are in `results/logs/*.log` and `results/tables/`.

## Report

The submitted report (CVPR two-column, 6 pages) uses only numbers and figures produced by this repository.
