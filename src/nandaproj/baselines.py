"""Direction 1: does the masked J-lens lie classifier beat simple baselines?

Pre-registered in `misc/prereg_detector_baselines.md`. 08's headline (balanced
0.94) mixed H, C1 and D rows, so a detector of the lie *instruction* scored
0.90 on it. Here every row carries the D prompt and the label is whether the
model lied, so the instruction is constant and cannot be what a detector reads.

Every detector goes through one path, `cv_scores`: a per-fold feature builder,
fold-internal standardisation, and the same L2 logistic. Detectors differ only
in their features, so a gap between two of them is a gap in what the features
carry, not in how they were fitted.

Numpy only; runs on the laptop from cached plan-2 artifacts.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from nandaproj import probe, tokenfeat

#: `(train_idx, test_idx) -> (X_train, X_test)`, unscaled. Anything learned from
#: data (a vocabulary, idf weights) must come from `train_idx` only.
FoldFeatures = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def balanced_weights(Y: np.ndarray) -> np.ndarray:
    """Per-column class-balancing sample weights, shaped like `Y`, mean 1 per column."""
    Y = np.asarray(Y, dtype=bool)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = len(Y)
    n_pos = np.maximum(Y.sum(0), 1)
    n_neg = np.maximum((~Y).sum(0), 1)
    return np.where(Y, n / (2 * n_pos), n / (2 * n_neg))


def fit_logistic_many(X: np.ndarray, Y: np.ndarray, l2: float = 1.0,
                      steps: int = 300, lr: float = 0.1,
                      weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`probe.fit_logistic` on every column of `Y` at once.

    The null refits the same features against 200 permuted labellings; with the
    features shared, the 200 fits are one `[d, R]` weight matrix and the cost
    of the null is close to the cost of one fit.

    `weights`, shaped like `Y`, scale each row's gradient (see `balanced_weights`);
    None leaves every row at weight 1.
    """
    Y = np.asarray(Y, dtype=X.dtype)       # float32 features keep the null fit in float32
    if Y.ndim == 1:
        Y = Y[:, None]
    if weights is not None:
        weights = np.asarray(weights, dtype=X.dtype).reshape(Y.shape)
    n, d = X.shape
    W = np.zeros((d, Y.shape[1]), dtype=X.dtype)
    b = np.zeros(Y.shape[1], dtype=X.dtype)
    for _ in range(steps):
        P = 1.0 / (1.0 + np.exp(-(X @ W + b)))
        G = P - Y
        if weights is not None:
            G = G * weights
        W -= lr * (X.T @ G / n + l2 * W / n)
        b -= lr * G.mean(0)
    return W, b


def cv_scores(features: FoldFeatures, Y: np.ndarray, groups: Sequence[str],
              l2: float = 1.0, folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
              balance: bool = False) -> np.ndarray:
    """Cross-validated decision scores `[n, R]`, one column per labelling in `Y`.

    Scores rather than labels, so AUROC is available; `scores > 0` is the
    prediction. `folds` defaults to leave-one-group-out; `balance` weights the
    training rows of each column so both classes count equally.
    """
    Y = np.asarray(Y, dtype=bool)
    if Y.ndim == 1:
        Y = Y[:, None]
    S = np.zeros(Y.shape)
    for train, test in (folds if folds is not None else probe.grouped_folds(groups)):
        Xtr, Xte = features(train, test)
        mu, sd = tokenfeat._standardize(Xtr)
        w = balanced_weights(Y[train]) if balance else None
        W, b = fit_logistic_many((Xtr - mu) / sd, Y[train], l2=l2, weights=w)
        S[test] = ((Xte - mu) / sd) @ W + b
    return S


# --------------------------------------------------------------------------
# Feature builders
# --------------------------------------------------------------------------

def fixed_features(X: np.ndarray) -> FoldFeatures:
    """Features that need nothing learned: a residual, a one-hot, a scalar."""
    X = np.asarray(X)
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float64)
    if X.ndim == 1:
        X = X[:, None]
    return lambda train, test: (X[train], X[test])


def topk_features(ids: np.ndarray, lp: np.ndarray, k_top: int,
                  banned: Sequence[int] = ()) -> FoldFeatures:
    """08's lens-token features: a vocabulary from the training rows' top-`k_top`."""
    def build(train, test):
        vocab = tokenfeat.union_vocab(ids[train], k_top, banned)
        return (tokenfeat.feature_matrix(ids[train], lp[train], vocab),
                tokenfeat.feature_matrix(ids[test], lp[test], vocab))
    return build


def words(text: str) -> list[str]:
    """Lower-cased word unigrams and bigrams."""
    unigrams = re.findall(r"[a-z0-9']+", text.lower())
    return unigrams + [f"{a} {b}" for a, b in itertools.pairwise(unigrams)]


def tfidf_features(texts: Sequence[str], min_df: int = 2) -> FoldFeatures:
    """TF-IDF of the prompt text, vocabulary and idf from the training rows.

    `min_df` drops terms seen in fewer than two training prompts: a term
    private to one prompt can only memorise that row.
    """
    bags = [Counter(words(t)) for t in texts]

    def build(train, test):
        df = Counter(term for i in train for term in bags[i])
        vocab = sorted(term for term, c in df.items() if c >= min_df)
        idf = np.log((1 + len(train)) / (1 + np.array([df[t] for t in vocab]))) + 1
        col = {t: j for j, t in enumerate(vocab)}

        def rows(idx):
            X = np.zeros((len(idx), len(vocab)))
            for r, i in enumerate(idx):
                for term, count in bags[i].items():
                    if term in col:
                        X[r, col[term]] = count
            X *= idf
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            return X / np.where(norms == 0, 1.0, norms)

        return rows(train), rows(test)
    return build


def one_hot(columns: Sequence[Sequence[str]]) -> np.ndarray:
    """`[n, sum of levels]` indicator matrix from categorical columns."""
    blocks = []
    for values in columns:
        values = [str(v) for v in values]          # a missing field (None) is its own level
        levels = sorted(set(values))
        blocks.append(np.array([[v == lv for lv in levels] for v in values], dtype=float))
    return np.hstack(blocks)


# --------------------------------------------------------------------------
# Metrics, intervals, null
# --------------------------------------------------------------------------

def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties; nan if a class is absent."""
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for s in np.unique(scores):                    # average ranks over ties
        tie = scores == s
        ranks[tie] = ranks[tie].mean()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def balanced(pred: np.ndarray, y: np.ndarray) -> float:
    """Balanced accuracy; nan when either class is absent (a resample can drop one)."""
    pred, y = np.asarray(pred, dtype=bool), np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return float("nan")
    return tokenfeat.balanced_accuracy(pred, y)


def summarize(scores: np.ndarray, y: np.ndarray, arm: np.ndarray) -> dict[str, float]:
    """Overall metrics plus balanced accuracy inside each polarity arm.

    `arm` is True for Yes-true rows. Inside one arm the truth is constant, so a
    detector that only reads polarity scores 0.5 there; `arm_mean` is the
    polarity-controlled number.
    """
    y, arm = np.asarray(y, dtype=bool), np.asarray(arm, dtype=bool)
    pred = scores > 0
    out = {"balanced": balanced(pred, y), "auroc": auroc(scores, y),
           "balanced_yes_true": balanced(pred[arm], y[arm]),
           "balanced_no_true": balanced(pred[~arm], y[~arm])}
    out["arm_mean"] = (out["balanced_yes_true"] + out["balanced_no_true"]) / 2
    return out


def bootstrap_ci(scores: np.ndarray, y: np.ndarray, arm: np.ndarray,
                 groups: Sequence[str], n_boot: int, rng: np.random.Generator,
                 alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """Percentile CI of every `summarize` metric, resampling whole pairs.

    The held-out scores are fixed; refitting inside each resample is out of
    budget. So the interval covers which pairs were drawn, not fit variance,
    and is somewhat narrow for that reason.
    """
    groups = np.asarray(groups)
    members = [np.flatnonzero(groups == g) for g in dict.fromkeys(groups)]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
        draws.append(summarize(scores[idx], y[idx], arm[idx]))
    return {m: (float(np.nanpercentile([d[m] for d in draws], 100 * alpha / 2)),
                float(np.nanpercentile([d[m] for d in draws], 100 * (1 - alpha / 2))))
            for m in draws[0]}


def permute_within_arms(y: np.ndarray, arm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels among rows with the same `arm` value.

    Keeps each arm's lie rate, so a detector that reads polarity keeps its
    score under the null; only signal beyond polarity is destroyed. `arm` may
    be any cell key (bool, or a combined code for several variables).
    """
    y = np.asarray(y, dtype=bool).copy()
    arm = np.asarray(arm)
    for a in np.unique(arm)[::-1]:               # True before False, as before
        idx = np.flatnonzero(arm == a)
        y[idx] = y[rng.permutation(idx)]
    return y


def null_p95(features: FoldFeatures, y: np.ndarray, arm: np.ndarray, groups: Sequence[str],
             n: int, rng: np.random.Generator) -> dict[str, float]:
    """p95 of `balanced` and `arm_mean` over `n` within-arm permutations, refit each time."""
    Ys = np.stack([permute_within_arms(y, arm, rng) for _ in range(n)], axis=1)
    S = cv_scores(features, Ys, groups)
    draws = [summarize(S[:, r], Ys[:, r], arm) for r in range(n)]
    return {m: float(np.nanpercentile([d[m] for d in draws], 95)) for m in ("balanced", "arm_mean")}
