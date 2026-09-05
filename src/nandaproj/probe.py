"""Linear probes on the answer-slot residual, without the lens in the loop.

04 read "belief" through the J-lens and the readout turned out to track which
token the lens can decode at a layer, not what is true. This module asks the
same question with the standard instrument instead: is the true answer
*linearly decodable* from `h_l` at the answer slot, and does that hold under
the deceptive prompt on the items where the model lied?

The v2 bank makes this sharper than the usual truth probe. Every item has a
twin sharing its context byte-for-byte with the question reversed, so the two
twins carry opposite labels over identical context. A probe scored
leave-one-pair-out therefore cannot get credit for recognising the context; it
has to read how the question was combined with it.

Numpy only. `d_model` is 2560 and `n` is at most a few hundred, so a closed
form or a few hundred gradient steps is all that is needed, and nothing here
should require the box.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: A fitted probe: weight vector, bias. `score = X @ w + b`, positive = class 1.
Probe = tuple[np.ndarray, float]
Fit = Callable[[np.ndarray, np.ndarray], Probe]


# --------------------------------------------------------------------------
# Two fits. Mean-difference is the robust one at n << d; logistic is the
# conventional one. Both are reported, neither is tuned per layer.
# --------------------------------------------------------------------------

def fit_mean_diff(X: np.ndarray, y: np.ndarray) -> Probe:
    """Mass-mean probe: `w = mean(X | y=1) - mean(X | y=0)`, threshold halfway.

    No covariance estimate, so it cannot overfit 2560 dimensions on 100 points
    the way a discriminant would, and its direction is exactly the object 04b
    called `d_yesno` when the labels are the answer token.
    """
    y = np.asarray(y).astype(bool)
    if not y.any() or y.all():
        raise ValueError("fit needs both classes present")
    w = X[y].mean(0) - X[~y].mean(0)
    proj = X @ w
    b = -0.5 * (proj[y].mean() + proj[~y].mean())
    return w, float(b)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 steps: int = 300, lr: float = 0.1) -> Probe:
    """L2-regularised logistic regression by plain gradient descent.

    `X` is expected standardised (see `standardize`), which is what makes one
    `lr` and one `l2` serve every layer. Features are 2560-wide and rows are
    ~100, so this converges in a few hundred steps and needs no solver.
    """
    y = np.asarray(y).astype(float)
    if y.min() == y.max():
        raise ValueError("fit needs both classes present")
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return w, float(b)


def standardize(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and std from the *training* rows only."""
    mu = X_train.mean(0)
    sd = X_train.std(0) + 1e-6
    return mu, sd


def predict(probe: Probe, X: np.ndarray) -> np.ndarray:
    w, b = probe
    return (X @ w + b) > 0


def accuracy(probe: Probe, X: np.ndarray, y: np.ndarray) -> float:
    return float((predict(probe, X) == np.asarray(y).astype(bool)).mean())


# --------------------------------------------------------------------------
# Folds and nulls that respect the pair structure
# --------------------------------------------------------------------------

def grouped_folds(groups: Sequence[str]) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-group-out: every row of a group is held out together.

    With `groups = pair_id` this is leave-one-pair-out. Holding out one twin
    while training on the other would let the probe learn that context and
    score the held-out twin by elimination -- the leak this exists to close.
    """
    groups = np.asarray(groups)
    for g in dict.fromkeys(groups):          # first-seen order, deterministic
        test = np.nonzero(groups == g)[0]
        train = np.nonzero(groups != g)[0]
        yield train, test


def cv_accuracy(X: np.ndarray, y: np.ndarray, groups: Sequence[str],
                fit: Fit = fit_mean_diff, scale: bool = True) -> float:
    """Leave-one-group-out accuracy of `fit`, standardising inside each fold."""
    y = np.asarray(y).astype(bool)
    correct = 0
    for train, test in grouped_folds(groups):
        Xtr, Xte = X[train], X[test]
        if scale:
            mu, sd = standardize(Xtr)
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        probe = fit(Xtr, y[train])
        correct += int((predict(probe, Xte) == y[test]).sum())
    return correct / len(y)


def cv_logistic(X: np.ndarray, y: np.ndarray, groups: Sequence[str], l2: float = 1.0,
                steps: int = 300, lr: float = 0.1) -> float:
    """`cv_accuracy(..., fit=fit_logistic)` with every fold trained at once.

    Same folds, same standardisation-inside-the-fold, same optimiser and the
    same numbers to float tolerance -- but the 50 folds are a leading axis of
    one `[F, n, d]` array, so the 300 steps are 300 batched matmuls instead of
    15,000 small ones. On the box this is the difference between 38 s and well
    under a second per layer.
    """
    y = np.asarray(y).astype(np.float32)
    folds = list(grouped_folds(groups))
    n, d = X.shape
    F = len(folds)
    train_mask = np.ones((F, n), dtype=np.float32)
    for f, (_, test) in enumerate(folds):
        train_mask[f, test] = 0.0
    n_train = train_mask.sum(1, keepdims=True)                       # [F, 1]

    # Per-fold standardisation without materialising a [F, n, d] array: with
    # Z_f = (X - mu_f) / sd_f, both Z_f @ w_f and Z_f^T g_f reduce to matmuls on
    # the shared X plus rank-1 corrections, so every step is two [n, d] x [d, F]
    # products instead of a 50 MB sweep. Same arithmetic, same answer.
    Xf = X.astype(np.float32)                                        # [n, d]
    M = train_mask.T                                                 # [n, F]
    nt = n_train.T                                                   # [1, F]
    mu = (Xf.T @ M) / nt                                             # [d, F]
    ex2 = (Xf.T ** 2 @ M) / nt
    sd = np.sqrt(np.maximum(ex2 - mu ** 2, 0.0)) + 1e-6              # [d, F]

    W = np.zeros((d, F), dtype=np.float32)
    b = np.zeros((1, F), dtype=np.float32)
    yy = y[:, None]                                                  # [n, 1]

    def logits(W, b):
        V = W / sd                                                   # [d, F]
        return Xf @ V - (mu * V).sum(0, keepdims=True) + b           # [n, F]

    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-logits(W, b)))
        G = (p - yy) * M                                             # held-out rows contribute 0
        gW = (Xf.T @ G - mu * G.sum(0, keepdims=True)) / sd          # = Z_f^T g_f, per fold
        W -= lr * (gW / nt + l2 * W / nt)
        b -= lr * G.sum(0, keepdims=True) / nt

    scores = logits(W, b)                                            # [n, F]
    correct = 0
    for f, (_, test) in enumerate(folds):
        correct += int(((scores[test, f] > 0) == (y[test] > 0.5)).sum())
    return correct / n


def transfer(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
             fit: Fit = fit_mean_diff, scale: bool = True) -> np.ndarray:
    """Fit on one condition's residuals, return predictions on another's.

    The primary Exp 3 measurement: train on H, where truth and emitted answer
    coincide, and read what the probe says about the D residual of an item
    where they do not. Standardisation comes from the training rows only.
    """
    if scale:
        mu, sd = standardize(X_train)
        X_train, X_test = (X_train - mu) / sd, (X_test - mu) / sd
    return predict(fit(X_train, np.asarray(y_train).astype(bool)), X_test)


def flip_within_pairs(y: np.ndarray, groups: Sequence[str],
                      rng: np.random.Generator) -> np.ndarray:
    """The pair-respecting label permutation.

    Twins carry opposite labels by construction, so a global shuffle would
    produce label sets no bank could have and make the null too easy. Flipping
    the *whole pair* with probability 1/2 keeps every pair opposite and every
    context balanced, and only breaks the link between question and label.
    """
    y = np.asarray(y).astype(bool).copy()
    groups = np.asarray(groups)
    for g in dict.fromkeys(groups):
        if rng.random() < 0.5:
            m = groups == g
            y[m] = ~y[m]
    return y


def permutation_null(X: np.ndarray, y: np.ndarray, groups: Sequence[str],
                     fit: Fit = fit_mean_diff, n: int = 200, seed: int = 0,
                     scale: bool = True) -> np.ndarray:
    """`n` leave-one-pair-out accuracies under pair-flipped labels."""
    rng = np.random.default_rng(seed)
    return np.array([cv_accuracy(X, flip_within_pairs(y, groups, rng), groups,
                                 fit=fit, scale=scale) for _ in range(n)])


# --------------------------------------------------------------------------
# Residual store: one array per condition, [n_layers, n_items, d_model]
# --------------------------------------------------------------------------

@dataclass
class Residuals:
    item_ids: list[str]
    layers: list[int]
    by_condition: dict[str, np.ndarray]      # cond -> [L, n, d] float32

    def X(self, condition: str, layer: int) -> np.ndarray:
        return self.by_condition[condition][self.layers.index(layer)]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez_compressed(
            path, item_ids=np.array(self.item_ids), layers=np.array(self.layers),
            conditions=np.array(list(self.by_condition)),
            **{f"h_{c}": v.astype(np.float16) for c, v in self.by_condition.items()})
        return path

    @classmethod
    def load(cls, path: str | Path) -> Residuals:
        d = np.load(path)
        conds = [str(c) for c in d["conditions"]]
        return cls(item_ids=[str(i) for i in d["item_ids"]],
                   layers=[int(l) for l in d["layers"]],
                   by_condition={c: d[f"h_{c}"].astype(np.float32) for c in conds})


def layer_sweep(res: Residuals, layers: Sequence[int],
                score: Callable[[int], float]) -> dict[int, float]:
    """`{layer: score(layer)}` -- a tiny helper so notebook cells read as tables."""
    return {l: float(score(l)) for l in layers}


# --------------------------------------------------------------------------
# 04e: signed projections onto a probe axis, and the mass of a direction
# inside a subspace. Both are the arithmetic behind a figure, kept here so
# the figure's numbers have a test.
# --------------------------------------------------------------------------

def unit(w: np.ndarray) -> np.ndarray:
    """`w / ||w||`."""
    w = np.asarray(w, dtype=float)
    return w / np.linalg.norm(w)


def signed_projection(probe: Probe, X: np.ndarray) -> np.ndarray:
    """Signed distance of every row from the probe's decision boundary.

    `(X @ w + b) / ||w||`: positive means the probe reads class 1 (Yes, or
    "said Yes"), zero is the threshold, and the unit is one step along the
    probe axis in whatever space `X` lives in. Scale-free in the probe, so a
    mean-difference fit and the same fit times ten give the same picture.
    """
    w, b = probe
    return (X @ w + b) / np.linalg.norm(w)


def subspace_mass(V: np.ndarray, W: np.ndarray, ks: Sequence[int]) -> np.ndarray:
    """`rho_k(w) = ||V_k^T w||^2 / ||w||^2` for every column `w` of `W`, every `k`.

    `V` is `[d, r]` with orthonormal columns ordered top-first (the right
    singular vectors of `J_l`), so `V_k = V[:, :k]` and the mass is a running
    sum of squared coefficients -- one matmul for every `k` at once. `W` is
    `[d]` or `[d, n]`; the result is `[len(ks), n]`. Under an isotropic
    direction `E[rho_k] = k / d`, which is the number the test checks.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        W = W[:, None]
    ks = [int(k) for k in ks]
    if max(ks) > V.shape[1] or min(ks) < 1:
        raise ValueError(f"k must lie in [1, {V.shape[1]}], got {ks}")
    coef = V.T @ W                                # [r, n]
    cum = np.cumsum(coef ** 2, axis=0)            # [r, n]
    norms = (W ** 2).sum(0)                       # [n]
    return np.stack([cum[k - 1] / norms for k in ks])
