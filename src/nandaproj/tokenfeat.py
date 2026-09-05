"""Exp 4: a lying-vs-honest classifier on the J-lens's top tokens.

04d asked whether the *truth* is linearly decodable from the 2560-d residual.
This asks a smaller, readable question: at the answer slot, do the tokens the
J-lens decodes at layer l separate the forward pass where the model lied from
the ones where it did not, and which tokens carry that separation?

Three things this module is built around, all from `misc/exp4_spec.md`:

1. **Features are a per-fold vocabulary.** The token set at a layer is the
   union of the training rows' top-K, built inside every CV fold, so a
   held-out row's private token can never become a feature.
2. **The answer tokens are removable.** The lying set is polarity-skewed
   (37 No-true vs 16 Yes-true), so reading " Yes"/" No" alone gets ~0.7 on the
   raw label. `answer_token_ids` finds every yes/no spelling in the vocab and
   the primary fit drops them.
3. **The null shuffles within an item.** Rows are H, C1 and D of the same
   item; the null permutes labels among those three rows, so a lying item
   keeps exactly one positive row but the classifier no longer knows which
   condition it was. `probe`'s pair-flip null would make H and C1 rows
   positive, which no bank produces.

Storage is top-200 ids and log-probs per row per layer, for both lenses, from
one `Reader.readout` call each. The full 262k-vocab vector per row would be
~10 GB across the bank; the top-200 is a few MB and the feature floor below
is rarely hit at K <= 50.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nandaproj import probe
from nandaproj.lens_readout import _progress

LOG_FLOOR = 1e-38          # log(0) guard on float32 softmax outputs
SD_FLOOR = 1e-3            # a feature constant on the training rows is left unscaled


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------

@dataclass
class TopK:
    """Top-`k` tokens per (row, layer) for the J-lens and the logit lens.

    A row is one (item, condition) readout. `ids`/`lp` are `[n, L, k]`, sorted
    by descending log-prob along the last axis, so `lp[:, :, -1]` is the floor
    a row assigns to any token outside its list.
    """

    item_ids: list[str]
    conditions: list[str]
    layers: list[int]
    j_ids: np.ndarray
    j_lp: np.ndarray
    l_ids: np.ndarray
    l_lp: np.ndarray

    def __post_init__(self):
        n, L, _ = self.j_ids.shape
        assert len(self.item_ids) == len(self.conditions) == n, "row count mismatch"
        assert len(self.layers) == L, "layer count mismatch"
        assert self.j_lp.shape == self.j_ids.shape == self.l_ids.shape == self.l_lp.shape

    @property
    def n(self) -> int:
        return len(self.item_ids)

    @property
    def k(self) -> int:
        return self.j_ids.shape[-1]

    def lens(self, which: str) -> tuple[np.ndarray, np.ndarray]:
        """`(ids, lp)` for `"j"` or `"l"`."""
        if which == "j":
            return self.j_ids, self.j_lp
        if which == "l":
            return self.l_ids, self.l_lp
        raise ValueError(f"lens must be 'j' or 'l', got {which!r}")

    def at(self, which: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """`(ids [n, k], lp [n, k])` at one layer."""
        ids, lp = self.lens(which)
        li = self.layers.index(layer)
        return ids[:, li], lp[:, li]

    def rows(self, condition: str | None = None) -> np.ndarray:
        """Row indices, optionally of one condition."""
        if condition is None:
            return np.arange(self.n)
        return np.flatnonzero(np.asarray(self.conditions) == condition)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez_compressed(
            path, item_ids=np.array(self.item_ids), conditions=np.array(self.conditions),
            layers=np.array(self.layers),
            j_ids=self.j_ids.astype(np.int32), j_lp=self.j_lp.astype(np.float32),
            l_ids=self.l_ids.astype(np.int32), l_lp=self.l_lp.astype(np.float32))
        return path

    @classmethod
    def load(cls, path: str | Path) -> TopK:
        d = np.load(path)
        return cls(item_ids=[str(i) for i in d["item_ids"]],
                   conditions=[str(c) for c in d["conditions"]],
                   layers=[int(l) for l in d["layers"]],
                   j_ids=d["j_ids"], j_lp=d["j_lp"], l_ids=d["l_ids"], l_lp=d["l_lp"])


def top_k_of(probs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """`(ids, log_probs)` of the `k` largest entries, descending."""
    k = min(k, probs.shape[-1])
    idx = np.argpartition(-probs, k - 1)[:k]
    idx = idx[np.argsort(-probs[idx], kind="stable")]
    return idx.astype(np.int32), np.log(np.maximum(probs[idx], LOG_FLOOR)).astype(np.float32)


def capture_topk(reader, items: Iterable, conditions: Sequence[str] = ("H", "C1", "D"),
                 k: int = 200, save_to: str | Path | None = None,
                 skip_missing: bool = True) -> TopK:
    """One `reader.readout` per (item, condition); keep the top-`k` per layer.

    Written after **every item** when `save_to` is given, so a sweep that dies
    part-way leaves what it had. Two lens passes per row, unbatched, so this
    is minutes per hundred rows and carries a progress bar.
    """
    from nandaproj import items as items_mod

    items = list(items)
    jobs = [(it, c) for it in items for c in conditions
            if not (skip_missing and c not in it.prompts)]
    layers = list(reader.layers)
    item_ids, conds, J_ids, J_lp, L_ids, L_lp = [], [], [], [], [], []

    def _flush() -> TopK:
        store = TopK(item_ids, conds, layers, np.stack(J_ids), np.stack(J_lp),
                     np.stack(L_ids), np.stack(L_lp))
        if save_to is not None:
            store.save(save_to)
        return store

    for it, cond in _progress(jobs, total=len(jobs), desc=f"top-{k} readout"):
        j_probs, l_probs, _ = reader.readout(items_mod.render(reader.tok, it, cond))
        ji, jl = zip(*(top_k_of(j_probs[l], k) for l in layers))
        li, ll = zip(*(top_k_of(l_probs[l], k) for l in layers))
        item_ids.append(it.item_id)
        conds.append(cond)
        J_ids.append(np.stack(ji)); J_lp.append(np.stack(jl))
        L_ids.append(np.stack(li)); L_lp.append(np.stack(ll))
        _flush()
    return _flush()


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def union_vocab(ids: np.ndarray, k_top: int, banned: Iterable[int] = ()) -> np.ndarray:
    """Sorted unique token ids among the top-`k_top` of the given rows, minus `banned`.

    `ids` is `[n, k]` (already descending), so the top-`k_top` is a slice.
    """
    vocab = np.unique(ids[:, :k_top])
    banned = np.asarray(sorted(set(int(b) for b in banned)), dtype=vocab.dtype)
    return vocab[~np.isin(vocab, banned)]


def feature_matrix(ids: np.ndarray, lp: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """`[n, |vocab|]` log-probs; a token absent from a row's list takes that row's floor.

    The floor is the row's last (smallest) stored log-prob: the true value is
    at most that, and at K << k_store the substitution is rare.
    """
    n = ids.shape[0]
    X = np.repeat(lp[:, -1:], len(vocab), axis=1).astype(np.float32)
    pos = np.searchsorted(vocab, ids)                      # [n, k]
    pos_c = np.minimum(pos, len(vocab) - 1)
    hit = (vocab[pos_c] == ids) if len(vocab) else np.zeros_like(ids, dtype=bool)
    rows = np.repeat(np.arange(n)[:, None], ids.shape[1], axis=1)
    X[rows[hit], pos_c[hit]] = lp[hit]
    return X


def answer_token_ids(tok, words: Sequence[str] = ("yes", "no")) -> list[int]:
    """Every vocab id whose decoded string, stripped and lower-cased, is in `words`.

    A full scan of the vocabulary rather than a guessed list of spellings, so
    the mask cannot miss " YES" or "no" because nobody typed them.
    """
    words = {w.lower() for w in words}
    out = []
    for i in range(len(tok)):
        if tok.decode([i]).strip().lower() in words:
            out.append(i)
    return out


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd < SD_FLOOR, 1.0, sd)
    return mu, sd


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

@dataclass
class Fitted:
    """A classifier plus everything needed to apply it to new rows."""

    vocab: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    b: float

    def predict(self, ids: np.ndarray, lp: np.ndarray) -> np.ndarray:
        X = (feature_matrix(ids, lp, self.vocab) - self.mu) / self.sd
        return probe.predict((self.w, self.b), X)

    def top_weights(self, tok, n: int = 15) -> list[tuple[str, float]]:
        """`[(token_string, weight)]` by |weight|, positive = towards "lie"."""
        order = np.argsort(-np.abs(self.w))[:n]
        return [(tok.decode([int(self.vocab[i])]), float(self.w[i])) for i in order]


def fit(ids: np.ndarray, lp: np.ndarray, y: np.ndarray, k_top: int,
        banned: Iterable[int] = (), l2: float = 1.0, **kw) -> Fitted:
    """Vocabulary from these rows, standardise, L2 logistic."""
    vocab = union_vocab(ids, k_top, banned)
    X = feature_matrix(ids, lp, vocab)
    mu, sd = _standardize(X)
    w, b = probe.fit_logistic((X - mu) / sd, y, l2=l2, **kw)
    return Fitted(vocab, mu, sd, w, b)


def fold_vocabs(ids: np.ndarray, groups: Sequence[str], k_top: int,
                banned: Iterable[int] = ()):
    """Per leave-one-group-out fold: `(train, test, vocab)` with vocab from `train` only."""
    for train, test in probe.grouped_folds(groups):
        yield train, test, union_vocab(ids[train], k_top, banned)


def cv_predict(ids: np.ndarray, lp: np.ndarray, y: np.ndarray, groups: Sequence[str],
               k_top: int, banned: Iterable[int] = (), l2: float = 1.0, **kw) -> np.ndarray:
    """Leave-one-group-out predictions, vocabulary and scaling fold-internal.

    Returns the predicted label per row, so the caller scores any subset
    (all rows, the lying set, one polarity arm) from one pass.
    """
    y = np.asarray(y).astype(bool)
    pred = np.zeros(len(y), dtype=bool)
    for train, test, vocab in fold_vocabs(ids, groups, k_top, banned):
        Xtr = feature_matrix(ids[train], lp[train], vocab)
        Xte = feature_matrix(ids[test], lp[test], vocab)
        mu, sd = _standardize(Xtr)
        w, b = probe.fit_logistic((Xtr - mu) / sd, y[train], l2=l2, **kw)
        pred[test] = probe.predict((w, b), (Xte - mu) / sd)
    return pred


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((np.asarray(pred).astype(bool) == np.asarray(y).astype(bool)).mean())


def balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    """Mean of true-positive and true-negative rate.

    The primary metric: 53 of 300 rows are lies, so plain accuracy is 0.82 for
    a classifier that never says "lie", and the null's p95 would sit there too.
    """
    pred, y = np.asarray(pred).astype(bool), np.asarray(y).astype(bool)
    return float((pred[y].mean() + (~pred[~y]).mean()) / 2)


def cv_accuracy(ids, lp, y, groups, k_top, banned=(), metric=accuracy, **kw) -> float:
    return metric(cv_predict(ids, lp, y, groups, k_top, banned, **kw), y)


# --------------------------------------------------------------------------
# The null
# --------------------------------------------------------------------------

def shuffle_within_items(y: np.ndarray, item_ids: Sequence[str],
                         rng: np.random.Generator) -> np.ndarray:
    """Permute labels among the rows that share an item id.

    Every item keeps its number of positive rows; only which condition carries
    the label is randomised.
    """
    y = np.asarray(y).copy()
    item_ids = np.asarray(item_ids)
    for it in dict.fromkeys(item_ids):
        m = np.flatnonzero(item_ids == it)
        y[m] = y[rng.permutation(m)]
    return y


def permutation_null(ids, lp, y, groups, item_ids, k_top, banned=(), n: int = 100,
                     seed: int = 0, metric=accuracy, desc: str | None = "null",
                     **kw) -> np.ndarray:
    """`n` CV scores under within-item label shuffles. `desc=None` hides the bar."""
    rng = np.random.default_rng(seed)
    draws = range(n) if desc is None else _progress(range(n), total=n, desc=desc)
    return np.array([cv_accuracy(ids, lp, shuffle_within_items(y, item_ids, rng),
                                 groups, k_top, banned, metric=metric, **kw)
                     for _ in draws])


# --------------------------------------------------------------------------
# Batched CV: every fold, and every label set, as one tensor operation.
#
# `cv_predict` is one logistic fit per fold and is the reference. The null
# needs 50 draws x 50 folds x 33 layers of it, which is minutes on a CPU. The
# same arithmetic batches: build X once over the union of every row's top-K,
# and give each fold a 0/1 feature mask that is 1 only for tokens in the
# *training* rows' top-K. A masked feature contributes 0 to every logit and
# receives 0 gradient, so its weight stays at 0 and the fit is the fold-vocab
# fit exactly. Label sets (the observed labels plus every null draw) share the
# masked, standardised X, so they are a second batch axis.
# --------------------------------------------------------------------------

def _batched_inputs(ids, lp, groups, k_top, banned):
    """`(Z [F, n, D], train_mask [F, n], folds)`: standardised, fold-masked features."""
    vocab = union_vocab(ids, k_top, banned)                     # superset over all rows
    X = feature_matrix(ids, lp, vocab)                          # [n, D]
    folds = list(probe.grouped_folds(groups))
    F, (n, D) = len(folds), X.shape
    Z = np.empty((F, n, D), dtype=np.float32)
    train_mask = np.ones((F, n), dtype=np.float32)
    for f, (train, test) in enumerate(folds):
        train_mask[f, test] = 0.0
        fold_vocab = union_vocab(ids[train], k_top, banned)
        feat = np.isin(vocab, fold_vocab).astype(np.float32)   # [D]
        mu, sd = _standardize(X[train])
        Z[f] = ((X - mu) / sd) * feat
    return Z, train_mask, folds


def _gd_numpy(Z, train_mask, Y, l2, steps, lr):
    F, n, D = Z.shape
    R = Y.shape[0]
    nt = train_mask.sum(1)[None, :, None]                       # [1, F, 1]
    W = np.zeros((R, F, D), dtype=np.float32)
    b = np.zeros((R, F, 1), dtype=np.float32)
    Yb = Y[:, None, :].astype(np.float32)                       # [R, 1, n]
    M = train_mask[None]                                        # [1, F, n]
    for _ in range(steps):
        logits = np.einsum("fnd,rfd->rfn", Z, W) + b
        G = (1.0 / (1.0 + np.exp(-logits)) - Yb) * M            # [R, F, n]
        W -= lr * (np.einsum("rfn,fnd->rfd", G, Z) / nt + l2 * W / nt)
        b -= lr * G.sum(2, keepdims=True) / nt
    return np.einsum("fnd,rfd->rfn", Z, W) + b                  # [R, F, n]


def _gd_torch(Z, train_mask, Y, l2, steps, lr, device):
    import torch

    Z = torch.as_tensor(Z, device=device)
    M = torch.as_tensor(train_mask, device=device)[None]
    Yb = torch.as_tensor(Y, device=device, dtype=torch.float32)[:, None, :]
    F, n, D = Z.shape
    R = Yb.shape[0]
    nt = M.sum(2, keepdim=True)                                 # [1, F, 1]
    W = torch.zeros((R, F, D), device=device)
    b = torch.zeros((R, F, 1), device=device)
    for _ in range(steps):
        logits = torch.einsum("fnd,rfd->rfn", Z, W) + b
        G = (torch.sigmoid(logits) - Yb) * M
        W -= lr * (torch.einsum("rfn,fnd->rfd", G, Z) / nt + l2 * W / nt)
        b -= lr * G.sum(2, keepdim=True) / nt
    return (torch.einsum("fnd,rfd->rfn", Z, W) + b).cpu().numpy()


def cv_predict_many(ids: np.ndarray, lp: np.ndarray, Ys: np.ndarray, groups: Sequence[str],
                    k_top: int, banned: Iterable[int] = (), l2: float = 1.0,
                    steps: int = 300, lr: float = 0.1, device: str | None = None) -> np.ndarray:
    """Leave-one-group-out predictions for several label sets at once.

    `Ys` is `[R, n]` (row 0 the observed labels, the rest null draws, say);
    returns `[R, n]` predicted labels. Same folds, same fold-internal
    vocabulary and standardisation, same optimiser as `cv_predict`, so the
    two agree to float tolerance. `device` picks torch (`"cuda"`) when it is
    available; `None` or `"cpu"` runs the numpy path.
    """
    Ys = np.asarray(Ys).astype(bool)
    if Ys.ndim == 1:
        Ys = Ys[None]
    Z, train_mask, folds = _batched_inputs(ids, lp, groups, k_top, banned)
    if device and device != "cpu":
        scores = _gd_torch(Z, train_mask, Ys, l2, steps, lr, device)
    else:
        scores = _gd_numpy(Z, train_mask, Ys, l2, steps, lr)
    pred = np.zeros(Ys.shape, dtype=bool)
    for f, (_, test) in enumerate(folds):
        pred[:, test] = scores[:, f, test] > 0
    return pred


def cuda_if_available() -> str | None:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else None
    except ImportError:
        return None
