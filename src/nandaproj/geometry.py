"""PCA of the residual stream, and what a principal component is *about*.

The plot is the easy half. The hard half is that a PCA of prompt activations
almost always finds the thing the prompts differ in most, which on this bank is
the **scenario** -- freezers, scaffolds, playparks -- and not the one system
turn that separates H from D. Two clouds that look separated in PC1/PC2 may be
separated by topic, and a scatter plot cannot tell the difference.

So every component here is reported with `eta_squared` against each label, which
is the fraction of that component's variance explained by the grouping. It turns
"they look separated" into a number that can be wrong.

**Two units of analysis, and they answer different questions.**

*Raw* slot residuals, one point per (item, condition), say what dominates the
geometry. *Paired* H->D differences, one point per item, cancel the scenario
exactly -- the two prompts share every token but the system turn -- and are the
same object `intervene.d_dim` builds a direction from. A condition effect that
is real but small shows up in the second and is invisible in the first.

Torch-free: takes anything `np.asarray` accepts, so the notebook converts its
bf16 tensors once and everything below is testable without a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCA:
    """A fitted PCA. `scores` is `n x k`, one row per input vector."""

    scores: np.ndarray
    components: np.ndarray            # k x d, unit rows
    explained: np.ndarray             # k, fraction of total variance each
    mean: np.ndarray                  # d
    total_variance: float

    @property
    def k(self) -> int:
        return self.components.shape[0]

    def summary(self) -> str:
        pcs = "  ".join(f"PC{i + 1} {v:.1%}" for i, v in enumerate(self.explained))
        return f"{pcs}  (cumulative {self.explained.sum():.1%})"


def pca(vectors: Sequence, k: int = 2) -> PCA:
    """Plain PCA by SVD of the centered matrix. No whitening, no scaling.

    Not scaled per-dimension on purpose: the residual stream's coordinates are
    not comparable features, they are one vector in one basis, and scaling them
    to unit variance would rotate the geometry into something the model never
    computes.

    `explained` is against the **total** variance of the input, so the numbers
    are readable as "PC1 is 12% of everything that varies here" rather than as a
    share of the k retained.
    """
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected an (n, d) matrix, got shape {x.shape}")
    if len(x) < 2:
        raise ValueError(f"PCA over {len(x)} vector(s) is not defined")
    mean = x.mean(axis=0)
    centered = x - mean
    # Total variance = sum of eigenvalues, whether or not k of them are kept.
    total = float((centered ** 2).sum() / max(len(x) - 1, 1))
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(k, vt.shape[0])
    var = s ** 2 / max(len(x) - 1, 1)
    return PCA(scores=centered @ vt[:k].T, components=vt[:k],
               explained=var[:k] / total if total else np.zeros(k),
               mean=mean, total_variance=total)


def eta_squared(values: Sequence[float], labels: Sequence) -> float:
    """Fraction of the variance in `values` explained by the grouping `labels`.

    One-way ANOVA's eta^2: between-group sum of squares over the total. 0 means
    the grouping says nothing about this component; 1 means the component *is*
    the grouping.

    This is the number that decides whether a separation on a scatter plot is
    the label or the scenario, and it is why the plot is never reported alone.
    A grouping with one level per point -- item id on raw residuals, say --
    scores 1.0 by construction and means nothing; the caller is responsible for
    not asking that question.
    """
    v = np.asarray(values, dtype=np.float64)
    labs = list(labels)
    if len(v) != len(labs):
        raise ValueError(f"{len(v)} values against {len(labs)} labels")
    if len(v) < 2:
        return float("nan")
    total = float(((v - v.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    between = 0.0
    for lab in set(labs):
        group = v[[i for i, x in enumerate(labs) if x == lab]]
        between += len(group) * (group.mean() - v.mean()) ** 2
    return float(between / total)


def label_report(fit: PCA, labels: Mapping[str, Sequence]) -> str:
    """Every retained PC against every labelling, as a table.

    Read the columns, not the plot: a PC with eta^2 0.9 on `item` and 0.02 on
    `condition` is a scenario axis, however cleanly the colours happen to fall.
    """
    names = list(labels)
    head = f"{'':<6} {'var':>7}  " + "  ".join(f"{n:>10}" for n in names)
    rows = [head, "-" * len(head)]
    for i in range(fit.k):
        cells = "  ".join(f"{eta_squared(fit.scores[:, i], labels[n]):>10.2f}"
                          for n in names)
        rows.append(f"{'PC' + str(i + 1):<6} {fit.explained[i]:>6.1%}  {cells}")
    return "\n".join(rows)


def paired_differences(vectors: Mapping[tuple, Sequence], item_ids: Sequence[str],
                       cond_from: str, cond_to: str) -> tuple[np.ndarray, list[str]]:
    """`v[(item, cond_to)] - v[(item, cond_from)]` for every item that has both.

    The scenario cancels: the two prompts differ only in the system turn, so
    whatever the context contributes to the residual is subtracted off. What
    remains is the displacement the condition causes, per item -- the object a
    "deception direction" would have to be, and the one `intervene.d_dim`
    averages.

    Items missing either condition are skipped rather than filled: a difference
    against a missing run is not a small error, it is a different vector.
    """
    out, kept = [], []
    for i in item_ids:
        a, b = vectors.get((i, cond_from)), vectors.get((i, cond_to))
        if a is None or b is None:
            continue
        out.append(np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64))
        kept.append(i)
    if not out:
        raise ValueError(
            f"no item has both {cond_from!r} and {cond_to!r}; nothing to difference")
    return np.asarray(out), kept


def cosine_to_mean(diffs: Sequence) -> np.ndarray:
    """Each difference vector's cosine with the mean difference.

    The one-number version of "is there a shared direction here?". A tight
    cluster of cosines near 1 is a direction every item moves along; a spread
    centred on 0 is item-specific displacement that a mean vector merely
    averages into something with no member near it.

    Leave-one-out: each vector is compared against the mean of the *others*, so
    a vector never contributes to the thing it is scored against -- otherwise a
    small n scores high on its own presence alone.
    """
    x = np.asarray(diffs, dtype=np.float64)
    n = len(x)
    if n < 2:
        raise ValueError("need at least two vectors to compare against a mean")
    total = x.sum(axis=0)
    out = np.empty(n)
    for i in range(n):
        others = (total - x[i]) / (n - 1)
        denom = np.linalg.norm(x[i]) * np.linalg.norm(others)
        out[i] = 0.0 if denom == 0 else float(x[i] @ others / denom)
    return out
