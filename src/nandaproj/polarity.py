"""Polarity controls for the paired bank (`misc/bank_v2_spec.md` 6).

The v1 bank balanced Yes and No across the bank, and the behavioral gate ate
the balance: 11 gated items came out 4 Yes / 7 No, and the four items carrying
04's headline result -- `a_H` legible under D at L19-22 -- were *exactly* the
four Yes-true ones. Under polarity-independent legibility that split has
probability 1/C(11,4) = 0.003. So on v1, "the honest answer is legible" and
"` Yes` is legible" make the same prediction, and no analysis downstream can
separate them.

v2 pairs every belief item with a polarity twin sharing its context and
persona verbatim. This module is what turns that pairing into numbers:

- `d_paired` replaces a pooled H-vs-D difference-in-means with a twin-averaged
  one, so the ` Yes`/` No` component cancels *exactly* rather than approximately;
- `d_yesno` estimates the polarity direction itself, so any direction used for
  steering can be reported with its cosine against the thing it must not be;
- `by_polarity` splits any per-item metric into its two arms, so a result is
  never reported without the number that would falsify it.

`intervene.d_jlens` needs nothing from here: it is already per-item and built
from that item's own `(a_H, a_D)` ids, so it is polarity-symmetric by
construction. It is `intervene.d_difference_in_means` that pools, and
`d_paired` is its controlled replacement.

Array-library agnostic: everything is written in `+ - * sum()` and works on a
torch tensor (on any device) or a numpy array without converting either. Only
`deception.DeceptionItem` and `items.Item` are understood as items, via
`pair_id`/`polarity` on the object or in its `meta`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _field(item: Any, name: str) -> Any:
    """`pair_id`/`polarity` from either item type.

    `deception.DeceptionItem` carries them as attributes; `items.Item` is the
    readout layer's shape and carries them in `meta`, because they arrive
    through the generic record loader. One accessor rather than two call sites
    that each know only one layer.
    """
    value = getattr(item, name, None)
    if value is not None:
        return value
    meta = getattr(item, "meta", None) or {}
    return meta.get(name)


def item_id(item: Any) -> str:
    return str(item.item_id)


def pair_index(items: Iterable[Any]) -> dict[str, tuple[Any, Any]]:
    """`{pair_id: (yes_true_item, no_true_item)}`.

    Ordered by polarity, not by input order, so every caller averages a pair the
    same way round and a per-pair difference has a fixed sign. Items with no
    `pair_id` -- the no-belief floor -- are skipped. A pair that is not exactly
    one Yes and one No raises: a half-pair silently dropped is how the v1 skew
    would come back.
    """
    groups: dict[str, list[Any]] = {}
    for it in items:
        pid = _field(it, "pair_id")
        if pid:
            groups.setdefault(str(pid), []).append(it)
    out: dict[str, tuple[Any, Any]] = {}
    for pid, group in groups.items():
        by_pol = {_field(it, "polarity"): it for it in group}
        if set(by_pol) != {"Yes", "No"} or len(group) != 2:
            raise ValueError(
                f"pair {pid!r} is {[(item_id(i), _field(i, 'polarity')) for i in group]}, "
                "not one Yes-true and one No-true item")
        out[pid] = (by_pol["Yes"], by_pol["No"])
    return out


def _require_pairs(items: Iterable[Any]) -> dict[str, tuple[Any, Any]]:
    """`pair_index`, but an empty result is an error with the cause named.

    Zero pairs almost always means the items came from a file that did not carry
    `pair_id` -- a gated bank written by a `to_json` that dropped `meta`. Left to
    itself that surfaces as "mean of no vectors" several frames later, which
    says nothing about the file.
    """
    items = list(items)
    index = pair_index(items)
    if not index:
        raise ValueError(
            f"no polarity pairs among {len(items)} items: none carries a `pair_id`. "
            "If these came from a gated bank, it was written before `items.to_json` "
            "round-tripped it -- repair with `items.attach_meta(bank, items.load_bank())`."
        )
    return index


def _mean(vectors: Sequence[Any]) -> Any:
    if not vectors:
        raise ValueError("mean of no vectors")
    total = vectors[0]
    for v in vectors[1:]:
        total = total + v
    return total / len(vectors)


def _require(residuals: Mapping[str, Any], item: Any) -> Any:
    key = item_id(item)
    if key not in residuals:
        raise KeyError(f"no residual for {key!r}; have {len(residuals)} items")
    return residuals[key]


def d_yesno(residuals: Mapping[str, Any], items: Iterable[Any]) -> Any:
    """`mean h(Yes-true) - mean h(No-true)` within one condition.

    The direction the bank exists to control for. Estimated over whole pairs, so
    every context and persona contributes to both arms and the only thing that
    differs between them is which word is true.
    """
    index = _require_pairs(items)
    yes = [_require(residuals, y) for y, _ in index.values()]
    no = [_require(residuals, n) for _, n in index.values()]
    return _mean(yes) - _mean(no)


def d_paired(
    residuals_a: Mapping[str, Any],
    residuals_b: Mapping[str, Any],
    items: Iterable[Any],
) -> Any:
    """Twin-averaged difference-in-means: `mean_pairs[ h_pair(a) - h_pair(b) ]`.

    `h_pair(c)` is the mean of the two twins under condition `c`. Because the
    twins are one Yes-true and one No-true over an identical context and
    persona, that inner mean has no polarity component, so neither does the
    difference -- exactly, not on average over a large enough sample.

    Pass H residuals as `a` and D as `b` for the polarity-free version of
    `intervene.d_difference_in_means`.
    """
    per_pair = []
    for yes, no in _require_pairs(items).values():
        a = _mean([_require(residuals_a, yes), _require(residuals_a, no)])
        b = _mean([_require(residuals_b, yes), _require(residuals_b, no)])
        per_pair.append(a - b)
    return _mean(per_pair)


def norm(v: Any) -> float:
    return float(((v * v).sum()) ** 0.5)


def cosine(a: Any, b: Any) -> float:
    """`cos(a, b)`. Report this against `d_yesno` for any steering direction.

    Chance is `1/sqrt(d_model)` = 0.0198 at `d_model` = 2560, which is the
    number a reported cosine has to beat before it means anything.
    """
    denom = norm(a) * norm(b)
    if denom == 0.0:
        raise ValueError("cosine with a zero vector")
    return float((a * b).sum()) / denom


def project_out(d: Any, reference: Any) -> Any:
    """`d` with its `reference` component removed.

    For asking whether a direction still steers once the polarity component is
    gone. Not applied by default anywhere -- a direction that only works *with*
    its polarity component is a result, and silently stripping it would hide it.
    """
    scale = norm(reference)
    if scale == 0.0:
        raise ValueError("cannot project out a zero vector")
    unit = reference / scale
    return d - unit * float((d * unit).sum())


@dataclass(frozen=True)
class PolaritySplit:
    """One per-item metric, split into its two arms."""

    yes: dict[str, float]
    no: dict[str, float]

    @property
    def mean_yes(self) -> float:
        return sum(self.yes.values()) / len(self.yes)

    @property
    def mean_no(self) -> float:
        return sum(self.no.values()) / len(self.no)

    @property
    def difference(self) -> float:
        """`mean_yes - mean_no`. Near zero is the result the pairing is for."""
        return self.mean_yes - self.mean_no

    def __str__(self) -> str:
        return (f"Yes-true n={len(self.yes)} mean={self.mean_yes:.4g} | "
                f"No-true n={len(self.no)} mean={self.mean_no:.4g} | "
                f"difference={self.difference:+.4g}")


def by_polarity(metric: Mapping[str, float], items: Iterable[Any]) -> PolaritySplit:
    """Split a per-item metric into Yes-true and No-true arms.

    Use it on anything reported per item -- peak `P_J(a_H)`, the crossover layer
    `l*`, a flip rate. If the two arms differ, the metric is at least partly
    about the answer token, whatever else it is about.
    """
    # Split on each item's own declared `polarity`, not on `pair_index`: the
    # arms are a property of the item, and requiring whole pairs here made this
    # raise on any gated arm where twins did not survive together (05 on the
    # alleged bank: 32 pair ids, 0 whole). Pair-level cancellation is
    # `d_paired`'s job; this only reports the two arms.
    items = list(items)
    yes = {item_id(it): float(metric[item_id(it)]) for it in items
           if _field(it, "polarity") == "Yes" and item_id(it) in metric}
    no = {item_id(it): float(metric[item_id(it)]) for it in items
          if _field(it, "polarity") == "No" and item_id(it) in metric}
    if not yes or not no:
        raise ValueError(
            f"metric covers {len(yes)} Yes-true and {len(no)} No-true items; "
            "a one-armed split is not a control")
    return PolaritySplit(yes=yes, no=no)


def axis_fraction(directions: Mapping[str, Any], items: Iterable[Any]) -> tuple[float, Any]:
    """How much of a per-item direction is one shared answer-token axis.

    `d_jlens` is `grad_h [ lens_logit(a_H) - lens_logit(a_D) ]`, and on a Yes/No
    bank `(a_H, a_D)` is `(Yes, No)` for one twin and `(No, Yes)` for the other.
    So the *item-independent* part of `d_J` is exactly the Yes/No axis wearing a
    per-item sign, and "truth direction" and "answer-token axis" are separated
    only by whatever item-specific residual `J_l` and the norm introduce.

    This measures that. Unit-normalise each item's direction, flip the sign of
    the No-true ones so every vector points the same way along the token axis,
    and average:

        m    = mean_i( sign_i * unit(d_i) )        the shared axis
        ||m||^2                                    its share of each d_i

    `||m||^2 = 1` means every direction is the same vector up to sign, and the
    per-item structure is nil. `||m||^2 ~ 1/n` is what unrelated directions give.
    Returns `(fraction, m)`; `m` is the axis itself, for `residual_direction`.
    """
    index = pair_index(items)
    signed = []
    for yes, no in index.values():
        for it, sign in ((yes, 1.0), (no, -1.0)):
            d = _require(directions, it)
            signed.append((d / norm(d)) * sign)
    if not signed:
        raise ValueError("no directions to decompose")
    m = _mean(signed)
    return float(norm(m) ** 2), m


def residual_direction(direction: Any, item: Any, axis: Any) -> Any:
    """`unit(d_i)` with the shared token axis removed -- the item-specific part.

    The only part of `d_J` that could carry something other than the answer
    token, and therefore the only part whose steering result would be about a
    belief. Returned unnormalised: its norm is the interesting quantity (a tiny
    residual means there was nothing item-specific to steer), and a caller that
    wants a unit vector should say so.
    """
    sign = 1.0 if _field(item, "polarity") == "Yes" else -1.0
    return direction / norm(direction) - axis * sign


def whole_pairs(items: Iterable[Any], keep: Iterable[str]) -> list[str]:
    """The item ids of pairs where **both** twins are in `keep`.

    The pair-level gate (`bank_v2_spec` 5). A pair with one surviving twin is
    dropped whole -- keeping it is exactly how v1's balance died at the gate.
    """
    kept = set(keep)
    out: list[str] = []
    for yes, no in pair_index(items).values():
        if item_id(yes) in kept and item_id(no) in kept:
            out += [item_id(yes), item_id(no)]
    return out


def gate_report(items: Iterable[Any], keep: Iterable[str]) -> str:
    """What a pair-level gate kept, and what the polarity asymmetry cost.

    `broken` -- pairs where exactly one twin passed -- is the number v1 could
    not see. A large count is itself a finding about the model's polarity
    asymmetry, and it is the reason the surviving set must be trimmed to whole
    pairs rather than reported as it stands.
    """
    kept = set(keep)
    index = pair_index(items)
    both = [p for p, (y, n) in index.items()
            if item_id(y) in kept and item_id(n) in kept]
    broken = {p: (item_id(y) if item_id(y) in kept else item_id(n))
              for p, (y, n) in index.items()
              if (item_id(y) in kept) != (item_id(n) in kept)}
    neither = len(index) - len(both) - len(broken)
    lines = [
        (f"pair gate: {len(both)}/{len(index)} pairs kept "
        f"({2 * len(both)} items), {len(broken)} broken by one twin, "
        f"{neither} lost entirely"),
    ]
    for pid, survivor in sorted(broken.items()):
        lines.append(f"  broken {pid}: only {survivor} passed -- pair dropped")
    return "\n".join(lines)
