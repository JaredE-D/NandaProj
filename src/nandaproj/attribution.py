"""Component attribution: which heads and MLPs close the belief-report gap.

PLAN2.md 4.3 and [`misc/exp2_spec.md`](../../misc/exp2_spec.md), gate **V5** and
the first half of **V6**. 04 located the flip at `l* = 25` and the honest
answer's peak legibility at L19-22; this module ranks the components that carry
the flip out.

The measurement, stated once so the code can be read against it:

> `LD = logit(a_H) - logit(a_D)` at the answer slot. Cache the **honest** run's
> component outputs there, replace one of them in the **deceptive** run, and ask
> how far `LD` moves back toward the honest value.

    recovery(c) = (LD_patched - LD_D) / (LD_H - LD_D)

0 = the patch did nothing; 1 = it fully restored the honest logit difference.

**Two halves, deliberately.** Everything above `Patcher` is torch-free: the
arithmetic that turns per-component numbers into a claim -- the bars, the nulls,
the set selection, the V6 overlap -- and that arithmetic is where a wrong result
would look like a finding rather than like an error. It is tested locally in
`tests/test_attribution.py` with no GPU. `Patcher` is the half that needs a
model and is exercised by `05_attribution.ipynb`'s assert cells on the box.
This is the same split `lens_readout.py` uses and for the same reason.

**Why nnsight.** `infra/probe_env.py --all` settled it on the box: nnsight runs
the *real* HF module, so it is numerically the object 04 measured `l*` on, and
its per-head split came out exact to 4.8e-07 with a bit-identical no-op trace.
transformer_lens re-implements and folds layernorms (0.875 max log-prob
disagreement, unattributed); it is the fallback, not the choice.

**Three architecture facts, each a silent bug if assumed** (all read off the
config at runtime, never hardcoded):

1. `text_config.layer_types` names the local/global alternation, so sliding and
   full attention heads are never pooled (PLAN2.md R4).
2. Gemma 3 puts `post_attention_layernorm` **between** the attention output and
   the residual add. See `Patcher.dla` for what that does to the decomposition.
3. GQA: the `o_proj` input is `n_q_heads x head_dim` wide, which is **not**
   `d_model`. Head slicing uses `head_dim`, never `d_model // n_heads`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# `recovery` is a ratio of logit differences in bf16. Where the honest and
# deceptive runs barely differ, the denominator is noise and the ratio is
# meaningless -- a 0.3 logit gap divided into a 0.3 logit patch effect reads as
# recovery 1.0 on an item that never had a gap to close. Such items are flagged
# and excluded from medians rather than averaged in (exp2_spec.md 10).
MIN_LD_GAP = 1.0

# exp2_spec.md 5.3, fixed before the sweep. Do not tune these to a result.
MIN_RECOVERY = 0.20        # a single component must move LD this far, median over the family
NULL_PERCENTILE = 95.0     # ...and clear this percentile of its own wrong-source null
SET_NULL_PERCENTILE = 99.0 # a set must clear this percentile of random sets of equal size
SET_TARGET = 0.5           # `k` is the smallest set reaching this joint recovery
SET_K_MAX = 10             # ...capped here

HEAD = "head"
MLP = "mlp"


# ==========================================================================
# Torch-free: components, results, bars, nulls, sets
# ==========================================================================


@dataclass(frozen=True, order=True)
class Component:
    """One patchable unit: an attention head, or a layer's MLP block.

    `layer_type` carries the sliding/full distinction from the model config so
    that R4 is enforceable downstream -- a top set that is secretly all
    sliding-window heads is a different claim from one that mixes them, and
    that is only checkable if the label travels with the component.
    """

    layer: int
    kind: str = HEAD
    head: int | None = None
    # `compare=False` keeps the label out of `__eq__`, `__hash__` and ordering.
    # L24H3 is L24H3 whether or not whoever built it happened to know the layer
    # was sliding-window. With the label in identity, a lookup needs the label
    # -- and worse, if the null sweep and the real sweep ever labelled a layer
    # differently, `rank` would silently treat one component as two and compute
    # both bars on half the rows.
    layer_type: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.kind not in (HEAD, MLP):
            raise ValueError(f"kind must be {HEAD!r} or {MLP!r}, got {self.kind!r}")
        if (self.kind == HEAD) != (self.head is not None):
            raise ValueError(f"{self.kind!r} component with head={self.head!r}")

    @property
    def name(self) -> str:
        return f"L{self.layer}H{self.head}" if self.kind == HEAD else f"L{self.layer}M"

    @classmethod
    def parse(cls, name: str, layer_type: str = "") -> Component:
        body = name[1:]
        if body.endswith("M"):
            return cls(int(body[:-1]), MLP, None, layer_type)
        layer, head = body.split("H")
        return cls(int(layer), HEAD, int(head), layer_type)

    def __str__(self) -> str:
        return self.name


@dataclass
class PatchResult:
    """One (item, condition, component, source) patch, and what it did to LD.

    `source` is the item whose honest run supplied the patched activation:
    the item's own id for the real measurement, a *different* item's id for the
    wrong-source null (5.3). Carrying it on the row rather than in a separate
    table is what makes the null impossible to mix into the ranking by accident.
    """

    item_id: str
    condition: str
    component: Component
    ld_deceptive: float
    ld_honest: float
    ld_patched: float
    source: str = ""
    family: str = ""

    @property
    def gap(self) -> float:
        return self.ld_honest - self.ld_deceptive

    @property
    def usable(self) -> bool:
        """False when the two runs barely differ -- see `MIN_LD_GAP`."""
        return abs(self.gap) >= MIN_LD_GAP

    @property
    def recovery(self) -> float:
        if not self.usable:
            return float("nan")
        return (self.ld_patched - self.ld_deceptive) / self.gap

    @property
    def is_null(self) -> bool:
        return bool(self.source) and self.source != self.item_id


def recoveries(rows: Iterable[PatchResult]) -> np.ndarray:
    """Recovery values, with the unusable-denominator rows dropped."""
    vals = np.array([r.recovery for r in rows], dtype=float)
    return vals[~np.isnan(vals)]


def median_recovery(rows: Iterable[PatchResult]) -> float:
    vals = recoveries(rows)
    return float(np.median(vals)) if vals.size else float("nan")


def by_component(rows: Iterable[PatchResult]) -> dict[Component, list[PatchResult]]:
    out: dict[Component, list[PatchResult]] = {}
    for r in rows:
        out.setdefault(r.component, []).append(r)
    return out


def split_null(rows: Iterable[PatchResult]) -> tuple[list[PatchResult], list[PatchResult]]:
    """(real, null) -- rows patched from the item's own honest run, and not."""
    rows = list(rows)
    return [r for r in rows if not r.is_null], [r for r in rows if r.is_null]


@dataclass(frozen=True)
class Ranked:
    """A component's standing against its own null. `hit` applies 5.3's bars.

    The bars are stored **on the row**, not read from module constants at call
    time. A sensitivity check run at a different bar produces rows that say so,
    and a table cannot be silently re-judged against bars it was not built
    with -- which is the failure mode that turns a pre-registration into a
    description.
    """

    component: Component
    n_items: int
    median: float
    null_median: float
    null_bar: float
    min_recovery: float = MIN_RECOVERY
    percentile: float = NULL_PERCENTILE
    # `None` means "not computed" -- not `nan`, because `nan != nan` makes two
    # identical rankings compare unequal, and a table that cannot be compared
    # to its own reloaded copy cannot be checked at all.
    median_yes: float | None = None
    median_no: float | None = None
    arms_required: bool = False

    @property
    def hit(self) -> bool:
        """Both bars, and -- when the arms are known -- both polarity arms.

        `LD = logit(a_H) - logit(a_D)` is polarity-signed per item, so a
        component that only ever writes `' Yes'` scores positive recovery on
        Yes-true items and *negative* on their No-true twins. On a balanced set
        those cancel and the pooled median is already near zero, which is why
        05 is largely insulated from the bank's Yes-bias. But the cancellation
        is only exact while the set stays balanced, and `MIN_LD_GAP` drops one
        twin without its partner. Requiring both arms makes the protection
        explicit instead of incidental: a component that restores the belief
        works in both directions, and a token-writer cannot clear both.
        """
        base = (self.median >= self.min_recovery) and (self.median > self.null_bar)
        if not (base and self.arms_required):
            return base
        if self.median_yes is None or self.median_no is None:
            return False
        return (self.median_yes >= self.min_recovery
                and self.median_no >= self.min_recovery)

    @property
    def arm_gap(self) -> float:
        """`median_yes - median_no`. Near zero is what a belief component looks
        like; a large magnitude is the signature of a component that writes an
        answer token rather than restoring an answer."""
        if self.median_yes is None or self.median_no is None:
            return float("nan")
        return self.median_yes - self.median_no

    @property
    def margin(self) -> float:
        """How far above its own null bar, in recovery units."""
        return self.median - self.null_bar


YES, NO = "Yes", "No"


def arm_medians(rows: Sequence[PatchResult], polarity_of: Mapping[str, str]
                ) -> tuple[float | None, float | None]:
    """(median over Yes-true items, median over No-true items).

    An empty arm is `None`, not `nan`: `Ranked.hit` refuses a `None` arm, so a
    component measured on one arm only is never promoted -- and `None` compares
    equal to itself, which `nan` does not, so an armed ranking can still be
    checked against its own reloaded copy.

    A one-armed median is exactly the quantity the polarity pairing exists to
    stop anyone reporting, so the safe direction is to refuse rather than to
    fall back on the arm that happens to have rows.
    """
    def arm(label: str) -> float | None:
        vals = recoveries([r for r in rows if polarity_of.get(r.item_id) == label])
        return float(np.median(vals)) if vals.size else None

    return arm(YES), arm(NO)


def rank(rows: Iterable[PatchResult], *, min_recovery: float = MIN_RECOVERY,
         percentile: float = NULL_PERCENTILE,
         polarity_of: Mapping[str, str] | None = None) -> list[Ranked]:
    """Rank components by median recovery, each against its **own** null.

    The null is the wrong-source patch: the same component, patched with the
    honest activation of a *different* item. Same component, same perturbation
    magnitude, wrong content -- which is what separates "this component carries
    the honest answer" from "poking this component moves the output". A
    zero-ablation null cannot make that distinction, and it is the distinction
    V5 is about.

    A component with no null rows gets `null_bar = inf` and can never be a hit:
    unranked is the safe direction, and a missing null is a pipeline bug rather
    than a licence to promote the component.

    `polarity_of` maps an item id to `"Yes"` or `"No"` (which answer is *true*
    for it). Given it, each component is also scored on the two arms separately
    and must clear the bar in **both** -- see `Ranked.hit` for why the pooled
    median alone is not sufficient protection once `MIN_LD_GAP` has dropped a
    twin. Without it the ranking is pooled-only, which is what it was before
    the bank was paired; the ranking rows say which happened.
    """
    out: list[Ranked] = []
    for comp, comp_rows in by_component(rows).items():
        real, null = split_null(comp_rows)
        null_vals = recoveries(null)
        arms = (arm_medians(real, polarity_of) if polarity_of is not None
                else (None, None))
        out.append(Ranked(
            component=comp,
            n_items=len(recoveries(real)),
            median=median_recovery(real),
            null_median=float(np.median(null_vals)) if null_vals.size else float("nan"),
            null_bar=float(np.percentile(null_vals, percentile)) if null_vals.size
            else float("inf"),
            min_recovery=min_recovery,
            percentile=percentile,
            median_yes=arms[0],
            median_no=arms[1],
            arms_required=polarity_of is not None,
        ))
    out.sort(key=lambda r: (-(r.median if np.isfinite(r.median) else -np.inf), r.component))
    return out


def hits(ranked: Sequence[Ranked]) -> list[Ranked]:
    return [r for r in ranked if r.hit]


def select_set(ordered: Sequence[Component],
               joint_recovery: Callable[[Sequence[Component]], float],
               *, target: float = SET_TARGET, k_max: int = SET_K_MAX
               ) -> tuple[list[Component], list[float]]:
    """Smallest prefix of `ordered` whose **joint** recovery reaches `target`.

    Joint, not summed: recoveries of separate patches do not add, and treating
    them as if they did is how a set of ten heads that each "recover 0.1" gets
    reported as a full restoration. `joint_recovery` is supplied by the caller
    -- on the box it patches the whole set at once and measures; in tests it is
    a stub -- which is what keeps this function torch-free and testable.

    Returns the set and the dose-response curve behind it (PLAN2.md 4.4 asks
    for that curve, so it is produced here rather than recomputed later).
    """
    curve: list[float] = []
    for k in range(1, min(k_max, len(ordered)) + 1):
        curve.append(float(joint_recovery(ordered[:k])))
        if curve[-1] >= target:
            return list(ordered[:k]), curve
    return list(ordered[: min(k_max, len(ordered))]), curve


def random_sets(pool: Sequence[Component], k: int, n: int = 200,
                seed: int = 0) -> list[list[Component]]:
    """`n` random component sets of size `k` -- the set-level null (PLAN2.md 5).

    Seeded, because "any k heads would do this" has to be re-runnable by
    someone reading the repo.
    """
    rng = np.random.default_rng(seed)
    pool = list(pool)
    if k > len(pool):
        raise ValueError(f"cannot draw {k} components from a pool of {len(pool)}")
    return [[pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
            for _ in range(n)]


def set_bar(null_recoveries: Sequence[float],
            percentile: float = SET_NULL_PERCENTILE) -> float:
    vals = np.array([v for v in null_recoveries if np.isfinite(v)], dtype=float)
    return float(np.percentile(vals, percentile)) if vals.size else float("inf")


def jaccard(a: Iterable[Component], b: Iterable[Component]) -> float:
    """|A n B| / |A u B|. 1.0 = the same components; 0.0 = disjoint.

    V6 in one number: if the deception set and the inversion set are identical
    this is 1.0, and the object is renamed *output-override components*
    throughout (PLAN2.md 10). Two empty sets are 1.0 by convention -- and a V6
    read off two empty sets is not a result, which is why the notebook prints
    the sizes next to it.
    """
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def overlap_null(pool: Sequence[Component], k: int, n: int = 200,
                 seed: int = 0) -> np.ndarray:
    """Jaccard between two independent random size-`k` sets, `n` times.

    The number V6's overlap has to beat. Two sets drawn from a 300-component
    pool overlap a little by chance, and "the sets share three heads" means
    nothing without knowing what chance gives.
    """
    left = random_sets(pool, k, n, seed=seed)
    right = random_sets(pool, k, n, seed=seed + 1)
    return np.array([jaccard(a, b) for a, b in zip(left, right)], dtype=float)


# -- families ---------------------------------------------------------------

LEGIBLE = "legible"      # a_H legible in the J-lens under D at some layer (04: 4/11)
NEVER = "never"          # a_H never legible under D (04: 7/11)


def families(curves, condition: str = "D") -> dict[str, list[str]]:
    """Split 04's curves into the two families, by whether `a_H` was ever legible.

    Fixed **before** any patch runs (exp2_spec.md 2a). The `NEVER` family is not
    filler: if patching recovers `a_H` there too, the belief was present in a
    form the J-lens could not read -- a finding about the instrument, and one
    that is invisible if those items are dropped for having no signal.
    """
    out: dict[str, list[str]] = {LEGIBLE: [], NEVER: []}
    for c in curves:
        if c.condition != condition:
            continue
        readable = c.readable()
        led = bool(np.any(readable & (c.j_margin > 0)))
        out[LEGIBLE if led else NEVER].append(c.item_id)
    return out


# -- persistence ------------------------------------------------------------

def save_results(rows: Sequence[PatchResult], path: str | Path) -> Path:
    """Write the patch table as JSON.

    Called after **every item**, not at the end. 04 lost its first V2 gate
    result to a save cell that sat at the bottom of the notebook and was never
    reached; a save that only runs when nothing went wrong is the opposite of
    what a save is for.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{
        "item_id": r.item_id, "condition": r.condition, "component": r.component.name,
        "layer_type": r.component.layer_type, "ld_deceptive": r.ld_deceptive,
        "ld_honest": r.ld_honest, "ld_patched": r.ld_patched,
        "source": r.source, "family": r.family,
    } for r in rows]
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_results(path: str | Path) -> list[PatchResult]:
    return [PatchResult(
        item_id=d["item_id"], condition=d["condition"],
        component=Component.parse(d["component"], d.get("layer_type", "")),
        ld_deceptive=d["ld_deceptive"], ld_honest=d["ld_honest"],
        ld_patched=d["ld_patched"], source=d.get("source", ""),
        family=d.get("family", ""),
    ) for d in json.loads(Path(path).read_text())]


def _num(value: float | None) -> str:
    return "     -- " if value is None else f"{value:>7.3f}"


def report(ranked: Sequence[Ranked], limit: int = 20) -> str:
    """The ranking table, hits marked. Layer type is printed: R4.

    When the arms are known, both are printed next to the pooled median. A
    component that clears pooled and fails one arm is the case worth seeing --
    it is what a `' Yes'`-writer looks like on a set the gate left slightly
    unbalanced -- and printing only `hit` would hide it behind a blank column.
    """
    armed = bool(ranked) and ranked[0].arms_required
    header = (f"{'component':<10} {'type':<18} {'n':>3} {'median':>8} "
              + (f"{'Yes-arm':>8} {'No-arm':>8} {'arm gap':>8} " if armed else "")
              + f"{'null p95':>9} {'margin':>8}  hit")
    lines = [header]
    for r in ranked[:limit]:
        lines.append(
            f"{r.component.name:<10} {r.component.layer_type:<18} {r.n_items:>3} "
            f"{r.median:>8.3f} "
            + (f"{_num(r.median_yes)} {_num(r.median_no)} {r.arm_gap:>8.3f} "
               if armed else "")
            + f"{r.null_bar:>9.3f} {r.margin:>8.3f}  "
            + ("YES" if r.hit else
               "one arm" if armed and r.median >= r.min_recovery
               and r.median > r.null_bar else ""))
    n_hits = len(hits(ranked))
    bar = ranked[0].min_recovery if ranked else MIN_RECOVERY
    pct = ranked[0].percentile if ranked else NULL_PERCENTILE
    lines.append(f"\n{n_hits} of {len(ranked)} components clear "
                 f"{'all three bars' if armed else 'both bars'} "
                 f"(median >= {bar}, above own p{pct:g} null"
                 + (", and >= the bar in BOTH polarity arms)" if armed else ")"))
    return "\n".join(lines)


# ==========================================================================
# Torch: the patcher itself
# ==========================================================================
#
# Nothing below imports torch at module level. `config.py` says why: this
# module is imported by the local test suite, which has no torch at all.


@dataclass
class DLAResult:
    """Direct logit attribution over every component, and its honesty check.

    `residual_frac` is what makes the table usable or not. The decomposition
    below is **exact for the observed run** -- see `Patcher.dla` -- so a large
    residual means something in the assumed graph is wrong (softcapping, a
    sublayer that does not write where we think it does), not that the
    approximation is loose. exp2_spec.md 5.1 fixes the bar at 5 %.
    """

    contributions: dict[Component, float]
    embed: float
    ld_true: float
    layer_split_err: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(self.contributions.values()) + self.embed)

    @property
    def residual(self) -> float:
        return self.ld_true - self.total

    @property
    def residual_frac(self) -> float:
        denom = abs(self.ld_true)
        return abs(self.residual) / denom if denom > 1e-6 else float("inf")

    @property
    def complete(self) -> bool:
        return self.residual_frac <= 0.05

    def ordered(self) -> list[tuple[Component, float]]:
        return sorted(self.contributions.items(), key=lambda kv: -abs(kv[1]))


class Patcher:
    """Read and write one component at the answer slot, via nnsight.

    Every module path is **discovered** from `named_modules`, never hardcoded:
    `gemma-3-4b-it` loads as the multimodal wrapper, so its decoder layers sit
    one level deeper than 270m's, and a hardcoded `model.model.layers` is a bug
    that only appears once a 4090 is already billing.
    """

    def __init__(self, model, tok):
        from nnsight import LanguageModel

        self.model = model
        self.tok = tok
        self.lm = LanguageModel(model, tokenizer=tok)

        names = dict(model.named_modules())
        o_proj = next((n for n in names if n.endswith("layers.0.self_attn.o_proj")), None)
        if o_proj is None:
            raise RuntimeError("no `layers.0.self_attn.o_proj`; unsupported architecture")
        self.prefix = o_proj[: -len(".0.self_attn.o_proj")]
        self.stem = self.prefix.rsplit(".layers", 1)[0]

        cfg = getattr(model.config, "text_config", model.config)
        self.cfg = cfg
        self.n_layers = cfg.num_hidden_layers
        self.d_model = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // self.n_heads)

        # R4. `layer_types` is authoritative where the transformers version
        # provides it; the pattern fallback is derived, and labelled as such so
        # a claim about sliding vs full heads is never made on a guess.
        types = getattr(cfg, "layer_types", None)
        if types and len(types) == self.n_layers:
            self.layer_types = list(types)
        else:
            period = getattr(cfg, "sliding_window_pattern", None) or 6
            self.layer_types = ["full_attention" if (i + 1) % period == 0
                                else "sliding_attention?" for i in range(self.n_layers)]

        self.norm_name = f"{self.stem}.norm" if f"{self.stem}.norm" in names else None
        self.has_post_ffw = f"{self.prefix}.0.post_feedforward_layernorm" in names
        self.softcap = getattr(cfg, "final_logit_softcapping", None)

    # -- naming ------------------------------------------------------------

    def _o_proj(self, layer: int) -> str:
        return f"{self.prefix}.{layer}.self_attn.o_proj"

    def _mlp(self, layer: int) -> str:
        return f"{self.prefix}.{layer}.mlp"

    def _post_attn_ln(self, layer: int) -> str:
        return f"{self.prefix}.{layer}.post_attention_layernorm"

    def _resid_write(self, layer: int) -> str:
        """The module whose output *is* the MLP's residual increment.

        Gemma 3 norms the MLP output before adding it back, so on Gemma the
        residual increment is `post_feedforward_layernorm`'s output, not the
        MLP's. On an architecture without that module they are the same thing.
        """
        return (f"{self.prefix}.{layer}.post_feedforward_layernorm"
                if self.has_post_ffw else self._mlp(layer))

    def _envoy(self, path: str):
        obj = self.lm
        for part in path.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        return obj

    def _module(self, path: str):
        return dict(self.model.named_modules())[path]

    def head_slice(self, head: int) -> slice:
        """GQA: the o_proj input is `n_q_heads x head_dim`, NOT `d_model`."""
        return slice(head * self.head_dim, (head + 1) * self.head_dim)

    def o_proj_weight(self, layer: int):
        """`W_O` for a layer, as `[d_model, n_q_heads * head_dim]`."""
        return self._module(self._o_proj(layer)).weight

    def top_tokens(self, prompt: str, k: int = 2) -> list[int]:
        """The model's own top-`k` ids at the slot -- a probe pair with no
        vocabulary assumption. `" Berlin"` being one token is a guess; the
        model's own top-2 never is."""
        return [int(i) for i in self.slot_logits(prompt).topk(k).indices]

    def components(self, kinds: Sequence[str] = (HEAD, MLP)) -> list[Component]:
        out: list[Component] = []
        for layer in range(self.n_layers):
            lt = self.layer_types[layer]
            if HEAD in kinds:
                out += [Component(layer, HEAD, h, lt) for h in range(self.n_heads)]
            if MLP in kinds:
                out.append(Component(layer, MLP, None, lt))
        return out

    # -- forward -----------------------------------------------------------

    def token_id(self, text: str) -> int:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{text!r} is {len(ids)} tokens, not 1; the slot trick needs 1")
        return ids[0]

    def slot_logits(self, prompt: str):
        import torch

        with torch.no_grad():
            enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
            return self.model(**enc).logits[0, -1].float()

    def logit_diff(self, prompt: str, id_honest: int, id_lie: int) -> float:
        lg = self.slot_logits(prompt)
        return float(lg[id_honest] - lg[id_lie])

    def cache_slot(self, prompt: str) -> dict[Component, object]:
        """Every component's output at the answer slot, in one forward pass.

        This is the honest run's contribution to the patch: what gets written
        into the deceptive run, one component at a time.
        """
        saved: dict[tuple, object] = {}
        with self.lm.trace(prompt):
            for layer in range(self.n_layers):
                z = _tensor(self._envoy(self._o_proj(layer)).input)
                saved[(HEAD, layer)] = z[0, -1].save()
                m = _tensor(self._envoy(self._mlp(layer)).output)
                saved[(MLP, layer)] = m[0, -1].save()

        out: dict[Component, object] = {}
        for layer in range(self.n_layers):
            lt = self.layer_types[layer]
            z = saved[(HEAD, layer)]
            for h in range(self.n_heads):
                out[Component(layer, HEAD, h, lt)] = z[self.head_slice(h)].clone()
            out[Component(layer, MLP, None, lt)] = saved[(MLP, layer)].clone()
        return out

    def cache_resid(self, prompt: str, layers: Sequence[int], position: int = -1):
        """The **residual stream** after each of `layers`, at one position.

        `layer l` here means *the output of decoder block `l`* -- what the next
        block reads. That convention is stated because it is the one thing about
        this method that can be silently wrong: 04b spends a whole gate (V-I1)
        on the same question, because a residual taken one block away from where
        it is reported is a confident wrong answer rather than an error.

        Returned as float32 numpy, not as a bf16 tensor. Everything downstream
        of this is `geometry`, which is torch-free and tested without a GPU, and
        converting once here is what keeps that true.

        `position=-1` is the answer slot, the same position `cache_slot` and
        `logit_diff` read, so a geometry claim and a logit claim are about the
        same vector.
        """
        with self.lm.trace(prompt):
            saved = {l: _tensor(self._envoy(f"{self.prefix}.{l}").output)[0, position].save()
                     for l in layers}
        return {l: v.detach().float().cpu().numpy() for l, v in saved.items()}

    def patched_logit_diff(self, prompt: str, patches, id_honest: int, id_lie: int) -> float:
        """Run `prompt` with each component's slot value replaced, return LD.

        `patches` maps Component -> the value cached from the other run. An
        empty mapping is a no-op trace and must reproduce `logit_diff` exactly;
        05 asserts that before any science, because a write that silently does
        nothing is indistinguishable from a component that does not matter.
        """
        with self.lm.trace(prompt):
            for comp, value in patches.items():
                if comp.kind == HEAD:
                    z = _tensor(self._envoy(self._o_proj(comp.layer)).input)
                    z[0, -1, self.head_slice(comp.head)] = value
                else:
                    m = _tensor(self._envoy(self._mlp(comp.layer)).output)
                    m[0, -1, :] = value
            logits = self.lm.output.logits[0, -1].save()
        lg = logits.float()
        return float(lg[id_honest] - lg[id_lie])

    # -- DLA ---------------------------------------------------------------

    def dla(self, prompt: str, id_honest: int, id_lie: int) -> DLAResult:
        """Direct contribution of every component to `LD`, at the answer slot.

        **Frozen-scale, and what that does and does not mean.** Gemma 3 writes
        `resid += post_attention_layernorm(attn_out)`, and RMSNorm is not
        linear -- so a head's contribution cannot simply be read off its output.
        Taking each norm's scale factor *from the observed forward pass* and
        holding it fixed makes the split linear again, because
        `RMSNorm(x) = x * s(x) * (1 + w)` and `s` is a scalar: distributing it
        over the heads that sum to `x` is an **identity**, not an
        approximation. This decomposition is therefore exact for the run it
        describes. What it is *not* is a counterfactual -- removing a head
        would change `s` -- which is precisely why PLAN2.md 4.3 has DLA propose
        and patching rank.

        The completeness residual is printed for that reason: it cannot detect
        a bad approximation (there isn't one), but it does detect a wrong graph
        -- a sublayer that writes somewhere other than where this code thinks.
        """
        import torch

        if self.softcap:
            raise RuntimeError(
                f"final_logit_softcapping={self.softcap} makes the logits a nonlinear "
                "function of the residual stream; DLA on raw logits is invalid here")
        if self.norm_name is None:
            raise RuntimeError(f"no final norm at {self.stem}.norm; cannot unembed")

        saved: dict[tuple, object] = {}
        with self.lm.trace(prompt):
            for layer in range(self.n_layers):
                saved[("z", layer)] = _tensor(
                    self._envoy(self._o_proj(layer)).input)[0, -1].save()
                saved[("attn_out", layer)] = _tensor(
                    self._envoy(self._o_proj(layer)).output)[0, -1].save()
                saved[("attn_add", layer)] = _tensor(
                    self._envoy(self._post_attn_ln(layer)).output)[0, -1].save()
                saved[("mlp_add", layer)] = _tensor(
                    self._envoy(self._resid_write(layer)).output)[0, -1].save()
            saved["resid_final"] = _tensor(self._envoy(self.norm_name).input)[0, -1].save()
            saved["logits"] = self.lm.output.logits[0, -1].save()

        W_U = self._module("lm_head").weight
        u = (W_U[id_honest].float() - W_U[id_lie].float())
        final_norm = self._module(self.norm_name)
        resid_final = saved["resid_final"].float()

        def to_logit_diff(v):
            """A residual-stream increment's direct effect on LD."""
            return float(_frozen_norm(v, final_norm, resid_final) @ u)

        contributions: dict[Component, float] = {}
        split_err = 0.0
        for layer in range(self.n_layers):
            lt = self.layer_types[layer]
            attn_out = saved[("attn_out", layer)].float()
            attn_add = saved[("attn_add", layer)].float()
            post_ln = self._module(self._post_attn_ln(layer))
            W_O = self._module(self._o_proj(layer)).weight.float()
            z = saved[("z", layer)].float()

            parts = []
            for h in range(self.n_heads):
                sl = self.head_slice(h)
                part = W_O[:, sl] @ z[sl]
                add = _frozen_norm(part, post_ln, attn_out)
                parts.append(add)
                contributions[Component(layer, HEAD, h, lt)] = to_logit_diff(add)
            # The identity above, checked rather than asserted in prose.
            split_err = max(split_err,
                            float((torch.stack(parts).sum(0) - attn_add).abs().max()))
            contributions[Component(layer, MLP, None, lt)] = to_logit_diff(
                saved[("mlp_add", layer)].float())

        written = sum(saved[("attn_add", l)].float() + saved[("mlp_add", l)].float()
                      for l in range(self.n_layers))
        embed = to_logit_diff(resid_final - written)

        lg = saved["logits"].float()
        return DLAResult(contributions=contributions, embed=embed,
                         ld_true=float(lg[id_honest] - lg[id_lie]),
                         layer_split_err=split_err)

    def describe(self) -> str:
        n_full = sum(t == "full_attention" for t in self.layer_types)
        return (f"{self.n_layers} layers x {self.n_heads} heads (head_dim={self.head_dim}, "
                f"d_model={self.d_model}) + {self.n_layers} MLPs = "
                f"{len(self.components())} components\n"
                f"layers at: model.{self.prefix}   final norm: model.{self.norm_name}\n"
                f"attention: {n_full} full, {self.n_layers - n_full} sliding "
                f"(R4: never pooled)\n"
                f"mlp residual increment read from: "
                f"{'post_feedforward_layernorm' if self.has_post_ffw else 'mlp'}")


def _tensor(value):
    """nnsight hands back a tuple for some modules and a tensor for others."""
    return value[0] if isinstance(value, tuple) else value


def _frozen_norm(part, norm_module, full):
    """`part`'s share of `RMSNorm(full)`, with the scale frozen at the observed run.

    `RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * (1 + w)` in Gemma. The scale is a
    scalar computed from the *whole* input, so distributing it across the parts
    that sum to that input is an identity. Gemma's `(1 + w)` is not the usual
    `w`; using `w` would scale every contribution wrongly and still look
    plausible, so it is read off the module rather than assumed.
    """
    import torch

    eps = getattr(norm_module, "eps", None)
    if eps is None:
        eps = getattr(norm_module, "variance_epsilon", 1e-6)
    scale = torch.rsqrt(full.float().pow(2).mean() + eps)
    weight = norm_module.weight.float()
    gain = (1.0 + weight) if _gemma_style(norm_module) else weight
    return part.float() * scale * gain


def _gemma_style(norm_module) -> bool:
    """Gemma RMSNorm stores `w` around zero and applies `(1 + w)`."""
    return type(norm_module).__name__.startswith("Gemma")


# ==========================================================================
# Drivers: the sweeps the notebook calls
# ==========================================================================


def render_default(tok, item, condition):
    from nandaproj import items as items_mod

    return items_mod.render(tok, item, condition)


def patch_sweep(patcher: Patcher, item_list: Sequence, ld: dict, *,
                condition: str = "D", components: Sequence[Component] | None = None,
                sources: dict | None = None, family_of: dict | None = None,
                by_id: dict | None = None, save_to: str | Path | None = None,
                desc: str = "patching") -> list[PatchResult]:
    """Patch every component, one at a time, for every item.

    `sources` maps an item id to the id of the item whose **honest** run supplies
    the activation. Identity (the default) is the real measurement; a derangement
    is the wrong-source null of `exp2_spec.md` §5.3. One code path produces both,
    which is the only way to be sure the null was computed the same way as the
    signal -- a null with its own loop is a null with its own bugs.

    `save_to` is written after **every item**, not at the end: a sweep that dies
    at item 8 of 11 leaves 8 on disk. 04 lost its first gate result to a save
    cell at the bottom of the notebook.
    """
    from nandaproj.lens_readout import _progress

    comps = list(components if components is not None else patcher.components())
    sources = sources or {}
    family_of = family_of or {}
    by_id = by_id or {it.item_id: it for it in item_list}

    rows: list[PatchResult] = []
    total = len(item_list) * len(comps)
    bar = _progress(range(total), total=total, desc=desc)
    step = iter(bar)

    for item in item_list:
        entry = ld[item.item_id]
        prompt = render_default(patcher.tok, item, condition)
        source_id = sources.get(item.item_id, item.item_id)
        cache = patcher.cache_slot(
            render_default(patcher.tok, by_id[source_id], "H"))

        if condition not in entry:
            # Falling back to entry["D"] here would compute C2's recoveries
            # against D's baseline -- every C2 number wrong, none of them
            # obviously so, and V6 decided on the mismatch.
            raise KeyError(
                f"no baseline LD for condition {condition!r} on {item.item_id}; "
                f"have {sorted(k for k in entry if k not in ('id_h', 'id_d'))}")

        for comp in comps:
            ld_patched = patcher.patched_logit_diff(
                prompt, {comp: cache[comp]}, entry["id_h"], entry["id_d"])
            rows.append(PatchResult(
                item_id=item.item_id, condition=condition, component=comp,
                ld_deceptive=entry[condition],
                ld_honest=entry["H"], ld_patched=ld_patched,
                source=source_id, family=family_of.get(item.item_id, ""),
            ))
            next(step, None)

        if save_to is not None:
            save_results(rows, save_to)
    return rows


def derangement(item_ids: Sequence[str], seed: int = 0) -> dict[str, str]:
    """Map each item to a *different* item -- the wrong-source null's pairing.

    A fixed point would patch an item with its own honest activation, which is
    the signal, and one such pair silently inflates the null bar it is supposed
    to define. Seeded so the null is re-runnable by someone reading the repo.
    """
    ids = list(item_ids)
    if len(ids) < 2:
        raise ValueError("a wrong-source null needs at least two items")
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        shuffled = list(rng.permutation(ids))
        if all(a != b for a, b in zip(ids, shuffled)):
            return dict(zip(ids, shuffled))
    raise RuntimeError("no derangement found")           # pragma: no cover


def joint_recovery_fn(patcher: Patcher, item_list: Sequence, ld: dict, *,
                      condition: str = "D") -> Callable[[Sequence[Component]], float]:
    """A callable that patches a whole set at once and returns median recovery.

    Handed to `select_set`. Joint, because separate recoveries do not add: ten
    heads that each "recover 0.1" are not a full restoration, and summing them
    is how a distributed non-result gets written up as a circuit.
    """
    def measure(comps: Sequence[Component]) -> float:
        vals = []
        for item in item_list:
            entry = ld[item.item_id]
            cache = patcher.cache_slot(render_default(patcher.tok, item, "H"))
            ld_p = patcher.patched_logit_diff(
                render_default(patcher.tok, item, condition),
                {c: cache[c] for c in comps}, entry["id_h"], entry["id_d"])
            vals.append(PatchResult(
                item_id=item.item_id, condition=condition, component=comps[0],
                ld_deceptive=entry[condition], ld_honest=entry["H"], ld_patched=ld_p,
                source=item.item_id).recovery)
        vals = np.array(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        return float(np.median(vals)) if vals.size else float("nan")

    return measure


def dla_sweep(patcher: Patcher, item_list: Sequence, ld: dict, *,
              condition: str = "D") -> tuple[dict[Component, float], list[DLAResult]]:
    """DLA for every item, and the median contribution per component.

    Proposes only. PLAN2.md §4.3: the causal ranking is patching, and DLA's job
    is to say which components are worth looking at first if the full sweep ever
    has to be cut short.
    """
    from nandaproj.lens_readout import _progress

    results: list[DLAResult] = []
    for item in _progress(item_list, total=len(item_list), desc="dla"):
        entry = ld[item.item_id]
        results.append(patcher.dla(
            render_default(patcher.tok, item, condition), entry["id_h"], entry["id_d"]))

    per_comp: dict[Component, list[float]] = {}
    for res in results:
        for comp, val in res.contributions.items():
            per_comp.setdefault(comp, []).append(val)
    return {c: float(np.median(v)) for c, v in per_comp.items()}, results
