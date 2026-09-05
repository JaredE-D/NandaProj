"""Which upstream run 05 attributes, and which of its items get swept.

Everything here is bookkeeping that happens **before** the first patch, and all
of it is the kind that fails silently. An arm that is empty because a file lost
a field, a subsample that quietly drops the scarce polarity, a pair index built
from half-pairs -- none of these raise, and each one turns into "no component
cleared the bar", which is a sentence 05 is otherwise in the business of
producing honestly.

So it lives in a module with tests rather than in a notebook cell.

**The two banks this has to serve.**

| run tag | bank | shape |
|---|---|---|
| `""` | 04's v2 bank | whole polarity pairs, 15 Yes-true and 15 No-true |
| `"alleged"` | 04c's alleged-fault arm | a gated *arm*: mostly No-true, no whole pairs |

The alleged arm is not paired because it could not be. Its Yes-true twin asks
the model to answer `" No"` about a sound object -- to invent a defect it has
just been told does not exist -- and across 390 authored pairs it did so 12
times. What survives is a set of lone twins, so anything that assumes whole
pairs has to degrade to a report rather than raise.

Torch-free by construction: this is the arithmetic that decides what gets
measured, and it is testable without a GPU.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nandaproj import items as items_mod
from nandaproj import polarity as polarity_mod

YES, NO = "Yes", "No"


def polarity_of(item: Any) -> str:
    """Which answer is **true** for `item`: `"Yes"` or `"No"`.

    Read from the bank's declared `polarity`, with the measured `answer_honest`
    as the fallback. The declared value is the one the item was authored
    against; deriving the arm from a measured answer would make the subgroup a
    function of the behaviour being measured, which is the one thing a subgroup
    must not be.

    On a gated bank the two agree by construction -- `usable` means the model's
    answer under H matched the stated one -- so a disagreement means the gate
    let something through, and it raises here rather than silently putting an
    item in the wrong arm.
    """
    declared = str(item.meta.get("polarity") or "").strip()
    measured = str(getattr(item, "answer_honest", None) or "").strip()
    if declared and measured and declared != measured:
        raise ValueError(
            f"{item.item_id}: declared polarity {declared!r} but the model answered "
            f"{measured!r} under H. The gate should have rejected this item.")
    if not (declared or measured):
        raise ValueError(
            f"{item.item_id} carries neither a declared `polarity` nor an "
            "`answer_honest`; it cannot be put in a polarity arm. This is usually a "
            "gated bank written before `to_json` round-tripped `meta` -- see "
            "`with_static_meta`.")
    return declared or measured


def with_static_meta(gated: Any, source: Any) -> Any:
    """`gated`, with the `meta` fields it lacks filled in from `source`.

    The gated bank is the right file to attribute -- it carries the answers the
    model actually gave -- but 04's `to_json` dropped `meta`, so
    `gated_bank_<lens>.json` has no `polarity` and no `pair_id`. Left alone that
    is not an error anywhere downstream: `attribution.rank` refuses to promote a
    component it cannot score on both arms, so the notebook prints a full
    ranking with no hits and no stated reason.

    The target's own meta always wins, so a measured value is never overwritten
    by the bank's stated one, and a bank that already round-trips its meta
    passes through unchanged.
    """
    return items_mod.LoadResult(
        items=items_mod.attach_meta(gated.items, source.items),
        used_templates=gated.used_templates,
        source=gated.source,
    )


def whole_pairs(items: Iterable[Any]) -> tuple[dict[str, tuple[Any, Any]], int]:
    """`(pair_index over complete pairs, number of pair ids seen)`.

    `polarity.pair_index` requires exactly one Yes-true and one No-true item per
    `pair_id` and raises otherwise. That is right for a paired bank -- a
    half-pair silently dropped is how a polarity skew comes back -- and wrong
    for a gated arm, where almost every `pair_id` is a lone survivor and the
    raise would take out the whole notebook.

    So the complete pairs are handed to `pair_index` (which still validates
    them), and the incomplete ones are counted. The count is not noise: it is
    the number that says *why* the within-pair control is unavailable.
    """
    groups: dict[str, list[Any]] = {}
    for it in items:
        pid = it.meta.get("pair_id")
        if pid:
            groups.setdefault(str(pid), []).append(it)
    complete = [g for g in groups.values()
                if len(g) == 2 and {polarity_of(i) for i in g} == {YES, NO}]
    return polarity_mod.pair_index([i for g in complete for i in g]), len(groups)


@dataclass(frozen=True)
class Selection:
    """The items 05 will sweep, and the rule that chose them.

    Written to disk *before* the first patch. A subsample chosen after the
    ranking has been looked at is not a subsample, and the only thing that makes
    that check possible later is the file's timestamp next to the sweep's.
    """

    item_ids: list[str]
    polarity_of: dict[str, str]
    family_of: dict[str, str]
    n_available: int
    seed: int
    max_items: int | None
    run_tag: str = ""

    @property
    def n_yes(self) -> int:
        return sum(v == YES for v in self.polarity_of.values())

    @property
    def n_no(self) -> int:
        return sum(v == NO for v in self.polarity_of.values())

    @property
    def one_sided(self) -> bool:
        """No arm contrast at all -- `attribution.rank` will refuse everything."""
        return min(self.n_yes, self.n_no) == 0

    def by_family(self, family: str) -> list[str]:
        return [i for i in self.item_ids if self.family_of.get(i) == family]

    def summary(self) -> str:
        head = (f"{len(self.item_ids)} of {self.n_available} gated items"
                + (f", subsampled at seed {self.seed}"
                   if len(self.item_ids) < self.n_available else ""))
        return f"{head} | {self.n_yes} Yes-true, {self.n_no} No-true"

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "run_tag": self.run_tag, "seed": self.seed, "max_items": self.max_items,
            "n_available": self.n_available, "item_ids": self.item_ids,
            "polarity_of": self.polarity_of, "family_of": self.family_of,
        }, indent=1))
        return path


def choose(families: Mapping[str, Sequence[str]],
           polarities: Mapping[str, str],
           *, max_items: int | None = None, seed: int = 0,
           run_tag: str = "") -> Selection:
    """Pick the sweep set: **every** minority-polarity item, then fill balanced.

    306 components x n items x 3 sweeps (null, D, C2) at ~30 ms a pass, so the
    item count is the cost. 30 items is ~2 1/4 h; the whole alleged bank is
    ~9 1/2 h.

    The stratification is by polarity **first** and family second, and that
    ordering is the point. 05's `hit` bar asks a component to clear
    `MIN_RECOVERY` on Yes-true *and* No-true items, which is what separates
    "restores the belief" from "writes `' Yes'`". On the alleged arm the
    Yes-true items are 14 of 141, so a family-stratified draw of 32 would take
    three or four of them and the bar would be decided by an arm of n=3. Keeping
    every scarce-polarity item and filling the remainder round-robin over the
    families spends the budget where the bar actually lives.

    The fill is round-robin rather than proportional so the families stay even
    as the budget shrinks: `exp2_spec.md` 2a fixed LEGIBLE/NEVER in advance and
    still reports them, and a fill weighted by availability would hand NEVER 3/4
    of a small budget purely because it is the bigger family.

    `max_items=None` (or a budget at least as large as the bank) sweeps
    everything and the seed does not matter.
    """
    ids = sorted({i for group in families.values() for i in group})
    missing = [i for i in ids if i not in polarities]
    if missing:
        raise ValueError(
            f"{len(missing)} items have no polarity, e.g. {missing[:5]}. An item with "
            "no arm can never be a hit, so this would silently shrink the measurement.")

    family_of = {i: fam for fam, group in families.items() for i in group}

    def selection(chosen: Sequence[str]) -> Selection:
        chosen = sorted(chosen)
        return Selection(
            item_ids=list(chosen),
            polarity_of={i: polarities[i] for i in chosen},
            family_of={i: family_of[i] for i in chosen},
            n_available=len(ids), seed=seed, max_items=max_items, run_tag=run_tag)

    if max_items is None or max_items >= len(ids):
        return selection(ids)

    counts = {YES: sum(polarities[i] == YES for i in ids),
              NO: sum(polarities[i] == NO for i in ids)}
    scarce = YES if counts[YES] <= counts[NO] else NO
    plentiful = NO if scarce == YES else YES

    rng = random.Random(seed)
    picked = [i for i in ids if polarities[i] == scarce][:max_items]
    pools = []
    for fam in sorted(families):
        pool = sorted(i for i in families[fam] if polarities[i] == plentiful)
        rng.shuffle(pool)
        pools.append(pool)
    while len(picked) < max_items and any(pools):
        for pool in pools:
            if pool and len(picked) < max_items:
                picked.append(pool.pop())
    return selection(picked)
