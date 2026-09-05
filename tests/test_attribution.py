"""Tests for the parts of `attribution` that do not need a GPU.

`Patcher` needs a model, so it is exercised by 05's assert cells on the box.
What *is* testable here is the arithmetic that turns per-component numbers into
a claim -- the recovery ratio, the bars, the wrong-source null, set selection,
the V6 overlap -- which is exactly where a wrong result would look like a
finding rather than like an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from nandaproj import attribution as at


def row(component=None, *, ld_d=0.0, ld_h=10.0, ld_p=5.0, item="I1",
        source=None, condition="D", family=at.LEGIBLE) -> at.PatchResult:
    comp = component if component is not None else at.Component(24, at.HEAD, 3)
    return at.PatchResult(
        item_id=item, condition=condition, component=comp,
        ld_deceptive=ld_d, ld_honest=ld_h, ld_patched=ld_p,
        source=item if source is None else source, family=family,
    )


# -- components ------------------------------------------------------------

def test_component_names_round_trip():
    for comp in (at.Component(24, at.HEAD, 3, "full_attention"),
                 at.Component(0, at.MLP, None, "sliding_attention")):
        assert at.Component.parse(comp.name, comp.layer_type) == comp


def test_a_head_component_must_carry_a_head_index():
    """A head with `head=None` would silently patch the whole o_proj input --
    every head at once -- and report it as one head's effect."""
    with pytest.raises(ValueError):
        at.Component(24, at.HEAD, None)
    with pytest.raises(ValueError):
        at.Component(24, at.MLP, 3)


def test_components_are_hashable_and_ordered():
    pool = {at.Component(24, at.HEAD, 3), at.Component(24, at.HEAD, 3)}
    assert len(pool) == 1
    assert at.Component(2, at.HEAD, 0) < at.Component(24, at.HEAD, 0)


# -- recovery --------------------------------------------------------------

def test_recovery_is_zero_when_the_patch_does_nothing():
    assert row(ld_d=0.0, ld_h=10.0, ld_p=0.0).recovery == 0.0


def test_recovery_is_one_when_the_patch_restores_the_honest_logit_diff():
    assert row(ld_d=0.0, ld_h=10.0, ld_p=10.0).recovery == 1.0


def test_recovery_handles_a_negative_gap():
    """Under D the model asserts a_D, so LD_D is negative and LD_H positive.
    The ratio must be signed by the gap, not by the raw difference."""
    assert row(ld_d=-6.0, ld_h=6.0, ld_p=0.0).recovery == 0.5


def test_recovery_is_nan_when_the_two_runs_barely_differ():
    """A 0.3-logit 'gap' divided into a 0.3-logit patch effect reads as a full
    recovery on an item that never had a gap to close (exp2_spec.md 10)."""
    r = row(ld_d=0.0, ld_h=0.3, ld_p=0.3)
    assert not r.usable
    assert np.isnan(r.recovery)


def test_unusable_rows_are_dropped_from_medians_not_averaged_in():
    rows = [row(ld_d=0.0, ld_h=10.0, ld_p=2.0), row(ld_d=0.0, ld_h=10.0, ld_p=4.0),
            row(ld_d=0.0, ld_h=0.1, ld_p=0.1)]          # unusable, would read as 1.0
    assert at.median_recovery(rows) == pytest.approx(0.3)


# -- the wrong-source null -------------------------------------------------

def test_a_row_patched_from_another_item_is_null_not_signal():
    assert not row(item="I1", source="I1").is_null
    assert row(item="I1", source="I2").is_null


def test_rank_measures_each_component_against_its_own_null():
    comp = at.Component(25, at.HEAD, 1)
    rows = [row(comp, ld_p=8.0, item=f"I{i}") for i in range(4)]          # median 0.8
    rows += [row(comp, ld_p=1.0, item=f"I{i}", source="OTHER") for i in range(4)]
    (r,) = at.rank(rows)
    assert r.median == pytest.approx(0.8)
    assert r.null_bar == pytest.approx(0.1)
    assert r.hit


def test_a_component_that_moves_the_output_regardless_of_content_is_not_a_hit():
    """The whole point of the wrong-source null: a big effect from someone
    else's activation means the component is perturbation-sensitive, not that
    it carries the honest answer."""
    comp = at.Component(25, at.HEAD, 2)
    rows = [row(comp, ld_p=7.0, item=f"I{i}") for i in range(4)]          # median 0.7
    rows += [row(comp, ld_p=7.5, item=f"I{i}", source="OTHER") for i in range(4)]
    (r,) = at.rank(rows)
    assert r.median >= at.MIN_RECOVERY
    assert not r.hit


def test_a_component_with_no_null_rows_can_never_be_a_hit():
    """A missing null is a pipeline bug, not a licence to promote a component."""
    comp = at.Component(25, at.HEAD, 3)
    (r,) = at.rank([row(comp, ld_p=10.0, item=f"I{i}") for i in range(4)])
    assert r.null_bar == float("inf")
    assert not r.hit


def test_a_strong_component_below_the_absolute_bar_is_not_a_hit():
    comp = at.Component(3, at.HEAD, 0)
    rows = [row(comp, ld_p=1.0, item=f"I{i}") for i in range(4)]          # 0.1 < 0.20
    rows += [row(comp, ld_p=0.0, item=f"I{i}", source="OTHER") for i in range(4)]
    assert not at.rank(rows)[0].hit


def test_bars_travel_on_the_row_so_a_table_cannot_be_re_judged():
    comp = at.Component(25, at.HEAD, 1)
    rows = [row(comp, ld_p=1.5, item=f"I{i}") for i in range(4)]          # 0.15
    rows += [row(comp, ld_p=0.0, item=f"I{i}", source="OTHER") for i in range(4)]
    assert not at.rank(rows)[0].hit
    loose = at.rank(rows, min_recovery=0.1)[0]
    assert loose.hit and loose.min_recovery == 0.1


def test_ranking_is_ordered_by_median_recovery():
    a, b = at.Component(25, at.HEAD, 0), at.Component(25, at.HEAD, 1)
    rows = [row(a, ld_p=2.0), row(b, ld_p=9.0),
            row(a, ld_p=2.0, source="X"), row(b, ld_p=1.0, source="X")]
    assert [r.component for r in at.rank(rows)] == [b, a]


# -- sets ------------------------------------------------------------------

def test_select_set_stops_at_the_smallest_set_reaching_the_target():
    pool = [at.Component(25, at.HEAD, h) for h in range(8)]
    joint = {1: 0.2, 2: 0.35, 3: 0.6}
    chosen, curve = at.select_set(pool, lambda s: joint[len(s)], target=0.5)
    assert chosen == pool[:3]
    assert curve == [0.2, 0.35, 0.6]


def test_select_set_respects_the_cap_and_still_returns_the_curve():
    pool = [at.Component(25, at.HEAD, h) for h in range(8)]
    chosen, curve = at.select_set(pool, lambda s: 0.01 * len(s), target=0.5, k_max=4)
    assert len(chosen) == 4 and len(curve) == 4


def test_joint_recovery_is_measured_not_summed():
    """Ten components each 'recovering 0.1' is not a full restoration. The
    callable exists so the set's effect is measured jointly on the box."""
    pool = [at.Component(25, at.HEAD, h) for h in range(10)]
    seen = []
    at.select_set(pool, lambda s: seen.append(len(s)) or 0.05, target=0.5)
    assert seen == list(range(1, 11))          # every prefix measured, none inferred


def test_random_sets_are_seeded_and_reproducible():
    pool = [at.Component(l, at.HEAD, h) for l in range(10) for h in range(4)]
    assert at.random_sets(pool, 5, 3, seed=7) == at.random_sets(pool, 5, 3, seed=7)
    assert at.random_sets(pool, 5, 3, seed=7) != at.random_sets(pool, 5, 3, seed=8)


def test_random_sets_draw_without_replacement():
    pool = [at.Component(l, at.HEAD, h) for l in range(10) for h in range(4)]
    for s in at.random_sets(pool, 6, 20, seed=1):
        assert len(set(s)) == 6


def test_random_sets_refuses_a_k_larger_than_the_pool():
    with pytest.raises(ValueError):
        at.random_sets([at.Component(0, at.HEAD, 0)], 5)


# -- V6 --------------------------------------------------------------------

def test_jaccard_is_one_for_identical_sets_and_zero_for_disjoint():
    a = [at.Component(25, at.HEAD, h) for h in range(3)]
    b = [at.Component(25, at.HEAD, h) for h in range(3, 6)]
    assert at.jaccard(a, a) == 1.0
    assert at.jaccard(a, b) == 0.0
    assert at.jaccard(a, a[:2] + b[:1]) == pytest.approx(2 / 4)


def test_overlap_null_is_small_but_not_zero_for_a_large_pool():
    """The number V6 has to beat. Two size-5 sets from 300 components share
    something occasionally, and 'the sets share a head' means nothing without
    it."""
    pool = [at.Component(l, at.HEAD, h) for l in range(34) for h in range(8)]
    null = at.overlap_null(pool, 5, n=200, seed=3)
    assert 0.0 <= null.mean() < 0.05
    assert at.set_bar(null, 99.0) < 0.5


# -- families --------------------------------------------------------------

class FakeCurve:
    """Enough of `lens_readout.Curves` for the family split."""

    def __init__(self, item_id, j_honest, j_lie, condition="D"):
        self.item_id, self.condition = item_id, condition
        self.j_honest = np.array(j_honest, dtype=float)
        self.j_lie = np.array(j_lie, dtype=float)

    @property
    def j_margin(self):
        return self.j_honest - self.j_lie

    def readable(self):
        return (self.j_honest + self.j_lie) >= 0.01


def test_families_split_on_whether_a_h_was_ever_legible():
    curves = [
        FakeCurve("LEG", [0.0, 0.9, 0.1], [0.0, 0.05, 0.9]),      # a_H leads at L1
        FakeCurve("NEV", [0.0, 1e-6, 1e-6], [0.0, 0.9, 1.0]),     # never leads
        FakeCurve("OTHER_COND", [0.0, 0.9, 0.9], [0.0, 0.0, 0.0], condition="C2"),
    ]
    fam = at.families(curves)
    assert fam[at.LEGIBLE] == ["LEG"]
    assert fam[at.NEVER] == ["NEV"]


def test_families_ignore_unreadable_layers():
    """Both answers at ~1e-13 is not a lead -- the same guard 04 needed for l*."""
    fam = at.families([FakeCurve("X", [5.3e-13, 1e-6], [2.0e-13, 0.9])])
    assert fam[at.NEVER] == ["X"]


# -- persistence -----------------------------------------------------------

def test_results_round_trip_through_disk(tmp_path):
    rows = [row(at.Component(24, at.HEAD, 3, "full_attention"), item="I1"),
            row(at.Component(24, at.MLP, None, "sliding_attention"), item="I2",
                source="I1", family=at.NEVER)]
    back = at.load_results(at.save_results(rows, tmp_path / "a.json"))
    assert back == rows
    assert back[1].is_null and back[1].component.layer_type == "sliding_attention"


def test_a_reloaded_table_ranks_identically(tmp_path):
    comp = at.Component(25, at.HEAD, 1)
    rows = [row(comp, ld_p=8.0, item=f"I{i}") for i in range(4)]
    rows += [row(comp, ld_p=1.0, item=f"I{i}", source="OTHER") for i in range(4)]
    back = at.load_results(at.save_results(rows, tmp_path / "a.json"))
    assert at.rank(back) == at.rank(rows)


# -- the null's pairing ----------------------------------------------------

def test_derangement_never_maps_an_item_to_itself():
    """A fixed point patches an item with its own honest activation -- that is
    the signal, and one such pair inflates the null bar it is meant to define."""
    ids = [f"I{i}" for i in range(11)]
    pairs = at.derangement(ids, seed=2)
    assert set(pairs) == set(ids)
    assert all(k != v for k, v in pairs.items())


def test_derangement_is_seeded():
    ids = [f"I{i}" for i in range(11)]
    assert at.derangement(ids, seed=2) == at.derangement(ids, seed=2)


def test_derangement_refuses_a_single_item():
    with pytest.raises(ValueError):
        at.derangement(["only"])


# -- identity --------------------------------------------------------------

def test_layer_type_is_metadata_not_identity():
    """`cache_slot` knows a layer is sliding-window; a lookup written by hand
    does not. If the label were part of the hash, the two would miss each other
    -- which is exactly how 05's first run died."""
    labelled = at.Component(24, at.HEAD, 3, "sliding_attention")
    bare = at.Component(24, at.HEAD, 3)
    assert labelled == bare
    assert hash(labelled) == hash(bare)
    assert {labelled: "cached"}[bare] == "cached"


def test_rows_labelled_differently_rank_as_one_component():
    """The quiet version of the same bug: one component split in two, and both
    bars computed on half the rows."""
    rows = [row(at.Component(25, at.HEAD, 1, "full_attention"), ld_p=8.0, item=f"I{i}")
            for i in range(3)]
    rows += [row(at.Component(25, at.HEAD, 1), ld_p=8.0, item=f"J{i}") for i in range(3)]
    rows += [row(at.Component(25, at.HEAD, 1, "full_attention"), ld_p=1.0,
                 item=f"I{i}", source="OTHER") for i in range(3)]
    ranked = at.rank(rows)
    assert len(ranked) == 1
    assert ranked[0].n_items == 6


# -- polarity arms ---------------------------------------------------------
#
# LD = logit(a_H) - logit(a_D) is polarity-signed per item, so a component that
# only writes ' Yes' recovers on Yes-true items and *anti*-recovers on their
# No-true twins. On a balanced set those cancel in the pooled median -- but
# MIN_LD_GAP can drop one twin without its partner, and then they do not.

def armed(comp, per_item, *, null=0.0):
    """Rows for one component with a stated recovery per item, plus a null."""
    rows = [row(comp, ld_p=ld, item=iid) for iid, ld in per_item.items()]
    rows += [row(comp, ld_p=null, item=iid, source="OTHER") for iid in per_item]
    return rows


POLARITY = {"Y1": at.YES, "Y2": at.YES, "N1": at.NO, "N2": at.NO}


def test_a_component_that_works_in_both_directions_is_a_hit():
    comp = at.Component(25, at.HEAD, 1)
    rows = armed(comp, {"Y1": 8.0, "Y2": 8.0, "N1": 7.0, "N2": 7.0})
    (r,) = at.rank(rows, polarity_of=POLARITY)
    assert r.median_yes == pytest.approx(0.8)
    assert r.median_no == pytest.approx(0.7)
    assert abs(r.arm_gap) < 0.2
    assert r.hit


def test_a_yes_writer_clears_pooled_but_fails_an_arm():
    """The whole point: strong on Yes-true, negative on No-true, and an
    unbalanced set leaves the pooled median above the bar anyway."""
    comp = at.Component(25, at.HEAD, 2)
    rows = armed(comp, {"Y1": 10.0, "Y2": 10.0, "N1": -2.0})     # 3 items, not 4
    (r,) = at.rank(rows, polarity_of=POLARITY)
    assert r.median >= at.MIN_RECOVERY                # pooled would promote it
    assert r.median_no < at.MIN_RECOVERY
    assert not r.hit


def test_a_component_measured_on_one_arm_only_is_never_promoted():
    comp = at.Component(25, at.HEAD, 3)
    rows = armed(comp, {"Y1": 9.0, "Y2": 9.0})
    (r,) = at.rank(rows, polarity_of=POLARITY)
    assert r.median_no is None
    assert not r.hit


def test_without_polarity_the_ranking_is_pooled_and_says_so():
    """Backwards compatible: the pre-pairing behaviour, and the rows record
    which bar they were judged against."""
    comp = at.Component(25, at.HEAD, 4)
    rows = armed(comp, {"Y1": 10.0, "Y2": 10.0, "N1": -2.0})
    (r,) = at.rank(rows)
    assert r.arms_required is False and r.median_yes is None
    assert r.hit


def test_the_report_prints_both_arms_when_they_are_known():
    comp = at.Component(25, at.HEAD, 5)
    rows = armed(comp, {"Y1": 10.0, "Y2": 10.0, "N1": -2.0})
    armed_report = at.report(at.rank(rows, polarity_of=POLARITY))
    assert "Yes-arm" in armed_report and "No-arm" in armed_report
    # A component that clears pooled and fails an arm must be visible as such,
    # not blank -- blank reads as "did not clear", which is a different fact.
    assert "one arm" in armed_report
    assert "Yes-arm" not in at.report(at.rank(rows))


def test_arm_medians_drop_the_unusable_denominators_too():
    comp = at.Component(25, at.HEAD, 6)
    rows = [row(comp, ld_p=8.0, item="Y1"), row(comp, ld_p=6.0, item="Y2"),
            row(comp, ld_d=0.0, ld_h=0.1, ld_p=0.1, item="N1"),   # unusable gap
            row(comp, ld_p=4.0, item="N2")]
    yes, no = at.arm_medians(rows, POLARITY)
    assert yes == pytest.approx(0.7)
    assert no == pytest.approx(0.4)          # N1 dropped, not averaged as 1.0


def test_an_armed_ranking_survives_a_round_trip(tmp_path):
    """A table that cannot be compared against its own reloaded copy cannot be
    checked at all -- which is why an empty arm is None and not nan."""
    comp = at.Component(25, at.HEAD, 7)
    rows = armed(comp, {"Y1": 8.0, "N1": 7.0})
    back = at.load_results(at.save_results(rows, tmp_path / "a.json"))
    assert at.rank(back, polarity_of=POLARITY) == at.rank(rows, polarity_of=POLARITY)
