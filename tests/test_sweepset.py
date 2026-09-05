"""Tests for the bookkeeping 05 does before its first patch.

Every failure mode here is silent. A subsample that drops the scarce polarity,
a bank that lost `polarity` on the way to disk, a pair index built from lone
survivors -- none of them raise in the notebook, and all of them come out as
"no component cleared the bar", which is a conclusion rather than an error.
"""

from __future__ import annotations

import json

import pytest

from nandaproj import sweepset


class FakeItem:
    """The three fields `sweepset` reads. Not `items.Item`: these tests are about
    the selection arithmetic, and building whole prompt sets to exercise it would
    hide the arithmetic behind a bank fixture."""

    def __init__(self, item_id, polarity=None, answer_honest=None, pair_id=None):
        self.item_id = item_id
        self.answer_honest = answer_honest
        self.meta = {k: v for k, v in
                     (("polarity", polarity), ("pair_id", pair_id)) if v}


# --------------------------------------------------------------------------
# polarity_of
# --------------------------------------------------------------------------

def test_declared_polarity_wins_and_the_measured_answer_is_the_fallback():
    assert sweepset.polarity_of(FakeItem("a", polarity="Yes")) == "Yes"
    assert sweepset.polarity_of(FakeItem("b", answer_honest=" No")) == "No"
    assert sweepset.polarity_of(
        FakeItem("c", polarity="No", answer_honest=" No")) == "No"


def test_a_declared_polarity_that_contradicts_the_measured_answer_raises():
    # The gate's whole job is to reject these, so reaching here means the bank
    # and the model disagree about what the item is -- silently arming it would
    # put the item in the wrong arm.
    with pytest.raises(ValueError, match="gate should have rejected"):
        sweepset.polarity_of(FakeItem("d", polarity="Yes", answer_honest=" No"))


def test_an_item_with_no_polarity_at_all_raises_and_names_the_cause():
    with pytest.raises(ValueError, match="to_json"):
        sweepset.polarity_of(FakeItem("e"))


# --------------------------------------------------------------------------
# whole_pairs -- the guard that lets a gated arm through
# --------------------------------------------------------------------------

def test_whole_pairs_indexes_complete_pairs_and_counts_every_pair_id():
    its = [FakeItem("y", polarity="Yes", pair_id="p1"),
           FakeItem("n", polarity="No", pair_id="p1"),
           FakeItem("lonely", polarity="No", pair_id="p2")]
    index, n_groups = sweepset.whole_pairs(its)
    assert set(index) == {"p1"}
    assert n_groups == 2                      # p2 is counted, not indexed


def test_an_arm_of_lone_survivors_yields_no_pairs_rather_than_raising():
    # 04c's bank: 12 of 390 pairs had both twins gate. `polarity.pair_index`
    # raises on a half-pair by design; here that would take out the notebook.
    its = [FakeItem(f"i{k}", polarity="No", pair_id=f"p{k}") for k in range(5)]
    index, n_groups = sweepset.whole_pairs(its)
    assert index == {}
    assert n_groups == 5


def test_items_with_no_pair_id_are_skipped_not_grouped_together():
    its = [FakeItem("a", polarity="Yes"), FakeItem("b", polarity="No")]
    assert sweepset.whole_pairs(its) == ({}, 0)


# --------------------------------------------------------------------------
# choose -- the subsample
# --------------------------------------------------------------------------

def bank(n_yes_legible, n_yes_never, n_no_legible, n_no_never):
    families = {
        "legible": [f"yl{k}" for k in range(n_yes_legible)]
                   + [f"nl{k}" for k in range(n_no_legible)],
        "never": [f"yn{k}" for k in range(n_yes_never)]
                 + [f"nn{k}" for k in range(n_no_never)],
    }
    polarities = {i: ("Yes" if i.startswith("y") else "No")
                  for group in families.values() for i in group}
    return families, polarities


def test_no_budget_sweeps_every_item():
    families, polarities = bank(2, 2, 5, 5)
    sel = sweepset.choose(families, polarities)
    assert len(sel.item_ids) == 14
    assert sel.n_available == 14


def test_a_budget_bigger_than_the_bank_is_not_a_subsample():
    families, polarities = bank(1, 1, 2, 2)
    sel = sweepset.choose(families, polarities, max_items=99)
    assert len(sel.item_ids) == 6


def test_every_scarce_polarity_item_survives_the_budget():
    # The alleged arm's shape: 14 Yes-true of 141. A family-stratified draw of
    # 32 would take ~3 of them and decide the arm bar on n=3.
    families, polarities = bank(4, 10, 33, 94)
    sel = sweepset.choose(families, polarities, max_items=32)
    assert len(sel.item_ids) == 32
    assert sel.n_yes == 14                    # all of them
    assert sel.n_no == 18


def test_the_plentiful_arm_is_filled_evenly_across_the_families():
    families, polarities = bank(4, 10, 33, 94)
    sel = sweepset.choose(families, polarities, max_items=32)
    n_no_legible = sum(1 for i in sel.by_family("legible") if polarities[i] == "No")
    n_no_never = sum(1 for i in sel.by_family("never") if polarities[i] == "No")
    # Round-robin, not proportional: `never` is 3x the size of `legible` and
    # must not take 3x the budget with it.
    assert abs(n_no_legible - n_no_never) <= 1


def test_the_scarce_side_is_whichever_one_is_scarce_not_always_yes():
    families, polarities = bank(20, 20, 1, 1)
    sel = sweepset.choose(families, polarities, max_items=10)
    assert sel.n_no == 2                      # both No-true items kept
    assert sel.n_yes == 8


def test_a_budget_smaller_than_the_scarce_arm_does_not_overfill():
    families, polarities = bank(10, 10, 50, 50)
    sel = sweepset.choose(families, polarities, max_items=5)
    assert len(sel.item_ids) == 5


def test_the_same_seed_picks_the_same_set_and_a_different_one_does_not():
    families, polarities = bank(2, 2, 20, 20)
    a = sweepset.choose(families, polarities, max_items=12, seed=0)
    b = sweepset.choose(families, polarities, max_items=12, seed=0)
    c = sweepset.choose(families, polarities, max_items=12, seed=1)
    assert a.item_ids == b.item_ids
    assert a.item_ids != c.item_ids


def test_an_item_with_no_polarity_stops_the_selection_rather_than_shrinking_it():
    families, polarities = bank(1, 1, 2, 2)
    del polarities["nl0"]
    with pytest.raises(ValueError, match="no polarity"):
        sweepset.choose(families, polarities, max_items=4)


def test_a_one_sided_bank_is_flagged_as_such():
    families, polarities = bank(0, 0, 10, 10)
    sel = sweepset.choose(families, polarities, max_items=8)
    assert sel.one_sided
    assert "0 Yes-true" in sel.summary()


def test_a_two_sided_selection_is_not_flagged():
    families, polarities = bank(2, 2, 10, 10)
    assert not sweepset.choose(families, polarities, max_items=8).one_sided


def test_the_selection_round_trips_through_disk(tmp_path):
    families, polarities = bank(2, 2, 10, 10)
    sel = sweepset.choose(families, polarities, max_items=8, seed=3, run_tag="alleged")
    path = sel.to_json(tmp_path / "sub" / "sel.json")
    payload = json.loads(path.read_text())
    assert payload["item_ids"] == sel.item_ids
    assert payload["seed"] == 3
    assert payload["run_tag"] == "alleged"
    assert payload["n_available"] == 24
    assert set(payload["family_of"]) == set(sel.item_ids)


def test_families_of_the_selection_are_the_selection_not_the_bank():
    families, polarities = bank(2, 2, 10, 10)
    sel = sweepset.choose(families, polarities, max_items=8)
    assert set(sel.by_family("legible")) <= set(sel.item_ids)
    assert len(sel.by_family("legible")) + len(sel.by_family("never")) == 8
