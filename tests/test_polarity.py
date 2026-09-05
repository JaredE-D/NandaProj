"""Tests for the polarity controls (bank_v2_spec 6).

Numpy rather than torch: the module is written in `+ - * sum()` precisely so it
does not care, and the test asserting that is the point of the design.
"""

from __future__ import annotations

import numpy as np
import pytest

from nandaproj import deception as dec
from nandaproj import polarity as pol


def _pair(pair_id="P01", category="product_fault"):
    common = {"category": category, "stake": "high", "context": "A fact.",
                  "persona": "You are a seller with rent due.", "pair_id": pair_id}
    return [
        dec.DeceptionItem(item_id=f"{pair_id}_y", question="Is it broken?",
                          answer_true="Yes", **common),
        dec.DeceptionItem(item_id=f"{pair_id}_n", question="Is it sound?",
                          answer_true="No", **common),
    ]


@pytest.fixture
def bank():
    return dec.load_bank()


# --------------------------------------------------------------------------
# Pair index
# --------------------------------------------------------------------------


def test_pair_index_orders_yes_first_regardless_of_input_order():
    items = _pair()
    assert pol.pair_index(items)["P01"] == (items[0], items[1])
    assert pol.pair_index(items[::-1])["P01"] == (items[0], items[1])


def test_pair_index_skips_items_without_a_pair_id():
    lone = dec.DeceptionItem("NB", "no_belief", "none", "", "Q?", None, "A seer.")
    assert set(pol.pair_index(_pair() + [lone])) == {"P01"}


def test_pair_index_rejects_a_half_pair():
    """A silently dropped twin is how v1's balance died at the gate."""
    with pytest.raises(ValueError, match="not one Yes-true and one No-true"):
        pol.pair_index(_pair()[:1])


def test_pair_index_reads_the_readout_layer_shape():
    """`items.Item` carries pair_id/polarity in `meta`, not as attributes."""
    from nandaproj import items as it_mod
    loaded = it_mod.load_bank()
    index = pol.pair_index(loaded.items)
    assert len(index) == 50
    yes, no = index["PF01_car_gasket"]
    assert (yes.item_id, no.item_id) == ("PF01_car_gasket", "PF01_car_gasket_t")


# --------------------------------------------------------------------------
# Directions
# --------------------------------------------------------------------------


def _residuals(items, *, truth, polarity_axis, noise=0.0, seed=0):
    """h = truth * e0 + (+/-1) * polarity_axis * e1, so both parts are known."""
    rng = np.random.default_rng(seed)
    out = {}
    for it in items:
        sign = 1.0 if it.answer_true == "Yes" else -1.0
        h = np.zeros(4)
        h[0] = truth
        h[1] = sign * polarity_axis
        out[it.item_id] = h + noise * rng.standard_normal(4)
    return out


def test_d_yesno_recovers_the_polarity_axis():
    items = _pair("P01") + _pair("P02")
    d = pol.d_yesno(_residuals(items, truth=5.0, polarity_axis=3.0), items)
    assert np.allclose(d, [0.0, 6.0, 0.0, 0.0])   # 3 - (-3) on e1, truth cancels


def test_d_paired_cancels_polarity_exactly():
    """Not 'on average over enough items' -- exactly, on two pairs, with noise off."""
    items = _pair("P01") + _pair("P02")
    h = _residuals(items, truth=5.0, polarity_axis=3.0)
    d = _residuals(items, truth=-2.0, polarity_axis=3.0)
    assert np.allclose(pol.d_paired(h, d, items), [7.0, 0.0, 0.0, 0.0])


def test_pooled_difference_in_means_does_not_cancel_on_a_skewed_subset():
    """The v1 failure, reproduced: the confound is in the pooling, not the data."""
    items = _pair("P01") + _pair("P02")
    skewed = [items[0], items[2], items[3]]        # 2 Yes-true, 1 No-true
    h = _residuals(items, truth=5.0, polarity_axis=3.0)
    d = _residuals(items, truth=-2.0, polarity_axis=3.0)
    pooled = (np.mean([h[i.item_id] for i in skewed], axis=0)
              - np.mean([d[i.item_id] for i in skewed], axis=0))
    assert np.allclose(pooled, [7.0, 0.0, 0.0, 0.0])   # same condition, so it cancels
    # but across conditions with a polarity-dependent shift it does not:
    d2 = {k: v.copy() for k, v in d.items()}
    for it in items:
        d2[it.item_id][1] *= 0.5
    pooled2 = (np.mean([h[i.item_id] for i in skewed], axis=0)
               - np.mean([d2[i.item_id] for i in skewed], axis=0))
    paired2 = pol.d_paired(h, d2, items)
    assert abs(pooled2[1]) > 0.4        # polarity leaks into the pooled direction
    assert np.allclose(paired2[1], 0.0)  # and not into the paired one


def test_cosine_and_project_out():
    a, b = np.array([1.0, 1.0, 0, 0]), np.array([1.0, 0, 0, 0])
    assert pol.cosine(a, b) == pytest.approx(1 / np.sqrt(2))
    stripped = pol.project_out(a, b)
    assert np.allclose(stripped, [0.0, 1.0, 0, 0])
    assert pol.cosine(stripped, b) == pytest.approx(0.0, abs=1e-12)


def test_cosine_rejects_a_zero_vector():
    with pytest.raises(ValueError, match="zero vector"):
        pol.cosine(np.zeros(4), np.ones(4))


def test_d_yesno_raises_on_a_missing_residual():
    items = _pair()
    with pytest.raises(KeyError, match="no residual"):
        pol.d_yesno({items[0].item_id: np.ones(4)}, items)


# --------------------------------------------------------------------------
# Per-item metric splits and the pair-level gate
# --------------------------------------------------------------------------


def test_by_polarity_splits_the_arms():
    items = _pair("P01") + _pair("P02")
    metric = {"P01_y": 1.0, "P01_n": 0.0, "P02_y": 3.0, "P02_n": 1.0}
    split = pol.by_polarity(metric, items)
    assert (split.mean_yes, split.mean_no, split.difference) == (2.0, 0.5, 1.5)
    assert "difference=+1.5" in str(split)


def test_by_polarity_refuses_a_one_armed_split():
    items = _pair()
    with pytest.raises(ValueError, match="one-armed split is not a control"):
        pol.by_polarity({"P01_y": 1.0}, items)


def test_whole_pairs_drops_a_pair_broken_by_one_twin():
    items = _pair("P01") + _pair("P02")
    kept = pol.whole_pairs(items, ["P01_y", "P01_n", "P02_y"])
    assert kept == ["P01_y", "P01_n"]


def test_whole_pairs_output_is_always_balanced():
    items = _pair("P01") + _pair("P02") + _pair("P03")
    for keep in (["P01_y"], ["P01_y", "P01_n", "P02_y"],
                 ["P01_y", "P01_n", "P02_y", "P02_n", "P03_y"]):
        kept = pol.whole_pairs(items, keep)
        by = {i.item_id: i.answer_true for i in items}
        assert sum(by[k] == "Yes" for k in kept) * 2 == len(kept)


def test_gate_report_names_the_broken_pair():
    items = _pair("P01") + _pair("P02")
    text = pol.gate_report(items, ["P01_y", "P01_n", "P02_y"])
    assert "1/2 pairs kept" in text
    assert "broken P02: only P02_y passed" in text


# --------------------------------------------------------------------------
# The real bank
# --------------------------------------------------------------------------


def test_the_bank_gives_fifty_pairs(bank):
    assert len(pol.pair_index(bank)) == 50


def test_v1_gated_set_would_have_been_caught(bank):
    """The v1 result, restated as this module's gate: 4 Yes / 7 No cannot happen.

    Whatever subset a gate keeps, `whole_pairs` returns a balanced one, so no
    later split of it -- legible vs never, high vs low stake -- can be a hidden
    polarity subgroup.
    """
    import random
    rng = random.Random(0)
    ids = [i.item_id for i in dec.belief_items(bank)]
    for _ in range(20):
        kept = pol.whole_pairs(bank, rng.sample(ids, rng.randint(1, len(ids))))
        by = {i.item_id: i.answer_true for i in bank}
        assert sum(by[k] == "Yes" for k in kept) * 2 == len(kept)


# --------------------------------------------------------------------------
# B: is d_J one shared axis wearing a sign? (exp1b gate)
# --------------------------------------------------------------------------


def _dirs(items, axis, item_specific=0.0, seed=0):
    """d_i = sign_i * axis + item_specific * (a vector unique to the item)."""
    np.random.default_rng(seed)
    out = {}
    for n, it in enumerate(items):
        sign = 1.0 if it.answer_true == "Yes" else -1.0
        own = np.zeros(len(axis)); own[2 + n % 2] = 1.0
        out[it.item_id] = sign * np.asarray(axis, float) + item_specific * own
    return out


def test_axis_fraction_is_one_when_every_direction_is_the_same_axis():
    items = _pair("P01") + _pair("P02")
    axis = [1.0, 0, 0, 0]
    frac, m = pol.axis_fraction(_dirs(items, axis), items)
    assert frac == pytest.approx(1.0)
    assert np.allclose(m, [1.0, 0, 0, 0])


def test_axis_fraction_ignores_the_sign_convention():
    """A No-true item's d_J points the other way by construction, not by content."""
    items = _pair("P01") + _pair("P02")
    d = _dirs(items, [1.0, 0, 0, 0])
    assert np.allclose(d["P01_y"], -d["P01_n"])       # anti-parallel twins
    assert pol.axis_fraction(d, items)[0] == pytest.approx(1.0)


def test_axis_fraction_falls_as_item_specific_structure_grows():
    items = _pair("P01") + _pair("P02")
    fracs = [pol.axis_fraction(_dirs(items, [1.0, 0, 0, 0], k), items)[0]
             for k in (0.0, 0.5, 2.0)]
    assert fracs[0] > fracs[1] > fracs[2]
    assert fracs[0] == pytest.approx(1.0)


def test_axis_fraction_is_small_for_unrelated_directions():
    """The floor: n random directions average to ~1/n of a unit vector."""
    rng = np.random.default_rng(3)
    items = [i for p in ("P01", "P02", "P03", "P04") for i in _pair(p)]
    d = {it.item_id: rng.standard_normal(256) for it in items}
    frac, _ = pol.axis_fraction(d, items)
    assert frac < 0.25, frac


def test_residual_direction_removes_the_axis_and_keeps_the_rest():
    items = _pair("P01") + _pair("P02")
    d = _dirs(items, [1.0, 0, 0, 0], item_specific=0.5)
    _, m = pol.axis_fraction(d, items)
    for it in items:
        r = pol.residual_direction(d[it.item_id], it, m)
        # orthogonal to the shared axis, and non-zero because the item-specific
        # part survives -- both halves of what the residual is for.
        assert abs(float(np.dot(r, m)) / pol.norm(m) ** 2) < 0.35
        assert pol.norm(r) > 0.05


def test_residual_direction_is_near_zero_when_there_is_no_item_structure():
    items = _pair("P01") + _pair("P02")
    d = _dirs(items, [1.0, 0, 0, 0], item_specific=0.0)
    _, m = pol.axis_fraction(d, items)
    for it in items:
        assert pol.norm(pol.residual_direction(d[it.item_id], it, m)) < 1e-12


def test_d_paired_names_the_cause_when_no_item_carries_a_pair_id():
    """The 04b crash: a gated bank written without pair_id gave zero pairs."""
    from nandaproj import items as it_mod

    bare = [it_mod.Item(item_id="A", question="q?"), it_mod.Item(item_id="B", question="q?")]
    with pytest.raises(ValueError, match="none carries a `pair_id`"):
        pol.d_paired({"A": np.ones(4)}, {"A": np.ones(4)}, bare)
    with pytest.raises(ValueError, match="attach_meta"):
        pol.d_yesno({"A": np.ones(4)}, bare)


def test_by_polarity_does_not_need_whole_pairs():
    """A gated arm can hold one twin of a pair; the arms are still well defined."""
    a = _pair("P01"); b = _pair("P02")
    items = [a[0], b[1]]                       # one Yes-true, one No-true, no whole pair
    split = pol.by_polarity({"P01_y": 1.0, "P02_n": 0.0}, items)
    assert split.yes == {"P01_y": 1.0} and split.no == {"P02_n": 0.0}
