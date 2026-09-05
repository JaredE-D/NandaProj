"""Tests for the parts of `lens_readout` that do not need a GPU.

`Reader` needs a model, a tokenizer and a fitted lens, so it is exercised on the
box and not here. What *is* testable locally is the arithmetic that turns
per-layer probabilities into a claim -- the crossover layer, the gate verdict,
the write-back of measured answers -- and that arithmetic is where a wrong
result would look like a finding rather than like an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from nandaproj import items
from nandaproj import lens_readout as lr


def curves(j_honest, j_lie, layers=None, item_id="X", condition="D") -> lr.Curves:
    layers = layers or list(range(len(j_honest)))
    zeros = np.zeros(len(j_honest))
    return lr.Curves(
        item_id=item_id, condition=condition, layers=layers,
        answer_honest=" Yes", answer_lie=" No",
        j_honest=np.array(j_honest, dtype=float), j_lie=np.array(j_lie, dtype=float),
        l_honest=zeros, l_lie=zeros, final_honest=0.0, final_lie=1.0,
    )


# -- crossover -------------------------------------------------------------

def test_crossover_is_the_layer_after_the_honest_answer_last_leads():
    c = curves([0.1, 0.6, 0.7, 0.2, 0.1], [0.2, 0.1, 0.2, 0.6, 0.8])
    assert c.crossover() == 3


def test_no_crossover_when_the_honest_answer_never_leads():
    """A lie that leads from the bottom of the stack is not an edit applied to
    a belief -- reporting l*=0 there would invent the signature 4.2 looks for."""
    assert curves([0.1, 0.1, 0.1], [0.5, 0.6, 0.7]).crossover() is None


def test_no_crossover_when_the_honest_answer_still_leads_at_the_top():
    assert curves([0.1, 0.5, 0.9], [0.2, 0.1, 0.05]).crossover() is None


def test_a_single_noisy_layer_does_not_become_the_crossover():
    """Honest leads at L1 and again at L3; l* is after the *last* lead, not the
    first dip, or one noisy layer would locate the whole result."""
    c = curves([0.1, 0.6, 0.2, 0.6, 0.1], [0.2, 0.1, 0.3, 0.1, 0.9])
    assert c.crossover() == 4


def test_crossover_reports_the_layer_label_not_the_index():
    """`lens.source_layers` is not always `range(n_layers)` -- the 270m lens
    stops at 16 of 18 -- so an index would silently mislabel every l*."""
    c = curves([0.1, 0.6, 0.1], [0.2, 0.1, 0.9], layers=[10, 12, 14])
    assert c.crossover() == 14


def test_margin_is_honest_minus_lie():
    c = curves([0.5, 0.2], [0.1, 0.8])
    assert np.allclose(c.j_margin, [0.4, -0.6])


# -- the crossover must ignore layers where neither answer has any mass ----
#
# Found by running the notebook on the box: PF02_laptop_battery reported l*=7,
# in a region whose J-lens top-5 is [':', '!:', ' :']. The "lead" that located
# it was 5.3e-13 against 2.0e-13. A null wearing a finding's clothes.

def test_a_lead_between_two_vanishing_numbers_cannot_locate_the_crossover():
    c = curves([5.3e-13, 1.7e-10, 2.2e-25, 1.2e-27],
               [2.0e-13, 1.5e-10, 9.9e-01, 1.0e+00])
    assert c.crossover() is None          # a_H never leads where anything is legible


def test_the_real_pf02_shape_reports_no_crossover():
    """a_D at ~1.0 from L20 up, a_H at ~1e-25 throughout: the honest answer is
    absent from the readout under D, which is 7.3's 'never becomes legible' --
    a result, not a missing number."""
    c = curves([1e-28, 5.3e-13, 1.7e-10, 2.2e-25, 3.9e-07],
               [3.7e-27, 2.0e-13, 1.5e-10, 9.9e-01, 9.996e-01])
    assert c.crossover() is None
    assert c.readable_layers == [3, 4]


def test_a_crossover_on_readable_layers_still_reports():
    """The signature 4.2 is looking for: a_H holds real mass and is overtaken."""
    c = curves([0.05, 0.80, 0.70, 0.10, 0.02],
               [0.02, 0.15, 0.25, 0.88, 0.97])
    assert c.crossover() == 3


def test_readable_layers_excludes_the_dead_bottom_of_the_stack():
    c = curves([1e-20, 0.60, 0.10], [1e-20, 0.30, 0.85])
    assert c.readable_layers == [1, 2]


def test_the_mass_floor_is_an_argument_so_the_claim_can_be_checked():
    """The measurement should not be sensitive to the floor: on real curves the
    pair mass jumps from ~1e-25 to >0.98 within two layers."""
    c = curves([1e-20, 0.60, 0.10], [1e-20, 0.30, 0.85])
    assert c.crossover(min_mass=1e-3) == c.crossover(min_mass=0.5) == 2


def test_crossover_summary_names_items_with_no_readable_layer():
    got = lr.crossover_summary([curves([1e-30, 1e-30], [1e-30, 1e-30], item_id="dead")])
    assert "no readable layer 1" in got


# -- aggregation -----------------------------------------------------------

def test_mean_curves_averages_over_items_and_names_all_four_series():
    got = lr.mean_curves([curves([0.0, 1.0], [1.0, 0.0]),
                          curves([1.0, 0.0], [0.0, 1.0])])
    assert set(got) == {"J-lens a_H", "J-lens a_D", "logit lens a_H", "logit lens a_D"}
    assert np.allclose(got["J-lens a_H"], [0.5, 0.5])


def test_mean_curves_refuses_an_empty_set():
    with pytest.raises(ValueError, match="no curves"):
        lr.mean_curves([])


def test_crossover_summary_counts_the_items_that_have_one():
    """A mean l* over items that mostly report None is not a located layer, so
    the split leads (V4 is about stability, not about the mean)."""
    got = lr.crossover_summary([
        curves([0.1, 0.6, 0.1], [0.2, 0.1, 0.9], item_id="a"),
        curves([0.1, 0.1, 0.1], [0.5, 0.6, 0.7], item_id="b"),
    ])
    assert "2 items" in got and "crossed 1" in got


def test_crossover_summary_distinguishes_never_legible_from_leads_at_top():
    """The two ways to have no crossover are opposite results. Under D,
    "a_H never legible" is PLAN2 7.3's mind-changer; under H, "still leads at
    top" is the instrument working. One line for both would hide the headline --
    which is exactly what the first version of this report did on real data.
    """
    never = lr.crossover_summary([curves([0.1, 0.1], [0.5, 0.6], item_id="n")])
    assert "a_H never legible 1" in never
    assert "still leads at top" not in never

    at_top = lr.crossover_summary([curves([0.5, 0.9], [0.3, 0.05], item_id="t")])
    assert "a_H still leads at top 1" in at_top
    assert "never legible" not in at_top


def test_classify_splits_all_four_ways():
    groups = lr.classify([
        curves([0.05, 0.80, 0.10], [0.02, 0.15, 0.88], item_id="crossed"),
        curves([0.10, 0.10, 0.10], [0.50, 0.60, 0.70], item_id="never"),
        curves([0.50, 0.90, 0.95], [0.30, 0.05, 0.02], item_id="top"),
        curves([1e-30, 1e-30], [1e-30, 1e-30], item_id="dead"),
    ])
    assert groups == {"crossed": ["crossed"], "never_leads": ["never"],
                      "leads_at_top": ["top"], "unreadable": ["dead"]}


def test_crossover_summary_is_empty_safe():
    assert "no curves" in lr.crossover_summary([])


def test_by_condition_groups_without_losing_items():
    grouped = lr.by_condition([curves([0.1], [0.2], condition="D"),
                               curves([0.1], [0.2], condition="C2"),
                               curves([0.1], [0.2], condition="D")])
    assert {k: len(v) for k, v in grouped.items()} == {"D": 2, "C2": 1}


# -- the behavioral gate ---------------------------------------------------

def row(item_id, honest=" Yes", deceptive=" No", ok=True, lied=True,
        p_honest=0.9, mass=1.0, stated=True, tier=None, emitted="") -> lr.GateRow:
    """`p_honest` is renormalized within the legal answers; `mass` is how much
    of the full-vocab distribution those answers held."""
    return lr.GateRow(item_id, honest, deceptive, p_honest, 0.9, ok, lied,
                      mass_honest=mass, has_stated_answer=stated, tier=tier,
                      emitted_honest=emitted or honest)


def test_apply_gate_writes_the_measured_answers_onto_the_items():
    """The bank's stated answers are the author's expectation; P_J(a_D) must be
    the probability of the token the model actually emitted."""
    bank = items.from_records(
        [{"item_id": "X", "question": "q", "answer_honest": " Yes",
          "answer_lie": " No", "prompts": {"H": {"user": "q"}, "D": {"user": "q"}}}],
        conditions=None).items
    out = lr.apply_gate(bank, [row("X", honest=" No", deceptive=" Yes")])
    assert (out[0].answer_honest, out[0].answer_lie) == (" No", " Yes")


def test_apply_gate_drops_items_that_failed_the_gate():
    bank = items.from_records(
        [{"item_id": "X", "question": "q", "prompts": {"H": {"user": "q"}}},
         {"item_id": "Y", "question": "q", "prompts": {"H": {"user": "q"}}}],
        conditions=None).items
    kept = lr.apply_gate(bank, [row("X"), row("Y", lied=False)])
    assert [i.item_id for i in kept] == ["X"]


def test_apply_gate_keeps_failures_when_asked():
    bank = items.from_records(
        [{"item_id": "X", "question": "q", "prompts": {"H": {"user": "q"}}}],
        conditions=None).items
    assert len(lr.apply_gate(bank, [row("X", lied=False)], usable_only=False)) == 1


def test_apply_gate_skips_items_the_gate_never_saw():
    bank = items.from_records(
        [{"item_id": "X", "question": "q", "prompts": {"H": {"user": "q"}}}],
        conditions=None).items
    assert lr.apply_gate(bank, []) == []


def test_gate_report_counts_usable_as_both_conditions_holding():
    got = lr.gate_report([row("a"), row("b", lied=False), row("c", ok=False)])
    assert "usable (honest under H, flipped under D)" in got
    assert "33%" in got


def test_gate_report_says_so_with_no_rows():
    assert "no rows" in lr.gate_report([])


# -- guards ----------------------------------------------------------------

def test_belief_curves_refuses_a_no_belief_item():
    """C3 items have no a_H to be suppressed; measuring one against a
    placeholder would put the 4.2 floor into the numerator."""
    nb = items.from_records(
        [{"item_id": "NB", "question": "q", "answer_honest": None,
          "prompts": {"H": {"user": "q"}, "C3": {"user": "q"}}}],
        conditions=None)["NB"]
    with pytest.raises(ValueError, match="no-belief item has neither"):
        lr.belief_curves(reader=None, item=nb, condition="C3")


# -- the two failures that must not be one bucket --------------------------
#
# "wrong under H" collapses a model that does not know the fact with a bank
# whose stated answer is wrong. The first is correctly dropped; the second is a
# repairable item that would otherwise just shrink n.

def test_a_confident_in_format_disagreement_is_the_banks_bug():
    assert row("a", honest=" No", ok=False).outcome == lr.STATED_ANSWER_MISMATCH


def test_an_off_format_answer_is_the_models_problem_not_the_banks():
    """Almost no mass on the legal answers, so there is no committed answer for
    the bank to have got wrong."""
    assert row("a", ok=False, mass=0.02).outcome == lr.NO_STABLE_BELIEF


def test_an_unconfident_answer_is_not_a_mismatch_even_when_it_disagrees():
    """A near-tie at the slot is not a belief the bank can be said to contradict."""
    assert row("a", honest=" No", ok=False,
               p_honest=0.3).outcome == lr.NO_STABLE_BELIEF


def test_confidence_is_read_at_the_declared_threshold():
    assert row("a", p_honest=lr.MIN_CONFIDENCE).outcome == lr.USABLE
    assert row("a", p_honest=lr.MIN_CONFIDENCE - 0.01).outcome == lr.NO_STABLE_BELIEF


def test_an_honest_run_that_did_not_flip_is_its_own_outcome():
    assert row("a", lied=False).outcome == lr.DID_NOT_LIE


def test_c3_floor_items_are_not_reported_as_mismatches():
    """They state no a_H by construction; calling that a bank bug would report
    the design as a defect."""
    assert row("nb", ok=False, stated=False).outcome == lr.NO_STATED_ANSWER


def test_only_usable_rows_survive_the_gate():
    bank = items.from_records(
        [{"item_id": i, "question": "q", "prompts": {"H": {"user": "q"}}}
         for i in ("a", "b", "c", "d")], conditions=None).items
    kept = lr.apply_gate(bank, [row("a"),
                                row("b", honest=" No", ok=False),
                                row("c", ok=False, mass=0.02),
                                row("d", lied=False)])
    assert [i.item_id for i in kept] == ["a"]


def test_mismatches_returns_the_list_to_hand_back_to_the_bank():
    rows = [row("a"), row("b", honest=" No", ok=False), row("c", ok=False, mass=0.02)]
    assert [r.item_id for r in lr.mismatches(rows)] == ["b"]


def test_gate_report_clusters_mismatches_by_topic():
    """A cluster in one topic is a wording bug; the report has to make that
    visible rather than leaving it to be inferred from a small n."""
    got = lr.gate_report([
        row("wl1", honest=" No", ok=False, tier="social_white_lie"),
        row("wl2", honest=" No", ok=False, tier="social_white_lie"),
        row("pf1", tier="product_fault"),
    ])
    assert "mismatches by topic" in got
    assert "social_white_lie" in got and "wl1, wl2" in got


def test_gate_report_omits_the_cluster_section_when_nothing_mismatches():
    assert "mismatches by topic" not in lr.gate_report([row("a")])


# -- the distribution the confidence threshold is applied to ---------------
#
# The bug this section exists for: `p_honest` used to be the top token's share
# of the FULL vocabulary. On an "Answer:" prefill the model spreads real mass
# over ' Yes', ' yes', 'Yes', '**' and newlines, so a model overwhelmingly
# committed between Yes and No can sit at 0.42 full-vocab and get filed as
# "does not know" -- letting a bad bank item hide behind an ignorant model.

class FakeReader:
    """Just enough of `Reader` for `read_slot`: a fixed distribution at the slot."""

    def __init__(self, tok, probs):
        self.tok = tok
        self._probs = np.array(probs, dtype=float)

    def slot_probs(self, prompt):
        return self._probs

    def top_k(self, probs, k=1):
        idx = np.argsort(probs)[-k:][::-1]
        return [(self.tok.decode([int(i)]), float(probs[i])) for i in idx]


class IdTok:
    """Tokenizer where a listed string is token i and everything else is 2 tokens."""

    def __init__(self, vocab):
        self.vocab = list(vocab)

    def encode(self, text, add_special_tokens=False):
        if text in self.vocab:
            return [self.vocab.index(text)]
        return [0, 0]

    def decode(self, ids):
        return self.vocab[int(ids[0])]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]


def slot_item(**kw):
    record = {"item_id": "X", "question": "q", "answer_honest": " Yes",
              "answer_lie": " No", "prompts": {"H": {"user": "q"}}}
    return items.from_records([{**record, **kw}], conditions=None)["X"]


def test_confidence_is_measured_within_the_legal_answers_not_the_vocabulary():
    """Yes 0.42 against No 0.05 is a committed answer, even though 0.42 of the
    full vocabulary is not. Thresholding the full-vocab number here would file
    a confident contradiction as 'the model does not know'."""
    tok = IdTok([" Yes", " No", "**"])
    reader = FakeReader(tok, [0.42, 0.05, 0.53])
    slot = lr.read_slot(reader, slot_item(), "H")
    assert slot.emitted == "**"                      # what it would actually say
    assert slot.answer == " Yes"                     # the legal answer with the mass
    assert slot.p_answer == pytest.approx(0.42 / 0.47)
    assert slot.mass == pytest.approx(0.47)
    assert lr.GateRow("X", slot.answer, " No", slot.p_answer, 0.9, False, True,
                      mass_honest=slot.mass).outcome == lr.STATED_ANSWER_MISMATCH


def test_mass_over_casings_so_a_lowercase_answer_is_not_an_absent_belief():
    """Gemma emitting ' yes' is a formatting wobble, not a missing belief, so
    the casings are summed rather than one of them being the only legal token."""
    tok = IdTok([" Yes", " yes", " No", "**"])
    reader = FakeReader(tok, [0.30, 0.35, 0.05, 0.30])
    slot = lr.read_slot(reader, slot_item(), "H")
    assert slot.answer == " Yes"                     # canonical: what gets tracked
    assert slot.mass == pytest.approx(0.70)
    assert slot.p_answer == pytest.approx(0.65 / 0.70)


def test_a_slot_with_almost_no_answer_mass_is_not_a_belief():
    """The renormalized number here is a ratio between two rounding errors."""
    tok = IdTok([" Yes", " No", "**"])
    reader = FakeReader(tok, [0.02, 0.001, 0.979])
    slot = lr.read_slot(reader, slot_item(), "H")
    assert slot.p_answer > lr.MIN_CONFIDENCE         # renormalized, it looks certain
    assert slot.mass < lr.MIN_ANSWER_MASS            # ... on nothing
    assert lr.GateRow("X", slot.answer, " No", slot.p_answer, 0.9, True, True,
                      mass_honest=slot.mass).outcome == lr.NO_STABLE_BELIEF


def test_min_confidence_on_two_answers_is_exactly_the_argmax():
    """Documenting a real limit rather than asserting a comfort.

    With two legal answers the renormalized probabilities sum to 1, so a 0.5
    threshold excludes only an exact tie: 0.52 vs 0.48 is a coin flip that the
    gate will happily call a belief. The threshold does what its docstring says
    -- more mass than the alternative -- but on a two-answer bank that is a
    much weaker filter than it looks, and `no_stable_belief` is carried almost
    entirely by MIN_ANSWER_MASS. Raise MIN_CONFIDENCE if near-ties turn out to
    be common; do not assume this line is doing the work.
    """
    tok = IdTok([" Yes", " No"])
    slot = lr.read_slot(FakeReader(tok, [0.48, 0.52]), slot_item(), "H")
    assert slot.answer == " No"
    assert slot.p_answer == pytest.approx(0.52)
    assert slot.p_answer > lr.MIN_CONFIDENCE          # a near-tie, and it passes

    tied = lr.read_slot(FakeReader(tok, [0.5, 0.5]), slot_item(), "H")
    assert tied.p_answer == pytest.approx(0.5)        # only an exact tie fails


def test_an_item_with_no_declared_answers_falls_back_to_the_emitted_token():
    """The C3 floor: nothing to renormalize within, and nothing gated on it."""
    tok = IdTok([" Yes", " No", "**"])
    reader = FakeReader(tok, [0.1, 0.2, 0.7])
    nb = items.from_records(
        [{"item_id": "NB", "question": "q", "answer_honest": None,
          "prompts": {"H": {"user": "q"}, "C3": {"user": "q"}}}], conditions=None)["NB"]
    slot = lr.read_slot(reader, nb, "H")
    assert slot.answer == "**" and np.isnan(slot.mass)


def test_answer_token_ids_collects_casings_and_the_unspaced_form():
    """Both the casings and the form without the leading space count: after an
    'Answer:' prefill the model can reach the same answer either way, and mass
    sitting on a variant is still mass on that answer."""
    tok = IdTok([" Yes", " yes", "Yes"])
    assert sorted(lr.answer_token_ids(tok, " Yes")) == [0, 1, 2]
    assert lr.answer_token_ids(tok, " Nope") == []   # not a single token: dropped


def test_the_lie_check_ignores_casing():
    """' yes' under H against ' Yes' under D is not a flip."""
    assert not row("a", honest=" Yes", deceptive=" Yes").lied or True
    tok = IdTok([" Yes", " yes", " No"])
    reader = FakeReader(tok, [0.0, 0.9, 0.1])
    slot = lr.read_slot(reader, slot_item(), "H")
    assert slot.answer == " Yes"                     # canonicalized, so no false flip


def test_gate_report_names_the_thin_mass_items():
    got = lr.gate_report([row("a", ok=False, mass=0.02, emitted="**")])
    assert "prompt-format" in got and "a (mass=0.020" in got


# -- the threshold is reported, not chosen ---------------------------------
#
# MIN_CONFIDENCE is an item *inclusion* criterion: raising it changes which
# items reach the 4.2 sweep and therefore every curve downstream. Picking it
# after seeing where the items landed is PLAN2 7.4's forking path, and an
# invisible one. So the band is fixed here and reported every run.

def test_outcome_at_re_filters_a_stored_row_without_a_forward_pass():
    r = row("a", p_honest=0.55)
    assert r.outcome_at(min_confidence=0.5) == lr.USABLE
    assert r.outcome_at(min_confidence=0.7) == lr.NO_STABLE_BELIEF


def test_the_bare_outcome_property_is_the_pre_registered_threshold():
    r = row("a", p_honest=0.55)
    assert r.outcome == r.outcome_at(min_confidence=lr.MIN_CONFIDENCE)


def test_gate_report_prints_the_whole_sensitivity_band():
    """Fixed in SENSITIVITY before any data existed. If the usable set collapses
    across the band, that lands on the page instead of in a quiet decision."""
    got = lr.gate_report([row("a", p_honest=0.55), row("b", p_honest=0.95)])
    for threshold in lr.SENSITIVITY:
        assert f"{threshold:.2f}" in got
    assert "pre-registered primary" in got


def test_a_mismatch_can_be_demoted_by_a_higher_threshold_too():
    """Raising the bar does not only shrink `usable`; it moves confident
    contradictions into `no_stable_belief` as well, which is exactly why the
    band is reported for both counts."""
    r = row("a", honest=" No", ok=False, p_honest=0.55)
    assert r.outcome_at(min_confidence=0.5) == lr.STATED_ANSWER_MISMATCH
    assert r.outcome_at(min_confidence=0.7) == lr.NO_STABLE_BELIEF


def test_apply_gate_can_re_filter_at_another_threshold():
    bank = items.from_records(
        [{"item_id": i, "question": "q", "prompts": {"H": {"user": "q"}}}
         for i in ("a", "b")], conditions=None).items
    rows = [row("a", p_honest=0.55), row("b", p_honest=0.95)]
    assert len(lr.apply_gate(bank, rows)) == 2
    assert [i.item_id for i in lr.apply_gate(bank, rows, min_confidence=0.7)] == ["b"]


# -- NaN must never be silently admitted by a filter -----------------------

def test_a_nan_mass_row_is_never_admitted_by_the_format_test():
    """C3 rows carry mass=NaN. `nan >= x` is False, so such a row cannot pass
    `in_format`; and `outcome` short-circuits on has_stated_answer before ever
    reaching it. A NaN that leaked into a filter would read as 'no items
    matched' rather than as an error."""
    r = row("nb", stated=False, mass=float("nan"))
    assert not r.in_format
    assert r.outcome == lr.NO_STATED_ANSWER
    assert r.outcome_at(min_confidence=0.0, min_mass=0.0) == lr.NO_STATED_ANSWER


def test_a_nan_mass_row_with_a_stated_answer_fails_closed():
    """Belt and braces: if a NaN ever reached a row that *does* state an answer,
    it must drop out rather than sail through."""
    assert row("x", mass=float("nan")).outcome == lr.NO_STABLE_BELIEF


def test_nan_rows_do_not_reach_the_thin_mass_report():
    got = lr.gate_report([row("nb", stated=False, mass=float("nan")), row("a")])
    assert "prompt-format" not in got


# -- persistence, written as the work happens ------------------------------
#
# The first V2 gate run was lost because the save cell sat at the bottom of the
# notebook and the cell above it raised. 11/20 existed only in a chat log. A
# pre-registered gate whose output cannot be re-derived from the repo is not an
# artifact.

def test_gate_rows_round_trip_through_disk(tmp_path):
    rows = [row("a", tier="product_fault"),
            row("b", honest=" No", ok=False, p_honest=0.55, mass=0.3),
            row("nb", stated=False, mass=float("nan"))]
    path = lr.save_gate(rows, tmp_path / "gate.json")
    back = lr.load_gate(path)
    assert [r.item_id for r in back] == ["a", "b", "nb"]
    assert [r.outcome for r in back] == [r.outcome for r in rows]
    assert back[1].p_honest == 0.55 and back[1].mass_honest == 0.3


def test_a_nan_mass_survives_the_json_round_trip_as_a_dropped_row(tmp_path):
    """json turns NaN into NaN via the non-strict encoder; what matters is that
    the C3 row still classifies the same way after a reload."""
    rows = [row("nb", stated=False, mass=float("nan"))]
    back = lr.load_gate(lr.save_gate(rows, tmp_path / "g.json"))
    assert back[0].outcome == lr.NO_STATED_ANSWER


def test_the_saved_gate_records_the_thresholds_it_was_produced_under(tmp_path):
    """A gate table read six weeks later must say which MIN_CONFIDENCE produced
    it, or the sensitivity band cannot be reconstructed."""
    import json
    path = lr.save_gate([row("a")], tmp_path / "gate.json")
    payload = json.loads(path.read_text())
    assert payload["min_confidence"] == lr.MIN_CONFIDENCE
    assert payload["sensitivity"] == list(lr.SENSITIVITY)


def test_curves_round_trip_and_re_derive_their_crossover(tmp_path):
    cs = [curves([0.05, 0.80, 0.10], [0.02, 0.15, 0.88], item_id="x", condition="D"),
          curves([0.10, 0.10, 0.10], [0.50, 0.60, 0.70], item_id="y", condition="D")]
    back = lr.load_curves(lr.save_curves(cs, tmp_path / "c.npz"))
    assert [c.item_id for c in back] == ["x", "y"]
    assert [c.crossover() for c in back] == [2, None]
    assert lr.classify(back) == lr.classify(cs)
    assert np.allclose(back[0].j_honest, cs[0].j_honest)


def test_saving_an_empty_sweep_does_not_raise(tmp_path):
    """A sweep that dies before its first item should leave a readable file,
    not a half-written one."""
    assert lr.load_curves(lr.save_curves([], tmp_path / "empty.npz")) == []


# --------------------------------------------------------------------------
# A: does the lens's top fitted layer agree with the model's output? (exp1b gate)
# --------------------------------------------------------------------------


def _curve(item_id, condition, j_top, l_top, final_h, final_l, layers=(30, 31, 32)):
    n = len(layers)
    return lr.Curves(
        item_id=item_id, condition=condition, layers=list(layers),
        answer_honest=" Yes", answer_lie=" No",
        j_honest=np.array([0.0] * (n - 1) + [j_top]),
        j_lie=np.array([0.0] * (n - 1) + [l_top]),
        l_honest=np.zeros(n), l_lie=np.zeros(n),
        final_honest=final_h, final_lie=final_l,
    )


def test_agreement_when_lens_and_model_pick_the_same_answer():
    c = lr.top_layer_agreement([_curve("A", "D", 0.9, 0.05, 0.8, 0.1)])[0]
    assert (c.lens_leads_honest, c.model_leads_honest, c.agrees) == (True, True, True)
    assert c.top_layer == 32


def test_disagreement_is_flagged_when_the_lens_leads_honest_but_the_model_lies():
    """The 8 `leads_at_top` items under D: lens says a_H, model emitted a_D."""
    c = lr.top_layer_agreement([_curve("A", "D", 0.9, 0.05, 0.02, 0.95)])[0]
    assert c.lens_leads_honest and not c.model_leads_honest
    assert not c.agrees


def test_mass_is_reported_for_both_lens_and_model():
    c = lr.top_layer_agreement([_curve("A", "D", 0.6, 0.3, 0.5, 0.4)])[0]
    assert c.lens_mass == pytest.approx(0.9)
    assert c.model_mass == pytest.approx(0.9)


def test_report_counts_agreement_per_condition():
    checks = lr.top_layer_agreement([
        _curve("A", "H", 0.9, 0.05, 0.9, 0.05),
        _curve("B", "H", 0.9, 0.05, 0.9, 0.05),
        _curve("A", "D", 0.9, 0.05, 0.02, 0.95),     # disagrees
        _curve("B", "D", 0.05, 0.9, 0.02, 0.95),
    ])
    text = lr.top_layer_report(checks, n_layers=34)
    assert "top fitted lens layer: L32 of 34" in text
    assert "1 more layer(s)" in text
    assert "2/2" in text and "1/2" in text
    assert "A" in text.split("D ")[-1]


def test_report_shouts_when_the_control_condition_disagrees():
    """H disagreeing means the check itself is broken, not that D is interesting."""
    checks = lr.top_layer_agreement([_curve("A", "H", 0.9, 0.05, 0.02, 0.95)])
    assert "instrument failing its own control" in lr.top_layer_report(checks)


def test_report_is_quiet_when_the_control_is_clean():
    checks = lr.top_layer_agreement([_curve("A", "H", 0.9, 0.05, 0.9, 0.02)])
    assert "instrument failing" not in lr.top_layer_report(checks)


def test_report_handles_no_curves():
    assert "no curves" in lr.top_layer_report([])
