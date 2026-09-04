"""Local tests for the synthetic confidence prompts (01b).

Runs in the local venv, which has numpy and nothing else -- no torch, no
transformers. The tokenizer is stubbed; `score_candidates` is the one function
that needs a real model and is exercised on the box instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from nandaproj import synthetic


class FakeTokenizer:
    """Digit-splitting tokenizer, the behaviour Gemma is expected to have."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Each character is its own token; digits land on ids 0-9.
        return [int(c) if c.isdigit() else 100 + ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + ("<model>" if add_generation_prompt else "")


class MultiTokenNumberTokenizer(FakeTokenizer):
    """A tokenizer that keeps '7' fused with something -- must be rejected."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text == "7":
            return [900, 901]
        return super().encode(text, add_special_tokens)


# --------------------------------------------------------------------------
# Item banks
# --------------------------------------------------------------------------


def test_dictated_items_default_to_the_single_digit_scale():
    items = synthetic.dictated_items()
    assert [i.target for i in items] == list(synthetic.NINE_TARGETS)
    assert all(i.scale == "nine" for i in items)
    assert all(i.tier == "S1_dictated" for i in items)


def test_dictated_items_carry_exact_targets_on_percent():
    items = synthetic.dictated_items(scale="percent")
    assert [i.target for i in items] == list(synthetic.DICTATED_TARGETS)
    assert all(i.scale == "percent" for i in items)


def test_nine_targets_avoid_the_models_default_answers():
    """5 is the midpoint reflex and 9 the overconfident one; a hit on either
    could be the model's prior rather than the dictated number."""
    assert 5 not in synthetic.NINE_TARGETS
    assert 9 not in synthetic.NINE_TARGETS
    assert all(1 <= t <= 9 for t in synthetic.NINE_TARGETS)


def test_nine_scale_targets_are_single_digit():
    for item in synthetic.dictated_items():
        assert len(synthetic.digits_of(item.target)) == 1


def test_dictated_targets_have_distinct_first_and_second_digits():
    """A target like 88 cannot distinguish digit one from digit two."""
    for target in synthetic.DICTATED_TARGETS:
        text = str(target)
        if len(text) > 1:
            assert text[0] != text[1], f"{target} has a repeated leading digit"


def test_dictated_instruction_names_the_number():
    item = synthetic.dictated_items([85], scale="percent")[0]
    assert "85" in item.instruction and "%" in item.instruction


def test_nine_dictated_instruction_asks_for_a_bare_digit():
    item = synthetic.dictated_items([4])[0]
    assert "4" in item.instruction
    assert "%" not in item.instruction


def test_nine_free_instruction_names_the_range():
    item = synthetic.forced_extreme_items()[0]
    assert "1" in item.instruction and "9" in item.instruction
    assert "%" not in item.instruction


def test_forced_extreme_items_are_balanced_two_arms():
    items = synthetic.forced_extreme_items()
    easy = [i for i in items if i.tier == "S2_easy"]
    hard = [i for i in items if i.tier == "S2_unanswerable"]
    assert len(easy) == len(hard) > 0
    assert all(i.target is None for i in items)


def test_item_ids_are_unique():
    ids = [i.item_id for i in synthetic.all_items()]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_render_ends_at_the_confidence_slot():
    tok = FakeTokenizer()
    item = synthetic.dictated_items([85])[0]
    prompt = synthetic.render(tok, item, answer="Paris.")
    assert prompt.endswith(synthetic.CONFIDENCE_PREFIX)
    # The next token must be a digit, so no trailing digit may already be there.
    assert not prompt.rstrip().endswith("%")


def test_render_includes_answer_and_question():
    tok = FakeTokenizer()
    item = synthetic.forced_extreme_items()[0]
    prompt = synthetic.render(tok, item, answer="  Four.  ")
    assert item.question in prompt
    assert "Four." in prompt


# --------------------------------------------------------------------------
# Digit readout
# --------------------------------------------------------------------------


def test_digit_token_ids_maps_all_ten():
    ids = synthetic.digit_token_ids(FakeTokenizer())
    assert ids == {d: d for d in range(10)}


def test_digit_token_ids_rejects_multi_token_digits():
    with pytest.raises(ValueError, match="not a single token"):
        synthetic.digit_token_ids(MultiTokenNumberTokenizer())


def test_digit_mass_sums_only_digit_tokens():
    probs = np.zeros(200)
    probs[3] = 0.4
    probs[7] = 0.1
    probs[150] = 0.5  # a word token, must not count
    ids = synthetic.digit_token_ids(FakeTokenizer())
    assert synthetic.digit_mass(probs, ids) == pytest.approx(0.5)


def test_digit_distribution_renormalises_within_digits():
    probs = np.zeros(200)
    probs[8] = 0.2
    probs[150] = 0.8
    dist = synthetic.digit_distribution(probs, synthetic.digit_token_ids(FakeTokenizer()))
    assert dist.sum() == pytest.approx(1.0)
    assert dist[8] == pytest.approx(1.0)


def test_digit_distribution_survives_zero_digit_mass():
    probs = np.zeros(200)
    probs[150] = 1.0
    dist = synthetic.digit_distribution(probs, synthetic.digit_token_ids(FakeTokenizer()))
    assert dist.sum() == 0.0  # no NaNs


@pytest.mark.parametrize("value,expected", [(5, 5), (37, 3), (100, 1), (0, 0)])
def test_first_digit(value, expected):
    assert synthetic.first_digit(value) == expected


def test_first_digit_rejects_out_of_range():
    with pytest.raises(ValueError):
        synthetic.first_digit(101)


# --------------------------------------------------------------------------
# Candidate distribution
# --------------------------------------------------------------------------


def test_confidence_candidates_default_to_one_through_nine():
    assert synthetic.confidence_candidates() == list(range(1, 10))


def test_confidence_candidates_span_the_range_inclusive_on_percent():
    cands = synthetic.confidence_candidates(step=5, scale="percent")
    assert cands[0] == 0 and cands[-1] == 100
    assert len(cands) == 21


def test_confidence_candidates_rejects_non_dividing_step():
    with pytest.raises(ValueError):
        synthetic.confidence_candidates(step=7, scale="percent")


@pytest.mark.parametrize("value,expected", [
    (5, [5]), (37, [3, 7]), (85, [8, 5]), (100, [1, 0, 0]), (0, [0]),
])
def test_digits_of_returns_every_digit_in_order(value, expected):
    assert synthetic.digits_of(value) == expected


def test_digits_of_rejects_out_of_range():
    with pytest.raises(ValueError):
        synthetic.digits_of(101)


def test_slot_distribution_reads_one_position_over_one_to_nine():
    probs = np.zeros(200)
    probs[4] = 0.3          # digit 4
    probs[7] = 0.1          # digit 7
    probs[150] = 0.6        # a word token, excluded
    dist = synthetic.slot_distribution(probs, synthetic.digit_token_ids(FakeTokenizer()))
    assert len(dist) == 9
    assert dist.sum() == pytest.approx(1.0)
    assert dist[3] == pytest.approx(0.75)   # value 4 -> index 3
    assert dist[6] == pytest.approx(0.25)   # value 7 -> index 6


def test_slot_distribution_excludes_zero():
    """0 is not on the 1-9 scale; probability there must not leak in."""
    probs = np.zeros(200)
    probs[0] = 0.9
    probs[4] = 0.1
    dist = synthetic.slot_distribution(probs, synthetic.digit_token_ids(FakeTokenizer()))
    assert dist[3] == pytest.approx(1.0)


def test_slot_distribution_survives_zero_mass():
    probs = np.zeros(200)
    probs[150] = 1.0
    dist = synthetic.slot_distribution(probs, synthetic.digit_token_ids(FakeTokenizer()))
    assert dist.sum() == 0.0


def test_normalise_is_a_distribution_and_stable_on_large_logprobs():
    probs = synthetic.normalise(np.array([-1000.0, -1001.0, -1002.0]))
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(probs))
    assert probs[0] > probs[1] > probs[2]


def test_expected_confidence_matches_hand_computation():
    cands = [0, 50, 100]
    probs = np.array([0.25, 0.25, 0.5])
    assert synthetic.expected_confidence(cands, probs) == pytest.approx(62.5)


def test_expected_confidence_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        synthetic.expected_confidence([0, 50], np.array([1.0]))


# --------------------------------------------------------------------------
# The module must stay importable where torch is absent (config.py's rule)
# --------------------------------------------------------------------------


def test_module_does_not_import_torch_at_module_level():
    import sys
    assert "torch" not in sys.modules
