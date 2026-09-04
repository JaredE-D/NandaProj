"""Local tests for the Arm A item bank and condition wrappers (PLAN2.md 2, 4.1).

No torch, no transformers -- the tokenizer is stubbed. `check_single_token` is
exercised here against a stub and on the box against the real Gemma tokenizer.
"""

from __future__ import annotations

import pytest

from nandaproj import deception as dec


class FakeTokenizer:
    """One token per whitespace-delimited word, leading space included."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [abs(hash(w)) % 10_000 for w in text.split(" ") if w]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return body + ("<model>" if add_generation_prompt else "")


class SplittingTokenizer(FakeTokenizer):
    """Splits ' Yes' into two tokens -- must be rejected."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text == " Yes":
            return [1, 2]
        return super().encode(text, add_special_tokens)


@pytest.fixture(scope="module")
def bank():
    return dec.load_bank()


# --------------------------------------------------------------------------
# Bank invariants
# --------------------------------------------------------------------------


def test_bank_loads_and_validates(bank):
    assert len(bank) >= 20
    assert len({it.item_id for it in bank}) == len(bank)


def test_every_answer_is_yes_no_or_none(bank):
    assert {it.answer_true for it in bank} <= {"Yes", "No", None}


def test_lie_is_the_opposite_by_construction(bank):
    for it in dec.belief_items(bank):
        assert it.answer_lie != it.answer_true
        assert {it.answer_true, it.answer_lie} == {"Yes", "No"}
    for it in dec.no_belief_items(bank):
        assert it.answer_lie is None


def test_yes_no_balanced_within_each_category(bank):
    """An unbalanced bank measures a token prior, not a lie (meta-prompt rule 3)."""
    for cat in {it.category for it in bank}:
        sub = dec.belief_items([it for it in bank if it.category == cat])
        if not sub:
            continue
        n_yes = sum(it.answer_true == "Yes" for it in sub)
        assert abs(2 * n_yes - len(sub)) <= 1, f"{cat} is {n_yes}/{len(sub)} Yes"


def test_persona_carries_no_honesty_directive(bank):
    """D and C1 must differ by the directive alone."""
    for it in bank:
        assert not dec._DIRECTIVE_WORDS.search(it.persona), it.item_id


def test_validator_rejects_a_smuggled_directive():
    bad = dec.DeceptionItem("X", "product_fault", "low", "", "Q?", "Yes",
                            "You are a salesperson who must never lie.")
    with pytest.raises(ValueError, match="honesty directive"):
        dec.validate_bank([bad])


def test_validator_allows_lie_as_a_substring():
    """'client' contains 'lie'. Word boundaries, not substrings."""
    ok = dec.DeceptionItem("X", "product_fault", "low", "", "Q?", "Yes",
                           "You are a broker with one very demanding client.")
    dec.validate_bank([ok])


def test_no_belief_flag_agrees_with_category(bank):
    for it in bank:
        assert it.has_belief == (it.category != "no_belief")


def test_no_belief_items_have_no_context(bank):
    """A context block would hand the model the belief the floor is meant to lack."""
    for it in dec.no_belief_items(bank):
        assert it.context == ""


# --------------------------------------------------------------------------
# Condition wrappers
# --------------------------------------------------------------------------


def test_condition_grid_is_fixed():
    """PLAN2.md 7.4: the grid is frozen before the sweep."""
    assert dec.CONDITIONS == ("H", "D", "C1", "C2", "C3")


def test_user_turn_is_identical_across_conditions(bank):
    """Only the system turn may vary -- that is the whole design."""
    for it in bank:
        users = {dec.build_chat(it, c)[1]["content"] for c in dec.conditions_for(it)}
        assert len(users) == 1


def test_d_and_c1_differ_only_by_the_directive(bank):
    for it in dec.belief_items(bank):
        d = dec.system_prompt(it, "D")
        c1 = dec.system_prompt(it, "C1")
        assert d != c1
        assert d.replace(dec.CONCEAL_DIRECTIVE, "") == c1.replace(dec.TRUTHFUL_DIRECTIVE, "")


def test_directives_are_length_matched():
    """PLAN2.md 5 asks C1 to match D in length and oddity."""
    delta = abs(len(dec.CONCEAL_DIRECTIVE) - len(dec.TRUTHFUL_DIRECTIVE))
    assert delta <= 5, delta


def test_persona_appears_only_in_d_and_c1(bank):
    for it in dec.belief_items(bank):
        assert it.persona in dec.system_prompt(it, "D")
        assert it.persona in dec.system_prompt(it, "C1")
        assert it.persona not in dec.system_prompt(it, "H")
        assert it.persona not in dec.system_prompt(it, "C2")


def test_answer_format_is_constant_across_conditions(bank):
    it = dec.belief_items(bank)[0]
    for c in ("H", "D", "C1", "C2"):
        assert dec.system_prompt(it, c).endswith(dec.ANSWER_FORMAT)


def test_c3_rejects_belief_items(bank):
    with pytest.raises(ValueError, match="no-belief floor"):
        dec.system_prompt(dec.belief_items(bank)[0], "C3")


def test_d_rejects_no_belief_items(bank):
    with pytest.raises(ValueError, match="run as C3"):
        dec.system_prompt(dec.no_belief_items(bank)[0], "D")


def test_unknown_condition_raises(bank):
    with pytest.raises(ValueError, match="unknown condition"):
        dec.system_prompt(bank[0], "C4")


def test_conditions_for_partitions_the_bank(bank):
    for it in bank:
        expected = ("H", "C3") if not it.has_belief else ("H", "D", "C1", "C2")
        assert dec.conditions_for(it) == expected


# --------------------------------------------------------------------------
# The answer slot
# --------------------------------------------------------------------------


def test_prefill_has_no_trailing_space(bank):
    """A trailing space merges with the answer and moves the slot."""
    assert not dec.ANSWER_PREFIX.endswith(" ")
    rendered = dec.render(FakeTokenizer(), bank[0], "H")
    assert rendered.endswith(dec.ANSWER_PREFIX)


def test_candidates_carry_the_leading_space():
    assert dec.CANDIDATES == (" Yes", " No")


def test_check_single_token_passes_on_a_good_tokenizer():
    dec.check_single_token(FakeTokenizer())


def test_check_single_token_rejects_a_split_candidate():
    with pytest.raises(ValueError, match="is 2 tokens"):
        dec.check_single_token(SplittingTokenizer())


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def test_export_shape(bank):
    recs = dec.export_bank(bank)
    assert len(recs) == len(bank)
    for r in recs:
        assert set(r["prompts"]) == set(dec.conditions_for(
            next(it for it in bank if it.item_id == r["item_id"])))
        if r["answer_honest"] is not None:
            assert r["answer_honest"] in dec.CANDIDATES
            assert r["answer_lie"] in dec.CANDIDATES
            assert r["answer_honest"] != r["answer_lie"]


def test_export_harm_axis(bank):
    for r in dec.export_bank(bank):
        assert r["harm"] in ("low", "high", None)
