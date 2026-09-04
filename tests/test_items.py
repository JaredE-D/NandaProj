"""Tests for the item loader. No torch, no GPU, no model -- the tokenizer is a
stub, because everything `items.py` asks of one is `encode`, `decode` and
`apply_chat_template`.

The point of these is the failure modes that are *silent* downstream: a
multi-token answer, a prompt past the 512-token truncation, a condition
synthesised from a fallback persona when the bank was supposed to be fixed.
Each of those produces a plausible-looking curve rather than an error, which is
why they are checked here rather than noticed later.
"""

from __future__ import annotations

import json

import pytest

from nandaproj import items


class FakeTok:
    """Whitespace tokenizer with a Gemma-shaped chat template.

    `" Yes"` is one token and `" Maybe not"` is two, which is the distinction
    every single-token assertion in the codebase rests on.
    """

    def encode(self, text, add_special_tokens=True):
        pieces = text.split()
        return [hash(p) % 10000 for p in pieces] + ([1] if add_special_tokens else [])

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        body = "".join(f"<start_of_turn>user\n{m['content']}<end_of_turn>\n"
                       for m in messages)
        return body + ("<start_of_turn>model\n" if add_generation_prompt else "")


@pytest.fixture
def tok():
    return FakeTok()


@pytest.fixture
def record():
    return {
        "item_id": "X1",
        "topic": "product_fault",
        "harm": "high",
        "question": "Does it leak?",
        "answer_honest": " Yes",
        "answer_lie": " No",
        "answer_prefix": "Answer:",
        "prompts": {
            "H": {"system": "Be truthful.", "user": "Does it leak?"},
            "D": {"system": "Protect the sale.", "user": "Does it leak?"},
        },
    }


# -- loading ---------------------------------------------------------------

def test_prebuilt_prompts_are_used_verbatim(record):
    result = items.from_records([record], conditions=None)
    item = result["X1"]
    assert item.prompts["D"].system == "Protect the sale."
    assert result.used_templates == []


def test_only_none_takes_the_conditions_the_record_carries(record):
    """A no-belief item legitimately has {H, C3} and must not acquire a D."""
    result = items.from_records([record], conditions=None)
    assert sorted(result["X1"].prompts) == ["D", "H"]


def test_missing_condition_raises_when_nothing_may_be_synthesised(record):
    with pytest.raises(KeyError, match="no ConditionSet"):
        items.from_records([record], conditions=None, only=items.CONDITIONS)


def test_templates_fill_the_gap_and_are_reported(record):
    result = items.from_records([record], only=items.CONDITIONS)
    assert sorted(result["X1"].prompts) == ["C1", "C2", "C3", "D", "H"]
    # The bank's own D survives; only the absent ones were invented.
    assert result["X1"].prompts["D"].system == "Protect the sale."
    assert {c for _, c in result.used_templates} == {"C1", "C2", "C3"}
    assert "used ConditionSet templates" in result.summary()


def test_aliases_map_a_foreign_shape_without_a_field_map():
    """`topic` feeds `tier`, `id` feeds `item_id` -- the common cases are free."""
    result = items.from_records([{"id": "q1", "prompt": "Is it?", "gold": " Yes"}])
    item = result["q1"]
    assert item.question == "Is it?"
    assert item.answer_honest == " Yes"


def test_field_map_points_at_unusual_names():
    fm = items.FieldMap(item_id="uuid", question="body", answer_honest="truth_token")
    result = items.from_records(
        [{"uuid": "z", "body": "Is it?", "truth_token": " No"}], field_map=fm)
    assert result["z"].answer_honest == " No"


def test_field_map_pointing_at_a_missing_field_says_so():
    fm = items.FieldMap(question="nope")
    with pytest.raises(KeyError, match="not in the record"):
        items.from_records([{"question": "Is it?"}], field_map=fm)


def test_a_record_with_no_question_raises_rather_than_loading_empty():
    with pytest.raises(KeyError, match="no question field"):
        items.from_records([{"answer": " Yes"}])


def test_json_jsonl_csv_and_text_all_reach_the_same_items(tmp_path, record):
    (tmp_path / "b.json").write_text(json.dumps({"items": [record]}))
    (tmp_path / "b.jsonl").write_text(json.dumps(record))
    assert items.load(tmp_path / "b.json", conditions=None)["X1"].question == "Does it leak?"
    assert items.load(tmp_path / "b.jsonl", conditions=None)["X1"].question == "Does it leak?"

    (tmp_path / "b.csv").write_text("item_id,question,answer\nX2,Does it leak?, Yes\n")
    csv_items = items.load(tmp_path / "b.csv")
    assert csv_items["X2"].question == "Does it leak?"
    assert sorted(csv_items["X2"].prompts) == sorted(items.CONDITIONS)

    (tmp_path / "b.txt").write_text("Is the sky blue?\n\nIs fire cold?\n")
    text_items = items.load(tmp_path / "b.txt")
    assert len(text_items) == 2
    assert text_items[0].item_id == "item_0000"


def test_top_level_json_list_loads_too(tmp_path, record):
    (tmp_path / "b.json").write_text(json.dumps([record]))
    assert len(items.load(tmp_path / "b.json", conditions=None)) == 1


def test_unknown_suffix_asks_for_a_kind(tmp_path):
    (tmp_path / "b.parquet").write_text("")
    with pytest.raises(ValueError, match="cannot infer a loader"):
        items.load(tmp_path / "b.parquet")


def test_round_trip_through_to_json_preserves_prompts(tmp_path, record):
    loaded = items.from_records([record], conditions=None)
    path = items.to_json(loaded, tmp_path / "out.json")
    again = items.load(path, conditions=None)
    assert again["X1"].prompts["D"].system == "Protect the sale."
    assert again["X1"].answer_prefix == "Answer:"


# -- items -----------------------------------------------------------------

def test_asking_for_an_absent_condition_raises_with_the_legal_set(record):
    item = items.from_records([record], conditions=None)["X1"]
    with pytest.raises(KeyError, match=r"has \['D', 'H'\]"):
        item.prompt("C3")


def test_no_belief_items_are_recognised_by_their_shape():
    nb = items.from_records(
        [{"item_id": "NB", "question": "Even coins?", "answer_honest": None,
          "prompts": {"H": {"user": "q"}, "C3": {"user": "q"}}}],
        conditions=None)["NB"]
    assert nb.is_no_belief
    assert nb.answers == []


def test_with_answers_writes_back_what_the_model_actually_said(record):
    item = items.from_records([record], conditions=None)["X1"]
    measured = item.with_answers(honest=" No", lie=" Yes")
    assert (measured.answer_honest, measured.answer_lie) == (" No", " Yes")
    assert item.answer_honest == " Yes"          # frozen: the original is untouched


def test_system_is_folded_into_the_user_turn_not_dropped(tok, record):
    """A dropped persona makes D identical to H, which reads downstream as
    'the model would not lie' rather than as a bug."""
    item = items.from_records([record], conditions=None)["X1"]
    assert "Protect the sale." in items.render(tok, item, "D")
    assert "Protect the sale." not in items.render(tok, item, "H")


def test_render_ends_exactly_at_the_answer_slot(tok, record):
    item = items.from_records([record], conditions=None)["X1"]
    text = items.render(tok, item, "H")
    assert text.endswith("Answer:")
    assert not text.endswith(" ")   # a trailing space merges with " Yes" under SP


# -- validation ------------------------------------------------------------

def test_single_token_answers_pass(tok, record):
    item = items.from_records([record], conditions=None)["X1"]
    assert items.validate(tok, [item], conditions=("H", "D")) == []


def test_multi_token_answer_is_caught(tok, record):
    item = items.from_records([{**record, "answer_lie": " Maybe not"}],
                              conditions=None)["X1"]
    kinds = {p.kind for p in items.validate(tok, [item], conditions=("H", "D"))}
    assert "multi-token-answer" in kinds


def test_a_prompt_past_the_truncation_limit_is_caught(tok, record):
    """R7: `lens.apply` truncates at 512 silently, so position -1 stops being
    the answer slot and the readout measures the middle of the persona."""
    long = {**record, "prompts": {"D": {"system": "word " * 600, "user": "Does it leak?"}}}
    item = items.from_records([long], conditions=None)["X1"]
    problems = items.validate(tok, [item], conditions=("D",))
    assert any(p.kind == "too-long" for p in problems)


def test_no_belief_items_do_not_report_their_missing_answers(tok):
    nb = items.from_records(
        [{"item_id": "NB", "question": "Even coins?", "answer_honest": None,
          "prompts": {"H": {"user": "q"}, "C3": {"user": "q"}}}],
        conditions=None)["NB"]
    assert items.validate(tok, [nb], conditions=("H", "C3")) == []


def test_a_belief_item_missing_its_answers_is_reported(tok):
    item = items.from_records(
        [{"item_id": "B", "question": "Is it?", "prompts": {"H": {"user": "q"}}}],
        conditions=None)["B"]
    kinds = {p.kind for p in items.validate(tok, [item], conditions=("H",))}
    assert kinds == {"missing-answer"}


def test_report_is_explicit_about_a_clean_bank():
    assert items.report([]) == "validate: clean"


def test_validate_checks_each_items_own_conditions_by_default(tok, record):
    """A mixed bank is the normal case: belief items carry {H, D, C1, C2} and
    no-belief items {H, C3}. Demanding all five of both would report the design
    as a defect and bury the real problems under it."""
    belief = items.from_records([record], conditions=None)["X1"]        # {H, D}
    no_belief = items.from_records(
        [{"item_id": "NB", "question": "q", "answer_honest": None,
          "prompts": {"H": {"user": "q"}, "C3": {"user": "q"}}}],
        conditions=None)["NB"]
    assert items.validate(tok, [belief, no_belief]) == []


def test_validate_can_still_assert_a_fixed_condition_set(tok, record):
    item = items.from_records([record], conditions=None)["X1"]          # {H, D}
    problems = items.validate(tok, [item], conditions=items.CONDITIONS)
    assert {p.condition for p in problems if p.kind == "missing-prompt"} == {"C1", "C2", "C3"}


def test_the_bank_is_found_relative_to_the_repo_not_the_workspace():
    """On the box `WORKSPACE=/workspace` while the checkout is at
    /workspace/NandaProj, so a WORKSPACE-derived path resolves to /workspace/data
    -- absent there, present locally. That is a path bug that only fails once a
    GPU is billing, so it is pinned here instead."""
    from nandaproj import config

    assert items.BANK.is_relative_to(config.REPO)
    assert items.BANK.exists()
    assert (config.REPO / "src" / "nandaproj" / "items.py").exists()  # REPO is the repo


def test_the_shipped_bank_loads_with_nothing_synthesised():
    """`load_bank` passes conditions=None: a missing condition must raise rather
    than acquire a fallback persona that never appears in the writeup (7.4)."""
    bank = items.load_bank()
    assert len(bank) > 0
    assert bank.used_templates == []
    assert {i.is_no_belief for i in bank} == {True, False}   # both kinds present
