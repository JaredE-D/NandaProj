"""Tests for the answers-only pilot: prompts, the cities sample, the report logic. No model."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from nandaproj import deception, pilot


@pytest.fixture(scope="module")
def v2():
    return pilot.v2_items()


def test_yesno_is_the_plan2_prompt_byte_for_byte(v2):
    for it in v2[:10]:
        for cond in pilot.CONDITIONS:
            assert pilot.build_chat(it.item, cond, "yesno") == deception.build_chat(it.item, cond)


@pytest.mark.parametrize("fmt, first", [("ab_yesA", "A) Yes\nB) No"), ("ab_yesB", "A) No\nB) Yes")])
def test_ab_swaps_only_the_format_sentence_and_appends_the_options(v2, fmt, first):
    item = v2[0].item
    for cond in pilot.CONDITIONS:
        plan2 = deception.build_chat(item, cond)
        ab = pilot.build_chat(item, cond, fmt)
        assert ab[0]["content"] == plan2[0]["content"].replace(deception.ANSWER_FORMAT, pilot.AB_FORMAT)
        assert ab[1]["content"] == f"{plan2[1]['content']}\n\n{first}"


def test_candidates_name_the_letter_that_means_yes():
    assert pilot.candidates("yesno") == (" Yes", " No")
    assert pilot.candidates("ab_yesA") == (" A", " B")
    assert pilot.candidates("ab_yesB") == (" B", " A")
    with pytest.raises(ValueError):
        pilot.candidates("free_text")


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["statement", "label", "city", "country", "correct_country"])
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def got_files(tmp_path):
    aff, neg = [], []
    for c in ("Oslo", "Lima", "Quito", "Hanoi"):
        aff += [{"statement": f"The city of {c} is in X.", "label": "1", "city": c, "country": "X", "correct_country": "X"},
                {"statement": f"The city of {c} is in Y.", "label": "0", "city": c, "country": "Y", "correct_country": "X"}]
        neg += [{"statement": f"The city of {c} is not in X.", "label": "0", "city": c, "country": "X", "correct_country": "X"},
                {"statement": f"The city of {c} is not in Y.", "label": "1", "city": c, "country": "Y", "correct_country": "X"}]
    # A city missing from neg_cities must never be sampled: its four cells are not the same fact.
    aff.append({"statement": "The city of Rome is in Z.", "label": "1", "city": "Rome", "country": "Z", "correct_country": "Z"})
    _write(tmp_path / "cities.csv", aff)
    _write(tmp_path / "neg_cities.csv", neg)
    return tmp_path / "cities.csv", tmp_path / "neg_cities.csv"


def test_city_items_are_complete_four_cell_blocks(got_files):
    out = pilot.city_items(*got_files, n_cities=3, rng=np.random.default_rng(0))
    assert len(out) == 12
    by_pair = {}
    for it in out:
        by_pair.setdefault(it.item.pair_id, set()).add((it.negated, it.truth_yes))
        assert it.target in ("true", "false")
        assert it.item.persona == pilot.QUIZ_PERSONA.format(target=it.target)
        assert it.item.context.startswith("Statement: The city of")
    assert "city_rome" not in by_pair
    assert all(cells == {(False, True), (False, False), (True, True), (True, False)}
               for cells in by_pair.values())


def test_city_sample_is_reproducible_from_the_seed(got_files):
    a = pilot.city_items(*got_files, n_cities=3, rng=np.random.default_rng(7))
    b = pilot.city_items(*got_files, n_cities=3, rng=np.random.default_rng(7))
    assert [(i.item.item_id, i.target) for i in a] == [(i.item.item_id, i.target) for i in b]


def test_conflict_means_the_target_is_not_the_truth(got_files):
    for it in pilot.city_items(*got_files, n_cities=4, rng=np.random.default_rng(1)):
        assert it.conflict == ((it.target == "true") != it.truth_yes)


def test_every_v2_item_is_a_conflict_item(v2):
    assert len(v2) == 100 and all(it.conflict for it in v2)


def _row(item_id, cond, fmt, yes):
    return {"item_id": item_id, "condition": cond, "format": fmt,
            "p_yes": 0.9 if yes else 0.1, "p_no": 0.1 if yes else 0.9}


def test_outcomes_drop_h_wrong_and_no_conflict_items_and_label_lies(v2):
    yes_true = next(it for it in v2 if it.truth_yes)
    no_true = next(it for it in v2 if not it.truth_yes)
    wrong_h = next(it for it in v2 if it.truth_yes and it is not yes_true)
    rows = [_row(yes_true.item.item_id, "H", "yesno", True), _row(yes_true.item.item_id, "D", "yesno", False),
            _row(no_true.item.item_id, "H", "yesno", False), _row(no_true.item.item_id, "D", "yesno", False),
            _row(wrong_h.item.item_id, "H", "yesno", False), _row(wrong_h.item.item_id, "D", "yesno", False)]
    items = {it.item.item_id: it for it in (yes_true, no_true, wrong_h)}
    got = {it.item.item_id: lied for it, _, lied in pilot.outcomes(rows, items, ["yesno"])}
    assert got == {yes_true.item.item_id: True, no_true.item.item_id: False}


def test_passes_needs_both_outcomes_in_every_arm():
    ok = {"a": pilot.cell([True] * 3 + [False] * 5), "b": pilot.cell([True] * 5 + [False] * 3)}
    assert pilot.passes(ok)
    skewed = {"a": ok["a"], "b": pilot.cell([True] * 9 + [False] * 1)}
    assert not pilot.passes(skewed)
    assert not pilot.passes({"a": ok["a"], "b": pilot.cell([])})


def test_wilson_brackets_the_rate():
    lo, hi = pilot.wilson(3, 10)
    assert lo < 0.3 < hi and 0 <= lo and hi <= 1


def test_alleged_items_are_the_full_authored_bank_and_all_conflict():
    alleged = pilot.alleged_items()
    assert len(alleged) == 880
    assert sum(it.truth_yes for it in alleged) == 440
    assert all(it.source == "alleged" and it.conflict for it in alleged)
    assert pilot.SOURCE_CONDITIONS["alleged"] == ("H", "D")


def test_robust_labels_separate_mapping_flips_from_stable_outcomes(v2):
    stable_lie, stable_honest, flip, h_wrong = [it for it in v2 if it.truth_yes][:4]
    rows = []
    for it, d_yes in ((stable_lie, {"ab_yesA": False, "ab_yesB": False}),
                      (stable_honest, {"ab_yesA": True, "ab_yesB": True}),
                      (flip, {"ab_yesA": True, "ab_yesB": False})):
        for fmt, yes in d_yes.items():
            rows += [_row(it.item.item_id, "H", fmt, True), _row(it.item.item_id, "D", fmt, yes)]
    # H wrong under one mapping: no robust label at all.
    rows += [_row(h_wrong.item.item_id, "H", "ab_yesA", False), _row(h_wrong.item.item_id, "D", "ab_yesA", False),
             _row(h_wrong.item.item_id, "H", "ab_yesB", True), _row(h_wrong.item.item_id, "D", "ab_yesB", False)]
    items = {it.item.item_id: it for it in (stable_lie, stable_honest, flip, h_wrong)}
    assert pilot.robust_labels(rows, items) == {stable_lie.item.item_id: "lied",
                                                stable_honest.item.item_id: "honest",
                                                flip.item.item_id: "flips"}


def test_render_turns_qwen_thinking_off_and_ends_at_the_answer_prefix(v2):
    seen = {}

    class FakeTok:
        def apply_chat_template(self, chat, tokenize, add_generation_prompt, **kwargs):
            seen.update(kwargs)
            return "<rendered>"

    out = pilot.render(FakeTok(), v2[0].item, "D", "ab_yesA")
    assert seen == {"enable_thinking": False}
    assert out == "<rendered>" + deception.ANSWER_PREFIX


def test_lens_paths_for_gemma_and_qwen_presets():
    from nandaproj import config

    assert config.get_model_config("escalate").lens_path == \
        "gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt"
    assert config.get_model_config("qwen14b").lens_path == \
        "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt"
