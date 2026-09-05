"""Tests for the parts of `intervene` that do not need a GPU.

The hook machinery needs a model, a tokenizer and a fitted lens, so it is
exercised on the box by `intervene.self_check` -- which is written as a hard
gate rather than a diagnostic for exactly that reason. What is testable here is
the accounting that turns forward passes into a claim: which items are eligible
to be flipped, what counts as truthful, and whether a saved grid reloads as the
same grid. That accounting is where a wrong result would look like a finding
rather than like an error.
"""

from __future__ import annotations

import math

import pytest

from nandaproj import intervene as iv


def trial(item_id="X", *, p_honest=0.9, kind="add", subgroup="legible",
          condition="D", layer=25, alpha=1.0) -> iv.Trial:
    return iv.Trial(
        item_id=item_id, condition=condition, kind=kind, direction="jlens",
        layer=layer, alpha=alpha, p_honest=p_honest, p_lie=1.0 - p_honest,
        mass=0.99, emitted=" Yes", answer=" Yes", subgroup=subgroup,
    )


def test_truthful_is_the_honest_answer_leading_the_legal_ones():
    assert trial(p_honest=0.51).truthful
    assert not trial(p_honest=0.49).truthful


def test_a_tie_is_not_truthful():
    # The honest answer has to *win*, not draw. A 50/50 slot is the model
    # declining to commit, and counting it as a flip would inflate every rate.
    assert not trial(p_honest=0.5).truthful


def test_flip_rate_counts_only_items_that_were_lying_to_begin_with():
    # An item already answering honestly under the edit's condition cannot be
    # "flipped back". Counting it would let a null edit score 100% on a bank
    # where the model happened not to lie.
    baseline = {"A": trial("A", p_honest=0.0), "B": trial("B", p_honest=1.0)}
    edited = [trial("A", p_honest=1.0), trial("B", p_honest=1.0)]
    assert iv.flip_rate(edited, baseline) == 1.0


def test_flip_rate_is_zero_when_nothing_moved():
    baseline = {"A": trial("A", p_honest=0.0), "B": trial("B", p_honest=0.0)}
    edited = [trial("A", p_honest=0.0), trial("B", p_honest=0.1)]
    assert iv.flip_rate(edited, baseline) == 0.0


def test_flip_rate_averages_over_the_eligible_items_only():
    baseline = {"A": trial("A", p_honest=0.0), "B": trial("B", p_honest=0.0),
                "C": trial("C", p_honest=1.0)}
    edited = [trial("A", p_honest=1.0), trial("B", p_honest=0.0),
              trial("C", p_honest=1.0)]
    assert iv.flip_rate(edited, baseline) == 0.5


def test_flip_rate_is_nan_when_no_item_was_eligible():
    # Not 0.0. A rate over an empty set is undefined, and reporting it as zero
    # would read as "the edit did nothing" when the truth is "nothing could
    # have been measured".
    baseline = {"A": trial("A", p_honest=1.0)}
    assert math.isnan(iv.flip_rate([trial("A", p_honest=1.0)], baseline))


def test_flip_rate_ignores_trials_with_no_baseline():
    # A trial whose item never got an unedited run is not evidence either way.
    baseline = {"A": trial("A", p_honest=0.0)}
    edited = [trial("A", p_honest=1.0), trial("Z", p_honest=0.0)]
    assert iv.flip_rate(edited, baseline) == 1.0


def test_trials_round_trip_through_disk(tmp_path):
    grid = [trial("A", alpha=a, subgroup=s)
            for a in (-1.0, 0.0, 1.0) for s in ("legible", "not_legible")]
    path = iv.save_trials(grid, tmp_path / "trials.json")
    assert iv.load_trials(path) == grid


def test_saving_creates_the_results_directory(tmp_path):
    path = iv.save_trials([trial()], tmp_path / "nested" / "trials.json")
    assert path.exists()


def test_the_alpha_grid_is_signed_and_contains_the_no_op():
    # alpha=0 is the internal control: the `add` machinery running with no dose
    # must reproduce the baseline, and if it does not the hook is not clean.
    # Both signs, because a direction that only works one way is a different
    # object from one that works in both.
    assert 0.0 in iv.ALPHAS
    assert all(-a in iv.ALPHAS for a in iv.ALPHAS)


def test_the_layer_grid_brackets_the_crossover_from_04():
    # l* = 25. A grid that only reached up to it could not show the effect
    # decaying above it, which is half of the predicted layer profile.
    assert 25 in iv.LAYERS
    assert min(iv.LAYERS) < 25 < max(iv.LAYERS)


@pytest.mark.parametrize("name", ["ablate", "add", "replace", "editing", "self_check"])
def test_the_torch_paths_are_importable_without_torch(name):
    # Every torch import in this module is function-local so that the accounting
    # above is testable off the box. If one migrates to module scope this test
    # fails here rather than at `import nandaproj.intervene` in CI.
    assert callable(getattr(iv, name))


# --------------------------------------------------------------------------
# The scalar solve behind `set_gap`. Pure arithmetic, so it is testable here;
# the model-dependent `f` is supplied on the box.
# --------------------------------------------------------------------------

def test_bisect_finds_the_root_of_a_monotone_function():
    assert iv.bisect(lambda x: 2 * x, target=6.0, lo=-50, hi=50) == pytest.approx(3.0)


def test_bisect_hits_a_nonzero_target():
    assert iv.bisect(lambda x: x + 1, target=0.0, lo=-50, hi=50) == pytest.approx(-1.0)


def test_bisect_handles_a_decreasing_function():
    # The lens gap falls with alpha for half the items -- a_H and a_D swap roles
    # with the item's polarity -- so a solve that assumed increasing would return
    # the wrong edge on half the bank.
    assert iv.bisect(lambda x: -3 * x, target=6.0, lo=-50, hi=50) == pytest.approx(-2.0)


def test_bisect_returns_the_nearest_edge_when_the_target_is_out_of_reach():
    # An unreachable gap is a fact about the layer, not an error: the caller gets
    # the closest achievable edit and an alpha sitting at the bracket edge, which
    # is what flags it as unreachable in the report.
    assert iv.bisect(lambda x: x, target=500.0, lo=-50, hi=50) == 50


def test_bisect_returns_the_other_edge_for_an_unreachable_low_target():
    assert iv.bisect(lambda x: x, target=-500.0, lo=-50, hi=50) == -50


def test_bisect_is_exact_enough_for_a_logit_gap():
    # 40 halvings of a 100-wide bracket is ~1e-10, far below the precision the
    # gap itself carries in bf16.
    got = iv.bisect(lambda x: x**3, target=27.0, lo=-50, hi=50)
    assert got == pytest.approx(3.0, abs=1e-6)


# --------------------------------------------------------------------------
# CellStats -- telling indifference apart from wreckage
# --------------------------------------------------------------------------


def _trial(item_id, p_h, mass, alpha=0.0, answer=" Yes", **kw):
    return iv.Trial(
        item_id=item_id, condition="D", kind="set_gap_flip", direction="jlens",
        layer=25, alpha=alpha, p_honest=p_h, p_lie=1.0 - p_h, mass=mass,
        emitted=answer, answer=answer, subgroup="legible", **kw)


def test_p_honest_at_half_with_mass_is_a_genuinely_undecided_model():
    rows = [_trial("A", 0.5, 0.97), _trial("B", 0.5, 0.95)]
    stats = iv.summarize(rows, {})
    assert stats.p_honest == pytest.approx(0.5)
    assert not stats.degenerate
    assert "!" not in str(stats)


def test_p_honest_at_half_with_no_mass_is_flagged_as_out_of_format():
    """The set_gap flip rows: |a| ~ 45 replaces the residual, both answers go to
    ~0, and their ratio prints as a perfectly undecided model."""
    rows = [_trial("A", 0.5, 1e-4, alpha=45.0), _trial("B", 0.5, 3e-4, alpha=48.0)]
    stats = iv.summarize(rows, {})
    assert stats.degenerate
    assert str(stats).endswith("!")
    assert stats.alpha == pytest.approx(46.5)


def test_mass_is_a_median_so_one_wrecked_item_is_not_averaged_away():
    rows = [_trial(c, 0.5, m) for c, m in zip("ABCDE", [0.9, 0.9, 0.9, 0.9, 0.0])]
    assert iv.summarize(rows, {}).mass == pytest.approx(0.9)
    rows = [_trial(c, 0.5, m) for c, m in zip("ABCDE", [0.9, 0.0, 0.0, 0.0, 0.0])]
    assert iv.summarize(rows, {}).mass == pytest.approx(0.0)


def test_summarize_of_an_empty_group_is_nan_not_a_crash():
    stats = iv.summarize([], {})
    assert stats.n == 0
    assert stats.p_honest != stats.p_honest          # NaN


def test_the_bar_is_the_gate_s_own_bar_not_a_new_one():
    from nandaproj import lens_readout
    assert iv.MIN_ANSWER_MASS is lens_readout.MIN_ANSWER_MASS


# --------------------------------------------------------------------------
# `rescale_to`. The nulls are drawn on the CPU and norm-matched against an
# item's `d_J`, which lives wherever the model does -- so the scale has to
# leave the reference's device behind.
# --------------------------------------------------------------------------

def test_rescale_to_matches_the_reference_norm():
    torch = pytest.importorskip("torch")
    out = iv.rescale_to(torch.tensor([3.0, 4.0]), torch.tensor([0.0, 7.0]))
    assert float(out.norm()) == pytest.approx(7.0)
    assert float(out[0] / out[1]) == pytest.approx(3.0 / 4.0)


def test_rescale_to_keeps_the_direction_on_its_own_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU to have two devices to confuse")
    out = iv.rescale_to(torch.tensor([3.0, 4.0]),
                        torch.tensor([0.0, 7.0], device="cuda"))
    assert out.device.type == "cpu"
    assert float(out.norm()) == pytest.approx(7.0)
