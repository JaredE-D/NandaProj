"""Tests for the residual-stream PCA and the numbers that keep it honest.

A scatter plot cannot be wrong out loud. `eta_squared` and `cosine_to_mean` are
the two numbers that can, so they are what is tested here: a PC that separates
the conditions and a PC that separates the scenarios look identical on a plot
and differ by an order of magnitude in these.
"""

from __future__ import annotations

import numpy as np
import pytest

from nandaproj import geometry


def two_clusters(n=10, d=6, gap=10.0, seed=0):
    """`2n` points in two clusters separated along dimension 0."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, d))
    b = rng.normal(size=(n, d))
    b[:, 0] += gap
    return np.vstack([a, b]), ["A"] * n + ["B"] * n


# --------------------------------------------------------------------------
# pca
# --------------------------------------------------------------------------

def test_pc1_is_the_axis_the_data_actually_varies_along():
    x, _ = two_clusters(gap=50.0)
    fit = geometry.pca(x, k=2)
    # The separation is along dimension 0, so PC1 must be (anti)parallel to it.
    assert abs(fit.components[0, 0]) > 0.95
    assert fit.explained[0] > 0.9


def test_explained_is_a_share_of_the_total_not_of_the_kept():
    x, _ = two_clusters(gap=3.0, d=8)
    two = geometry.pca(x, k=2)
    four = geometry.pca(x, k=4)
    # Adding components must not change what the first two are worth.
    assert two.explained[:2] == pytest.approx(four.explained[:2])
    assert four.explained.sum() <= 1.0 + 1e-9


def test_scores_are_the_data_projected_onto_the_components():
    x, _ = two_clusters()
    fit = geometry.pca(x, k=3)
    assert fit.scores == pytest.approx((x - fit.mean) @ fit.components.T)


def test_scores_are_centred():
    x, _ = two_clusters()
    fit = geometry.pca(x, k=2)
    assert fit.scores.mean(axis=0) == pytest.approx(np.zeros(2), abs=1e-9)


def test_k_is_capped_at_the_rank_the_data_has():
    fit = geometry.pca(np.eye(3)[:3], k=10)
    assert fit.k <= 3


def test_a_pca_of_one_vector_raises_rather_than_returning_a_point():
    with pytest.raises(ValueError, match="not defined"):
        geometry.pca([[1.0, 2.0]])


def test_a_non_matrix_raises_with_the_shape_named():
    with pytest.raises(ValueError, match="expected an"):
        geometry.pca(np.zeros((2, 3, 4)))


# --------------------------------------------------------------------------
# eta_squared -- the number that says what a PC is about
# --------------------------------------------------------------------------

def test_a_component_that_is_the_grouping_scores_one():
    assert geometry.eta_squared([1, 1, 1, 5, 5, 5], list("AAABBB")) == pytest.approx(1.0)


def test_a_grouping_that_says_nothing_scores_zero():
    assert geometry.eta_squared([1, 5, 1, 5], list("AABB")) == pytest.approx(0.0)


def test_the_separated_axis_scores_high_and_the_others_low():
    x, labels = two_clusters(gap=30.0)
    fit = geometry.pca(x, k=3)
    assert geometry.eta_squared(fit.scores[:, 0], labels) > 0.9
    assert geometry.eta_squared(fit.scores[:, 1], labels) < 0.3


def test_a_scenario_axis_and_a_condition_axis_are_told_apart():
    # The case the module exists for: 8 scenarios, each run under 2 conditions.
    # Scenario spread is 20x the condition displacement, so a plot shows eight
    # tidy pairs and PC1 is topic, not condition.
    rng = np.random.default_rng(1)
    scenario = rng.normal(scale=20.0, size=(8, 5))
    shift = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    rows, conds, items = [], [], []
    for s, base in enumerate(scenario):
        for cond, sign in (("H", -1), ("D", 1)):
            rows.append(base + sign * shift)
            conds.append(cond)
            items.append(f"item{s}")
    fit = geometry.pca(rows, k=2)
    assert geometry.eta_squared(fit.scores[:, 0], items) > 0.9      # scenario
    assert geometry.eta_squared(fit.scores[:, 0], conds) < 0.2      # not condition


def test_values_and_labels_of_different_lengths_raise():
    with pytest.raises(ValueError, match="against"):
        geometry.eta_squared([1, 2, 3], list("AB"))


def test_a_constant_component_explains_nothing_rather_than_dividing_by_zero():
    assert geometry.eta_squared([2.0, 2.0, 2.0, 2.0], list("AABB")) == 0.0


def test_label_report_names_every_pc_and_every_labelling():
    x, labels = two_clusters()
    fit = geometry.pca(x, k=2)
    text = geometry.label_report(fit, {"cond": labels})
    assert "PC1" in text and "PC2" in text and "cond" in text


# --------------------------------------------------------------------------
# paired_differences -- the scenario cancels
# --------------------------------------------------------------------------

def test_differencing_cancels_the_scenario_exactly():
    shift = np.array([0.0, 3.0, 0.0])
    vectors = {}
    for s in range(4):
        base = np.array([100.0 * s, 0.0, -7.0 * s])
        vectors[(f"i{s}", "H")] = base
        vectors[(f"i{s}", "D")] = base + shift
    diffs, kept = geometry.paired_differences(
        vectors, [f"i{s}" for s in range(4)], "H", "D")
    assert kept == ["i0", "i1", "i2", "i3"]
    for row in diffs:
        assert row == pytest.approx(shift)


def test_an_item_missing_a_condition_is_skipped_not_filled():
    vectors = {("a", "H"): [1.0], ("a", "D"): [2.0], ("b", "H"): [1.0]}
    diffs, kept = geometry.paired_differences(vectors, ["a", "b"], "H", "D")
    assert kept == ["a"]
    assert len(diffs) == 1


def test_no_complete_pair_raises_rather_than_returning_an_empty_matrix():
    with pytest.raises(ValueError, match="nothing to difference"):
        geometry.paired_differences({("a", "H"): [1.0]}, ["a"], "H", "D")


# --------------------------------------------------------------------------
# cosine_to_mean -- is there a shared direction at all?
# --------------------------------------------------------------------------

def test_a_shared_direction_gives_cosines_near_one():
    diffs = [[1.0, 0.0, 0.0]] * 5
    assert geometry.cosine_to_mean(diffs) == pytest.approx(np.ones(5))


def test_item_specific_displacement_does_not_look_like_a_direction():
    # Orthogonal displacements: the mean has a norm, but no vector is near it.
    cos = geometry.cosine_to_mean(np.eye(6))
    assert abs(cos).max() < 0.1


def test_the_score_is_leave_one_out_so_a_vector_never_scores_itself():
    # Four aligned and one opposed. Under a self-inclusive mean the opposed
    # vector still drags the mean toward itself; leave-one-out puts it at -1.
    diffs = [[1.0, 0.0]] * 4 + [[-1.0, 0.0]]
    cos = geometry.cosine_to_mean(diffs)
    assert cos[-1] == pytest.approx(-1.0)
    assert cos[0] == pytest.approx(1.0, abs=1e-9)


def test_a_zero_vector_scores_zero_rather_than_nan():
    assert geometry.cosine_to_mean([[0.0, 0.0], [1.0, 0.0]])[0] == 0.0


def test_one_vector_has_no_mean_to_compare_against():
    with pytest.raises(ValueError, match="at least two"):
        geometry.cosine_to_mean([[1.0, 0.0]])
