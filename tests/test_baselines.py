"""Tests for the Direction-1 detector comparison. Synthetic data, numpy only."""

from __future__ import annotations

import itertools

import numpy as np

from nandaproj import baselines, probe, tokenfeat


def planted(n_pairs=30, d=40, signal=2.0, seed=0):
    """Two rows per pair, labels random, a planted label direction scaled by `signal`."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    y = rng.random(2 * n_pairs) < 0.5
    X = rng.normal(size=(2 * n_pairs, d)) + np.where(y, signal, -signal)[:, None] * direction
    groups = [f"P{i // 2:02d}" for i in range(2 * n_pairs)]
    arm = np.tile([True, False], n_pairs)
    return X, y, groups, arm


def test_fit_logistic_many_matches_the_single_column_fit_per_column():
    X, y, _, _ = planted()
    X = (X - X.mean(0)) / X.std(0)
    rng = np.random.default_rng(1)
    Y = np.stack([y, rng.random(len(y)) < 0.5], axis=1)
    W, b = baselines.fit_logistic_many(X, Y)
    for r in range(2):
        w_ref, b_ref = probe.fit_logistic(X, Y[:, r])
        np.testing.assert_allclose(W[:, r], w_ref, rtol=1e-10, atol=1e-12)
        assert abs(b[r] - b_ref) < 1e-10


def test_cv_scores_find_a_planted_signal_and_not_a_shuffled_one():
    X, y, groups, arm = planted()
    feats = baselines.fixed_features(X)
    good = baselines.summarize(baselines.cv_scores(feats, y, groups)[:, 0], y, arm)
    assert good["balanced"] > 0.9 and good["auroc"] > 0.95

    noise = np.random.default_rng(2).normal(size=X.shape)
    bad = baselines.summarize(
        baselines.cv_scores(baselines.fixed_features(noise), y, groups)[:, 0], y, arm)
    assert 0.25 < bad["balanced"] < 0.75


def test_cv_scores_columns_are_independent_fits():
    X, y, groups, _ = planted()
    feats = baselines.fixed_features(X)
    y2 = ~y
    both = baselines.cv_scores(feats, np.stack([y, y2], axis=1), groups)
    np.testing.assert_allclose(both[:, 0], baselines.cv_scores(feats, y, groups)[:, 0])
    np.testing.assert_allclose(both[:, 1], baselines.cv_scores(feats, y2, groups)[:, 0])


def test_topk_vocabulary_comes_from_training_rows_only():
    ids = np.array([[1, 2], [1, 3], [9, 2]])          # token 9 appears only in row 2
    lp = np.log(np.array([[0.6, 0.3], [0.5, 0.4], [0.7, 0.2]]))
    Xtr, Xte = baselines.topk_features(ids, lp, k_top=2)(np.array([0, 1]), np.array([2]))
    assert Xtr.shape[1] == Xte.shape[1] == 3         # vocab {1, 2, 3}, no 9
    np.testing.assert_array_equal(
        tokenfeat.union_vocab(ids[[0, 1]], 2), [1, 2, 3])


def test_topk_banned_tokens_never_become_features():
    ids = np.array([[1, 2], [1, 3]])
    lp = np.log(np.array([[0.6, 0.3], [0.5, 0.4]]))
    Xtr, _ = baselines.topk_features(ids, lp, k_top=2, banned=[1])(np.array([0, 1]), np.array([0]))
    assert Xtr.shape[1] == 2                          # {2, 3}


def test_tfidf_drops_test_only_and_single_prompt_terms():
    texts = ["the car is fine", "the car is broken", "the boat leaks", "zebra"]
    Xtr, Xte = baselines.tfidf_features(texts, min_df=2)(np.array([0, 1, 2]), np.array([3]))
    # Terms in >= 2 training prompts: "the", "car", "is", "the car", "car is".
    assert Xtr.shape[1] == 5
    assert not Xte.any()                              # "zebra" was never in training
    np.testing.assert_allclose(np.linalg.norm(Xtr, axis=1), 1.0)


def test_auroc_matches_a_brute_force_pair_count_with_ties():
    rng = np.random.default_rng(3)
    scores = rng.integers(0, 4, size=25).astype(float)   # many ties
    y = rng.random(25) < 0.4
    pairs = list(itertools.product(scores[y], scores[~y]))
    expected = np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs])
    assert abs(baselines.auroc(scores, y) - expected) < 1e-12


def test_a_polarity_only_score_is_at_chance_inside_each_arm():
    # Lies mostly on No-true rows: polarity predicts the label overall, nothing within an arm.
    arm = np.array([True] * 10 + [False] * 10)
    y = np.array([True] * 2 + [False] * 8 + [True] * 8 + [False] * 2)
    scores = np.where(arm, -1.0, 1.0)
    s = baselines.summarize(scores, y, arm)
    assert s["balanced"] == 0.8
    assert s["balanced_yes_true"] == 0.5 and s["balanced_no_true"] == 0.5
    assert s["arm_mean"] == 0.5


def test_permute_within_arms_keeps_each_arms_label_counts():
    _, y, _, arm = planted()
    yp = baselines.permute_within_arms(y, arm, np.random.default_rng(4))
    for a in (True, False):
        assert yp[arm == a].sum() == y[arm == a].sum()


def test_bootstrap_ci_brackets_the_point_estimate():
    X, y, groups, arm = planted(signal=0.6)
    scores = baselines.cv_scores(baselines.fixed_features(X), y, groups)[:, 0]
    point = baselines.summarize(scores, y, arm)
    ci = baselines.bootstrap_ci(scores, y, arm, groups, n_boot=300, rng=np.random.default_rng(5))
    for m in ("balanced", "auroc"):
        assert ci[m][0] <= point[m] <= ci[m][1]
