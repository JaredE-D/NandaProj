"""Tests for the helpers behind scripts/lie_probe_12b.py. Synthetic data, numpy only."""

from __future__ import annotations

import numpy as np

from nandaproj import baselines, probe


def test_weighted_fit_none_matches_unit_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5)).astype(np.float32)
    Y = rng.random((60, 3)) < 0.5
    W0, b0 = baselines.fit_logistic_many(X, Y)
    W1, b1 = baselines.fit_logistic_many(X, Y, weights=np.ones(Y.shape))
    assert np.allclose(W0, W1, atol=1e-6) and np.allclose(b0, b1, atol=1e-6)


def test_balanced_weights_equalise_class_mass():
    y = np.array([True] * 10 + [False] * 90)
    w = baselines.balanced_weights(y)[:, 0]
    assert np.isclose(w[y].sum(), w[~y].sum()) and np.isclose(w.mean(), 1.0)


def test_balanced_weights_fix_the_majority_class_predictor():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 4))
    y = rng.random(400) < 0.1
    X[y, 0] += 1.5
    Xs = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    plain = baselines.fit_logistic_many(Xs, y)
    weighted = baselines.fit_logistic_many(Xs, y, weights=baselines.balanced_weights(y))
    recall = lambda W, b: ((Xs @ W + b)[y, 0] > 0).mean()
    assert recall(*weighted) > recall(*plain)


def test_grouped_kfold_keeps_groups_together_and_covers_every_row():
    groups = np.repeat([f"P{i}" for i in range(23)], 3)
    folds = probe.grouped_kfold(groups, 10, seed=0)
    assert len(folds) == 10
    tests = np.concatenate([te for _, te in folds])
    assert sorted(tests) == list(range(len(groups)))
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, probe.grouped_kfold(groups, 10, 0)))


def test_permutation_within_cells_preserves_per_cell_counts():
    rng = np.random.default_rng(2)
    cell = rng.integers(0, 4, size=200)
    y = rng.random(200) < 0.3
    for _ in range(5):
        yp = baselines.permute_within_arms(y, cell, rng)
        assert not np.array_equal(yp, y)
        for c in range(4):
            assert yp[cell == c].sum() == y[cell == c].sum()


def test_cv_scores_accepts_explicit_folds():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] > 0
    folds = [(np.arange(20, 40), np.arange(20))]
    S = baselines.cv_scores(baselines.fixed_features(X), y, [str(i) for i in range(40)],
                            folds=folds, balance=True)
    assert np.all(S[20:] == 0) and ((S[:20, 0] > 0) == y[:20]).mean() > 0.8


def test_item_permutation_keeps_item_labels_shared_and_arm_counts():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from lie_probe_12b import permute_items_within_arms

    rng = np.random.default_rng(1)
    items = np.repeat([f"i{k}" for k in range(40)], 2)       # two mappings per item
    truth = np.repeat(np.arange(40) % 2 == 0, 2)
    y = np.repeat(rng.random(40) < 0.3, 2)
    for _ in range(20):
        yp = permute_items_within_arms(y, items, truth, rng)
        assert np.array_equal(yp[0::2], yp[1::2])               # both rows of an item agree
        for arm in (True, False):
            assert yp[truth == arm].sum() == y[truth == arm].sum()
