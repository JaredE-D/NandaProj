"""Tests for the pair-aware linear probes. Synthetic data, no GPU, no torch."""

from __future__ import annotations

import numpy as np
import pytest

from nandaproj import probe


def planted(n_pairs=40, d=64, signal=3.0, seed=0):
    """Pairs of rows sharing a context vector, labels opposite, plus a planted
    truth direction scaled by `signal`. Returns X, y, groups, direction."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=d)
    truth /= np.linalg.norm(truth)
    X, y, groups = [], [], []
    for p in range(n_pairs):
        context = rng.normal(size=d) * 2.0
        for label in (True, False):
            noise = rng.normal(size=d) * 0.5
            X.append(context + noise + (signal if label else -signal) * truth)
            y.append(label)
            groups.append(f"P{p:02d}")
    return np.array(X), np.array(y), groups, truth


def test_mean_diff_recovers_the_planted_direction():
    X, y, _, truth = planted()
    w, _ = probe.fit_mean_diff(X, y)
    cos = w @ truth / np.linalg.norm(w)
    assert cos > 0.95


def test_both_fits_need_both_classes():
    X = np.zeros((4, 3))
    with pytest.raises(ValueError):
        probe.fit_mean_diff(X, np.ones(4, dtype=bool))
    with pytest.raises(ValueError):
        probe.fit_logistic(X, np.zeros(4))


def test_grouped_folds_hold_out_whole_pairs():
    groups = ["A", "A", "B", "B", "C", "C"]
    for train, test in probe.grouped_folds(groups):
        held = {groups[i] for i in test}
        assert len(held) == 1
        assert held.isdisjoint({groups[i] for i in train})
        assert len(test) == 2


def test_cv_accuracy_is_high_with_signal_and_chance_without():
    X, y, groups, _ = planted(signal=4.0)
    assert probe.cv_accuracy(X, y, groups) > 0.9
    assert probe.cv_accuracy(X, y, groups, fit=probe.fit_logistic) > 0.9
    X0, y0, g0, _ = planted(signal=0.0, seed=1)
    acc = probe.cv_accuracy(X0, y0, g0)
    assert 0.3 < acc < 0.7


def test_batched_logistic_cv_matches_the_per_fold_version():
    X, y, groups, _ = planted(n_pairs=12, d=32, signal=1.0, seed=2)
    slow = probe.cv_accuracy(X, y, groups, fit=probe.fit_logistic)
    fast = probe.cv_logistic(X, y, groups)
    assert abs(slow - fast) <= 1.0 / len(y)      # at most one row differs at float32


def test_flip_within_pairs_keeps_twins_opposite_and_changes_something():
    _, y, groups, _ = planted(n_pairs=30)
    rng = np.random.default_rng(3)
    flipped = probe.flip_within_pairs(y, groups, rng)
    g = np.asarray(groups)
    for pid in set(groups):
        a, b = flipped[g == pid]
        assert a != b
    assert (flipped != y).any() and (flipped == y).any()


def test_permutation_null_sits_at_chance_even_with_signal():
    # The null breaks question-label linkage while keeping pair structure, so
    # a strongly planted direction must not survive it.
    X, y, groups, _ = planted(signal=3.0)
    null = probe.permutation_null(X, y, groups, n=30, seed=0)
    assert null.shape == (30,)
    assert abs(null.mean() - 0.5) < 0.15
    assert np.percentile(null, 95) < 0.9


def test_transfer_reads_the_planted_label_on_new_rows():
    X, y, _, truth = planted(seed=0)
    X2, y2, _, _ = planted(seed=1)
    # Same direction in both draws only if we plant the same one; do that.
    rng = np.random.default_rng(5)
    X2 = rng.normal(size=X2.shape) + np.where(y2[:, None], 3.0, -3.0) * truth
    pred = probe.transfer(X, y, X2)
    assert (pred == y2).mean() > 0.9


def test_residuals_round_trip(tmp_path):
    res = probe.Residuals(
        item_ids=["a", "b"], layers=[0, 5],
        by_condition={"H": np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3),
                      "D": np.ones((2, 2, 3), dtype=np.float32)})
    path = res.save(tmp_path / "r.npz")
    back = probe.Residuals.load(path)
    assert back.item_ids == ["a", "b"] and back.layers == [0, 5]
    assert set(back.by_condition) == {"H", "D"}
    np.testing.assert_allclose(back.X("H", 5), res.X("H", 5))


# --------------------------------------------------------------------------
# 04e helpers
# --------------------------------------------------------------------------

def test_unit_has_norm_one_and_keeps_direction():
    w = np.array([3.0, 4.0])
    u = probe.unit(w)
    assert abs(np.linalg.norm(u) - 1) < 1e-12
    assert np.allclose(u, [0.6, 0.8])


def test_signed_projection_signs_follow_labels_and_ignore_probe_scale():
    X, y, _, _ = planted(signal=4.0)
    w, b = probe.fit_mean_diff(X, y)
    s = probe.signed_projection((w, b), X)
    assert ((s > 0) == y).mean() > 0.95
    np.testing.assert_allclose(s, probe.signed_projection((10 * w, 10 * b), X))
    # zero is the probe's own threshold
    np.testing.assert_array_equal(s > 0, probe.predict((w, b), X))


def _orthonormal(d, r, seed=0):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, r)))
    return Q


def test_subspace_mass_on_basis_vectors_is_a_step():
    V = _orthonormal(40, 40)
    ks = (1, 5, 6, 20, 40)
    rho = probe.subspace_mass(V, V[:, 5], ks)          # 6th column: inside V_k iff k >= 6
    assert rho.shape == (len(ks), 1)
    np.testing.assert_allclose(rho[:, 0], [0.0, 0.0, 1.0, 1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(probe.subspace_mass(V, 3.0 * V[:, 0], ks)[:, 0], 1.0)


def test_subspace_mass_is_k_over_d_for_isotropic_directions():
    d, n = 64, 4000
    V = _orthonormal(d, d)
    W = np.random.default_rng(1).normal(size=(d, n))
    ks = (4, 16, 32, 64)
    rho = probe.subspace_mass(V, W, ks)
    assert rho.shape == (len(ks), n)
    np.testing.assert_allclose(rho.mean(1), np.array(ks) / d, atol=0.02)
    np.testing.assert_allclose(rho[-1], 1.0, atol=1e-10)   # full basis holds everything
    assert (np.diff(rho, axis=0) >= -1e-12).all()          # monotone in k


def test_subspace_mass_rejects_k_outside_the_basis():
    V = _orthonormal(10, 4)
    with pytest.raises(ValueError):
        probe.subspace_mass(V, np.ones(10), (2, 8))
    with pytest.raises(ValueError):
        probe.subspace_mass(V, np.ones(10), (0,))
