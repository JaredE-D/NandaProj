"""Tests for the top-token classifier pipeline. Synthetic stores, no GPU, no torch."""

from __future__ import annotations

import numpy as np
import pytest

from nandaproj import tokenfeat


class FakeTok:
    """Decodes ids from a fixed table; everything else is a blank."""

    table = {0: " Yes", 1: " No", 2: "yes", 3: " NO", 4: " maybe", 5: " the", 6: ":", 7: " true"}

    def __len__(self):
        return 10

    def decode(self, ids):
        return "".join(self.table.get(i, " x") for i in ids)


def toy_store(n_items=20, layers=(0, 1), k=6, seed=0):
    """Rows H, C1, D per item. Token 7 (' true') is high on lying D rows only;
    token 0/1 (Yes/No) is the emitted answer, alternating by polarity."""
    rng = np.random.default_rng(seed)
    item_ids, conds, ids, lp, y = [], [], [], [], []
    for i in range(n_items):
        lying = i % 2 == 0
        polarity_yes = (i // 2) % 2 == 0
        for c in ("H", "C1", "D"):
            emitted = 0 if polarity_yes else 1
            if c == "D" and lying:
                emitted = 1 - emitted
            # Token 7 is in every row's list, so banning it removes the signal
            # without changing which tokens are present (a missing token would
            # leak the label through the floor, which is a property of the
            # feature map and not what this toy is testing).
            row_ids = [emitted, 7, 4, 5, 6, 8]
            row_lp = [-0.1, -6.5, -3.0, -4.0, -5.0, -6.0]
            if c == "D" and lying:
                row_lp[1] = -1.0          # ' true' rises into the top-3
            row_lp = np.array(row_lp) + rng.normal(scale=0.05, size=6)
            order = np.argsort(-row_lp)
            item_ids.append(f"I{i:02d}")
            conds.append(c)
            ids.append(np.stack([np.array(row_ids)[order]] * len(layers)))
            lp.append(np.stack([row_lp[order]] * len(layers)))
            y.append(c == "D" and lying)
    ids = np.array(ids).astype(np.int32)
    lp = np.array(lp).astype(np.float32)
    store = tokenfeat.TopK(item_ids, conds, list(layers), ids, lp, ids.copy(), lp.copy())
    groups = [f"P{i // 2:02d}" for i in range(n_items) for _ in range(3)]
    return store, np.array(y), groups


def test_store_round_trip(tmp_path):
    store, _, _ = toy_store()
    path = store.save(tmp_path / "topk.npz")
    back = tokenfeat.TopK.load(path)
    assert back.item_ids == store.item_ids and back.conditions == store.conditions
    assert back.layers == store.layers and back.k == store.k
    np.testing.assert_array_equal(back.j_ids, store.j_ids)
    np.testing.assert_allclose(back.l_lp, store.l_lp)
    assert list(back.rows("D")) == list(range(2, back.n, 3))


def test_top_k_of_is_descending_and_logged():
    p = np.array([0.1, 0.5, 0.0, 0.4])
    ids, lp = tokenfeat.top_k_of(p, 3)
    assert list(ids) == [1, 3, 0]
    np.testing.assert_allclose(lp, np.log([0.5, 0.4, 0.1]), rtol=1e-5)
    ids, lp = tokenfeat.top_k_of(p, 4)
    assert np.isfinite(lp).all()          # the zero entry does not become -inf


def test_union_vocab_is_top_k_only_and_masked():
    ids = np.array([[3, 1, 9], [3, 5, 2]])
    assert list(tokenfeat.union_vocab(ids, 2)) == [1, 3, 5]
    assert list(tokenfeat.union_vocab(ids, 2, banned=[3])) == [1, 5]


def test_feature_matrix_uses_row_floor_for_absent_tokens():
    ids = np.array([[3, 1], [5, 2]])
    lp = np.array([[-0.1, -2.0], [-0.5, -9.0]], dtype=np.float32)
    X = tokenfeat.feature_matrix(ids, lp, np.array([1, 3, 5]))
    np.testing.assert_allclose(X[0], [-2.0, -0.1, -2.0])
    np.testing.assert_allclose(X[1], [-9.0, -9.0, -0.5])


def test_feature_matrix_empty_vocab():
    ids = np.array([[3, 1]])
    lp = np.array([[-0.1, -2.0]], dtype=np.float32)
    assert tokenfeat.feature_matrix(ids, lp, np.array([], dtype=int)).shape == (1, 0)


def test_answer_token_ids_scans_every_spelling():
    assert tokenfeat.answer_token_ids(FakeTok()) == [0, 1, 2, 3]


def test_fold_vocab_excludes_held_out_private_token():
    store, y, groups = toy_store()
    ids, _ = store.at("j", 0)
    ids = ids.copy()
    ids[0, 0] = 99                          # a token only row 0 has
    for train, test, vocab in tokenfeat.fold_vocabs(ids, groups, k_top=3):
        if 0 in test:
            assert 99 not in vocab
        else:
            assert 99 in vocab


def test_shuffle_within_items_keeps_per_item_counts():
    store, y, _ = toy_store()
    rng = np.random.default_rng(1)
    ys = tokenfeat.shuffle_within_items(y, store.item_ids, rng)
    items = np.asarray(store.item_ids)
    for it in set(items):
        assert ys[items == it].sum() == y[items == it].sum()
    assert (ys != y).any()


def test_classifier_reads_planted_token_and_loses_it_when_masked():
    store, y, groups = toy_store(n_items=24)
    ids, lp = store.at("j", 0)
    banned = tokenfeat.answer_token_ids(FakeTok())
    full = tokenfeat.cv_accuracy(ids, lp, y, groups, k_top=3, banned=banned)
    assert full > 0.95
    # Ban the planted token too: nothing left but the emitted answer, which
    # this toy balances across polarity, so the classifier falls to the base rate.
    none = tokenfeat.cv_accuracy(ids, lp, y, groups, k_top=3, banned=banned + [7])
    assert none < 0.85


def test_fit_predict_and_top_weights_name_the_planted_token():
    store, y, groups = toy_store(n_items=24)
    ids, lp = store.at("j", 0)
    fitted = tokenfeat.fit(ids, lp, y, k_top=3, banned=[0, 1, 2, 3])
    assert (fitted.predict(ids, lp) == y).mean() > 0.95
    token, weight = fitted.top_weights(FakeTok(), n=1)[0]
    assert token == " true" and weight > 0


def test_balanced_accuracy_ignores_the_base_rate():
    y = np.array([True] + [False] * 9)
    assert tokenfeat.accuracy(np.zeros(10, bool), y) == 0.9
    assert tokenfeat.balanced_accuracy(np.zeros(10, bool), y) == 0.5
    assert tokenfeat.balanced_accuracy(y, y) == 1.0


def test_null_is_near_chance_when_signal_is_present():
    store, y, groups = toy_store(n_items=16)
    ids, lp = store.at("j", 0)
    null = tokenfeat.permutation_null(ids, lp, y, groups, store.item_ids, k_top=3,
                                      banned=[0, 1, 2, 3], n=5, steps=50,
                                      metric=tokenfeat.balanced_accuracy, desc=None)
    assert null.shape == (5,)
    assert null.max() < 0.9


def test_batched_cv_matches_per_fold_reference():
    store, y, groups = toy_store(n_items=24)
    ids, lp = store.at("j", 0)
    banned = [0, 1, 2, 3]
    ref = tokenfeat.cv_predict(ids, lp, y, groups, k_top=3, banned=banned)
    rng = np.random.default_rng(0)
    Ys = np.stack([y, tokenfeat.shuffle_within_items(y, store.item_ids, rng)])
    got = tokenfeat.cv_predict_many(ids, lp, Ys, groups, k_top=3, banned=banned, device="cpu")
    assert got.shape == (2, len(y))
    np.testing.assert_array_equal(got[0], ref)
    ref_null = tokenfeat.cv_predict(ids, lp, Ys[1], groups, k_top=3, banned=banned)
    np.testing.assert_array_equal(got[1], ref_null)


def test_batched_cv_fold_mask_hides_private_token():
    # Row 0 alone carries token 99 in its top-3; with row 0 held out the token is
    # masked, so the batched fit must not use it there. Equality with the
    # per-fold reference (which builds its vocab without row 0) proves that.
    store, y, groups = toy_store(n_items=12)
    ids, lp = store.at("j", 0)
    ids = ids.copy(); ids[0, 0] = 99
    ref = tokenfeat.cv_predict(ids, lp, y, groups, k_top=3, banned=[0, 1, 2, 3])
    got = tokenfeat.cv_predict_many(ids, lp, y, groups, k_top=3, banned=[0, 1, 2, 3], device="cpu")
    np.testing.assert_array_equal(got[0], ref)
