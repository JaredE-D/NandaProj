"""Same-token lie detection on gemma-3-12b-it: residual probe vs baselines and controls.

Pre-registration: `misc/prereg_12b_same_token.md` (success bar, controls, null).
Reads the capture script's output; numpy only, one (layer, position) slice in
memory at a time.

  results/capture_12b_<lens>.npz   resid [N, 2, L, d] float16, positions ("pre", "slot")
  results/capture_12b_<lens>.json  one row per (item, condition, mapping): labels and metadata

Primary rows are D. H rows carry the same item labels and feed the propensity
control: if H residuals predict the D label as well as D residuals do, the probe
reads which items the model lies on, not the lie.

Writes results/lie_probe_12b_<lens>.json and pubplots/lie_probe_12b.png.

Run: `uv run --extra plot python scripts/lie_probe_12b.py`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pubplots as pp
from nandaproj import baselines, probe

RESULTS = pp.RESULTS
N_PCA = 256
N_FOLDS = 10
N_BOOT = 2000
SEED = 20261018
HELD_OUT_CATEGORIES = ("alleged_fault", "alleged_hazard", "alleged_slight")
POSITIONS = ("pre", "slot")


def pca_scores(X: np.ndarray, n_components: int = N_PCA) -> np.ndarray:
    """Top principal-component scores of all rows. Unsupervised, so no label leaks across folds.

    Gemma 3 has one massive-activation channel per layer that overflows the float16
    capture to inf in some layers; any non-finite column is dropped (at most 1 of 3840).
    """
    X = X[:, np.isfinite(X).all(0)]
    centred = X - X.mean(0)
    U, S, _ = np.linalg.svd(centred, full_matrices=False)
    k = min(n_components, len(X) - 1)
    return U[:, :k] * S[:k]


def fast_balanced_ci(scores: np.ndarray, y: np.ndarray, pair_codes: np.ndarray,
                     draws: np.ndarray) -> tuple[float, float]:
    """95% CI of balanced accuracy over pair resamples `draws` [B, n_pairs] (held-out scores fixed).

    `baselines.bootstrap_ci` recomputes AUROC per resample, which is too slow for
    ~200 probes x 2000 draws; balanced accuracy only needs per-pair confusion counts.
    """
    pred = scores > 0
    n_pairs = draws.shape[1]
    counts = np.stack([np.bincount(pair_codes, weights=w, minlength=n_pairs)
                       for w in (pred & y, ~pred & y, ~pred & ~y, pred & ~y)], axis=1)
    tp, fn, tn, fp = counts[draws].sum(1).T
    bal = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    return float(np.nanpercentile(bal, 2.5)), float(np.nanpercentile(bal, 97.5))


def permute_items_within_arms(y: np.ndarray, items: np.ndarray, truth: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels across items inside each truth arm; an item's rows keep one shared label.

    Both mappings of an item carry its single robust label, so the item is the
    exchangeable unit. Shuffling rows would give each item two independent null
    labels and a null narrower than the real one.
    """
    codes = np.unique(items, return_inverse=True)[1]
    first = np.unique(codes, return_index=True)[1]          # one row per item
    item_y, item_truth = y[first].copy(), truth[first]
    for arm in (True, False):
        m = np.flatnonzero(item_truth == arm)
        item_y[m] = item_y[rng.permutation(m)]
    return item_y[codes]


def balanced_columns(S: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.array([baselines.balanced(S[:, r] > 0, Y[:, r]) for r in range(Y.shape[1])])


def load_capture(npz_path: Path, meta_path: Path):
    meta = json.loads(meta_path.read_text())
    assert list(meta["positions"]) == list(POSITIONS), meta["positions"]
    resid = np.load(npz_path, mmap_mode="r")["resid"]
    rows = meta["rows"]
    assert resid.shape[0] == len(rows), (resid.shape, len(rows))
    return meta, resid, rows


def baseline_detectors(rows: list[dict]) -> dict[str, baselines.FoldFeatures]:
    truth = np.array([r["truth_yes"] for r in rows])
    mapping = np.array([r["format"] == "ab_yesA" for r in rows])
    letter = np.array([r["emitted_letter"] == "A" for r in rows])
    meaning = np.array([r["emitted_yes"] for r in rows])
    singles = np.stack([truth, mapping, letter, meaning], axis=1).astype(float)
    metadata = np.hstack([baselines.one_hot([[r["category"] for r in rows],
                                             [r["stake"] for r in rows]]), singles])
    return {
        "truth": baselines.fixed_features(truth),
        "mapping": baselines.fixed_features(mapping),
        "emitted_letter": baselines.fixed_features(letter),
        "emitted_meaning": baselines.fixed_features(meaning),
        "metadata": baselines.fixed_features(metadata),
        "tfidf": baselines.tfidf_features([r["prompt"] for r in rows]),
    }


def cell_balanced(scores: np.ndarray, y: np.ndarray, truth: np.ndarray,
                  letter_a: np.ndarray) -> dict[str, float]:
    """Balanced accuracy inside each (truth arm x emitted letter) cell."""
    out = {}
    for t in (True, False):
        for a in (True, False):
            m = (truth == t) & (letter_a == a)
            out[f"truth_{'yes' if t else 'no'}_emit_{'A' if a else 'B'}"] = \
                baselines.balanced(scores[m] > 0, y[m])
    return out


def run_slices(resid, d_idx, h_idx, layers, y, y_xor, y_h, items, truth, letter_a, groups, groups_h,
               pair_codes, pair_codes_h, folds, folds_h, draws, draws_h, n_null, rng):
    """Probe, XOR control, H propensity control and null for every (position, layer)."""
    Y = np.column_stack([y] + [permute_items_within_arms(y, items, truth, rng) for _ in range(n_null)])
    Y_xor = y_xor[:, None]
    probes, xors, propensity = {}, {}, {}
    null_max = np.full(n_null, -np.inf)
    t0 = time.time()
    total = len(POSITIONS) * len(layers)
    for p, position in enumerate(POSITIONS):
        for layer in layers:
            X = pca_scores(np.asarray(resid[d_idx, p, layer, :], dtype=np.float32)).astype(np.float32)
            feats = baselines.fixed_features(X)
            S = baselines.cv_scores(feats, Y, groups, folds=folds, balance=True)
            null_max = np.maximum(null_max, balanced_columns(S[:, 1:], Y[:, 1:]))
            s = S[:, 0]
            probes[(position, layer)] = {
                "balanced": baselines.balanced(s > 0, y), "auroc": baselines.auroc(s, y),
                "ci": fast_balanced_ci(s, y, pair_codes, draws),
                "cells": cell_balanced(s, y, truth, letter_a)}
            sx = baselines.cv_scores(feats, Y_xor, groups, folds=folds, balance=True)[:, 0]
            xors[(position, layer)] = {
                "balanced": baselines.balanced(sx > 0, y_xor),
                "ci": fast_balanced_ci(sx, y_xor, pair_codes, draws)}
            Xh = pca_scores(np.asarray(resid[h_idx, p, layer, :], dtype=np.float32)).astype(np.float32)
            sh = baselines.cv_scores(baselines.fixed_features(Xh), y_h, groups_h, folds=folds_h,
                                     balance=True)[:, 0]
            propensity[(position, layer)] = {
                "balanced": baselines.balanced(sh > 0, y_h),
                "ci": fast_balanced_ci(sh, y_h, pair_codes_h, draws_h)}
            done = len(probes)
            if done % 4 == 0 or done == total:
                print(f"  {done}/{total} slices ({time.time() - t0:.0f} s)", flush=True)
    return probes, xors, propensity, null_max


def bar_checks(probe_slice: dict, baseline_ci_high: float, null_p95: float) -> dict[str, bool]:
    return {
        "ci_low_above_baselines": probe_slice["ci"][0] > baseline_ci_high,
        "above_null_p95": probe_slice["balanced"] > null_p95,
        "all_four_cells_above_chance": all(v > 0.5 for v in probe_slice["cells"].values()),
    }


def held_out_category(resid, d_idx, position, layer, y, categories, groups):
    """Train on two categories, test on the third, for each rotation; balanced accuracy of the held-out one."""
    X = pca_scores(np.asarray(resid[d_idx, POSITIONS.index(position), layer, :], dtype=np.float32))
    feats = baselines.fixed_features(X.astype(np.float32))
    out = {}
    for held in HELD_OUT_CATEGORIES:
        test = np.flatnonzero(categories == held)
        train = np.flatnonzero((categories != held) & np.isin(categories, HELD_OUT_CATEGORIES))
        s = baselines.cv_scores(feats, y, groups, folds=[(train, test)], balance=True)[test, 0]
        out[held] = baselines.balanced(s > 0, y[test])
    return out


def plot(layers, probes, xors, propensity, null_p95, baseline_band, n_info):
    fig, axes = pp.plt.subplots(1, 2, figsize=(2 * pp.FIG_W, pp.FIG_H), sharey=True)
    for ax, position in zip(axes, POSITIONS):
        ax.set_title(f"{position} position", loc="left", color=pp.TEXT)
        ax.set_xlabel("layer")
        ax.grid(True, color=pp.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axhspan(*baseline_band, color="#7a7975", alpha=0.18, linewidth=0,
                   label="best baseline 95% CI")
        for colour, label, curve in ((pp.SERIES[0], "lie probe", probes),
                                     (pp.SERIES[1], "XOR control", xors),
                                     (pp.SERIES[2], "H rows, D label (propensity)", propensity)):
            ax.plot(layers, [curve[(position, l)]["balanced"] for l in layers], color=colour,
                    linewidth=2, marker="o", markersize=3, label=label)
            ax.fill_between(layers, [curve[(position, l)]["ci"][0] for l in layers],
                            [curve[(position, l)]["ci"][1] for l in layers],
                            color=colour, alpha=0.16, linewidth=0)
        ax.axhline(null_p95, color=pp.SERIES[0], linestyle=":", linewidth=1.3,
                   label="null p95 (max over layers x positions)")
        ax.axhline(0.5, color=pp.MUTED, linewidth=0.8)
    axes[0].set_ylabel("balanced accuracy")
    axes[0].legend(frameon=False, labelcolor=pp.MUTED, fontsize=9, loc="upper left")
    fig.suptitle(f"Lie vs honest under the same D prompt, 10-fold grouped CV ({n_info})",
                 x=0.06, ha="left", fontsize=15)
    pp.save(fig, "lie_probe_12b")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lens-id", default="gemma-3-12b-it")
    ap.add_argument("--npz", type=Path)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--null-draws", type=int, default=100)
    ap.add_argument("--layer-stride", type=int, default=1, help="use every k-th layer (quick runs)")
    args = ap.parse_args()
    npz = args.npz or RESULTS / f"capture_12b_{args.lens_id}.npz"
    meta_path = args.meta or RESULTS / f"capture_12b_{args.lens_id}.json"

    meta, resid, all_rows = load_capture(npz, meta_path)
    layers = list(range(0, resid.shape[2], args.layer_stride))
    rng = np.random.default_rng(SEED)
    d_idx = np.array([i for i, r in enumerate(all_rows) if r["condition"] == "D"])
    h_idx = np.array([i for i, r in enumerate(all_rows) if r["condition"] == "H"])
    rows = [all_rows[i] for i in d_idx]
    rows_h = [all_rows[i] for i in h_idx]
    y_h = np.array([r["label"] == "lied" for r in rows_h])
    groups_h = np.array([r["pair_id"] for r in rows_h])
    pair_codes_h = np.unique(groups_h, return_inverse=True)[1]
    folds_h = probe.grouped_kfold(groups_h, N_FOLDS, SEED)
    draws_h = rng.integers(pair_codes_h.max() + 1, size=(N_BOOT, pair_codes_h.max() + 1))

    y = np.array([r["label"] == "lied" for r in rows])
    truth = np.array([r["truth_yes"] for r in rows])
    letter_a = np.array([r["emitted_letter"] == "A" for r in rows])
    stake_high = np.array([r["stake"] == "high" for r in rows])
    y_xor = truth ^ stake_high
    categories = np.array([r["category"] for r in rows])
    groups = np.array([r["pair_id"] for r in rows])
    pair_codes = np.unique(groups, return_inverse=True)[1]
    n_pairs = pair_codes.max() + 1
    items = np.array([r["item_id"] for r in rows])
    folds = probe.grouped_kfold(groups, N_FOLDS, SEED)
    draws = rng.integers(n_pairs, size=(N_BOOT, n_pairs))

    counts = {"n": len(y), "pairs": int(n_pairs), "lied": int(y.sum()), "honest": int((~y).sum())}
    print(f"{counts}; {len(layers)} layers x {len(POSITIONS)} positions, {args.null_draws} null draws")

    base = {}
    for name, feats in baseline_detectors(rows).items():
        s = baselines.cv_scores(feats, y, groups, folds=folds, balance=True)[:, 0]
        base[name] = {"balanced": baselines.balanced(s > 0, y), "auroc": baselines.auroc(s, y),
                      "ci": fast_balanced_ci(s, y, pair_codes, draws)}
        print(f"  baseline {name:<16} {base[name]['balanced']:.3f}  CI {base[name]['ci']}")
    baseline_ci_high = max(b["ci"][1] for b in base.values())

    probes, xors, propensity, null_max = run_slices(
        resid, d_idx, h_idx, layers, y, y_xor, y_h, items, truth, letter_a, groups, groups_h,
        pair_codes, pair_codes_h, folds, folds_h, draws, draws_h, args.null_draws, rng)
    null_p95 = float(np.percentile(null_max, 95))

    best = max(probes, key=lambda k: probes[k]["balanced"])
    held_out = held_out_category(resid, d_idx, best[0], best[1], y, categories, groups)
    checks = {k: bar_checks(v, baseline_ci_high, null_p95) for k, v in probes.items()}
    passing = [k for k, c in checks.items() if all(c.values())]

    b = probes[best]
    print(f"\nnull p95 (max over slices): {null_p95:.3f}; highest baseline CI high: {baseline_ci_high:.3f}")
    print(f"best probe: {best[0]} L{best[1]} balanced {b['balanced']:.3f} CI "
          f"[{b['ci'][0]:.3f}, {b['ci'][1]:.3f}] AUROC {b['auroc']:.3f}; "
          f"XOR there {xors[best]['balanced']:.3f}; H-row propensity there {propensity[best]['balanced']:.3f}")
    print(f"cells: { {k: round(v, 3) for k, v in b['cells'].items()} }")
    print(f"held-out category: { {k: round(v, 3) for k, v in held_out.items()} }")
    for name, ok in checks[best].items():
        print(f"  bar {name}: {'PASS' if ok else 'FAIL'}")
    print(f"bar met at best slice: {all(checks[best].values())}; "
          f"slices meeting the full bar: {[f'{p} L{l}' for p, l in passing] or 'none'}")

    key = lambda k: f"{k[0]}_L{k[1]}"
    out = RESULTS / f"lie_probe_12b_{args.lens_id}.json"
    out.write_text(json.dumps({
        "prereg": "misc/prereg_12b_same_token.md", "lens": args.lens_id, "model": meta["model"],
        "n_boot": N_BOOT, "n_null": args.null_draws, "seed": SEED, "counts": counts,
        "null_p95": null_p95, "null_max": null_max.tolist(), "baselines": base,
        "baseline_ci_high": baseline_ci_high,
        "probe": {key(k): v for k, v in probes.items()},
        "xor_control": {key(k): v for k, v in xors.items()},
        "propensity_control_H": {key(k): v for k, v in propensity.items()},
        "best": key(best), "held_out_category": held_out,
        "bar_checks": {key(k): v for k, v in checks.items()},
        "bar_met_slices": [key(k) for k in passing],
    }, indent=1))
    print(f"  -> {out.relative_to(pp.ROOT)}")

    band = (min(b["ci"][0] for b in base.values() if b["ci"][1] == baseline_ci_high),
            baseline_ci_high)
    plot(layers, probes, xors, propensity, null_p95, band,
         f"n={counts['n']}: {counts['lied']} lied / {counts['honest']} honest")


if __name__ == "__main__":
    main()
