"""J-lens on the clean contrast: does the masked J-lens detect lies where the probe does?

Addendum to `misc/prereg_12b_same_token.md` (see its "J-lens addendum"). Rows: every
lied item plus a seeded sample of twice as many honest items, D condition, both A/B
mappings. On exactly those rows the masked J-lens top-K, the masked logit lens and
the residual probe go through the same CV, baselines and null as `lie_probe_12b.py`.

  run     on box:  just ssh "bash -lc 'cd /workspace/NandaProj && /venv/main/bin/python scripts/jlens_12b.py run'"
  report  locally: uv run --extra plot python scripts/jlens_12b.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nandaproj import config, pilot

LENS = "gemma-3-12b-it"
CAPTURE = config.RESULTS / f"capture_12b_{LENS}"
TOPK = config.RESULTS / f"jlens_topk_12b_{LENS}.npz"
OUT = config.RESULTS / f"jlens_12b_{LENS}.json"
SEED = 20261018
K_STORE = 50
K_TOP = 20                       # plan 2's primary K
POST_HOC_PCA = 16                # the registered PCA-256 probe overfits ~380 training rows per fold
MASK_WORDS = ("yes", "no", "a", "b")
LAYER_STRIDE = 2


def sampled_rows(rows: list[dict]) -> np.ndarray:
    """Capture-row indices: D rows of every lied item and of 2x as many honest items (seeded)."""
    d = [i for i, r in enumerate(rows) if r["condition"] == "D"]
    lied = sorted({rows[i]["item_id"] for i in d if rows[i]["label"] == "lied"})
    honest = sorted({rows[i]["item_id"] for i in d if rows[i]["label"] == "honest"})
    keep = set(lied) | set(np.random.default_rng(SEED).choice(honest, 2 * len(lied), replace=False))
    return np.array([i for i in d if rows[i]["item_id"] in keep])


# --------------------------------------------------------------------------
# run (box)
# --------------------------------------------------------------------------

def run(_args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nandaproj import items as items_mod, tokenfeat
    from nandaproj.lens_readout import Reader, _progress

    rows = json.loads(CAPTURE.with_suffix(".json").read_text())["rows"]
    idx = sampled_rows(rows)
    cfg = config.get_model_config("escalate")
    tok = AutoTokenizer.from_pretrained(cfg.name, cache_dir=str(config.HF_CACHE))
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name, cache_dir=str(config.HF_CACHE), dtype=torch.bfloat16, device_map="auto").eval()
    reader = Reader.load(model, tok, cfg)
    reader.layers = reader.layers[::LAYER_STRIDE]
    print(reader.describe())
    a_id, b_id = items_mod.token_id(tok, " A"), items_mod.token_id(tok, " B")

    # lens.apply tokenizes the string itself, which is where plan 2's double <bos>
    # came from. Pick the variant whose lens-side final distribution reproduces the
    # captured p(A), p(B) on the first row, and refuse to run if neither does.
    first = rows[idx[0]]
    with_bos = first["prompt"]
    without_bos = with_bos.removeprefix(tok.bos_token)
    strip = None
    for candidate, flag in ((without_bos, True), (with_bos, False)):
        _, _, final = reader.readout(candidate, layers=reader.layers[:1])
        if abs(final[a_id] - first["p_a"]) < 0.02 and abs(final[b_id] - first["p_b"]) < 0.02:
            strip = flag
            break
    assert strip is not None, "neither prompt variant reproduces the captured slot distribution"
    print(f"lens prompt: {'<bos> stripped' if strip else 'as rendered'}")

    banned = tokenfeat.answer_token_ids(tok, MASK_WORDS)
    J_ids, J_lp, L_ids, L_lp, worst = [], [], [], [], 0.0
    for i in _progress(idx, total=len(idx), desc="J-lens readout"):
        r = rows[i]
        prompt = r["prompt"].removeprefix(tok.bos_token) if strip else r["prompt"]
        j_probs, l_probs, final = reader.readout(prompt)
        worst = max(worst, abs(final[a_id] - r["p_a"]), abs(final[b_id] - r["p_b"]))
        ji, jl = zip(*(tokenfeat.top_k_of(j_probs[l], K_STORE) for l in reader.layers))
        li, ll = zip(*(tokenfeat.top_k_of(l_probs[l], K_STORE) for l in reader.layers))
        J_ids.append(np.stack(ji)); J_lp.append(np.stack(jl))
        L_ids.append(np.stack(li)); L_lp.append(np.stack(ll))
    print(f"max |p_lens - p_capture| over A/B: {worst:.4f} (expect < 0.02)")
    np.savez_compressed(TOPK, rows=idx, layers=np.array(reader.layers), banned=np.array(banned),
                        j_ids=np.stack(J_ids), j_lp=np.stack(J_lp),
                        l_ids=np.stack(L_ids), l_lp=np.stack(L_lp), max_p_diff=worst)
    print(f"-> {TOPK}")


# --------------------------------------------------------------------------
# report (local)
# --------------------------------------------------------------------------

def report(args) -> None:
    import lie_probe_12b as lp
    from nandaproj import baselines, probe

    all_rows = json.loads(CAPTURE.with_suffix(".json").read_text())["rows"]
    T = np.load(TOPK)
    idx, layers, banned = T["rows"], [int(l) for l in T["layers"]], T["banned"]
    rows = [all_rows[i] for i in idx]
    print(f"{len(rows)} rows; lens/capture max p diff {float(T['max_p_diff']):.4f}; "
          f"{len(banned)} masked ids; layers {layers}")

    y = np.array([r["label"] == "lied" for r in rows])
    truth = np.array([r["truth_yes"] for r in rows])
    letter_a = np.array([r["emitted_letter"] == "A" for r in rows])
    items = np.array([r["item_id"] for r in rows])
    groups = np.array([r["pair_id"] for r in rows])
    pair_codes = np.unique(groups, return_inverse=True)[1]
    rng = np.random.default_rng(SEED)
    folds = probe.grouped_kfold(groups, lp.N_FOLDS, SEED)
    draws = rng.integers(pair_codes.max() + 1, size=(lp.N_BOOT, pair_codes.max() + 1))
    Y = np.column_stack([y] + [lp.permute_items_within_arms(y, items, truth, rng)
                               for _ in range(args.null_draws)])

    def score(feats) -> tuple[dict, np.ndarray]:
        S = baselines.cv_scores(feats, Y, groups, folds=folds, balance=True)
        s = S[:, 0]
        return ({"balanced": baselines.balanced(s > 0, y), "auroc": baselines.auroc(s, y),
                 "ci": lp.fast_balanced_ci(s, y, pair_codes, draws),
                 "cells": lp.cell_balanced(s, y, truth, letter_a)},
                lp.balanced_columns(S[:, 1:], Y[:, 1:]))

    base = {}
    for name, feats in lp.baseline_detectors(rows).items():
        s = baselines.cv_scores(feats, y, groups, folds=folds, balance=True)[:, 0]
        base[name] = {"balanced": baselines.balanced(s > 0, y),
                      "ci": lp.fast_balanced_ci(s, y, pair_codes, draws)}
    baseline_ci_high = max(b["ci"][1] for b in base.values())

    resid = np.load(CAPTURE.with_suffix(".npz"))["resid"]
    curves, null_max, t0 = {"jlens": {}, "logit": {}, "probe": {}, "probe_pca16": {}}, {}, time.time()
    for li, layer in enumerate(layers):
        X = np.asarray(resid[idx, 1, layer, :], dtype=np.float32)
        feats = {
            "jlens": baselines.topk_features(T["j_ids"][:, li], T["j_lp"][:, li], K_TOP, banned),
            "logit": baselines.topk_features(T["l_ids"][:, li], T["l_lp"][:, li], K_TOP, banned),
            "probe": baselines.fixed_features(lp.pca_scores(X).astype(np.float32)),
            # Post hoc, not registered: the same probe sized for n.
            "probe_pca16": baselines.fixed_features(lp.pca_scores(X, POST_HOC_PCA).astype(np.float32)),
        }
        for name, f in feats.items():
            curves[name][layer], nulls = score(f)
            null_max[name] = np.maximum(null_max.get(name, -np.inf), nulls)
        print(f"  L{layer}: J {curves['jlens'][layer]['balanced']:.3f}  logit "
              f"{curves['logit'][layer]['balanced']:.3f}  probe {curves['probe'][layer]['balanced']:.3f}"
              f"  ({time.time() - t0:.0f} s)", flush=True)

    summary = {}
    for name in curves:
        p95 = float(np.percentile(null_max[name], 95))
        best = max(curves[name], key=lambda l: curves[name][l]["balanced"])
        c = curves[name][best]
        checks = lp.bar_checks(c, baseline_ci_high, p95)
        summary[name] = {"best_layer": best, "null_p95": p95, **c, "bar": checks,
                         "layers_meeting_bar": [l for l in layers if all(
                             lp.bar_checks(curves[name][l], baseline_ci_high, p95).values())]}
        print(f"{name:<6} best L{best}: {c['balanced']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}] "
              f"null p95 {p95:.3f}; bar {'MET' if all(checks.values()) else 'not met'} "
              f"{ {k: v for k, v in checks.items()} }")
    print(f"highest baseline CI high on these rows: {baseline_ci_high:.3f}")

    OUT.write_text(json.dumps({
        "prereg": "misc/prereg_12b_same_token.md (J-lens addendum)", "k_top": K_TOP,
        "n_null": args.null_draws, "n_rows": len(rows), "baselines": base,
        "baseline_ci_high": baseline_ci_high, "summary": summary,
        "curves": {n: {str(l): v for l, v in c.items()} for n, c in curves.items()}}, indent=1))
    print(f"-> {OUT}")
    plot(layers, curves, summary, base, baseline_ci_high)


def plot(layers, curves, summary, base, baseline_ci_high) -> None:
    import pubplots as pp

    fig, ax = pp.plt.subplots(figsize=(pp.FIG_W, pp.FIG_H))
    ax.grid(True, color=pp.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    lo = min(b["ci"][0] for b in base.values() if b["ci"][1] == baseline_ci_high)
    ax.axhspan(lo, baseline_ci_high, color="#7a7975", alpha=0.18, linewidth=0, label="best baseline 95% CI")
    for colour, name, label in ((pp.SERIES[0], "probe", "residual probe, PCA-256 (registered)"),
                                (pp.SERIES[1], "jlens", "masked J-lens top-20"),
                                (pp.SERIES[2], "logit", "masked logit lens top-20"),
                                (pp.SERIES[0], "probe_pca16", "residual probe, PCA-16 (post hoc)")):
        c = curves[name]
        ax.plot(layers, [c[l]["balanced"] for l in layers], color=colour, linewidth=2,
                marker="o", markersize=3, label=label,
                linestyle="--" if name == "probe_pca16" else "-")
        ax.fill_between(layers, [c[l]["ci"][0] for l in layers], [c[l]["ci"][1] for l in layers],
                        color=colour, alpha=0.16, linewidth=0)
        ax.axhline(summary[name]["null_p95"], color=colour, linestyle=":", linewidth=1.1)
    ax.axhline(0.5, color=pp.MUTED, linewidth=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("balanced accuracy")
    ax.legend(frameon=False, labelcolor=pp.MUTED, fontsize=9, loc="upper left")
    fig.suptitle("Lie vs honest at the answer slot, same rows, same CV (dotted: null p95)",
                 x=0.06, ha="left", fontsize=13)
    pp.save(fig, "jlens_12b")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report"))
    ap.add_argument("--null-draws", type=int, default=100)
    args = ap.parse_args()
    {"run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
