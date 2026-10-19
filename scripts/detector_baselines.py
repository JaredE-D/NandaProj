"""Direction 1: masked J-lens lie classifier vs simple baselines, D rows only.

Pre-registration: `misc/prereg_detector_baselines.md` (P1-P4). Everything is read
from cached plan-2 artifacts, so this runs on the laptop in a few minutes:

  results/slot_emitted_<lens>.json     labels (H and D answer probabilities)
  results/slot_residuals_<lens>.npz    answer-slot residual, every block
  results/jtoken_topk_<lens>.npz       top-200 J-lens / logit-lens tokens per row
  results/answer_token_ids_<lens>.json the yes/no mask (from an ungated tokenizer mirror)

Writes results/detector_baselines_<lens>.json and two figures in pubplots/.

Run: `uv run --extra plot python scripts/detector_baselines.py`
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pubplots as pp
from nandaproj import baselines, items, tokenfeat

RESULTS = pp.RESULTS
LENS_ID = pp.LENS_ID
K_TOP = 20
N_BOOT = 2000
N_NULL = 200
SEED = 20261018

LAYER_DETECTORS = {"jlens": "J-lens tokens (masked)", "logit": "logit-lens tokens (masked)",
                   "residual": "residual probe"}
FLAT_DETECTORS = {"tfidf": "prompt TF-IDF (no model)", "metadata": "item metadata",
                  "h_confidence": "confidence under H"}
# Added after the first run, not pre-registered: the honest class is 22 Yes-true vs 3
# No-true, so polarity alone may carry most of what any detector scores.
EXPLORATORY = {"polarity": "polarity only (exploratory)"}


def load_population(drop_world_fact: bool):
    """D rows of v2 belief items answered correctly under H; label = lied under D."""
    bank = [i for i in items.load_bank() if not i.is_no_belief]
    emit = json.loads((RESULTS / f"slot_emitted_{LENS_ID}.json").read_text())
    truth = {i.item_id: i.meta["polarity"] == "Yes" for i in bank}
    said_yes = lambda c, i: emit[c][i]["p_yes"] > emit[c][i]["p_no"]

    keep = [i for i in bank if said_yes("H", i.item_id) == truth[i.item_id]
            and not (drop_world_fact and i.tier == "world_fact")]
    ids = [i.item_id for i in keep]
    y = np.array([said_yes("D", i) != truth[i] for i in ids])
    arm = np.array([truth[i] for i in ids])
    p_true_H = np.array([emit["H"][i]["p_yes"] if truth[i] else emit["H"][i]["p_no"] for i in ids])
    return keep, ids, y, arm, p_true_H


def build_detectors(keep, ids, p_true_H):
    """`{name: {layer: FoldFeatures}}`; flat detectors use the single key None."""
    res = np.load(RESULTS / f"slot_residuals_{LENS_ID}.npz")
    res_row = {str(i): r for r, i in enumerate(res["item_ids"])}
    h_D = res["h_D"][:, [res_row[i] for i in ids]].astype(np.float32)      # [L, n, d]

    store = tokenfeat.TopK.load(RESULTS / f"jtoken_topk_{LENS_ID}.npz")
    d_row = {store.item_ids[r]: r for r in store.rows("D")}
    rows = np.array([d_row[i] for i in ids])
    banned = json.loads((RESULTS / f"answer_token_ids_{LENS_ID}.json").read_text())["ids"]

    det = {
        "jlens": {l: baselines.topk_features(*(a[rows] for a in store.at("j", l)), K_TOP, banned)
                  for l in store.layers},
        "logit": {l: baselines.topk_features(*(a[rows] for a in store.at("l", l)), K_TOP, banned)
                  for l in store.layers},
        "residual": {int(l): baselines.fixed_features(h_D[li]) for li, l in enumerate(res["layers"])},
    }
    texts = [f"{it.prompt('D').system or ''}\n{it.prompt('D').user}" for it in keep]
    det["tfidf"] = {None: baselines.tfidf_features(texts)}
    det["metadata"] = {None: baselines.fixed_features(baselines.one_hot(
        [[it.meta["polarity"] for it in keep], [it.tier for it in keep], [it.harm for it in keep]]))}
    det["polarity"] = {None: baselines.fixed_features(
        baselines.one_hot([[it.meta["polarity"] for it in keep]]))}
    clipped = np.clip(p_true_H, 1e-6, 1 - 1e-6)
    det["h_confidence"] = {None: baselines.fixed_features(np.log(clipped / (1 - clipped)))}
    return det, store.layers


def evaluate(det, y, arm, groups, rng):
    out = {}
    t0 = time.time()
    for name, by_layer in det.items():
        out[name] = {}
        for layer, feats in by_layer.items():
            scores = baselines.cv_scores(feats, y, groups)[:, 0]
            out[name][str(layer)] = {
                "point": baselines.summarize(scores, y, arm),
                "ci": baselines.bootstrap_ci(scores, y, arm, groups, N_BOOT, rng),
                "null_p95": baselines.null_p95(feats, y, arm, groups, N_NULL, rng),
            }
        print(f"  {name:<13} {len(by_layer):>2} layer(s) done  ({time.time() - t0:.0f} s elapsed)",
              flush=True)
    return out


def best_layer(curve: dict, metric: str = "balanced") -> str:
    return max(curve, key=lambda l: curve[l]["point"][metric])


def verdicts(r: dict, lens_layers: list[int]) -> dict[str, str]:
    """P1-P4 and the decision rule, mechanically, on the primary population."""
    J, L, R = r["jlens"], r["logit"], r["residual"]
    out = {}

    above = [l for l in lens_layers if J[str(l)]["point"]["balanced"] > R[str(l)]["point"]["balanced"]]
    out["P1"] = (f"{'SUPPORTED' if not above else 'NOT SUPPORTED'}: J-lens > residual probe "
                 f"(balanced, point) at {len(above)}/{len(lens_layers)} layers {above}")

    lj, ll = best_layer(J), best_layer(L)
    cj, cl = J[lj]["ci"]["balanced"], L[ll]["ci"]["balanced"]
    overlap = cj[0] <= cl[1] and cl[0] <= cj[1]
    out["P2"] = (f"{'SUPPORTED' if overlap else 'NOT SUPPORTED'}: best J L{lj} "
                 f"{J[lj]['point']['balanced']:.3f} [{cj[0]:.3f}, {cj[1]:.3f}] vs best logit L{ll} "
                 f"{L[ll]['point']['balanced']:.3f} [{cl[0]:.3f}, {cl[1]:.3f}]")

    m = r["metadata"]["None"]
    out["P3"] = (f"{'SUPPORTED' if m['ci']['balanced'][0] > 0.5 else 'NOT SUPPORTED'}: metadata "
                 f"balanced {m['point']['balanced']:.3f} CI {m['ci']['balanced']}, "
                 f"arm_mean {m['point']['arm_mean']:.3f}")

    parts = []
    for a in ("balanced_yes_true", "balanced_no_true"):
        incl = sum(J[str(l)]["ci"][a][0] <= 0.5 <= J[str(l)]["ci"][a][1] for l in lens_layers)
        parts.append((a, incl))
    ok = all(incl > len(lens_layers) / 2 for _, incl in parts)
    out["P4"] = (f"{'SUPPORTED' if ok else 'NOT SUPPORTED'}: J CI includes 0.5 at "
                 + ", ".join(f"{a} {incl}/{len(lens_layers)} layers" for a, incl in parts))

    flat_hi = max(r[f]["None"]["ci"]["balanced"][1] for f in FLAT_DETECTORS)
    beats = cj[0] > flat_hi
    out["decision"] = (f"J-lens best-layer CI low {cj[0]:.3f} vs highest baseline-4-6 CI high "
                       f"{flat_hi:.3f}: {'separated -- carry to full bank' if beats else 'not separated'}")

    p = r["polarity"]["None"]
    out["exploratory_polarity"] = (f"polarity only: balanced {p['point']['balanced']:.3f} "
                                   f"CI [{p['ci']['balanced'][0]:.3f}, {p['ci']['balanced'][1]:.3f}]")
    return out


def plot(r: dict, n_info: str, metric: str, name: str, ylabel: str) -> None:
    title = (f"Lie vs honest under the same D prompt, leave-one-pair-out ({n_info}); "
             f"bands = 95% pair-bootstrap CI")
    fig, ax = pp.golden_axes(title, "layer", ylabel)
    flat_colours = ("#7a7975", "#a8a7a2", "#c9c8c3", "#d4380d")
    flat = {**FLAT_DETECTORS, **EXPLORATORY}
    for colour, f in zip(flat_colours, flat):
        pt, (lo, hi) = r[f]["None"]["point"][metric], r[f]["None"]["ci"][metric]
        ax.axhspan(lo, hi, color=colour, alpha=0.18, linewidth=0)
        ax.axhline(pt, color=colour, linewidth=1.6, linestyle="--", label=flat[f])
    for colour, (d, label) in zip(pp.SERIES, LAYER_DETECTORS.items()):
        layers = sorted(int(l) for l in r[d])
        pt = [r[d][str(l)]["point"][metric] for l in layers]
        lo = [r[d][str(l)]["ci"][metric][0] for l in layers]
        hi = [r[d][str(l)]["ci"][metric][1] for l in layers]
        ax.fill_between(layers, lo, hi, color=colour, alpha=0.16, linewidth=0)
        ax.plot(layers, pt, color=colour, linewidth=2, marker="o", markersize=3, label=label)
    J = r["jlens"]
    jl = sorted(int(l) for l in J)
    ax.plot(jl, [J[str(l)]["null_p95"][metric] for l in jl], color=pp.SERIES[0],
            linestyle=":", linewidth=1.3, label="J-lens null p95 (within-arm shuffle)")
    ax.axhline(0.5, color=pp.MUTED, linewidth=0.8)
    ax.set_ylim(0.2, 1.0)
    order = list(LAYER_DETECTORS.values()) + ["J-lens null p95 (within-arm shuffle)"] \
        + list(flat.values())
    pp.ordered_legend(ax, order, loc="lower right", frameon=False, labelcolor=pp.MUTED, fontsize=9)
    pp.save(fig, name)


def main() -> None:
    rng = np.random.default_rng(SEED)
    summary = {"prereg": "misc/prereg_detector_baselines.md", "lens": LENS_ID, "k_top": K_TOP,
               "n_boot": N_BOOT, "n_null": N_NULL, "seed": SEED, "populations": {}}
    for pop, drop_wf in (("minus_world_fact", True), ("all_tiers", False)):
        keep, ids, y, arm, p_true_H = load_population(drop_wf)
        groups = [it.meta["pair_id"] for it in keep]
        counts = {"n": len(y), "lied": int(y.sum()), "honest": int((~y).sum()),
                  "lied_yes_true": int((y & arm).sum()), "lied_no_true": int((y & ~arm).sum()),
                  "honest_yes_true": int((~y & arm).sum()), "honest_no_true": int((~y & ~arm).sum()),
                  "pairs": len(set(groups))}
        print(f"\n[{pop}] {counts}", flush=True)
        det, lens_layers = build_detectors(keep, ids, p_true_H)
        r = evaluate(det, y, arm, groups, rng)
        summary["populations"][pop] = {"counts": counts, "results": r,
                                       "verdicts": verdicts(r, lens_layers)}
        for k, v in summary["populations"][pop]["verdicts"].items():
            print(f"  {k}: {v}")
        if pop == "minus_world_fact":
            n_info = f"n={counts['n']}: {counts['lied']} lied / {counts['honest']} honest"
            plot(r, n_info, "balanced", "detector_baselines_balanced", "balanced accuracy")
            plot(r, n_info, "arm_mean", "detector_baselines_arm_mean",
                 "balanced accuracy, mean of the two polarity arms")
        # Written after every population: a crash in the second must not lose the first.
        out = RESULTS / f"detector_baselines_{LENS_ID}.json"
        out.write_text(json.dumps(summary, indent=1))
        print(f"  -> {out.relative_to(pp.ROOT)}", flush=True)


if __name__ == "__main__":
    main()
