"""Publication figures: golden-ratio, large type, and error bars where the data allows.

The notebooks render with plotly, on the box, and the writeup versions of those
figures were screenshots. This is the local, static-export path: read `results/`,
write PNG + SVG into `pubplots/`. No GPU, no notebook, no plotly.

Two of the four figures carry resampled error bars and two carry analytic ones,
and the difference matters when reading them:

  * `belief_H` / `belief_D` / `belief_H_vs_D` average 30 per-item curves, and every item is stored,
    so the band is a percentile bootstrap of the mean over items. At the plateaus
    the per-item values are saturated at 0 or 1 -- the mean of 0.5 is not a layer
    where the model is undecided, it is a layer where half the items have gone one
    way. The band is the only thing on the figure that says so.

  * `classifier_balanced` / `truth_probe_D` were saved as per-layer scalars; the
    per-fold predictions are gone. So their intervals are binomial approximations
    built from the stored rates, and leave-one-pair-out folds share training data,
    which makes those intervals mildly optimistic. `CAPTIONS.md` says so too, in
    the place a reader of the figures will actually be.

Run: `uv run --extra plot python scripts/pubplots.py`
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "pubplots"
LENS_ID = "gemma-3-4b-it"

BOOT_N = 10_000
SEED = 20260905

# Categorical slots, copied from `nandaproj.viz.SERIES` so the static figures and
# the notebook's plotly ones agree on which colour means which series. Assign by
# entity, never by rank.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
TEXT, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"

# The ask: bigger title, bigger axis labels, bigger ticks. Sizes are in points at
# the figure size below, so they hold up when the PNG is dropped into a two-column
# layout at half width.
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12.5,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
    "legend.fontsize": 11.5,
    "axes.titlepad": 12,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})


# Golden-ratio landscape: the plot box is 1/phi as tall as it is wide. A layer
# sweep is a long x-axis with a bounded y, so a square box squeezes the sweep and
# stretches the noise; this is the shape the data is.
PHI = (1 + 5**0.5) / 2
FIG_W, FIG_H = 8.0, 5.4

# Chars per title line at 15pt over the axes width. Titles are wrapped rather
# than shrunk: the ask was for larger type, so the line count gives, not the size.
TITLE_WRAP = 62


def golden_axes(title: str, xaxis: str, yaxis: str):
    """One golden-ratio panel with the recessive chrome the plotly helpers use."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_box_aspect(1 / PHI)  # shape of the *plot box*, not of the whole figure
    ax.set_title(textwrap.fill(title, TITLE_WRAP), loc="left", color=TEXT)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    return fig, ax


def ordered_legend(ax, order: list[str], **kw):
    """Legend with the series in a stated order, addressed by label.

    matplotlib orders handles by artist type -- fills and lines before errorbar
    containers -- so index arithmetic over `get_legend_handles_labels` silently
    reorders itself when a series changes how it is drawn. Naming the labels is
    the only ordering that survives that.
    """
    handles = dict(zip(*reversed(ax.get_legend_handles_labels())))
    missing = [name for name in order if name not in handles]
    assert not missing, f"legend order names series that were never drawn: {missing}"
    return ax.legend([handles[name] for name in order], order, **kw)


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", format=ext, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> pubplots/{name}.png, pubplots/{name}.svg")


# --------------------------------------------------------------------------
# Intervals.
# --------------------------------------------------------------------------

def bootstrap_mean_ci(rows: np.ndarray, rng: np.random.Generator, alpha: float = 0.05):
    """Percentile bootstrap of the column means of `rows` (items x layers).

    Resampling items, not layers: the curve is one draw of 30 items and the
    question is how much the mean curve would move under a different 30. A
    symmetric SEM band is wrong here because the per-item values pile up on 0
    and 1, so it would run outside the probability range at exactly the layers
    the figure is about.
    """
    n = rows.shape[0]
    idx = rng.integers(0, n, size=(BOOT_N, n))
    draws = rows[idx].mean(axis=1)  # (BOOT_N, layers)
    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def wilson_ci(p: np.ndarray, n: int, z: float = 1.96):
    """Wilson score interval -- the right one near 0 or 1, where normal fails."""
    p = np.asarray(p, dtype=float)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def balanced_acc_ci(tpr: np.ndarray, tnr: np.ndarray, n_pos: int, n_neg: int, z: float = 1.96):
    """Balanced accuracy is (TPR + TNR)/2, so its variance is a quarter of the sum.

    The two arms are independent binomials over disjoint rows, which is what lets
    the variances add. Clipped to [0, 1]; the caveat about fold dependence is in
    CAPTIONS.md, not fixable here.
    """
    tpr, tnr = np.asarray(tpr, float), np.asarray(tnr, float)
    se = 0.5 * np.sqrt(tpr * (1 - tpr) / n_pos + tnr * (1 - tnr) / n_neg)
    mid = (tpr + tnr) / 2
    return np.clip(mid - z * se, 0, 1), np.clip(mid + z * se, 0, 1)


# --------------------------------------------------------------------------
# 1 & 2: P(answer) at the slot, per condition, from the per-item store.
# --------------------------------------------------------------------------

CONDITION_NAMES = {"H": "honest", "D": "deceptive", "C1": "persona-truthful",
                   "C2": "instructed-inversion"}


BELIEF_NAMES = {"j_honest": "J-lens a_H", "j_lie": "J-lens a_D",
                "l_honest": "logit lens a_H", "l_lie": "logit lens a_D"}


def _load_belief():
    with np.load(RESULTS / f"belief_readout_{LENS_ID}.npz", allow_pickle=False) as f:
        return ([int(x) for x in f["layers"]], np.asarray(f["conditions"]),
                {k: f[k] for k in BELIEF_NAMES}, f["crossover"])


def draw_belief(ax, layers, curves, mask, crossover, cond_name: str) -> list[str]:
    """One condition's four curves with their bootstrap bands, onto a given axes.

    Shared by the single-condition figures and the side-by-side pair so the two
    cannot drift apart: a band drawn one way in one figure and another way in the
    other is worse than having only one of them.

    The bootstrap generator is seeded from the condition rather than passed in, so
    the H panel of the pair is bit-identical to the standalone `belief_H`. A shared
    stream would advance between the two figures and move the bands by a few
    thousandths -- invisible, and exactly the kind of thing that wastes an hour
    when two versions of a figure are compared later.
    """
    rng = np.random.default_rng(SEED + sum(cond_name.encode()))
    notes = []
    for slot, (key, label) in enumerate(BELIEF_NAMES.items()):
        rows = curves[key][mask]
        lo, hi = bootstrap_mean_ci(rows, rng)
        colour = SERIES[slot]
        ax.fill_between(layers, lo, hi, color=colour, alpha=0.16, linewidth=0)
        ax.plot(layers, rows.mean(axis=0), color=colour, linewidth=2, marker="o",
                markersize=4.5, label=label)
        notes.append(f"widest 95% band {float((hi - lo).max()):.2f} ({label})")

    found = crossover[mask]
    found = found[found >= 0]
    if len(found):
        l_star = float(found.mean())
        ax.axvline(l_star, color=MUTED, linestyle=":", linewidth=1.2)
        # l* lands late in the stack for D, so a label to the right of the
        # line would run into the axes edge: flip it to the left half there.
        right = l_star > (min(layers) + max(layers)) / 2
        ax.annotate(f"mean l* = {l_star:.1f}  (n={len(found)})",
                    xy=(l_star, 1.0), xytext=(-6 if right else 6, -2),
                    textcoords="offset points", va="top",
                    ha="right" if right else "left", fontsize=11, color=MUTED)
        notes.append(f"mean l* {l_star:.2f} over {len(found)}/{int(mask.sum())} items that cross")

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    return notes


def belief_figures() -> list[str]:
    layers, cond, curves, crossover = _load_belief()
    notes = []
    for c in ("H", "D"):
        mask = cond == c
        fig, ax = golden_axes(
            f"{c} ({CONDITION_NAMES[c]}): P(answer) at the slot, {int(mask.sum())} items",
            "layer", "probability",
        )
        notes += [f"{c}: {n}" for n in draw_belief(ax, layers, curves, mask, crossover, c)]
        ax.legend(loc="upper left", frameon=False, labelcolor=MUTED)
        save(fig, f"belief_{c}")
    return notes


def belief_pair_figure() -> None:
    """H and D side by side on a shared y-axis -- the comparison figure.

    The point of the pair is that a_H and a_D swap places between the panels, and
    that is only readable if both panels are on one probability scale with one
    legend: two separately-legended figures make the reader check twice that the
    colours mean the same thing. Each panel keeps the golden-ratio box, so this is
    the two single figures abutted, not a squashed version of them.
    """
    layers, cond, curves, crossover = _load_belief()
    fig, axes = plt.subplots(1, 2, figsize=(2 * FIG_W * 0.82, FIG_H),
                             sharey=True, constrained_layout=True)

    for ax, c in zip(axes, ("H", "D")):
        mask = cond == c
        ax.set_box_aspect(1 / PHI)
        ax.set_title(f"{c} ({CONDITION_NAMES[c]}), {int(mask.sum())} items",
                     loc="left", color=TEXT)
        ax.set_xlabel("layer")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        draw_belief(ax, layers, curves, mask, crossover, c)

    axes[0].set_ylabel("probability")
    fig.suptitle("P(answer) at the answer slot, by condition", x=0.01, ha="left",
                 fontsize=16, color=TEXT)
    # One legend for both panels: the series are identical, and a per-panel legend
    # would also cost the top-left corner that H's curves occupy.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4,
               frameon=False, labelcolor=MUTED)
    save(fig, "belief_H_vs_D")


# --------------------------------------------------------------------------
# 3: the lie-row classifier, balanced accuracy, with its null.
# --------------------------------------------------------------------------

NULL_LABEL = "below null p95 (within-item shuffle)"


def classifier_figure() -> list[str]:
    s = json.loads((RESULTS / f"jtoken_classifier_summary_{LENS_ID}.json").read_text())
    layers = s["layers"]
    n_pos = s["n_lie_rows"]
    n_neg = s["n_rows"] - n_pos

    # Masked and full-vocabulary track each other to within a point over most of
    # the stack, so colour alone cannot separate them: the mask is carried by
    # linestyle as a second encoding.
    configs = [("j_k20_masked", "J-lens, answer tokens masked", "-"),
               ("j_k20_full", "J-lens, full vocabulary", "--"),
               ("l_k20_masked", "logit lens, answer tokens masked", "-"),
               ("l_k20_full", "logit lens, full vocabulary", "--")]

    fig, ax = golden_axes(
        f"Lie-row classifier, leave-one-pair-out, K={s['k_primary']}",
        "layer", "balanced accuracy",
    )

    null = np.array([s["null_p95_j_k20_masked"][str(l)] for l in layers])
    ax.fill_between(layers, 0, null, color=MUTED, alpha=0.12, linewidth=0,
                    label=NULL_LABEL)

    for slot, (key, label, dash) in enumerate(configs):
        cv = s["cv"][key]
        bal = np.asarray(cv["balanced"], float)
        lo, hi = balanced_acc_ci(cv["tpr"], cv["tnr"], n_pos, n_neg)
        # The stored `balanced` must be the mean of the stored arms, or the
        # interval is being drawn around a different number than the curve.
        # Tolerance is 1e-4 because the summary rounds every rate to 4 dp, so
        # averaging two of them can miss by half a unit in the last place.
        assert np.allclose(bal, (np.asarray(cv["tpr"]) + np.asarray(cv["tnr"])) / 2, atol=1e-4), key
        colour = SERIES[slot]
        ax.fill_between(layers, lo, hi, color=colour, alpha=0.16, linewidth=0)
        ax.plot(layers, bal, color=colour, linewidth=2, linestyle=dash,
                marker="o", markersize=4.5, label=label)

    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    # Four long labels will not fit inside the panel without covering the null
    # band, so the legend goes underneath. The curves come before the null they
    # are being compared against.
    ordered_legend(ax, [label for _, label, _ in configs] + [NULL_LABEL],
                   loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2,
                   frameon=False, labelcolor=MUTED, handlelength=2.4, columnspacing=1.4)
    save(fig, "classifier_balanced")

    bal = np.asarray(s["cv"]["j_k20_masked"]["balanced"], float)
    lo, _ = balanced_acc_ci(s["cv"]["j_k20_masked"]["tpr"], s["cv"]["j_k20_masked"]["tnr"],
                            n_pos, n_neg)
    clears = [l for l, b, lo_ in zip(layers, bal, lo) if lo_ > s["null_p95_j_k20_masked"][str(l)]]
    return [f"classifier: {n_pos} lie rows vs {n_neg} others",
            (f"classifier: masked J-lens lower CI above null p95 at "
             f"{len(clears)}/{len(layers)} layers: {clears}")]


# --------------------------------------------------------------------------
# 4: the truth probe under D, logistic fits against the null.
# --------------------------------------------------------------------------

def truth_probe_figure() -> list[str]:
    s = json.loads((RESULTS / f"truth_probe_summary_{LENS_ID}.json").read_text())
    layers = s["layers"]
    n = s["n_items"]  # leave-one-pair-out: every item held out exactly once

    series = [("D truth (logistic)", s["cv_truth_logistic"]["D"], True),
              ("H truth (logistic)", s["cv_truth_logistic"]["H"], True),
              ("D null p95 (pair-flipped)", s["null_p95_mean_diff"]["D"], False)]

    fig, ax = golden_axes(
        # A noun phrase, not a finding. The previous title claimed "the two fits
        # disagree", but the mean-difference fit it referred to is not on this
        # panel -- only the logistic ones are -- so the figure could not support
        # its own title. Claims belong in the caption, where they can be qualified.
        "Layer-wise truth-probe accuracy under honest and deceptive prompting",
        "layer", "accuracy",
    )
    for slot, (label, ys, with_ci) in enumerate(series):
        ys = np.asarray(ys, float)
        colour = SERIES[slot]
        if with_ci:
            lo, hi = wilson_ci(ys, n)
            ax.fill_between(layers, lo, hi, color=colour, alpha=0.16, linewidth=0)
            ax.plot(layers, ys, color=colour, linewidth=2, marker="o",
                    markersize=4.5, label=label)
        else:
            # The null is a percentile of a permutation distribution, not an
            # estimate with sampling error of its own -- no band belongs on it,
            # and the dashed line says it is a threshold rather than a measurement.
            ax.plot(layers, ys, color=colour, linewidth=2, linestyle="--",
                    marker="o", markersize=4, label=label)

    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    # Same as the classifier: the two fits, then the null.
    ordered_legend(ax, [label for label, _, _ in series],
                   loc="lower right", frameon=False, labelcolor=MUTED)
    save(fig, "truth_probe_D")

    d = np.asarray(s["cv_truth_logistic"]["D"], float)
    lo, _ = wilson_ci(d, n)
    null = np.asarray(s["null_p95_mean_diff"]["D"], float)
    clears = [l for l, lo_, nu in zip(layers, lo, null) if lo_ > nu]
    return [(f"truth probe: n={n} held-out items ({s['n_pairs']} pairs), "
             f"{s['n_lying_under_D']} lying under D"),
            (f"truth probe: D logistic lower CI above null p95 at "
             f"{len(clears)}/{len(layers)} layers, first at L{clears[0] if clears else '-'}")]


CAPTIONS = """# Figure captions and error-bar provenance

Generated by `scripts/pubplots.py` from `results/`. Do not edit by hand.

## belief_H, belief_D, belief_H_vs_D -- P(answer) at the answer slot

Mean over {n_items} items of the per-item probability the lens assigns to the honest
answer (a_H) and the deceptive answer (a_D), at the answer slot, per layer. Bands are
95% percentile bootstrap intervals of the mean over items ({boot} resamples, seed
{seed}), computed from the per-item curves in
`results/belief_readout_{lens}.npz`.

Read the bands, not just the means. The per-item values are saturated near 0 and 1,
so a mean of 0.5 marks a layer where the items have split, not a layer where the
model is undecided. The dotted line in `belief_D` is the mean crossover layer over
only those items that cross; the count is on the figure.

`belief_H_vs_D` is the same two panels side by side on a shared probability axis
with one legend -- the comparison figure, since the point is that a_H and a_D swap
places between the conditions. The panels are bit-identical to the standalone
figures: the bootstrap is seeded per condition, not from a shared stream.

## classifier_balanced -- lie-row classifier

Balanced accuracy of a leave-one-pair-out logistic fit on top-K lens tokens, over
{n_pos} lie rows and {n_neg} others. The shaded region is at or below the p95 of the
within-item label shuffle for the primary (masked J-lens) configuration.

**Error bars are analytic, not resampled.** Only per-layer rates were saved, so the
interval is +/-1.96 SE with SE = 0.5*sqrt(TPR(1-TPR)/{n_pos} + TNR(1-TNR)/{n_neg}),
treating the two arms as independent binomials. Leave-one-pair-out folds share
training data, so these are approximate and mildly optimistic. Recovering honest
intervals means re-running 08 with per-fold predictions saved.

## truth_probe_D -- Test 1 under D

Leave-one-pair-out accuracy of the logistic truth probe, D and H, against the
pair-flipped null p95 under D. n = {n_items_probe} held-out items ({n_pairs} pairs).

**Error bars are analytic, not resampled**, for the same reason: Wilson 95% intervals
on n = {n_items_probe}, with the same fold-dependence caveat. The null carries no band
-- it is a percentile of a permutation distribution, not an estimate with sampling
error of its own.

Two provenance points before this figure is used to make a claim. First, the plotted
null is `null_p95_mean_diff`, computed for the *mean-difference* probe: it is the only
null in the summary, and there is no logistic null to compare the plotted logistic
curves against. It is a reasonable reference but not the matched one. Second, the
finding this figure used to assert in its title -- that the mean-difference and
logistic fits diverge under D -- cannot be read off it, because no mean-difference
curve is plotted. Both need `cv_truth_mean_diff` on the panel, or a re-run of 04d that
saves a logistic null.
"""


def main() -> None:
    notes: list[str] = []

    print("belief readout (per-item bootstrap)")
    notes += belief_figures()
    belief_pair_figure()
    print("lie-row classifier (analytic binomial)")
    notes += classifier_figure()
    print("truth probe (Wilson)")
    notes += truth_probe_figure()

    js = json.loads((RESULTS / f"jtoken_classifier_summary_{LENS_ID}.json").read_text())
    tp = json.loads((RESULTS / f"truth_probe_summary_{LENS_ID}.json").read_text())
    with np.load(RESULTS / f"belief_readout_{LENS_ID}.npz", allow_pickle=False) as f:
        n_items = int((np.asarray(f["conditions"]) == "H").sum())
    (OUT / "CAPTIONS.md").write_text(CAPTIONS.format(
        n_items=n_items, boot=f"{BOOT_N:,}", seed=SEED, lens=LENS_ID,
        n_pos=js["n_lie_rows"], n_neg=js["n_rows"] - js["n_lie_rows"],
        n_items_probe=tp["n_items"], n_pairs=tp["n_pairs"],
    ))
    print("  -> pubplots/CAPTIONS.md")

    print("\nnumbers worth checking against the writeup:")
    for line in notes:
        print(f"  {line}")


if __name__ == "__main__":
    main()
