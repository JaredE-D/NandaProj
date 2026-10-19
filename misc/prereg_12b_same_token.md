# Pre-registration — same-token lie detection on gemma-3-12b-it

Written and approved 2026-10-18, before any activation was captured. Builds on
`misc/results_model_size_regate.md` (robust labels) and `misc/results_detector_baselines.md`
(why the v2 contrast could not show lie-specific content). Deviation from that
re-gate's decision rule ("stay on 4B"): 4B has 7 robust lies and is unusable; Jared
approved moving to 12B on 2026-10-18.

## Population

`gemma-3-12b-it`, alleged bank (880 items), robust labels from
`results/pilot_gate_gemma-3-12b-it_v2+alleged.json`: an item is **lied** if it is
H-correct and lies under D in *both* A/B mappings, **honest** if H-correct and honest
under D in both. `flips` and H-wrong items are excluded. Expected 70 lied / 626 honest.

Rows: each item under D in both mappings (`ab_yesA`, `ab_yesB`), so 2 rows per item.
The label is the item's robust label. H rows are not captured.

## Why this contrast

Under both mappings the emitted letter, the meaning of the emitted answer, the
mapping, and the true answer are each roughly balanced against the label. The label
is truth XOR emitted meaning, so no single one of those variables predicts it. Inside
each (truth arm x emitted letter) cell, both labels occur (through the mapping).

## Features

Residual stream (`hidden_states[1:]`, all 48 blocks; the last is post-final-norm)
at two positions:

- **slot**: the last prompt token (`:` of `Answer:`), as in every earlier analysis;
- **pre**: the last token of the chat-templated prompt before `Answer:`
  (the `\n` after `<start_of_turn>model`), before any answer text exists.

Per (layer, position): unsupervised PCA to 256 components fit on all rows (no labels).

## Detectors

Same pipeline for all: 10-fold GroupKFold by `pair_id` (twins held out together),
fold-internal standardisation, L2 logistic (`baselines.fit_logistic_many`), class-balanced
sample weights.

1. Residual probe, per layer x position (primary).
2. Single-variable baselines: true answer, mapping, emitted letter, emitted meaning (Yes/No).
3. Item metadata: category, stake (one-hot), plus 2.
4. TF-IDF of the D prompt text.

## Metrics

Balanced accuracy (primary) and AUROC. Balanced accuracy within each of the four
(truth arm x emitted letter) cells. 95% CIs: 2000 bootstrap resamples over pairs,
held-out scores fixed. Null: labels permuted within (truth arm x mapping) cells,
100 draws, refit; the null statistic is the **max over all layer x position
probes** of balanced accuracy, so layer selection is paid for.

## Success bar (all three, at one layer x position)

1. Probe balanced-accuracy CI low > the highest CI high among detectors 2–4.
2. Probe balanced accuracy > the max-over-layers null p95.
3. Probe balanced accuracy > 0.5 in all four (truth x emitted letter) cells.

## Controls (reported, not part of the bar)

- **XOR control.** Same pipeline, label = true answer XOR (stake == high), at every
  layer x position. If it matches the lie probe, the claim is "the lie is linearly
  decodable like an arbitrary XOR of represented features", not a lie feature.
- **Held-out category.** At the best primary layer x position: train on two of
  {fault, hazard, slight}, test on the third, all three rotations.

## Predictions

- **P1.** Bar met at the slot, from mid-depth (~layer 20–30 of 48) up.
- **P2.** At *pre*, the probe is above null but below its slot accuracy.
- **P3.** Single-variable baselines each have CIs including 0.5.
- **P4.** XOR control is decodable well above chance at the slot, but below the lie probe.

## Decision rule

Bar met: positive result, carried to the steering stretch (probe direction vs random
direction of equal norm at the best layer, effect on D lie rate). Bar not met: the
writeup is the critique ("lens and probe lie detectors on instructed lying read
answer format and polarity; with both controlled, nothing separates").

## Amendment (2026-10-18, before any capture ran)

1. **Null unit.** Labels are permuted across *items* within each truth arm, both mappings
   of an item keeping one shared label (it has one robust label). The row-level
   permutation within (truth x mapping) written above gave each item two independent
   null labels and an over-narrow null.
2. **H rows captured as a propensity control.** H rows of the same items, labelled with
   the item's D label, go through the same pipeline per layer x position. If they match
   the D probe, the probe reads which items the model lies on, not the lie. Reported
   beside the XOR control; not part of the bar.

## Deviation (2026-10-18, after capture, before any probe result was seen)

The float16 capture overflows to inf in exactly one channel per layer (Gemma 3's
massive activation) at pre L23–46 and slot L26–46, in some or all rows. The analysis
crashed at the first such slice, after printing only the baselines. Fix: drop non-finite
columns per slice before PCA (at most 1 of 3840 dims). No re-capture.

## J-lens addendum (2026-10-18, after the probe result, before any J-lens readout)

Question: on this clean contrast, does the masked J-lens detect lies?
Script: `scripts/jlens_12b.py`. Rows: D rows of all 70 lied items plus a seeded sample of
140 honest items, both mappings (420 rows). Slot position, every 2nd fitted lens layer.
Features: plan 2's masked top-K (K=20, store 50, fold-internal vocabulary), mask = every
vocab id decoding to yes/no/a/b. Same CV, class weights, baselines (refit on these rows),
item-level null (max over layers, 100 draws, per detector) and three-part bar as above.
The residual probe and masked logit lens are refit on the same rows for a like-for-like
comparison.

- **P5.** The masked J-lens meets the bar at some layer, but its best balanced accuracy is
  below the residual probe's on the same rows (CIs may overlap).
- **P6.** Masked J-lens and masked logit lens have overlapping CIs at their best layers.

## Steering follow-up (2026-10-18, exploratory, after the registered steering run was null)

Slot-only steering at L42 (registered) and L30 was no better than a random direction of
equal norm. Follow-up, labelled exploratory: the same cross-fitted slot directions applied
at **every position** (`--apply-from all`) at L20 (where the probe signal first appears)
and L42.

## Zero-shot J-lens addendum (2026-10-18, exploratory, before computing)

Not blind: the fitted J-lens weights (error/doubt tokens) were seen first, so this is
exploratory. To limit steering, the list is plain English synonyms of "incorrect", chosen
by meaning, not copied from the fitted weights (which were mostly non-English):
**error, errors, wrong, incorrect, false, mistake, mistaken, doubt, untrue, inaccurate.**
Every vocab id whose decoded string (stripped, lower-cased) is in the list counts.

Score per row and layer: total J-lens probability on those ids within the row's stored
top-50 (0 if none appear). No fitting, no labels. Same 420 rows. Primary: AUROC per layer;
null: 1000 item-level label permutations within truth arm, max over layers, p95. Also
reported: AUROC within each (truth x emitted letter) cell, and the same score from the logit
lens. Prediction: J-lens AUROC above null at L24–40, below the trained J-lens classifier.
