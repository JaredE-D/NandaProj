# Pre-registration — Direction 1: does the J-lens beat simple baselines?

Written and approved 2026-10-18, before any of the analysis below was run.
Inputs are cached plan-2 artifacts only (gemma-3-4b-it, v2 bank); no GPU.

## Task

Under the D prompt only, predict whether the model lied (D answer != truth) or
answered honestly. Population: v2 belief items the model answers correctly under
H (an H-wrong item is neither a lie nor an honest answer under D, so it is
dropped; plan-2 notebook 08 counted it as a negative). Primary: minus
`world_fact` (the model lies on 1 of 20). Secondary: all tiers.

## Detectors (same leave-one-pair-out folds, same L2 logistic for all)

1. Masked J-lens top-20 tokens, per layer (08's features)
2. Masked logit-lens top-20 tokens, per layer
3. Raw answer-slot residual, per layer
4. TF-IDF of the D prompt text alone (no model)
5. Item metadata only: polarity, tier, harm (v2 has no stake field)
6. Model confidence under H: logit P(true answer | H)

## Metrics

Balanced accuracy (primary), AUROC, and balanced accuracy within each polarity
arm. 95% CIs by bootstrap over pairs (held-out predictions fixed). Null: labels
permuted within each polarity arm, refit, 200 draws, p95.

## Predictions

- **P1.** Masked J-lens <= residual probe at every layer.
- **P2.** Masked J-lens and logit lens have overlapping CIs at the best layer.
- **P3.** Metadata-only beats chance on overall balanced accuracy, via polarity.
- **P4.** Within each polarity arm, J-lens CI includes 0.5 at most layers.

## Decision rule

If the masked J-lens does not beat detectors 4–6 with separated CIs, the claim
becomes "lens-based lie classifiers on instructed lying mostly read the item and
prompt". If it does, it is the signal worth carrying to the full-bank collection.
