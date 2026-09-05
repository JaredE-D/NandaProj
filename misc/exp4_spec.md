# Exp 4 — A lying-vs-honest classifier on J-lens token features (notebook `08_jtoken_classifier`)

**Approved 2026-09-04** (C1 as honest negatives: yes; K = 20 with {5, 50} band: yes; fit on
all 100 belief items, score the lying set: yes). §8 is registered in
[`PLAN2.md`](../PLAN2.md) §13 and this file is the implementation reference.

**Status:** drafted and approved 2026-09-04. **Depends on:** the v2 bank, the 04d gate/emitted files in
`results/`, and the alleged arm (`data/alleged_arm_final.json`). Nothing from 05.
**Cost estimate:** ~600 lens readouts (two `lens.apply` passes each). At the ~1–2 s/readout 04
saw, 10–20 min on the current box, ≈ $0.10. No new box. Everything after capture is CPU.

**Relation to 04d.** 04d asked whether the *truth* is linearly decodable from the raw 2560-d
residual under D (yes, ~0.92). Exp 4 asks a different question with a different, much
smaller and human-readable feature set: does *what the J-lens says the model is about to say*
separate a lying forward pass from an honest one, and which tokens carry that separation.
The answer is interesting either way. If it works with the answer tokens removed, the
weighted tokens are a readable description of what the J-lens sees at the slot while the
model lies. If it collapses, the J-lens at the slot reads the emitted answer and nothing else,
which is the token-asymmetry finding stated one more way.

---

## 1. Question, in one sentence

At the answer slot, using only the J-lens's top-K token probabilities per layer as features,
can a linear classifier trained with leave-one-pair-out CV tell an item's D pass (model lied)
from its H pass (model honest), above a pair-flipped null, **after the Yes/No tokens are
removed from the feature set**, and does that classifier transfer unchanged to the alleged bank?

## 2. Two confounds fixed in advance

**(a) The lying set is polarity-skewed.** On the v2 bank the model lied on 53/100 belief items:
37 No-true (lies with " Yes") and 16 Yes-true (lies with " No"). The H row of a No-true item
emits " No" and its D row emits " Yes", so a classifier that reads the answer token gets
~70% on H-vs-D without learning anything about lying. Therefore:

- The **primary** number is the classifier fit with every Yes/No variant removed from the
  feature vocabulary (§4.3). The full-vocabulary fit is reported beside it as the thing being
  controlled for, never as the headline.
- Every accuracy is reported **split by polarity arm** (No-true / Yes-true), per the standing
  rule that a result carried by one arm is a token result.

**(b) H and D differ in prompt text, not only in whether the model lies.** D carries a persona
and a lie instruction; H has no system prompt. A classifier could learn "a persona is present".
C1 is the same persona instructed to be honest, and the model is honest under it (0.94 correct
under H, 0.94 under C1). So:

- Rows are **all 100 belief items × {H, C1, D}**. The label is *this row is a lie*: the D row
  of an item in 04d's `LYING` mask (honest under H, wrong under D). Every H and C1 row, and
  the D row of every item the model did **not** lie on, is label 0. The D rows with label 0
  (about 47) are what stop "the D persona is present" from being a sufficient feature.
- Accuracy is reported over all rows and, separately, on the lying set's rows.

Truth labels come from the bank's declared polarity; "lied" comes from the 04d emitted
probabilities (`slot_emitted_*.json`). No new gating.

## 3. Data

| set | items | rows | positives | groups |
|---|---|---|---|---|
| v2 main | 100 belief items | H, C1, D → 300 | 53 D rows (37 No-true, 16 Yes-true) | `pair_id` (both twins, all conditions of a pair held out together) |
| alleged arm | 141 belief items | H, D → 282 | 118 D rows | test only, never fitted on |

The alleged arm has no C1 prompts, so transfer is scored on its H and D rows.

## 4. Features

**4.1 Capture (GPU, once).** For every (item, condition) row: `reader.readout(prompt)` at the
answer slot, giving a full-vocab distribution per fitted layer for the J-lens and, from the
same call, the logit lens. Store per row, per layer, the **top-200** token ids and their
log-probs, for both lenses. Written after every item (04's rule). Two files:
`results/jtoken_topk_{lens_id}.npz` and `results/jtoken_topk_alleged_{lens_id}.npz`.
Storing 200 rather than the full 262k-vocab vector keeps the file at a few MB instead of
~10 GB, and 200 is far above the K used downstream so the floor in 4.2 is rarely hit.

**4.2 Feature matrix (CPU).** Per layer ℓ, the feature vocabulary `V_ℓ` is the union over
**training rows** of each row's top-K (K = 20 primary; K = 5 and 50 as a sensitivity band).
Feature value = the row's log-prob of that token; a token absent from a row's stored top-200
takes that row's 200th log-prob as a floor. Computing `V_ℓ` inside each CV fold keeps
held-out rows out of feature selection.

**4.3 Answer-token mask.** A token is an answer token if its decoded string, stripped and
lower-cased, is `yes` or `no`. The mask is applied to `V_ℓ` before fitting. The list of
masked ids is printed once so it can be checked (expected: " Yes", "Yes", " yes", " YES",
" No", "No", " no", " NO" and the like).

**4.4 Logit-lens twin.** The identical pipeline on the logit-lens top-K from the same
readouts. This costs nothing extra and is the codebase's standing control for whether the
Jacobian did any work. It is reported alongside, not as a gate you asked for.

## 5. Classifier and evaluation

**Metric: balanced accuracy** (mean of true-positive and true-negative rate). Only 53 of the
300 rows are lies, so plain accuracy is 0.82 for a classifier that never says "lie". Reported
alongside: TPR on each polarity arm, and TNR on the honest D rows specifically (the rows that
carry the D persona but no lie).

- `probe.cv_logistic` (L2 logistic, standardised inside each fold, leave-one-pair-out), per
  layer, over all fitted layers. Existing, tested code; nothing new here.
- **Null:** labels shuffled **within each item's rows** (`tokenfeat.shuffle_within_items`),
  so a lying item still has exactly one positive row but it lands on H, C1 or D at random.
  That keeps the number of positives, the per-item structure and the polarity balance, and
  breaks only the link between "which condition" and "lie". 100 draws, p95 per layer. The
  pair-flip null in `probe` is not used: with three rows per item it would flip an item's
  H and C1 rows to positive, a label set no bank could have.
- **D rows only (required, added 2026-09-05).** A classifier that only detects the lie
  instruction in the context scores balanced 0.90 on the all-rows metric (TPR 1.0, TNR 0.81:
  only 47 of 247 negatives carry the D prompt). The 47 honest D rows are the only rows that
  separate "told to lie" from "lied", so the same pipeline is run on the 100 D rows alone,
  53 lie vs 47 honest, leave-one-pair-out, with a label-permutation null. This is the number
  that decides whether anything beyond the instruction was read. Beside it: the tier and
  harm split of lied vs honest D items, because a bank where "honest" means one tier makes
  the D-only classifier a tier detector.
- **Polarity is confounded with the D-only label** (v2 gate: lied 37 No-true / 16 Yes-true,
  honest D 13 No-true / 34 Yes-true), and **tier is too** (world_fact: 1 lied / 19 honest).
  So the D-only comparison is also run without world_fact items, and every score carries
  balanced accuracy *within* each polarity arm, on which a polarity reader scores 0.5.
- Fitting is fold-internal throughout: the feature vocabulary is built from the training
  rows of each fold. `tokenfeat.cv_predict_many` batches every fold and every null draw
  of a layer into one tensor fit (torch on the box's GPU, numpy otherwise) using per-fold
  feature masks, and is tested to give predictions identical to the per-fold reference;
  the notebook re-checks that equality at one layer on the box. Null draws: 200.
- **Transfer:** at each layer, fit once on all v2 rows and apply to the alleged rows.
  Accuracy overall and per polarity arm.
- **Readout of the classifier:** at the best upper-stack layer, the 15 highest-|weight|
  tokens with sign, for the answer-masked fit. This is the deliverable if the classifier works.

## 6. What the outcomes mean

| answer-masked J-token CV accuracy (upper stack) | reading |
|---|---|
| ≈ null p95 (~0.6) while the full-vocab fit is high | the J-lens at the slot reads the emitted answer and nothing else; Exp 4 restates the token-asymmetry finding |
| well above null, but the logit-lens twin matches it | the lying signature is legible in the raw residual's vocab projection; J-space adds nothing |
| well above null and above the logit-lens twin, transfers to alleged | the J-lens carries a lying signature beyond the answer token, and the weighted tokens say what it is |
| high on v2, near chance on alleged | it learned the v2 bank's wording |

No layer is pre-selected. "Upper stack" for the readout in §5 means the argmax over
`reader.upper`, and the whole curve is plotted.

## 7. Deliverables

- `src/nandaproj/tokenfeat.py`: `TopK` store (save/load, per-row per-layer ids and log-probs),
  `capture_topk(reader, items, conditions, k, save_to)`, `feature_matrix(store, layer, rows,
  vocab, floor)`, `union_vocab(store, layer, rows, k)`, `answer_token_ids(tok)`. Tests for the
  store round-trip, the union/floor logic on a toy store, the mask, and the fold-internal
  vocabulary (a held-out row's private token must not appear in `V_ℓ`).
- `notebooks/08_jtoken_classifier.ipynb`: §1 capture (GPU), §2 features + CV + null (CPU),
  §3 transfer, §4 token readout, §5 summary JSON
  `results/jtoken_classifier_summary_{lens_id}.json`.
- Jared runs the notebook. Claude edits it with `NotebookEdit` only.

## 8. Registered text (PLAN2.md §13)

> **13. Exp 4 — lying-vs-honest classifier on J-lens tokens (`08_jtoken_classifier`).**
> Features are the J-lens top-K token log-probs at the answer slot, per layer, vocabulary
> built inside each CV fold; rows are the H, C1 and D passes of all 100 v2 belief items;
> the label is "this is the D row of an item the model lied on"; CV is leave-one-pair-out;
> the primary fit removes every Yes/No token from the vocabulary. Success is accuracy above
> the within-item-shuffled null p95 on the answer-masked fit, on both polarity arms, and
> transfer to the alleged arm. The logit-lens twin is reported alongside. Failure of the
> masked fit with success of the unmasked one is the expected deflationary outcome and is
> reported as such. K = 20 primary, {5, 50} as a band. No layer is pre-selected.

## 9. Decisions (closed 2026-09-04)

1. C1 rows as honest negatives: **yes**.
2. K = 20 primary, {5, 50} band: **yes**.
3. Fit on all 100 belief items × {H, C1, D}, score everything and the lying set: **yes**.
