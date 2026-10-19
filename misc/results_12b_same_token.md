# Results — same-token lie detection on gemma-3-12b-it

Run 2026-10-18 against `misc/prereg_12b_same_token.md` (with its pre-data amendment and
post-capture deviation). Capture: `scripts/capture_12b.py` on one 48 GB box. Analysis:
`scripts/lie_probe_12b.py`, numbers in `results/lie_probe_12b_gemma-3-12b-it.json`,
figure `pubplots/lie_probe_12b.png`.

## Population

70 lied / 626 honest alleged items (404 pairs), robust under both A/B mappings; D and H
rows in both mappings, 2,784 forward passes. Labels reproduce on re-capture: 1/1,392 D
rows and 0/1,392 H rows disagree with the gate.

## Baselines (D rows, 10-fold grouped CV, class-balanced)

| detector | balanced | 95% CI |
|---|---|---|
| true answer | 0.569 | [0.527, 0.611] |
| mapping | 0.443 | [0.371, 0.519] |
| emitted letter | 0.497 | [0.478, 0.520] |
| emitted meaning | 0.603 | [0.564, 0.642] |
| metadata (category, stake, + the four above) | 0.709 | [0.665, 0.754] |
| prompt TF-IDF | 0.681 | [0.605, 0.750] |

The answer-token and polarity baselines are near chance, as the design intended. The bar
is set by item metadata: lie propensity varies by category and stake.

## Probe — pre-registered bar MET

Null p95 (max over all 96 layer x position slices, 100 item-level permutations): 0.598.
Highest baseline CI high: 0.754 (metadata).

| | best slice | balanced [95% CI] | AUROC | min of 4 (truth x letter) cells |
|---|---|---|---|---|
| slot | L42 | **0.931 [0.899, 0.959]** | 0.97 | 0.87 |
| pre | L42 | 0.907 [0.86, 0.95] | 0.96 | 0.86 |

- 46/96 slices meet all three bar conditions: slot L23 and L25–47, pre L25 and L27–47.
- Profile (figure): through L19 the probe sits at 0.63–0.72, inside the metadata band, i.e.
  item content. It jumps at **L20** in both positions and is above 0.85 from ~L27 (slot)
  / ~L29 (pre).
- The lie is as decodable at *pre* — the `\n` after `<start_of_turn>model`, before any
  answer text — as at the slot (gap 0.02–0.04).

## Controls

| control (at slot L42) | balanced | reading |
|---|---|---|
| XOR control: truth XOR stake | 0.751 | an arbitrary XOR of represented features is decodable, but 0.18 below the lie probe; it rises later (from ~L22) and plateaus ~0.75 |
| H rows, D label (propensity) | 0.608 | without the lie prompt, residuals barely predict which items get lied on (≈ metadata) |
| held-out category (train 2, test 3rd) | fault 0.791, hazard 0.846, slight 0.778 | generalises across item categories |

## Predictions

- P1 (bar met at slot from mid-depth): **supported** (from L23).
- P2 (pre above null but below slot): **partly** — above null, but only marginally below slot.
- P3 (single-variable baselines include 0.5): **mostly** — truth [0.53, 0.61] and emitted
  meaning [0.56, 0.64] exclude 0.5 narrowly; letter and mapping include it.
- P4 (XOR above chance, below lie probe): **supported**.

## Reading

1. On a contrast where answer token, answer meaning, mapping and polarity are each near
   chance, a linear probe on the 12B residual reads lie vs honest at 0.93, above every
   baseline, the null, in all four (truth x letter) cells, and across held-out categories.
2. The lie signal appears at L20 (~40% depth) and is present before the answer is written.
3. The H-row control rules out "the probe recognises lie-prone items" as the main story.

## Caveats

- The XOR control uses stake, plausibly a weaker feature than the model's own answer, so it
  does not fully rule out "truth and the model's intended answer are both represented and
  their XOR is linearly readable". The *pre* result narrows but does not close this.
- "Lied" = gave the persona-serving answer against the truth (instructed / incentivised
  lying), one model, one bank. Not spontaneous lying.
- One float16-overflowed channel per affected layer was dropped (prereg deviation).

## J-lens on the clean contrast (J-lens addendum)

`scripts/jlens_12b.py`, `results/jlens_12b_gemma-3-12b-it.json`, `pubplots/jlens_12b.png`.
420 D rows (70 lied + 140 sampled honest items, both mappings), slot, every 2nd layer.
The lens's own final distribution reproduces the captured p(A), p(B) exactly (max diff 0.0000).
Masked ids: every spelling of yes/no/a/b (24). Highest baseline CI high on these rows: 0.805.

| detector (registered) | best layer | balanced [95% CI] | null p95 | bar |
|---|---|---|---|---|
| masked J-lens top-20 | L26 | **0.963 [0.936, 0.984]** | 0.614 | **MET** (L24–40) |
| masked logit lens top-20 | L32 | 0.964 [0.940, 0.984] | 0.607 | MET |
| residual probe, PCA-256 | L40 | 0.618 [0.580, 0.657] | 0.582 | not met |

**The registered probe failed on this subset by overfitting, not for lack of signal.**
256 components on ~380 training rows per fold. Post hoc, same rows and folds, slot:

| PCA components | 8 | 16 | 32 | 64 | 256 |
|---|---|---|---|---|---|
| L26 | 0.963 | 0.963 | 0.959 | 0.925 | 0.546 |
| L30 | 0.970 | 0.971 | 0.971 | 0.946 | 0.575 |
| L40 | 0.970 | 0.970 | 0.964 | 0.963 | 0.618 |

So on the clean contrast all three readouts — J-lens, logit lens, a small residual probe —
detect the lie at ~0.96–0.97. The lie lies along a top-variance direction of D-slot
activations (8 PCs suffice).

**Black-box confidence does not explain it.** The D answer's own |log p(A) − log p(B)|
scores 0.625 balanced (AUROC 0.656) on all D rows; A+B mass 0.582.

**What the J-lens reads (top-weighted tokens, logistic fit on all 420 rows):**

- L26 → lie: `Fehler` (error), `dúvida` (doubt), `Ошибка` (error), `Incorrect`, `Zweifel`
  (doubt), `dubbio` (doubt), `ошибка`, `противоре…` (contradict-), `Disagree`, `👎`, `contradicts`.
- L26 → honest: `Zustimmung` (agreement), `correctAnswer`, `Confident`, `Nobody`, `Keine`, `Aucun`.
- L30 → lie: `Incorrect`, `Guilty`, `Unsupported`, `Objection`, `falsch`, `Wrong`.
- L30 → honest: `correctAnswer`, `Zustimmung`, `Definitely`, `Agreed`, `Correct`.
- Logit lens L32: single letters and function words (`Z`, `X`, `The` vs `S`, `N`, `both`) —
  equally accurate, not readable.

Noise: some lie-related tokens weight *honest* (`謊` "lie", `মিথ্যা` "false", `FALSE`), so the
reading is a tendency over many tokens, not a clean single concept.

Predictions: **P5 not supported** as written — the J-lens is not below the probe; with the
probe sized for n, they tie. **P6 supported** — J-lens and logit lens CIs overlap.

## Steering (registered: slot L42; others exploratory)

Cross-fitted mean-difference directions vs random directions of equal norm, 70 lied + 70
honest items x both mappings. Lie rate after steering (dose = multiples of the mean
difference; at L42, 4x ≈ 80% of the median residual norm):

| layer, scope | lied items, probe dir x4 | lied, random x4 | honest, probe x4 | honest, random x4 |
|---|---|---|---|---|
| L42 slot (registered) | 136/140 | 134/140 | 6/140 | 2/140 |
| L30 slot | 135/140 | 129/140 | 5/140 | 7/140 |
| L42 all positions | 135/140 | 133/140 | 8/140 | 2/140 |

**Null.** The direction that reads the lie at 0.93 does not move the answer beyond a random
direction, at the slot or at every position. Readable, not (by this intervention) causal —
consistent with plan 2's null direction-level intervention on 4B. The direction is not
dominated by outlier channels (top-20 dims carry 5% of its squared norm at L42). L20
all-positions was planned but not run.

## Headline

On a lie-vs-honest contrast built so that answer token, answer meaning, letter mapping,
polarity and output confidence each sit near chance, gemma-3-12b-it's lies are linearly
readable from L20 onward (probe 0.93 on 1,392 rows; J-lens 0.96 on 420), before the answer
is written, across held-out item categories, and not from the same items' honest-prompt
activations. The J-lens adds what the probe cannot: a readable description — at the answer
slot of a lie, it decodes *error / doubt / incorrect* across languages. Steering along the
lie direction does not change the answer.
