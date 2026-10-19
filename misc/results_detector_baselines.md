# Results — Direction 1: does the J-lens beat simple baselines?

Run 2026-10-18, `scripts/detector_baselines.py`, against `misc/prereg_detector_baselines.md`.
Numbers: `results/detector_baselines_gemma-3-4b-it.json`. Figures:
`pubplots/detector_baselines_balanced.png`, `pubplots/detector_baselines_arm_mean.png`.
CIs: 2000 pair resamples, held-out scores fixed. Null: 200 within-arm permutations, refit.

## Primary population (D rows, H-correct, minus world_fact): n=77, 52 lied / 25 honest, 40 pairs

| | lied | honest |
|---|---|---|
| Yes-true | 16 | 22 |
| No-true | 36 | **3** |

| detector | best balanced [95% CI] | layer |
|---|---|---|
| masked J-lens | 0.731 [0.643, 0.822] | L20 |
| masked logit lens | 0.772 [0.685, 0.861] | L17 |
| residual probe | 0.893 [0.828, 0.950] | L31 |
| prompt TF-IDF | 0.562 [0.481, 0.651] | — |
| item metadata | 0.585 [0.470, 0.700] | — |
| confidence under H | 0.482 [0.434, 0.538] | — |
| *polarity only (exploratory)* | *0.786 [0.691, 0.873]* | — |

- **P1 NOT SUPPORTED** (narrowly): J-lens > residual probe at L1, 2, 3, 5 only; residual ahead from L6 up.
- **P2 SUPPORTED**: best J and best logit CIs overlap; logit nominally higher.
- **P3 NOT SUPPORTED**: metadata 0.585, CI includes 0.5. Its within-arm mean is 0.43 (below
  chance), the usual sign of leave-one-out anti-learning on small categorical cells; the bare
  polarity rule scores 0.786.
- **P4 SUPPORTED, but near-vacuous**: the No-true arm has 3 honest rows.
- **Decision rule: not separated.** J-lens CI low 0.643 < highest baseline CI high 0.700.
- Masked J-lens exceeds its within-arm null p95 at 5/33 layers (L2, 15, 20, 22, 24); logit
  4/33; residual 18/34 (every layer from L16 up, plus L14).

## Secondary (all tiers): n=93, 53 lied / 40 honest

Every model-based detector clears its null from **L0–L1**, before any computation on the
question, so this population is dominated by recognising `world_fact` items (1 lied of 20).
It does not bear on lie content.

## Reading

1. On this bank, "lied vs honest under the same prompt" is nearly the same split as
   "the true answer is No": the polarity rule beats every lens detector.
2. At the answer slot the label is exactly `truth XOR emitted answer`. Inside one polarity
   arm it *is* the emitted answer. The residual probe's climb after L25 tracks the unmasked
   answer becoming decodable, not a lie representation. Masking the yes/no spellings does
   not remove answer-correlated tokens (Certainly, Doubt, ...).
3. So no detector here can show lie-specific content until the bank has honest No-true items
   in number. That is a bank problem, not a model-size problem.
