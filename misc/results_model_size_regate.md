# Results — alleged-bank re-gate, gemma-3-4b-it vs gemma-3-12b-it

Run 2026-10-18 on one 48 GB box, pre-registered in `misc/prereg_model_size_regate.md`.
Files: `results/pilot_gate_gemma-3-{4b,12b}-it_v2+alleged{,_report}.json`.
Each model: 6,180 rows (v2 100 x H/D/C1, alleged 880 x H/D, x 3 formats).

## Pre-registered decision

| alleged bank                         | 4B         | 12B          |
|--------------------------------------|------------|--------------|
| robust lied / flips / honest         | 7 / 77 / 660 | 70 / 72 / 626 |
| robust lie rate, Yes-true / No-true  | 0.02 / 0.00 | 0.12 / 0.07 |
| usable yield min(lied, honest)       | 7          | 70           |
| P(D answer = A)                      | 0.43       | 0.46         |

1. yield >= 1.25x: **yes** (70 vs 7).
2. arm lie rate in [0.25, 0.75]: **no** (12B 0.12 / 0.07; 4B fails it too, 0.02 / 0.00).
3. letter bias no worse: **yes**.

**Pre-registered decision: stay on 4B.**

## What the decision rule did not anticipate

The rule takes 4B as the default, which assumed 4B was usable. It is not. Its 7 robust lies come
from 7 different pairs, so leave-one-pair-out CV has no lie twins at all.

Most of 4B's alleged "lies" in Yes/No are the D Yes-bias. They fall in the No-true arm (125/391,
against 17/403 for Yes-true) and disappear once the A/B mapping is controlled for. That casts
doubt on plan 2's 141 lie-gated alleged survivors on 4B.

12B fails criterion 2 because it lies *rarely* on this bank, not because one arm dominates.
- Its 70 robust lies span 53 pairs (17 with both twins lied).
- Categories: fault 31, hazard 22, slight 17. Stakes: low 38, high 32.
- The arm skew is 47 Yes-true vs 23 No-true. Among the honest items the split is 334 vs 292
  (53% Yes-true), so polarity is a weaker confound than v2's 22:3.

## v2 (reported, not decisive)

- 4B reproduces the first pilot: the D Yes-bias in Yes/No is 0.71, P(A) is 0.26, and the robust
  labels are 21 lied / 23 honest.
- 12B shows no format bias in any condition (D says Yes 0.48–0.53, P(A) 0.50) and lies almost
  always: 84 robust lied / 12 flips / 1 honest. So v2 on 12B is a clean *instructed-lie vs H*
  contrast, but it has no D-honest class.
- Format compliance: 4B has one row below 0.5 mass on the two candidates; 12B has none.

## Behavioural signal checks (exploratory, post-hoc, 12B robust labels: 70 lied / 626 honest)

- **The lies are confident.** The median P(lie answer) is 0.999, and 99% of lies are above 0.9.
- **H confidence does not predict lying** (AUROC 0.53), so these are not the items 12B is unsure
  about. On 4B the same AUROC is 0.82, but that rests on only n=7 lies.
- **Item text predicts which items get lied on.** Leave-one-pair-out TF-IDF scores:

  | text used    | AUROC | within-arm balanced accuracy |
  |--------------|-------|------------------------------|
  | persona only | 0.81  | 0.76                         |
  | full prompt  | 0.76  | 0.72                         |
  | context only | 0.62  | 0.59                         |
  | category + stake | 0.43 | 0.50                     |

- **Selling or competitive motives draw the lies.** Personas mentioning "the buyer" have a lie
  rate of 0.53 (n=17) and "rival" 0.41, against a base rate of 0.10.
- A D-only lie detector therefore has to beat the persona-text baseline (0.81), not 0.5.

## Caveats

- 12B's lie rate depends on format. On the alleged bank it lies 0.08 / 0.04 in Yes/No, against
  0.16 / 0.11 pooled over A/B. The robust labels use A/B only.
- The Q1 verdict prints "word bias SUPPORTED" for 12B (gap +0.06 -> -0.02). That is misleading,
  because there was no Yes/No gap to explain.
- The 4B Yes/No reproduction against plan-2 `slot_emitted` still shows a large maximum |dp_yes|
  (0.97, median 0). This is unresolved.
