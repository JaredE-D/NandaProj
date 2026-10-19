# PLAN3 — Does the J-lens lie signature survive a change of model family?

**Target:** MATS application (Neel Nanda stream), continuation. **Budget:** whatever remains of the
$50 (about $45; plan 2 spent roughly $5). **Status:** drafted 2026-09-12, awaiting approval.
**Supersedes** [`misc/plan2old.md`](misc/plan2old.md), which is kept intact for its
registrations (§7, §12, §13). Code and notebooks that cite "PLAN2 §n" mean that file.
**Branch:** `cross-model-qwen3`.

## 0. What changed and why

Plan 2 ran to completion on `gemma-3-4b-it`. Its results, in the order they should be read
([`misc/methods_summary.md`](misc/methods_summary.md), memory notes):

1. **Token asymmetry.** The J-lens decodes ` Yes` at the answer slot from L19 and ` No` only from
   L25, in every condition and on both banks. Plan 2's "belief legible at L19–22, crossover at L25"
   is this asymmetry restated, not a lie-specific event.
2. **Truth is decodable under D on a separate direction.** A D-fitted logistic probe reads the true
   answer at ≥0.85 from L16 up in all four (emitted × lied) cells, against a pair-flip null of
   ~0.58. Replicates on the alleged bank.
3. **The J-token classifier works with the answer tokens masked.** Balanced accuracy on
   lie-vs-honest rows peaks at 0.94 at L19 (masked J-lens, K=20), above the within-item null,
   on both polarity arms.
4. **Direction-level intervention was null** (Exp 1) and the `ablate d_J` arm was void by
   construction. Component attribution (05) found L19M/L25M with positive recovery in both arms,
   below the registered bar.

Every one of those numbers is one model, one tokenizer, one corpus-fitted lens. Finding 1 in
particular smells like a property of Gemma's unembedding and of a wikitext-fitted Jacobian, not of
lying. Findings 2 and 3 are the ones worth keeping if they survive a second family; finding 4 is
not carried forward.

**This plan replicates findings 1–3 on a second model family, and reports which of them are
model-specific.** Nothing new is invented. The instrument stays the same; the model changes.

## 1. The question

For each of the three findings, on `Qwen3-4B` versus `gemma-3-4b-it`:

| finding | replicates if | Gemma-specific if |
|---|---|---|
| **F1** lens token asymmetry | one answer token is decodable many layers before the other, in every condition | both tokens become decodable at the same relative depth, or the order flips with no condition dependence either way |
| **F2** truth decodable under D | D-fitted probe accuracy above the pair-flip null p95 in all four (emitted × lied) cells, from some layer up | at or near null in any cell |
| **F3** masked J-token classifier | balanced accuracy above the within-item null p95 on **both** polarity arms at some layer | above null on one arm only, or nowhere |

And for whatever replicates: **does it sit at the same relative depth?** Layer indices are
reported as `ℓ / n_layers` (Gemma 34 blocks, Qwen 36) so the two profiles can be overlaid.

The safety-relevant reading is unchanged from plan 2 and still deliberately narrow: a lie-detection
feature that exists in two families at a similar depth is a candidate for a general inference-time
monitor; one that exists in one family is a fact about that family.

## 2. Model choice

The hard constraint is a pre-fitted lens in `neuronpedia/jacobian-lens`. Every candidate with one,
in the 1.7B–9B range, checked 2026-09-12:

| lens dir | family | chat-tuned | verdict |
|---|---|---|---|
| `qwen3-4b` | Alibaba | yes (hybrid thinking) | **chosen** — different family, matched size, instruct, ungated |
| `qwen3.5-4b` | Alibaba | yes | newer arch, less exercised by nnsight/jlens; fallback only |
| `gemma-4-e4b` | Google | yes | same family; weak cross-family evidence |
| `qwen3-1.7b`, `qwen3-8b` | Alibaba | yes | size mismatch; 8b is a depth control if time allows |
| `qwen3.5-2b-pt`, `qwen3.5-9b-pt` | Alibaba | base only | will not follow a lie instruction |
| `gemma-2-2b-it`, `gemma-2-9b-it` | Google | yes | same family, wrong size |
| `llama3.1-8b-it`, `qwen2.5-7b-it` | Meta / Alibaba | yes | 2× the weights; ~16 GB bf16 on a 24 GB card leaves little for activations |

**`Qwen/Qwen3-4B`:** 36 decoder blocks, `d_model` 2560, 32 query heads / 8 KV heads (GQA), QK-norm,
standard RMSNorm (`w`, not Gemma's `1+w`), no sliding-window alternation, tied embeddings, ungated on
HF, ~8 GB bf16. Lens file: `qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt` (459 MB).
Same fit corpus as the Gemma lens, so the corpus-average caveat (plan 2 R5) is matched, not added.

**Thinking mode is off.** Every `apply_chat_template` call passes `enable_thinking=False`. Decided
2026-09-12. With thinking off, Qwen3's template inserts an empty `<think>\n\n</think>\n\n` block at
the start of the assistant turn; the `Answer:` prefill goes after it and position −1 remains the
answer slot. This is verified in V1, not assumed.

## 3. What has to change in the code

Found by reading, to be confirmed by running (CLAUDE.md: verify by executing). None is large.

| # | where | issue | change |
|---|---|---|---|
| P1 | [`config.py`](src/nandaproj/config.py) | lens path built from lowercase `lens_id`; Qwen file is `Qwen3-4B_…` | add `lens_file` field to `ModelConfig`, default derived from `lens_id`; new preset `qwen` |
| P2 | [`items.py:562`](src/nandaproj/items.py#L562), [`deception.py:304`](src/nandaproj/deception.py#L304), [`synthetic.py:197`](src/nandaproj/synthetic.py#L197) | Qwen3 emits `<think>` unless told not to | one `chat_template_kwargs(cfg)` helper in `config.py`; every render passes `**kwargs` |
| P3 | [`items.py:78`](src/nandaproj/items.py#L78) | system prompt folded into user turn because Gemma has no system role | **keep folding on both models.** Same prompt bytes, same comparison. Qwen's real system role is a follow-up, not a variable here |
| P4 | [`lens_readout.py:114`](src/nandaproj/lens_readout.py#L114) | `text_config` fallback already handles non-multimodal; fitted-layer list is read from the lens | expect no change; V1 prints `describe()` |
| P5 | [`attribution.py:852`](src/nandaproj/attribution.py#L852) | `_gemma_style` branches on class name | falls through to standard RMSNorm; only matters if 05 is re-run (it is not, §4) |
| P6 | [`tokenfeat.answer_token_ids`](src/nandaproj/tokenfeat.py) | full-vocab scan for Yes/No spellings | tokenizer-agnostic by design; V1 prints the masked set and its size for Qwen |
| P7 | [`data.py:34`](src/nandaproj/data.py#L34), [`data.py:162`](src/nandaproj/data.py#L162) | hardcoded 24 layers / 3072 hidden, wrong for both models | plan-1 Arm B code; grep for live importers, archive if none |
| P8 | every `results/*_{lens_id}.*` filename | keyed by `lens_id` | already model-keyed; Qwen results land beside Gemma's with no collision |

**Single-token check.** ` Yes` and ` No` must each be one token in Qwen's tokenizer after the
`Answer:` prefill. If not, the answer format changes for *both* models (e.g. `A`/`B`), and every
Gemma number is re-run. V1 checks this before anything else is built.

## 4. Method

Per model, the plan-2 pipeline minus the parts that were null or void:

**4.1 Behavioral gate** (notebook 04 path). H accuracy, D lie rate, C1 honesty, per item.
Same directives, same prefill, same candidates. Output: `gate_{lens_id}.json`, `slot_emitted_{lens_id}.json`.

**4.2 Lens token profile** (F1). For each condition and each layer, the J-lens and logit-lens
probability of ` Yes` and of ` No` at the slot, split by polarity arm. The plot is
`P_lens(token)` vs relative depth, one line per token, one panel per condition, both models
overlaid. This is the finding-1 replication and it needs nothing beyond 04's existing readout.

**4.3 Truth probe under D** (F2, notebook 04d path). D-fitted logistic, leave-one-pair-out, pair-flip
null, reported per (emitted × lied) cell. Also the H-fitted mean-difference probe applied to D,
because on Gemma that is where the L16–18 "truth on an orthogonal direction" result lives.

**4.4 Masked J-token classifier** (F3, notebook 08 path). Unchanged: rows = belief items ×
{H, C1, D}, label = D row of a lied-on item, K=20 with {5, 50} band, within-item null, masked fit
primary, balanced accuracy, both arms. **Save per-fold predictions this time** so the error bars in
[`pubplots/CAPTIONS.md`](pubplots/CAPTIONS.md) stop being analytic approximations.

**4.5 Cross-model comparison.** A classifier cannot transfer across tokenizers, so the comparison is:

- layer profiles of 4.2–4.4 on a shared relative-depth axis;
- the **top-weighted token strings** of the masked classifier at its peak layer, per model, as
  a table — the readable description of what the lens sees when the model lies, and whether the
  two models' descriptions overlap in words rather than in ids;
- the four gate counts (lied Yes-true / lied No-true / honest-D by arm), because a model that lies
  in only one polarity arm cannot give a two-arm result and that is a gate outcome, not a finding.

**Not carried forward:** Exp 1 (04b), component attribution (05), the alleged-bank readout (04c),
J-space ρ (04e §2). Each was null, void, or a control for a claim no longer being made. They stay
in the repo, unrun on Qwen, and the writeup says so.

## 5. Item bank

**Jared is revising the bank** (2026-09-12): more resilient items, plus examples ported from
external datasets. The bank is his; this plan constrains only what the comparison needs from it:

- **Polarity pairing stays.** Every belief item has a twin with the true answer flipped and the
  persona-serving answer flipped with it ([`misc/bank_v2_spec.md`](misc/bank_v2_spec.md) §2).
  It is the only thing that made F3's two-arm reporting possible.
- **Single-token answers stay**, whatever the token pair turns out to be.
- **Ported items carry a `source` field** (`hand`, or the dataset name) so any result can be split
  by provenance. If a finding is carried by hand-written items only, that is reported.
- **One bank, frozen before either model's 4.2 sweep, never extended** (plan 2 §7.4 stands).
- **Gate is run per model.** The comparison population is the items that pass on **both** models;
  each model's full gated set is reported beside it. A bank that gates down to fewer than ~15 lied
  items per arm on either model is a gate failure and stops the plan before 4.3.

Candidate external sources for Yes/No factual items, for Jared's consideration and not decided
here: the Azaria & Mitchell true/false statement sets, the Marks & Tegmark `cities` /
`sp_en_trans` / `larger_than` sets, and the TruthfulQA binary split. All have a declared truth
value per item; none has a persona or a stake, which the bank layer supplies.

## 6. Controls and the deflationary readings they test

| control | deflationary story |
|---|---|
| logit lens beside J-lens, per token | "the asymmetry is the unembedding, not the Jacobian" — on Gemma this was not separated cleanly and should be |
| C1 rows as honest negatives | "the classifier detects the persona" (carried) |
| within-item label shuffle | "the classifier detects the item" (carried) |
| honest-D rows as label 0 | "the classifier detects the lie instruction" (carried) |
| per-arm reporting everywhere | "the result is one token" (carried; F1 is exactly this on Gemma) |
| `source` split | "the result is the hand-written items" (new) |
| relative-depth axis | "same layer index" is meaningless across 34 vs 36 blocks |

## 7. Verification gates

| | gate | where |
|---|---|---|
| V1 | Qwen3-4B loads bf16 on the 4090; lens loads from the mixed-case path; `describe()` lists fitted layers; ` Yes`/` No` single tokens; thinking off verified on a rendered prompt; slot is position −1 | `09_qwen_smoke` |
| V2 | **Qwen lies under D** on ≥15 items per polarity arm and is ≥0.9 correct under H and C1 — hard gate | 04 path |
| V3 | F1 measured on both models with the logit lens beside it | 4.2 |
| V4 | F2 measured, all four cells, null fixed before looking | 4.3 |
| V5 | F3 measured, both arms, null fixed before looking, per-fold predictions saved | 4.4 |
| V6 | Overlay figures and the token-string table exist and the writeup states, per finding, *replicates* / *Gemma-specific* / *not evaluable* | 4.5 |

## 8. Pre-registration

**Registered 2026-09-12, before any Qwen3-4B forward pass. Do not edit below this line.**
Plan 2's registrations stand unedited in [`misc/plan2old.md`](misc/plan2old.md).

### 8.1 Predictions

| finding | prediction on Qwen3-4B |
|---|---|
| **F1** | **Does not replicate in form.** An asymmetry between ` Yes` and ` No` legibility will exist, but the order and gap will differ from Gemma's L19/L25, and the logit lens will show most of it. F1 is a tokenizer-and-lens fact, not a lying fact. |
| **F2** | **Replicates.** D-fitted probe reads truth in all four cells from roughly 40–50% depth, above null. The literature makes this the safe bet, which is why it is the least interesting of the three if it holds. |
| **F3** | **Replicates, weaker.** Masked classifier above null on both arms, with a peak balanced accuracy below Gemma's 0.94 and at a similar relative depth (Gemma's L19 is 0.56). |
| depth | F2 and F3 peaks within ±10% relative depth of Gemma's. |
| token strings | little overlap in the top-weighted non-answer tokens between models. The readable description will be model-specific even where the classifier accuracy is not. |
| gate | Qwen lies **more readily** than Gemma under D (stronger instruction-following at 4B), and more symmetrically across polarity arms. |

### 8.2 What would change my mind

- **F1 replicates exactly** (same token first, similar gap, absent from the logit lens): the
  asymmetry is a property of corpus-fitted Jacobians on Yes/No slots, and is itself a finding.
- **F2 fails**: the Gemma probe result was a Gemma fact, and the "truth survives the lie" framing
  is dropped from the writeup.
- **F3 above null on one arm only**: a token result on Qwen, and the Gemma two-arm result is
  demoted to "one of two models".
- **Qwen will not lie in one arm** (V2 fails on one polarity): not evaluable, reported as such;
  no persona search to fix it (plan 2 §7.4).

### 8.3 Forking paths

The model is fixed (Qwen3-4B, not qwen3.5, not a size sweep). The bank is fixed before the sweep.
K, the null, the metric, and the masked-primary rule are inherited unchanged from plan 2 §13.
No layer is pre-selected. `qwen3-8b` as a depth control is a labelled follow-up if hours remain,
not part of the registered comparison.

## 9. Budget and schedule

Two forward-pass jobs per model (gate + readouts, classifier capture), ~600 lens readouts each at
1–2 s. **Estimate: 1.5–2.5 GPU hours ≈ $1** for Qwen, plus a re-run of the Gemma pipeline on the
revised bank at the same cost. Idle instances remain the only real spend.

| hours | work | notebook | risk |
|---|---|---|---|
| 0–1 | P1–P2 code changes, tests locally | — | low |
| 1–2 | V1 smoke on the box | `09_qwen_smoke` | tokenizer surprises |
| 2–4 | gate on both models, revised bank; V2 | 04 path | **the plan lives or dies here** |
| 4–6 | lens token profile, truth probe; V3–V4 | 04 / 04d paths | low |
| 6–8 | classifier capture and CV; V5 | 08 path | null cost on CPU |
| 8–10 | overlay figures, token table, writeup delta; V6 | `scripts/pubplots.py` | — |

**Cut order:** `qwen3-8b` depth control (already out), then 4.5's token-string table, then the
H-fitted probe in 4.3. F1–F3 on both models is the deliverable.

## 10. Risks

- **R1 — Qwen refuses or moralises instead of lying.** Qwen3's post-training may decline the
  persona. V2 catches it; the answer is "not evaluable", not a stronger persona.
- **R2 — thinking-mode leakage.** A stray `<think>` shifts the slot. V1 asserts the rendered
  string ends in `Answer:` and that the top-2 next tokens are the answer candidates.
- **R3 — tokenizer.** ` Yes`/` No` may not be single tokens, or may have more spelling variants
  than Gemma's 12 masked ids. V1 prints the masked set; a larger set is fine, a non-single-token
  answer is a format change for both models.
- **R4 — the lens.** Fitted layer coverage on Qwen may differ (Gemma had 0–32 of 34). `describe()`
  says. jlens is unmaintained and truncates at 512 tokens silently (plan 2 R7); the length guard
  is re-checked because ported items may be longer.
- **R5 — bank revision moves the Gemma baseline.** Every Gemma number in §0 was on bank v2. The
  comparison is on the revised bank, both models, re-run. The v2 numbers are history, not a
  baseline; do not compare Qwen-on-v3 to Gemma-on-v2.
- **R6 — nnsight on Qwen3.** GQA and QK-norm are new to this code path. Only 4.3's residual
  capture needs it (04d uses hidden-state hooks), so exposure is small, but it is verified in V1.
- **R7 — two models in one 24 GB card.** They are not loaded together. One preset per kernel
  session; results are keyed by `lens_id`.

## 11. Language discipline

Carried from plan 2 §10 unchanged, plus:

- "Replicates" means *above the same null, on both arms, on a second family*. Not "similar curve".
- "Gemma-specific" is a claim about two models. Write "did not replicate on Qwen3-4B"; the
  generalisation is the reader's.
- A lens legibility claim is a token-readability claim until split by polarity (memory note,
  2026-09-05). F1 is the cautionary example and the writeup leads with it.
- Never "belief". "Linearly decodable true answer."

## 12. Deliverable

A delta to the existing writeup: one section per finding with the two-model overlay, the
token-string table, the gate counts, and a one-line verdict per finding from V6's three-way
vocabulary. Figures via `scripts/pubplots.py`, reproducible from `just up`.

## 13. Task decomposition

Each task is small enough to verify on its own. Jared runs anything that touches the box.

**A. Code (local, tested)**
- A1. `ModelConfig.lens_file` with a default of `f"{lens_id}_jacobian_lens.pt"`; `qwen` preset
  with `lens_file="Qwen3-4B_jacobian_lens.pt"`, `gated=False`. Test: path string for both presets.
- A2. `config.chat_template_kwargs(cfg)` returning `{"enable_thinking": False}` for Qwen presets,
  `{}` otherwise. Test: both branches.
- A3. Thread A2 through the three `apply_chat_template` sites. Test: a fake tokenizer records
  the kwargs it was called with.
- A4. `lens_readout.Reader.load` uses `cfg.lens_file`. Test: the filename passed to
  `from_pretrained` for both presets.
- A5. `data.py` is imported only by the plan-1 notebooks `01b` and `02` (checked 2026-09-12); archive it with them if Jared archives those notebooks, otherwise fix the two hardcodes.
- A6. README "Switching model scale" section: replace the stale `tiny`/`gemma` text with the
  real presets.

**B. Smoke (box, Jared runs)**
- B1. `09_qwen_smoke.ipynb`: load, `describe()`, single-token check, rendered prompt tail,
  top-2 next tokens on one H item. Prints only; no results file.

**C. Bank (Jared)**
- C1. Revised bank lands in `data/` with `source` fields and pairing intact. Local test: every
  belief item has a twin, every item has `source`.

**D. Runs (box, Jared runs; one preset per kernel)**
- D1. Gate, both models. D2. Lens token profile, both. D3. Truth probe, both. D4. Classifier
  capture and CV, both, per-fold predictions saved.

**E. Comparison (local)**
- E1. Overlay plotting in `scripts/pubplots.py` on a relative-depth axis. E2. Token-string
  table. E3. Writeup delta with the three verdicts.

## 14. Archived 2026-09-12

- `PLAN2.md` → [`misc/plan2old.md`](misc/plan2old.md), stamped as superseded, registrations intact.
- `DATA_ANALYSIS.md` → [`misc/data_analysis_plan1.md`](misc/data_analysis_plan1.md); plan-1 Arm B
  sizing with a stale layer count.
- Candidates **not** moved, for Jared's call: `notebooks/01b_number_readout.ipynb` and
  `02_behavioral.ipynb` (plan-1 Arm B, cut in plan 2), and the `alleged_v1/v2/v3/v5` bank
  variants in `data/` (only `alleged_final` is read).
