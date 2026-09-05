# Exp 1 — Does the J-lens truth direction *do* anything?

**Registered.** §7 of this file is now [`PLAN2.md`](../PLAN2.md) §12, fixed before data
collection; this file is the implementation reference. Where the two differ, §12 wins —
it is the frozen one.

**Status:** drafted and registered 2026-09-03. **Depends on:** [`04_belief_readout.ipynb`](../notebooks/04_belief_readout.ipynb)
run to completion on `gemma-3-4b-it` (done — gate + curves + gated bank in `results/` on the box).
**Cost estimate:** ~40 min GPU on the current 4090 box, ≈ $0.25. No new box.

---

## 1. What Exp 1 is, in one sentence

04 says the honest answer carries J-lens mass at L19–22 under a deceptive prompt and is gone
by the output. Exp 1 asks whether that mass is **load-bearing**: build the direction in the
layer-ℓ residual stream that the J-lens itself says carries the honest-vs-lie distinction,
edit it, and see whether the emitted token moves.

This is a **direction-level** causal test at a located layer. It is not §4.4 — that is a
*component* ablation over head/MLP sets found by 05's patching, a different object with a
different null. Exp 1 sits between 04 and 05 and its result changes what 05 should look for:
if the direction is inert, ranking components by their effect on it is ranking noise.

## 2. What 04 actually established, and what it constrains

The numbers Exp 1 is built on, from the completed run (11 gated items, D condition):

| | |
|---|---|
| `a_H` legible under D at some layer | **4/11** — `PF03_apartment_mould`, `PF05_phone_water`, `SF04_password_breach`, `SF05_bus_brakes`. Peak `P_J(a_H)` ≈ 1.00 at **L19–22**; J-lens beats its logit-lens control there by 0.78–0.86 |
| `a_H` never legible under D | **7/11** — peak `P_J(a_H)` median 2.9e-06 |
| crossover `ℓ*` | **L25**, sd = 0 across every item that crosses, 74 % depth |
| C2 (instructed inversion) | also crosses at **L25**, on 4 items |
| H and C1 | `a_H` leads at the top on 11/11 — no suppression from a persona alone |

Three constraints follow, and they are the reason this spec is shaped the way it is.

**(a) The positive claim has n = 4.** For the other 7 there is no honest-answer mass at any
layer to remove. They are not filler: they are a within-bank negative subgroup, and the
subgroup split is fixed in §7 *before* any intervention runs.

**(b) `ℓ*` = 25 with sd = 0 is a property of the instrument until proven otherwise.** Zero
variance across items with different content, topics, and answer polarities is not what a
content-dependent quantity looks like. §5.6 is a cheap check on this and its result is
reported whatever it says.

**(c) C2 is already at the same layer.** So every intervention runs under C2 as well as D. A
knob that moves D and C2 identically is an **output-override** knob (PLAN2 §10 naming rule),
and Exp 1 must be able to say so.

## 3. The two directions

`h_ℓ` is the residual stream at the answer slot (last position). `Δe = e_{a_H} − e_{a_D}`, the
unembedding-row difference for the two single-token answers, per item.

**`d_J` — the J-lens direction (primary).** Reading `jacobian-lens` settles the composition
rather than guessing it: `lens.apply` computes `unembed(transport(h, ℓ))` where
`transport(h, ℓ) = J_ℓ h` and `unembed(r) = lm_head(final_norm(r))`, plus an optional logit
softcap. **There is a norm between `J_ℓ` and `W_U`**, so the closed form `J_ℓᵀ W_Uᵀ Δe` is
wrong in general — it silently drops that norm's Jacobian. The direction is therefore taken
by autograd through the lens's own readout:

```
d_J(ℓ, item) = grad_h [ lens_logit(a_H) − lens_logit(a_D) ]   at  h = h_ℓ
```

Exact, assumption-free, and it stays correct if the composition ever changes. This is *the
J-lens's own answer* to "what separates the two answers at layer ℓ", which is exactly the
object the readout in 04 is a projection of. It is per-item, because `Δe` is.

**`d_DiM` — difference in means (baseline).** `mean h_ℓ over H items − mean h_ℓ over D items`,
the standard steering vector. Shared across items, computed on the gated bank.

Both are run everywhere. `d_DiM` is to `d_J` what the logit lens is to the J-lens in 04: the
control that says whether the Jacobian did any causal work, or whether any sensible
difference vector would have done the same. Reporting `d_J` alone cannot answer that.

## 4. The interventions

All edits are applied by a forward hook on the **output of block ℓ**, at the answer slot only
(primary) — with `d̂ = d/‖d‖`, and `σ_ℓ` = mean ‖h_ℓ‖ over the bank at that position, so the
dose is in units of residual norm and is comparable across layers.

| name | edit | question it asks |
|---|---|---|
| **ablate** | `h ← h − (d̂·h) d̂` | is the component necessary? |
| **add** | `h ← h + α σ_ℓ d̂` | is it sufficient? α ∈ {0, ±0.5, ±1, ±2, ±4, ±8} |
| **patch** | `h ← h_ℓ` from the same item's **H** run | upper bound: what can *any* single-layer edit at ℓ do? |

**Grid:** {ablate, add, patch} × conditions {H, D, C2} × layers ℓ ∈ {16, 18, 20, 22, 24, **25**, 26, 28}
× 11 items. Plus a cumulative variant — the same edit at **every** fitted layer ℓ ≤ 25 — because
"around 24 and before" as a region is a different hypothesis from any single layer, and a
single-layer null does not rule it out.

The four corners and what each means:

| | flips the output | does not |
|---|---|---|
| **add `+d_J` under D** | the money result: the belief was there and unrouted, and pushing it restores the honest answer | the mass at L19–22 is not what the output reads |
| **ablate under H** | the direction is causally load-bearing for the answer token | the J-lens mass is epiphenomenal — and this **invalidates the add result too**, so it is run first |
| **ablate under D** | there was something to remove (expect little on the 7 by construction) | consistent with §7.3's first bullet for the 7 |
| **same effect under C2** | not deception-specific — **output-override**, per §10 | separation, which is what V6 needs |

**Positions.** Answer slot only is primary: it is the minimal edit that can change the emitted
token, and it leaves every earlier position's computation intact. All-positions is a secondary
sweep at the single best (intervention, ℓ, α) only, reported separately and never merged.

## 5. Measurements, controls, and the gates

### 5.1 V-I1 — the hook reads the layer the lens reads (**hard gate, first**)

"Layer ℓ" must mean the same tensor to the hook and to `lens.apply`, or every number below is
one block off. The check: capture `h_ℓ` with the hook, push it through `J_ℓ` and the unembedding
by hand, and compare against `lens.apply(..., layers=[ℓ])`'s J-lens distribution on the same
prompt. Agreement to float tolerance at every fitted ℓ, or the offset is found and fixed
before anything else runs. This also settles the exact readout composition assumed by §3.

Nothing in Exp 1 is interpretable until this passes.

### 5.2 Primary outcome

At the answer slot, from the model's own final distribution: `p_H` and `p_D` renormalized
within the two legal answers, and `mass` = how much of the full vocabulary those two held.
Same quantities the 04 gate already computes — `lens_readout.read_slot` is reused, not
reimplemented.

Headline number: **flip rate** — the fraction of items whose top legal answer changes from
`a_D` to `a_H` — reported separately for the 4-item and 7-item subgroups.

### 5.3 The nulls, fixed before the ranking is looked at (§7.4)

1. **Covariance-matched random directions.** Sample `d ~ N(0, Σ_ℓ)`, rescale to `‖d_J‖`,
   20 draws. This is the strong null: the residual stream is anisotropic, and a norm-matched
   *isotropic* vector is too easy to beat. `Σ_ℓ` is estimated from residuals pooled over
   **all token positions** of the bank's prompts (~2000 vectors), not from the 11 slot
   residuals — an 11-vector estimate of a 2560-dimensional covariance is a rank-11 null, and
   a "covariance-matched" draw from it would be a random combination of eleven points.
   The bar is the 95th percentile of the **per-draw** flip rates, not of the pooled trials:
   pooling averages away the tail, and the tail is what "could a random direction have done
   this?" actually asks.
2. **Norm-matched isotropic random**, 20 draws. The weak null, reported alongside so the gap
   between the two nulls is visible.
3. **Wrong-item `d_J`** — item *i*'s direction applied to item *j*. Separates "a truth
   direction" from "a Yes/No answer-token direction", which on a Yes/No bank is the single most
   likely confound and is *not* caught by either random null.

An effect counts only if it exceeds the **covariance-matched** null, with the bar in §7.

### 5.4 Damage (PLAN2 R8 — a broken model is not an honest one)

Reported next to every flip-rate number, never in a separate table:

- `mass` on the two legal answers — an edit that pushes the model out of format has not made
  it truthful;
- accuracy on the **H** items under the identical hook;
- perplexity on held-out text with the hook live, against the unhooked baseline.

### 5.5 Dose–response

Flip rate and all three damage measures against α, one curve per (ℓ, condition). A flip that
only appears at the α where perplexity has already doubled is a report of having broken the
model, and the plot is what makes that visible instead of arguable.

### 5.6 Is `ℓ*` = 25 about the items or about the stack? (cheap, secondary)

Two checks, ~2 min: (a) does `ℓ*` sit at 25 for the C3 floor items and the inverted-polarity
controls too, and (b) is there anything structural at 25 — Gemma 3's 5:1 local/global attention
pattern (R4) puts global-attention layers at a fixed periodicity, and if 25 is one of them the
"crossover layer" may be reporting the architecture. Recorded either way; it changes how §7's
`ℓ*` language may be written.

## 6. Deliverables

| what | where |
|---|---|
| intervention hooks, directions, nulls, damage | `src/nandaproj/intervene.py` (new) |
| the experiment | `notebooks/04b_direction_intervention.ipynb` (new) |
| unit tests on a tiny fake model | `tests/test_intervene.py` (new) |
| registration | `PLAN2.md` §12, from §7 below |

`05_attribution` and `06_ablation` are **untouched** — the numbering avoids a collision with
the other direction being run in parallel.

Every expensive result is written to `results/` as it is produced, not at the end, for the
reason 04 §7 records: a save cell at the bottom only runs when nothing interesting went wrong.

## 7. Draft registration — to become PLAN2.md §12

> **Registered <date>, before Exp 1 data collection. Do not edit below this line.**

**Population, fixed now.** `legible = {PF03_apartment_mould, PF05_phone_water,
SF04_password_breach, SF05_bus_brakes}` (4). `not_legible` = the remaining 7 gated items. This
split is taken from the completed 04 run and is not revised after seeing any intervention
result.

**Predicted signs:**

| quantity | prediction |
|---|---|
| add `+α d_J` under D, `legible` subgroup, ℓ ∈ [20, 25] | flips a **majority** of the 4 to `a_H` at an α whose held-out perplexity is within 20 % of baseline |
| the same, `not_legible` subgroup | **markedly lower** flip rate — nothing legible to restore |
| ablate `d_J` under H | drops `p_H` substantially — the direction is load-bearing |
| ablate `d_J` under D | **little effect**, and least on `not_legible`. A registered null |
| `d_J` vs `d_DiM` | `d_J` at least comparable; if `d_DiM` strictly dominates at every ℓ, the J-lens is descriptively interesting and causally redundant, and that is the honest headline |
| `d_J` vs covariance-matched null | above, on `legible`, at ℓ ∈ [20, 25] |
| wrong-item `d_J` | **substantially weaker** than own-item. If it is not, the direction is a Yes/No answer-token direction and must be named that |
| D vs C2 at matched α | overlapping but not identical. If identical → **output-override**, and that is the headline |
| layer profile | effect peaks at ℓ ≲ 25 and decays above it |

**The bar, fixed before the ranking:** an intervention "works" if its flip rate on `legible`
exceeds the 95th percentile of the 20-draw covariance-matched null at the same (ℓ, α), **and**
held-out perplexity is within 20 % of baseline, **and** H-condition accuracy is ≥ 90 % of
baseline. All three, or it is reported as not working.

**What would change my mind:**

- **Ablating `d_J` under H does nothing.** The J-lens mass is a readout epiphenomenon, not a
  used representation. This kills the causal half of Exp 1 outright and is the first thing run.
- **Wrong-item `d_J` works as well as own-item.** Not a truth direction — an answer-token
  direction on a Yes/No bank.
- **The covariance-matched null flips as many items.** The effect is norm and anisotropy, not
  direction.
- **D and C2 respond identically at every α.** No deception-specific mechanism at the direction
  level; the object is renamed *output-override direction* throughout the writeup.

**Forking paths.** α, ℓ, and the choice of direction are three continuous knobs over which a
positive result can always be found. The grid in §4 and the bar above are fixed here and are
not extended after looking. Anything found outside this grid is a follow-up, labelled
exploratory, on fresh items.

## 8. Language discipline (PLAN2 §10, applied here)

- `d_J` is **the direction the J-lens says separates the two answer tokens at layer ℓ**. It is
  not "the model's belief" and it is not "a truth vector".
- Ablating it and seeing the output move shows that **this direction is used by the
  computation that selects the answer token**, and nothing more.
- Until D separates from C2, the object is an **output-override direction**. The word
  "deception" is not attached to any Exp 1 result before that separation exists.

## 9. Risks specific to Exp 1

- **E1 — layer-index off-by-one between hook and lens.** The single most likely way to produce
  a confident wrong number here. §5.1 is a hard gate for exactly this.
- **E2 — the Yes/No confound.** Every item has the same two answer tokens, so `d_J` may be a
  Yes/No direction shared across the bank. §5.3's wrong-item control is the test, and it is a
  control the steering literature routinely omits.
- **E3 — `J_ℓ` is corpus-average (R5).** A direction derived from it may miss precisely the
  contextual content a per-item lie consists of. This bounds `d_J`'s ceiling and is the caveat
  to quote if `d_DiM` beats it.
- **E4 — RMSNorm.** Gemma normalizes the residual before each block, so a large α is partly
  renormalized away and the dose is not linear in effect. α is reported in units of `σ_ℓ` and
  the dose–response curve is reported rather than a single α.
- **E5 — n = 4.** The positive subgroup is four items. No p-value on the flip rate will mean
  anything; the result is a per-item table plus the nulls, and it is reported that way.
- **E6 — two tokenization paths in the existing stack.** `HFLensModel.encode` tokenizes with
  `add_special_tokens` at its default (True) and `from_hf(force_bos=True)` sets
  `add_bos_token=True`; `Reader.slot_probs` passes `add_special_tokens=False`. `items.render`
  returns a chat-templated string that already carries `<bos>`, so the lens may see a
  **double-BOS** sequence where 04's gate saw one — meaning 04's curves and 04's gate answers
  could have come off different token sequences. jlens' own docstring hedges that
  `add_bos_token` "may have no effect for some fast-tokenizer configurations", so this is
  unknown until run. `intervene` routes everything through `encode`, so Exp 1 is internally
  consistent regardless; §5.1 check 5 reports which way it went, and if the paths differ it
  is a note about 04 for the writeup rather than a bug here.
