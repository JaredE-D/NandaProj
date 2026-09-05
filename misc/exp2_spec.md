# Exp 2 — Which components close the gap? (notebook `05_attribution`)

**Spec, for approval. Not yet registered.** On approval, §8 of this file is appended to
[`PLAN2.md`](../PLAN2.md) as the next free section after Exp 1's, and this file becomes the
implementation reference.

**Status:** drafted 2026-09-03. **Implements:** PLAN2 §4.3, gate **V5**, and the first half
of **V6**. **Depends on:** [`04_belief_readout.ipynb`](../notebooks/04_belief_readout.ipynb)
complete (gate + curves + gated bank in `results/`).
**Cost estimate:** ~50 min GPU on the current 4090, ≈ $0.32. No new box.

**Relation to [Exp 1](exp1_spec.md).** Exp 1 is a *direction*-level test at a located layer:
edit the J-lens honest-vs-lie direction, see if the token moves. Exp 2 is a *component*-level
test: find which heads and MLPs, when replaced with their honest-run values, restore `a_H`.
Neither depends on the other's result — they are causal tests at different resolutions and
either can run first. They **do** share one GPU and one `results/` directory; §7 says how.

---

## 1. What Exp 2 is, in one sentence

04 located the flip at **L25** and the honest answer's peak legibility at **L19–22**. Exp 2
asks *which components* carry out the flip, by ranking every attention head and MLP block on
whether patching its honest-run output into the deceptive run recovers `a_H`.

## 2. What 04 established, and the three constraints it imposes

| | |
|---|---|
| `a_H` legible under D at some layer | **4/11** — PF03, PF05, SF04, SF05. Peak `P_J(a_H)` ≈ 1.00 at **L19–22** |
| `a_H` never legible under D | **7/11** — peak `P_J(a_H)` median 2.9e-06 |
| crossover `ℓ*` | **L25**, sd = 0, 74 % depth |
| C2 (instructed inversion) | crosses at **L25** too, on 4 items |
| H, C1 | `a_H` leads at the top on 11/11 |

**(a) Two item families, fixed in advance.** `LEGIBLE` = the 4; `NEVER` = the 7. Every number
in Exp 2 is reported for both. The `NEVER` family is not filler — if patching recovers `a_H`
there too, the belief was present in a form the J-lens could not read, which is a finding
about the *instrument*, and it is invisible if those items are dropped for having no signal.

**(b) The window is a prediction, not a filter.** DLA and patching run over the **whole**
stack. L24–25 is what §8 predicts, and pre-selecting it would make the prediction
unfalsifiable and draw the null from a pre-selected pool (PLAN2 §7.4).

**(c) C2 crosses at the same layer as D.** The deflationary reading is already live at layer
resolution. V6 is decided on *component sets*, so C2 is a first-class condition here, not an
afterthought run if time allows.

## 3. Framework: nnsight, and why (settled empirically)

`infra/probe_env.py --all` on the box, 2026-09-03, `transformers` 5.16.1:

| | nnsight 0.7.0 | transformer_lens 3.8.1 |
|---|---|---|
| per-head split | exact, `4.8e-07` | structural (`hook_z` is `[b, p, head, d_head]`) |
| no-op is identity | **True** | n/a — it is a different forward pass |
| intervention lands | Δ = 4.06 | Δ = 3.56 |
| agrees with the model 04 measured | **by construction — it *is* that module** | `max|Δ log-prob| = 0.875`, unattributed |

TL re-implements the architecture and folds layernorms; it warns that this is lossy in bf16,
and `HookedTransformer.from_pretrained` is deprecated in 3.8. Its 0.875 may be bf16 noise on
junk tokens or real divergence — and diagnosing that is work that buys nothing, because 04's
`ℓ*` = 25 was measured on the HF object. **nnsight runs the real module.** TL is the fallback
if nnsight blocks on something; raw forward hooks are the fallback to that.

Three architecture facts the probe pinned down, each of which is a bug if assumed instead:

- `text_config.layer_types` names the local/global alternation directly → **R4** enforced
  from config, never from a hand-derived 5:1 pattern.
- Gemma 3 has `post_attention_layernorm` **between** attention and the residual add, so the
  per-head contribution is *not* linear in head outputs → §5.1.
- GQA: `o_proj` input width is `n_q_heads × head_dim`, which is **not** `d_model`. Head
  slicing uses `head_dim` from config.

## 4. Components, metric, and conditions

**Component pool.** For every layer `ℓ` in `range(n_layers)`: each attention head
`(ℓ, h)`, taken as its slice of the `o_proj` input; and the MLP block output `(ℓ, mlp)`.
On 4b that is `34 × n_heads + 34` ≈ 300 components. Each carries its `layer_type`
(`sliding_attention` / `full_attention`) so heads are never pooled across the two (R4).

**Position.** The answer slot only (position −1). PLAN2's design constraint is that `a_H` and
`a_D` occupy the identical slot; patching elsewhere is a different experiment.

**Metric.** `LD = logit(a_H) − logit(a_D)` at the answer slot. For a patch of component `c`:

```
recovery(c) = (LD_patched − LD_D) / (LD_H − LD_D)
```

0 = the patch did nothing; 1 = it fully restored the honest run's logit difference. Reported
per item, and as a median across each family — never as a mean across families.

**Conditions.** `D` (primary) and `C2` (V6) get the full sweep. `C1` gets only the top set
found under D, as the persona control — a persona that suppresses nothing should show no
recovery structure.

## 5. Method

### 5.1 DLA — proposes, does not rank

Per-component contribution to `LD`, at the answer slot, using the **frozen-scale** treatment:
take each RMSNorm's scale factor from the actual forward pass and hold it constant, which
restores linearity through Gemma 3's post-sublayer norms.

**Corrected during implementation, and the correction matters.** Frozen-scale is *not* an
approximation of the observed run. `RMSNorm(x) = x · s(x) · (1 + w)` and `s` is a **scalar**,
so distributing it over the heads that sum to `x` is an identity — the decomposition is
**exact for the run it describes**. What it is not is a *counterfactual*: removing a head
would change `s`. That is precisely why PLAN2 §4.3 has DLA propose and patching rank, and it
is why the DLA table never enters a causal claim.

The completeness residual is therefore a different instrument than first assumed: it cannot
detect a loose approximation, because there is none. It detects a **wrong graph** — logit
softcapping making the unembed nonlinear, or a sublayer writing somewhere other than where
the code thinks. The notebook prints `Σ(head DLA) + Σ(MLP DLA) + embed − true LD` per item,
plus a per-layer split error. The **5 %** bar stands: over it, the DLA table is *indicative
only* and plays no part in any claim, and the first thing to suspect is a bug. Patching is
unaffected either way — it is causal and assumes no linearity.

### 5.2 Patching H→D — the causal ranking

For each item, cache the H run's component outputs at the answer slot; re-run D replacing one
component; record `recovery`. ~300 components × 11 items × 2 conditions ≈ 6 600 forward
passes. A `tqdm` bar, per the progress rule, and incremental saves after every item — 04 lost
its first gate result to a save cell at the bottom.

### 5.3 The null, fixed before the ranking is looked at (V5)

Two nulls, both pre-registered here:

1. **Wrong-source patch.** Patch component `c` with the H-run activation of a *different,
   randomly chosen item*. Same component, same perturbation magnitude, wrong content. This
   is the null that separates "this component carries the honest answer" from "perturbing
   this component moves the output".
2. **Random sets of equal size** (PLAN2 §5), for any set-level claim: `n = 200` random
   component sets of size `k`, drawn from the same pool.

**Bars, fixed now:**

| claim | bar |
|---|---|
| a single component is a hit | median `recovery` ≥ **0.20** across the family **and** above the **95th** percentile of its own wrong-source null |
| a set of size `k` is real | its median `recovery` exceeds the **99th** percentile of the 200 random sets of size `k` |
| `k` for the V6 comparison | the smallest set whose cumulative recovery ≥ 0.5, capped at **10** |

### 5.4 Reverse patching D→H

For components that clear §5.3 only. Patch the deceptive run's component into the honest run
and check the honest answer degrades. A component that recovers `a_H` one way and does
nothing the other is a one-run artefact (PLAN2 §4.3.3).

### 5.5 V6 — does C2 separate from D?

Top-`k` sets under D and under C2, compared by Jaccard overlap, against a null of two random
size-`k` sets. **If the sets are the same, that is the headline** and the object is renamed
*output-override components* throughout (PLAN2 §10). This spec does not get to decide that;
§8 predicts and the run answers.

## 6. Code architecture

New module `src/nandaproj/attribution.py`, following `lens_readout.py`'s split exactly:

| layer | contents | tested |
|---|---|---|
| torch-free | `Component` (layer, kind, head, layer_type), `recovery()`, the null percentiles, set selection, Jaccard, save/load of the results table | **locally**, `tests/test_attribution.py`, no GPU |
| torch | `Patcher` — wraps an nnsight `LanguageModel`, discovers module paths by walking `named_modules` (never hardcoded, since 4b is the multimodal wrapper), caches and patches at the slot | on the box, via the notebook's assert cells |

`notebooks/05_attribution.ipynb` is **self-contained**: it loads its own model, reads 04's
`gated_bank_*.json` and `belief_readout_all_*.npz` off `results/`, and imports nothing from
04's kernel. Section 1 is three asserts before any science — head-split exactness, no-op patch
identity, DLA completeness — each printing a number, not a boolean alone.

## 7. Operating alongside 04 / Exp 1

One GPU, 23.5 GiB, and a 4b bf16 model is ~8.6 GiB plus activations.

- 05 runs in its **own kernel**. Cell 0 prints free VRAM and **refuses to load** below a
  threshold, so a collision fails in two seconds instead of OOM-ing four minutes in.
- Shut 04's (or Exp 1's) kernel before the 4b run. The notebook says so in cell 0.
- `results/` filenames are namespaced `attribution_*` — no collision with Exp 1's artifacts.
- `just down` when the box is idle. Budget: $9.20 of $50 spent as of drafting.

## 8. Pre-registration (appended to PLAN2 on approval)

**To be registered before the §5.2 sweep runs. Nothing below is edited after that.**

### 8.1 Predicted result

The components whose patching recovers `a_H` under D are **sparse** — fewer than 10 clear the
§5.3 bar — and concentrate at **L24–25**, the crossover 04 located, with a secondary group at
L19–22 where `a_H` peaks. Attention heads dominate MLPs.

**Where this is most likely wrong, stated in advance:** the effect may be **distributed** —
no single component clears 0.20 and only sets of 20+ recover anything. That would refute
"specific heads" as the right resolution (PLAN2 §7.3, bullet 3) and is reported as a miss,
not repaired by lowering the bar.

### 8.2 Predicted signs

| quantity | prediction |
|---|---|
| top components' layer | **L24–25**, matching `ℓ*` |
| head vs MLP | heads dominate |
| `LEGIBLE` (n=4) vs `NEVER` (n=7) recovery | higher in `LEGIBLE`; **non-zero in `NEVER`** |
| best single component vs wrong-source null | well above |
| D set vs C2 set (Jaccard) | **overlapping but not identical** — the falsifiable one |
| local vs global attention heads (R4) | top set contains both; no prediction of a split |
| DLA completeness residual | < 5 % of `|LD|` |

### 8.3 What would change my mind

- **No component clears the bar at any layer.** The edit is distributed; §4.4's ablation has
  nothing to ablate.
- **The top set is identical under D and C2.** No deception-specific mechanism. Headline.
- **Recovery is as strong in `NEVER` as in `LEGIBLE`.** Then patching is not recovering the
  thing the J-lens reads, and the two instruments are measuring different objects — which
  would undercut the bridge from 04 to 05 that this whole notebook assumes.
- **Top components sit below L15.** The lie is not a late edit; PLAN2 §7.3's first bullet
  arrives by a different route.

## 9. Task decomposition

| # | task | where | verified by |
|---|---|---|---|
| 1 | `Component`, `recovery`, nulls, set selection, Jaccard — torch-free | `attribution.py` | `tests/test_attribution.py`, local — **done**, 46 tests |
| 2 | `Patcher`: module discovery, slot caching, single-component patch; `patch_sweep` / `joint_recovery_fn` / `dla_sweep` drivers | `attribution.py` | notebook §1 asserts, plus `tests/test_attribution_pipeline.py` on a planted ground truth — **done** |
| 3 | Notebook §0–1: VRAM guard, model load, three asserts | `05_attribution.ipynb` | numbers printed |
| 4 | Notebook §2: load 04's gated bank + curves, rebuild the two families | " | family sizes = 4 / 7 |
| 5 | Notebook §3: DLA over all components + completeness residual | " | residual < 5 % |
| 6 | Notebook §4: wrong-source null, computed **before** the real sweep | " | null distribution printed |
| 7 | Notebook §5: H→D patching sweep, D and C2, incremental save | " | `attribution_*.npz` on disk |
| 8 | Notebook §6: ranking, bars applied, D→H reverse check on survivors | " | table + heatmap |
| 9 | Notebook §7: V6 Jaccard, C1 control | " | the V6 number |

Order matters at step 6: the null exists on disk before the ranking is looked at, which is
what makes §5.3's bars a pre-registration rather than a description.

## 10. Risks

- **R6 (PLAN2)** — ~300 components × 2 conditions is a multiple-comparisons machine. The
  wrong-source null and the fixed bars are the answer; confirmation is 06's ablation.
- **n = 4 in the `LEGIBLE` family.** Every positive claim about it is a claim about four
  items. Stated in the writeup wherever the number appears.
- **nnsight 0.7 is a young API.** Mitigated by §1's asserts running on every kernel start —
  a silent no-op write is the failure mode that looks exactly like "this head doesn't matter".
- **Frozen-scale DLA.** §5.1's residual bar; and DLA never ranks.
- **bf16.** `recovery` is a ratio of logit differences in bf16. The notebook reports
  `LD_H − LD_D` per item; where that denominator is small the ratio is unstable and the item
  is flagged rather than averaged in.
