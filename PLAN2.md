# PLAN2 — Where does a lie get made? Belief, report, and the edit between them

**Target:** MATS application (Neel Nanda stream). **Budget:** ~20 h, <$50 (est. ~$5).
**Status:** planned, 2026-09-02. **Supersedes** [`misc/plan1old.md`](misc/plan1old.md).

## 0. What changed and why

Plan 1 asked where the *confidence–faithfulness gap* sits relative to J-space, and split the
outcome into **availability** failure (the signal isn't in the reportable subspace) vs **routing**
failure (it is, and isn't used). That frame was sound but its primary arm inferred from **absence**
— a low ρ(v_acc) is a null on an instrument, not a finding (old §4.1, R8), and half the plan was
controls existing to make a null interpretable.

This plan keeps the frame and changes the phenomenon so the inference runs **positive**. Under an
instructed lie we know what the honest answer is, because we can ask the same model the same
question honestly. So the claim is no longer "X is missing from the reportable subspace" but
**"X is present in the reportable subspace at layer ℓ, and absent from the output"** — a detection,
with a located gap between two measurements, not an unfalsifiable silence.

Everything already built carries over: the J-lens `readout`/`layer_table`/`track`/`probe` stack in
[`07_jlens_experiments.ipynb`](notebooks/07_jlens_experiments.ipynb), the single-token slot trick in
[`synthetic.py`](src/nandaproj/synthetic.py), the model/lens config, the infra. Notebooks 03–06 were
unrun stubs for plan 1 and are in [`notebooks/archive/`](notebooks/archive/).

## 1. The question

Can we find cases where the model **internally represents X and asserts ¬X**, and then locate two
things separately:

1. **the belief** — where in the stack the true answer lives, and crucially whether it is still
   sitting in the *reportable* subspace (J-space) at the moment the model says something else;
2. **the edit** — the specific heads and MLP blocks that take that reportable belief and keep it
   out of the output.

**Hypothesis (registered in §7).** The model learns true belief states with good accuracy and
reasonable calibration, *and separately* learns sub-circuits that modify verbalization. Those
sub-circuits sit late — near the end of the stack — and are sparse enough to name specific
components. If so, ablating them should make the model tell the truth it was already holding.

The safety-relevant object is **self-report you can trust**. A model whose honest belief survives
to the top of the stack and is edited out by a nameable circuit is a very different object from one
whose deception rewrites the computation from the first layer: the first is detectable and
patchable at inference time, the second is not.

## 2. Two arms

Both arms are the same measurement on different sources of ground truth. Arm A is primary.

### Arm A — instructed / role-play deception (primary)

The system prompt asks for a lie; the same question asked plainly gives the belief. Ground truth is
**given by construction**, not inferred.

Design constraint inherited from 01b/07: **make the answer a single token.** Yes/No questions, or a
fixed A/B/C/D letter, put the honest answer and the lie at the *identical* slot, so the J-lens
readout and the model's own distribution are directly comparable with no sequence scoring in
between. This is the same reason the confidence work runs on a 1–9 scale.

| | Condition | What it is |
|---|---|---|
| **H** | honest | plain system prompt, plain question |
| **D** | deceptive | persona instructed to conceal or misstate ("you are selling this car and must not admit the fault") |
| **C1** | persona-truthful | persona of matched length and oddity, instructed to answer **truthfully** |
| **C2** | instructed inversion | "always answer with the opposite of the true answer" — no deceptive frame, same output override |
| **C3** | no-belief | questions the model genuinely cannot answer (verified in H) — nothing to suppress |

C2 is load-bearing and is the **most likely deflationary result**: if the components found under D
are the same ones found under C2, there is no deception-specific circuit, only generic
instruction-following/negation. That outcome must be reported as the headline, not a footnote.

**Harm axis (secondary, cheap).** The same items in a low-stakes framing (a white lie about a gift)
and a high-stakes one (concealing a safety-relevant fact). Question: one circuit or two? This is an
extra split over an existing item bank, so it costs generation time and nothing else. It is not
allowed to expand into its own project.

### Arm B — dishonesty about confidence

The lie is about the model's own epistemic state rather than about the world: the internal
correctness signal says one thing, the verbalized number says another. This reuses
[`synthetic.py`](src/nandaproj/synthetic.py), the digit slot, and the behavioral pipeline in
[`02_behavioral.ipynb`](notebooks/02_behavioral.ipynb) as they stand.

Weaker on ground truth — "what the model believed about its own accuracy" is a probe estimate, not
a given — which is exactly why it is the second arm. Its value is that it connects the result to
the calibration literature and to work already done, and it tests whether the Arm A components also
carry confidence-verbalization edits.

## 3. Why J-lens rather than a truth probe

Linear probes for truth and for lying already work: Azaria & Mitchell (a model's internal state
knows when it is lying), Burns et al. (CCS), Marks & Tegmark (geometry of truth), Zou et al.
(representation engineering, honesty direction), Goldowsky-Dill et al. (strategic-deception probes),
and the sandbagging literature. *Detecting* a lie from activations is not novel and this project
must not claim it is. **Verify every citation in §11 before using it — they are from memory.**

What that work does not do is ask whether the suppressed truth is **reportable at the moment of
suppression**. A probe direction says the information is *present*; it says nothing about whether it
is in the subspace the model can actually speak from. J-space is an independently-derived answer to
exactly that question, and it converts a vague "the model knows" into a specific claim:

> at layer ℓ, under the deceptive prompt, the honest answer carries mass in J-space — the model
> could have said it — and by the final layer it does not.

That gap is the object to localize. Plus the second half nobody in that literature does: naming the
heads and MLPs that close it, and showing causally that removing them reopens it.

## 4. Method

`h_ℓ(x)` = residual stream at layer ℓ at the **answer slot** (the prefilled position where the
single-token answer goes). `J_ℓ = E[∂h_final/∂h_ℓ]`, J-space(k) = top-k right singular vectors.
`a_H` = the model's honest answer token; `a_D` = the token it emits under D.

**4.0 Setup.** `gemma-3-4b-it`, bf16, the **`-it`** pre-fitted lens, debug on `gemma-3-270m-it`.
Unchanged from plan 1 §5.0, and already working in 07.

**4.1 Item bank and the behavioral gate.** Build paired H/D items with single-token answers.
Gate: the model must (a) answer correctly and stably under H, and (b) *actually lie* under D on a
usable fraction of items. **No lying, no project** — this is the analogue of plan 1's R1 and it is
checked in the first working session, before anything else is built. Escalation if 4B will not lie
convincingly: stronger personas first, `gemma-3-12b-it` only if that fails (~24 GB, ~$0.50/hr — but
note 07 records that 12b does not fit alongside activations on a 24 GB 4090, so this means a
different box, not a different flag).

**4.2 Belief readout — where the truth lives under D.** For each item, per layer:
`P_J(a_H)`, `P_J(a_D)`, and the same under the vanilla logit lens as the control that says whether
the Jacobian is doing any work (07 already computes both side by side). The signature being looked
for is `P_J(a_H)` rising and then being *overtaken* by `P_J(a_D)` late, with the crossover layer
`ℓ*` being the thing to localize around. Report per-item curves as well as means — 07 §6 already
demonstrates that a tier mean can hide a bimodal population.

Run the same sweep for H, C1, C2, C3. C3 is the floor: no belief, so no crossover.

**4.3 Attribution — which components close the gap.** J-lens is a layer-level instrument; it does
not say which head wrote what. On top of it, on the logit difference `logit(a_H) − logit(a_D)` at
the answer slot:

1. **Direct logit attribution** over every head and MLP output — one pass, ranks candidates.
2. **Activation patching H→D**, component by component: patch the honest run's head/MLP output into
   the deceptive run and measure recovery of `a_H`. This is the causal ranking; DLA only proposes.
   ~n_layers × n_heads plus MLPs, a few hundred forward passes — minutes on a 4090.
3. **Patching in the opposite direction (D→H)** for the components that survive, to check the
   effect is not an artefact of one run's noise.

**4.4 Ablation — the money experiment.** Zero or mean-ablate the top-ranked component set under D.
**Does the model tell the truth?** Report the dose-response as components are added to the set, the
fraction of items flipped back, and the damage to unrelated behaviour (perplexity on held-out text,
accuracy on H items) so that "it stopped lying" cannot be "it stopped working".

**4.5 Does the belief stay in J-space at ℓ\*?** The plan-1 question, now asked at a located layer
rather than everywhere: project `h_ℓ*` and the honest-answer direction against J-space(k), sweep
`k`, report a curve. Answers whether the suppressed truth is reportable-but-unrouted.

**4.6 Generality.** Do the same components appear across item types, across the harm axis, and in
Arm B? A circuit found on one item bank and nowhere else is a fit to that bank.

## 5. Controls, and the deflationary readings they test

| Control | Deflationary story it tests |
|---|---|
| **C2 instructed inversion** | "It's a negation/instruction-following circuit, not a deception circuit" — the strongest and most likely alternative |
| **C1 persona-truthful** | "It's persona-adoption, not lying" |
| **C3 no-belief items** | "The signature appears whenever the prompt is odd, belief or not" |
| **logit lens alongside J-lens** | "J-space isn't doing any work; the token was just there" (07 already runs this) |
| **held-out text perplexity under ablation** | "Ablation broke the model rather than making it honest" |
| **random component sets of equal size** | "Any k heads would do this" — the null for §4.4 |
| **covariance-matched random directions** | anisotropy, for any ρ in §4.5 (plan 1 R3) |

## 6. Verification gates

| | Gate |
|---|---|
| V1 | Lens loads, Judy reads (07 §3 — already passing) |
| V2 | **Model actually lies under D**, on a usable fraction of items, with a stable honest answer under H — *hard gate, first session* |
| V3 | `a_H` is legible in the J-lens above the logit-lens baseline at *some* layer under D |
| V4 | A crossover layer `ℓ*` exists and is stable across items — not an artefact of one prompt |
| V5 | Patching recovers `a_H` for at least one component well above the random-set null, with the null and the effect-size bar fixed **before** looking at the ranking |
| V6 | **C2 separates from D** — the component set for deception is not identical to the set for plain inversion. If it does not separate, that is the result |
| V7 | Ablation flips items back to truth without wrecking held-out perplexity or H-condition accuracy |
| V8 | Components transfer across item types / harm framing / Arm B |

V6 is the gate that decides whether the word "deception" is allowed in the writeup.

## 7. Pre-registration

**Registered 2026-09-02, before Arm A data collection. Do not edit below this line.**
Plan 1's registration is preserved unrun in [`misc/plan1old.md`](misc/plan1old.md) §7 and is
superseded, not deleted.

### 7.1 Predicted result

**The belief survives and is edited late.** `P_J(a_H)` rises through the mid-stack under D, remains
non-trivial into the upper layers, and is overtaken by `a_D` in the **last ~25%** of the stack. The
components responsible are **sparse** — a nameable handful of heads plus one or two MLP blocks —
and ablating them restores the honest answer on a substantial fraction of items.

**Where this prediction is most likely wrong, stated in advance:** "very late" is a specific
commitment and the steering/refusal literature more often finds behaviour-controlling directions at
roughly 60–70% depth, not at the top. A crossover at mid-stack would not refute the framework but
*would* refute the stated prediction, and will be reported as a miss.

### 7.2 Predicted signs

| Quantity | Prediction |
|---|---|
| `P_J(a_H)` under D vs under C3 (no belief) | clearly higher |
| `P_J(a_H)` vs logit-lens `P(a_H)` under D | higher — J-space carries the suppressed truth |
| crossover layer `ℓ*` | in the last quarter of the stack |
| patching best component vs random-set null | well above |
| deception component set vs C2 inversion set | **overlapping but not identical** — this is the falsifiable one |
| ablation | flips a substantial fraction of D items back to `a_H`, perplexity roughly intact |
| ρ(honest-answer direction) at `ℓ*` vs covariance-matched null | above — reportable but unrouted |
| harm axis | same components, different magnitude |

### 7.3 What would change my mind

- **`a_H` never becomes legible under D at any layer.** The lie is not an edit applied to a belief;
  the deceptive prompt changes the computation from early on. Real result, different paper, and it
  kills the "detect the suppressed truth at inference time" application.
- **D and C2 give the same components.** No deception-specific mechanism — instruction-following
  with a lie-shaped input. This is the headline if it happens.
- **Ablation does not restore truth at any set size**, or only does so by breaking the model. The
  effect is distributed; "specific heads" was the wrong resolution.
- **Crossover is not stable across items.** Nothing is localized; the per-item curves were noise
  averaged into a story.

### 7.4 Forking-paths discipline

Inherited from plan 1 R9, and it bites harder here because the space of personas is unbounded.
**The item bank and the five conditions H/D/C1/C2/C3 are fixed before the §4.2 sweep and are never
extended.** A persona that "works better", discovered after seeing curves, is a follow-up on fresh
items reported as exploratory. Component ranking is likewise a search over hundreds of candidates:
the random-set null and the bar for "above null" are fixed at V5, before the ranking is looked at.

## 8. Budget and schedule

Generation is small — item banks in the hundreds, five conditions, single-token answers. Patching is
a few hundred forward passes. The expensive thing remains idle instances. `just down`.

| Hours | Work | Notebook | Risk |
|---|---|---|---|
| 1–2 | Item bank, H/D pairing, **V2 behavioral gate** | `03_deception_items` | **the project lives or dies here** |
| 3–6 | J-lens belief readout, all conditions, `ℓ*` (V3–V4) | `04_belief_readout` | low — 07's infra already does this |
| 7–11 | DLA + activation patching, nulls fixed first (V5) | `05_attribution` | new instrument, budget for rough edges |
| 12–15 | Ablation, C2 comparison, damage checks (V6–V7) | `06_ablation` | the real result |
| 16–17 | ρ at `ℓ*`, k-sweep (§4.5) | `04_belief_readout` | inherited from plan 1 |
| 18–19 | Arm B + transfer (V8) | `02_behavioral`, 07 | first thing cut if time runs out |
| 20 | Writeup and plots | — | — |

Notebook numbers 03–06 are reused for the new pipeline; the old stubs live in
[`notebooks/archive/`](notebooks/archive/). [`07_jlens_experiments.ipynb`](notebooks/07_jlens_experiments.ipynb)
stays the playground and stays ungated.

**Cut order if hours run short:** Arm B first, then the harm axis, then §4.5. Arm A through V7 is
the deliverable and must not be traded against anything.

## 9. Risks

- **R1 — 4B may not lie convincingly.** Instruction-following at 4B under a deceptive persona is
  not guaranteed. V2, first session. Escalation costs a bigger box, not just a flag (07).
- **R2 — prompted lying ≠ emergent dishonesty.** The one caveat that must be in the abstract. A
  circuit that implements "say the thing the system prompt asked for" is evidence about instructed
  deception and only suggestive about the deception anyone is actually worried about. Do not
  overclaim the bridge; C2 is what keeps this honest.
- **R3 — the "belief" is a probe/lens readout, not a belief.** §10.
- **R4 — Gemma 3 alternates local sliding-window and global attention layers** (5:1). Head
  attribution must not silently mix the two; report them separately if the top set clusters.
- **R5 — J-lens is a corpus-average Jacobian** (plan 1 §4.1, and 07 §1 quotes `n_prompts`). It may
  discard exactly the contextual content a per-item lie consists of. This bounds §4.2's sensitivity
  and is the caveat to quote if the readout is weaker than expected.
- **R6 — patching over hundreds of components is a multiple-comparisons machine.** Random-set null,
  fixed bar, and confirmation by ablation rather than by rank.
- **R7 — `jacobian-lens` is unmaintained**, and `lens.apply` is unbatched and truncates silently at
  512 tokens (07 §2 guards this). Deceptive prompts are longer than plan 1's; **check the length
  guard before trusting any Arm A readout.**
- **R8 — ablation damage.** A component set that stops the lying by degrading the model is not a
  finding. Held-out perplexity and H accuracy are reported with every ablation number.
- **R9 — persona search.** §7.4.

## 10. Language discipline

Carried forward from plan 1 §10 and extended, because this topic invites worse writing than the
last one did.

- A probe or lens readout is **not a belief**. Write "the honest-answer token carries mass in the
  J-lens readout at layer ℓ", not "the model knows the truth and hides it".
- **"Lie circuit" must be earned.** Until V6 and V7 both pass, the object is "components whose
  patching recovers the honest answer under the deceptive prompt". If C2 does not separate, the
  honest name is *output-override components* and the writeup uses that name throughout.
- Do not write **"the model intends"**, "chooses to deceive", or "wants". Nothing here measures
  intent; it measures which activations, when replaced, change which token is emitted.
- Do not write **"architectural"** (plan 1 §10). Location is not permanence.
- Instructed deception is named as such in every claim. Never let "we found where the model lies"
  stand unqualified when what was studied is "where the model does what the system prompt asked".

## 11. Deliverable

Short writeup: the question, the two-measurement design that makes it a detection rather than a
null, the belief-readout curves, the located components, the ablation result, and the C2 comparison
stated wherever the word deception appears. Plots: `P_J(a_H)` vs `P_J(a_D)` per layer with the
logit-lens baseline and the C3 floor; patching heatmap over heads × layers; ablation dose-response
with damage curves; ρ vs `k` at `ℓ*`. Repo reproducible from `just up` to figures.

A negative result — no legible belief, or no separation from C2 — is written up with the same care.

**Citations to verify before the writeup** (recalled from memory, ids deliberately omitted): Azaria
& Mitchell, "The Internal State of an LLM Knows When It's Lying"; Burns et al., CCS /
discovering latent knowledge; Marks & Tegmark, the geometry of truth; Zou et al., representation
engineering; Goldowsky-Dill et al., detecting strategic deception with linear probes; van der Weij
et al., sandbagging. Carried from plan 1 and already cited in the repo: 2603.25052, 2604.01457,
2607.08046, Anthropic's Global Workspace post and the `jacobian-lens` repo.
