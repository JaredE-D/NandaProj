> **SUPERSEDED, 2026-09-02.** This is plan 1, kept verbatim and unrun. The project pivoted to
> instructed-deception localization; see `PLAN2.md` in the repo root. Section 7 below is a
> pre-registration that was never tested and is preserved rather than deleted -- deleting a
> registration when the question changes is the thing registration exists to prevent. Notebooks
> 03-06 written against this plan are in `notebooks/archive/`.

# PLAN — Where does the confidence–faithfulness gap live relative to J-space?

**Target:** MATS application (Neel Nanda stream). **Budget:** ~20 h, <$50 (est. ~$5). **Status:** planned.

## 1. The question

Verbalized confidence diverges from accuracy, and this is known to be a *readout* failure, not a
knowledge deficit: accuracy and verbalized-confidence are both linearly decodable from the residual
stream but sit in near-orthogonal directions (cos < 0.04). Nobody has explained why.

Separately, Anthropic's Jacobian lens identifies **J-space** — the verbalizable subspace, ~6–10% of
activation variance. This project locates both directions relative to J-space to separate two
mechanisms: the accuracy signal is **absent** from the reportable subspace (W1), or **present but
not routed** into verbalization (W2). They imply different classes of fix.

## 2. Prior work

| Work | Established | Leaves open |
|---|---|---|
| [2603.25052](https://arxiv.org/html/2603.25052) *Closing the Confidence-Faithfulness Gap* | Both directions linearly encoded, cos < 0.04; joint prompting shifts alignment +0.26 → −0.63; CAA steering repairs ECE with no retraining | No mechanistic account of the orthogonality |
| [2604.01457](https://arxiv.org/pdf/2604.01457) *Wired for Overconfidence* | Circuits inflating verbalized confidence; confidence cached at answer-adjacent positions | Which representations are reportable |
| [2607.08046](https://arxiv.org/pdf/2607.08046) *Forecasters* | Internal probes beat verbalized confidence on ECE | Same |
| [Global Workspace](https://transformer-circuits.pub/2026/workspace/index.html) + [jacobian-lens](https://github.com/anthropics/jacobian-lens) | J-space = verbalizable subspace; **sufficiency** shown (inject → model reports); pre-fitted lenses released | Never applied to confidence. **Necessity unestablished** (§4.1) |

**Verified gap:** no confidence paper cites J-space or the workspace work; the J-space work never
touches confidence. **Novelty, conservatively:** not discovering the gap — localizing it against an
independently-derived subspace.

**The existing fix is external.** A ridge probe reads accuracy; isotonic regression calibrates it;
an inverse transfer function picks a steering strength α; a CAA vector built from
*verbalized-confidence* contrasts writes it. The accuracy direction is read, the verbalization
direction is written, and nothing inside the model connects them — the bridge is four steps of
experimenter scaffolding. That is a routing failure described operationally by a paper that never
names it, and it is independent support for W2. §5.5 is defined against it.

Note the distinction: **bypass** (use the probe output as the number) gives *the researcher*
calibration; **repair** (steer so the model says it) gives a model whose self-report can be trusted.
Only the latter is the safety-relevant object.

## 3. Why it matters given that steering already works

Steering shows both *endpoints* work. It says nothing about whether the missing middle is absent or
merely unused — and that determines which interventions can work:

| | `v_acc` outside J-space (W1) | `v_acc` inside J-space (W2) |
|---|---|---|
| Can prompting elicit calibrated self-report? | No | Plausibly yes |
| Would finetuning-for-honesty close it? | No — must change what enters the workspace | Yes — wiring present |
| Class of fix | Architectural / training-objective | Routing; maybe reward shaping alone |
| Is the external scaffold necessary? | Yes, structurally | No |

Calibration is not the object of interest — **self-report is.** A steering fix repairs one output
channel on one task; a structural result about where reportable information lives transfers to
introspection on goals, deception, and self-report under pressure. If W2 holds, the sharper question
is why RLHF never learned to use an available, reportable signal — plausibly because raters prefer
confident answers, making the gap *learned* rather than incidental.

This project will not solve calibration. It can say which *kind* of failure it is.

## 4. Outcome space

`h_ℓ(x)` = residual stream at layer ℓ, read at the **answer-adjacent position** (per 2604.01457).
`v_acc` = probe direction for answer correctness; `v_verb` = for verbalized confidence.
`J_ℓ = E[∂h_final/∂h_ℓ]`; J-space(k) = top-k right singular vectors; `ρ(v) = ‖P_{J(k)}v‖²/‖v‖²`.

### 4.1 The inferential asymmetry — read before designing anything

The workspace evidence is **sufficiency** (inject → reported). "Not in J-space ⇒ not reportable"
requires the **converse**, which nothing establishes. So **a low ρ(v_acc) is a null on an
instrument, not a finding.** Three benign explanations: `J_ℓ` is a corpus average, and per-answer
confidence is exactly the contextual content that average discards; `k` truncation may hide a
functional low-variance direction; `v_acc` is a noisy probe estimate. The covariance-matched
baseline shows ρ is below chance — not why.

### 4.2 The four worlds

| | Outcome | Mechanism | Inference |
|---|---|---|---|
| **W1** | `v_verb` in, `v_acc` out | availability failure | **weak** — needs §5.4 controls *and* §5.5 |
| **W2** | both in | routing failure — available, unused | **strong** — positive detection |
| **W3** | both out | lens tracking nothing relevant | uninterpretable; check setup |
| **W4** | `v_verb` out | setup broken | should not occur |

**Prior leans W2:** steering repairs without retraining, joint prompting *inverts* rather than
weakens alignment, and both endpoints are demonstrably functional — a pathway that exists and is
modulated, not information that is missing.

These are corners of a **continuum**, not bins: report a position in ρ with error bars, not a label.
ρ is a property of the representation *under a given elicitation*, which is why §5.1 sweeps five
prompt conditions. **Falsifies the framing:** W3, or both ρ indistinguishable from the null — still
a real result, constraining the workspace story.

Independent secondary test: under joint prompting (alignment −0.63), ρ should shift relative to
separate prompting. This one does not inherit W1's null-inference problem.

## 5. Method

**5.0 Setup.** `gemma-3-4b-it`, bf16, ~8 GB. Pre-fitted
[`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens/tree/main) —
**the `-it` lens, not base.** Debug the whole pipeline on `gemma-3-270m-it` before renting.

**5.1 Behavioral data.** MMLU primary (4-way MC, exact-match); TriviaQA secondary for probe
transfer. ~5000 items. Record answer, correctness, verbalized confidence, and residual activations
across a layer sweep at the answer-adjacent position.

Five prompt conditions, **fixed in advance and never extended (R9)**:

| | Condition | Purpose |
|---|---|---|
| P1 | **Separate** — confidence in a fresh turn | 2603.25052 baseline |
| P2 | **Joint** — answer + confidence in one pass | reasoning-contamination arm |
| P3 | **Introspective** — "how confident, and how do you know?" | invites self-report |
| P4 | **Third-person** — "how likely is *a model* to get this right?" | decouples report from self-model |
| P5 | **Deferred** — confidence after a filler turn | does the cached signal persist? |

Adding a sixth after seeing results invalidates §7; it becomes a follow-up on fresh data, reported
as exploratory.

**5.2 Probes (replication floor).** Logistic → `v_acc`, ridge → `v_verb`, split by item. Reproduce
`cos(v_acc, v_verb)` per layer and condition; probe AUROC vs verbalized-confidence AUROC. *Complete
and defensible on its own — this is the safety net.*

**5.3 J-space projection.** SVD of `J_ℓ`; pick `k` to match ~6–10% variance, then **sweep `k` and
report a curve, never a point.** ρ(v_acc), ρ(v_verb) across layers and conditions.

**5.4 Controls — load-bearing, not calibration.** Without these W1 is uninterpretable. Run at
identical layer, `k`, and probe methodology.
- **Positive control** (sentiment/topic — known verbalizable). If the lens picks it up at high ρ,
  a low ρ for `v_acc` becomes discriminative rather than an instrument failure. **This licenses any
  W1 claim.**
- **Negative control** (token position, sequence length) — sets the floor.
- **Covariance-matched random directions** — activation space is anisotropic; an isotropic null
  inflates significance and a reviewer will say so.
- **Confound check on `v_acc`:** within-topic analysis and MMLU → TriviaQA transfer. If it only
  encodes difficulty or topic, say so.

**5.5 Causal injection — internal bridge vs external scaffold.** Inject along `v_acc` into J-space:
does verbalized confidence move *without* the probe/isotonic/α scaffold? **Moves** → wiring present
and unused, W1 collapses toward W2. **Doesn't** → causal evidence for genuine absence, not an
instrument null. This applies the workspace paper's own sufficiency method to our direction, turning
the weak arm positive, and reuses machinery §5.3 already needs.

**5.6 Optional, only if all the above finishes.** Gemma Scope 2 SAEs for Gemma 3 4B: do
accuracy-relevant SAE features sit similarly relative to J-space? **The core project needs no SAEs.**

## 6. Verification gates

| | Gate |
|---|---|
| V1 | Lens loads; jlens readout gives sensible top tokens |
| V2 | Verbalized confidence has **non-degenerate spread** at 4B — *hard gate, R1* |
| V3 | Correctness probe AUROC meaningfully above chance |
| V4 | `cos(v_acc, v_verb)` small — replicates 2603.25052 qualitatively |
| V5 | Covariance-matched null computed **before** looking at ρ(v_acc), *and* the effect size counting as "meaningfully above null" fixed at the same moment, in units of the null's SD (R10) |
| V6 | **Positive control separates from negative under matched methodology.** If not, no null is interpretable — stop, do not report W1 |
| V7 | ρ stable across the `k` sweep and across layers |
| V8 | Cross-dataset probe transfer holds |
| V9 | (if run) Injection gives a monotone dose-response |

## 7. Pre-registration

**Registered 2026-08-31, before any data collection. Do not edit below this line.**

### 7.1 Predicted world

**W2 — both directions exist in J-space, but are not connected.**

Refinements registered at the same time:

- **Graded, not binary.** One of the two may sit only *weakly* in J-space. The prediction is that
  `ρ(v_acc) > null` — meaningfully above the covariance-matched baseline — while still plausibly
  `ρ(v_acc) < ρ(v_verb)`. W1 and W2 are corners of a continuum in ρ, not a dichotomy, and the
  result should be reported as a position on that continuum.
- **Prompt-dependence.** There exists a set of prompt formats under which **both** directions sit
  in J-space more strongly than under the default format. ρ is predicted to be a property of the
  *representation under a given elicitation*, not a fixed property of the model.

### 7.2 Predicted signs

| Quantity | Prediction |
|---|---|
| `ρ(v_verb)` vs. null | well above (near-tautological, low evidential weight — R6) |
| `ρ(v_acc)` vs. null | **above** — this is the W2 commitment and the falsifiable one |
| `ρ(v_acc)` vs. `ρ(v_verb)` | lower, possibly much lower, but not at floor |
| Joint vs. separate prompting | ρ changes; direction not predicted in advance |
| Prompt sweep (§5.1) | at least one format raises `ρ(v_acc)` above the default condition |
| Injection test (§5.5) | verbalized confidence **moves** — the internal wiring is present and unused |

### 7.3 What would change my mind

- `ρ(v_acc)` at or below the covariance-matched null **across all layers, all `k`, and all prompt
  conditions**, with the positive control (V6) clearly separating. That is W1, and it would mean
  the accuracy signal genuinely is not in the reportable subspace.
- Injection along `v_acc` producing no movement in verbalized confidence at any dose. That would
  say the external scaffold is doing work no internal path can, and would contradict 7.1 directly.
- V6 failing — positive and negative controls not separating — in which case no claim is licensed
  in either direction and the honest report is that the instrument could not discriminate.

### 7.4 Why registration matters here

The prompt-dependence clause in 7.1 creates a **garden-of-forking-paths risk**: searching over
prompt formats until one shows the predicted effect would manufacture a result. See R9. The prompt
set must be fixed before looking at ρ, and that fixed list belongs in §5.1 before data collection
begins.

## 8. Budget and schedule

`gemma-3-4b-it` bf16 ~8 GB is comfortable on a 4090. Lens fitting is **$0** (pre-fitted).
Generation (~5000 items × 5 conditions × 2 datasets) ~1–2 h; probes, SVD, projections, injection
sweep are minutes. Total GPU across all sessions including re-runs: ~5–7 h ≈ **$2**. The budget
risk is idle instances, not the experiment. `just down`.

| Hours | Work | Risk |
|---|---|---|
| 1–3 | Setup, lens loads (V1) | low |
| 4–8 | Behavioral data, all conditions (V2) | **R1 lives here** |
| 9–12 | Probes, reproduce orthogonality (V3–V4) | low — **replication floor reached** |
| 13–15 | Controls first (V5–V6), *then* projections | gates the rest |
| 16–18 | ρ across layers and `k`; injection (§5.5) | the real result |
| 19–20 | Writeup and plots | — |

Controls run **before** the main measurement, so a failed V6 redirects the remaining hours instead
of producing an uninterpretable null at hour 19.

## 9. Risks

- **R1 — verbalized confidence may be degenerate at 4B.** Reference work uses 7–8B. If the model
  says "90%" to everything, the phenomenon doesn't exist at this scale. Check at hour ~6 (V2).
  Escalation: `gemma-3-12b-it` (lens exists; ~24 GB, A6000 ~$0.50/hr). Comparison point:
  [2604.24070](https://arxiv.org/pdf/2604.24070) works on Gemma 3 4B.
- **R2 — J-space definition sensitivity.** Sweep `k`, report the curve.
- **R3 — anisotropy.** Covariance-matched nulls, never isotropic.
- **R4 — lens/model mismatch.** Base and instruct lenses are different files. Verify the pairing.
- **R5 — probe confounds.** Within-topic analysis and cross-dataset transfer.
- **R6 — high `ρ(v_verb)` is near-tautological.** It is by construction the direction that gets
  spoken. Never a standalone finding; it earns its place as the positive half of the contrast.
- **R7 — `jacobian-lens` is explicitly unmaintained.** Budget time for rough edges.
- **R8 — inference from absence.** The central methodological risk (§4.1). Mitigations: V6, §5.5.
- **R9 — forking paths in the prompt sweep.** With enough formats, one will clear the bar by
  chance — the easiest way to manufacture a false positive here. Mitigations: the five conditions
  are fixed before data collection and never extended; correct for five comparisons; report ρ for
  **all five** whatever they show. One condition clearing the bar after correction is weak
  evidence and must be described that way.
- **R10 — the graded prediction is a wide target.** Fix the effect size counting as "meaningfully
  above null," in units of the null's SD, when the null is computed (V5) — before seeing ρ(v_acc).

## 10. Language discipline

A probe direction is not a thought. Write "the accuracy-predictive linear direction at
answer-adjacent positions", not "what the model really believes."

**Do not write "architectural limit."** An earlier draft called miscalibration architectural rather
than a training artifact. That was wrong: both the availability of a signal and its routing into
speech are learned. Nothing here separates "cannot" from "did not learn to." The defensible claim is
about **location**, not permanence.

## 11. Deliverable

Short writeup: motivation, outcome space, method, controls, which world we landed in, honest
limitations — with the necessity/sufficiency asymmetry stated explicitly, not buried. Plots: ρ vs
layer with null band and both controls; ρ vs `k`; `cos(v_acc, v_verb)` per condition; injection
dose-response if run. Repo reproducible from `just up` to figures.

A negative result is publishable here and gets written up with the same care as a positive one.
