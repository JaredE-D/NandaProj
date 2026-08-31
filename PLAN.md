# PLAN — Where does the confidence–faithfulness gap live relative to J-space?

**Target:** MATS application project (Neel Nanda stream).
**Budget:** ~20 hours of work, <$50 compute (realistic estimate: ~$5).
**Status:** planned, not started.

---

## 1. One-paragraph summary

Language models verbalize confidence that diverges sharply from their accuracy, and it is
established that this is a *readout* failure rather than a knowledge deficit: the accuracy signal
and the verbalized-confidence signal are both linearly decodable from the residual stream but sit
in near-orthogonal directions. Nobody has explained *why*. Separately, Anthropic's Jacobian lens
identifies a subspace — J-space — that is the **verbalizable** part of the representation, ~6–10%
of activation variance. This project locates both directions relative to J-space to distinguish
two mechanisms: the accuracy signal is **absent** from the reportable subspace, or it is **present
but not routed** into verbalization. The two imply different classes of fix (§3), which is the
reason to run it.

---

## 2. Prior work

| Work | What it established | What it leaves open |
|---|---|---|
| [arXiv 2603.25052](https://arxiv.org/html/2603.25052) *Closing the Confidence-Faithfulness Gap* | Accuracy and verbalized-confidence directions both linearly encoded, cosine sim **< 0.04**. Joint prompting shifts alignment **+0.26 → −0.63**. CAA steering repairs ECE **with no retraining**. Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.1; MATH, MMLU, TriviaQA, TruthfulQA. | **No mechanistic account of why they are orthogonal.** |
| [arXiv 2604.01457](https://arxiv.org/pdf/2604.01457) *Wired for Overconfidence* | Circuits that causally inflate verbalized confidence on wrong answers. Verbal confidence computed automatically, **cached at answer-adjacent positions**. | No theory of which representations are reportable. |
| [arXiv 2607.08046](https://arxiv.org/pdf/2607.08046) *What LLM Forecasters Know but Don't Say* | Internal probes beat verbalized confidence on ECE. | Same. |
| [Verbalizable Representations Form a Global Workspace](https://transformer-circuits.pub/2026/workspace/index.html) + [jacobian-lens](https://github.com/anthropics/jacobian-lens) | J-space = verbalizable subspace, ~6–10% of variance. **Sufficiency** shown: inject into J-space → model reports it. Pre-fitted lenses released. | Never applied to confidence, calibration, or honesty. **Necessity never established** (§4). |

**Verified gap:** none of the confidence papers cite the Jacobian lens, J-space, global workspace
theory, or SAEs. The J-space work never touches confidence.

**Novelty claim, stated conservatively:** we are not discovering the confidence–faithfulness gap.
We are localizing it against an independently-derived subspace and testing two mechanisms for it.

### 2.1 How the existing steering fix actually works — the probe reads, steering writes

This matters enough to state precisely, because it is the baseline the project is defined against.

**Stage 1 — read.** A ridge probe on activations *at prompt completion, before generation*
predicts empirical accuracy. Isotonic regression calibrates its output on validation data. In
their words, the prediction is *"the target confidence for that question: what the model should
say, given what its activations reveal about its likelihood of being correct."*

**Stage 2 — write.** The CAA steering vector is built from **verbalized-confidence** contrasts,
**not** the accuracy direction: a per-question contrast of mean activations where stated
confidence > 0.75 against where it is < 0.25. A transfer function maps steering strength α to mean
verbalized confidence; given Stage 1's target, its **inverse** yields the α for that question. 50
samples per question. The model then speaks an actual number.

**The consequence.** The accuracy direction is *read*; the verbalization direction is *written*;
and nothing connects them inside the model. The link is closed **externally**, through four steps
of experimenter scaffolding: probe → isotonic calibration → inverse transfer function → α. The
"separate generation pass" limitation exists precisely because joint prompting inverts the
relationship (reasoning contamination), so the passes cannot be merged.

That is a routing failure described in operational detail by a paper that never uses the term. It
is independent support for **W2** (§4) — and it is what Phase 4b is defined against.

### 2.2 Two things both called "matching verbalized confidence to accuracy"

| | Method | Is the model's own report fixed? |
|---|---|---|
| **Bypass** (2607.08046) | Use the probe output *as* the confidence number | **No** — better number, model still says wrong things |
| **Repair** (2603.25052) | Steer so the model *says* the calibrated value | **Yes**, but only with the external scaffold |

For safety framing this distinction is the whole point. Bypass gives *the researcher* calibration.
Only repair gives a model whose self-report can be trusted.

---

## 3. Why this matters given that steering already works

**The objection, stated fairly:** ECE improves 14.9 → 3.7 (Llama) and 35.1 → 3.3 (Mistral) with no
retraining. If the gap is already patchable, why care where it lives?

**What "already fixed" is actually worth.** Narrowly: the Stage 1 probe is trained on
ground-truth correctness labels, so calibration is only available on distributions where the
answers are already known — backwards from the deployment case. It needs two passes and 50 samples
per question. Qwen only reached ECE 10.7, still badly calibrated. Evaluation is restricted to QA
with verifiable answers, their own stated limitation.

**But that is an engineering complaint, and it is not the justification.** The justification is
that the two worlds make **different predictions about which interventions can work**:

| | If `v_acc` is **outside** J-space (W1) | If `v_acc` is **inside** J-space (W2) |
|---|---|---|
| Can prompting elicit calibrated self-report? | No — inaccessible for report | Plausibly yes, with the right elicitation |
| Would finetuning-for-honesty close the gap? | No — must change what enters the workspace | Yes — the wiring is present |
| Class of fix required | Architectural / training-objective | Routing; potentially reward shaping alone |
| Is the external steering scaffold *necessary*? | Yes, structurally | No — it substitutes for something the model has |

"We can patch it" and "we do not know which class of fix is required" are compatible. Steering
shows both *endpoints* work. It says nothing about whether the missing middle is absent or merely
unused.

**The part that generalizes.** Calibration is not the object of interest — **self-report is.** The
safety question is whether a model can accurately report its own internal states and what limits
that; confidence is one conveniently gradeable instance. A steering fix repairs one output channel
on one task. A structural result about where reportable information lives transfers to
introspection on goals, to deception, and to self-report under adversarial pressure.

**If W2 holds, a sharper question follows:** the accuracy signal is available *and* reportable, and
verbalization still ignores it — so **why did RLHF not learn to use it?** The likely answer
connects to the sycophancy literature: raters prefer confident-sounding answers, so the reward
signal favors the disconnect. That would make the gap **learned rather than incidental**, which is
a stronger and more troubling claim than "the model doesn't know."

**Honest scoping.** This project will not solve calibration. It can say which *kind* of failure it
is, on a phenomenon well-characterized behaviorally and unexplained mechanistically, by crossing
two literatures that have not been crossed. For an application project, that is the right size.

---

## 4. Outcome space

Let `h_ℓ(x)` be the residual stream at layer `ℓ`, read at the **answer-adjacent position**
(inherited from 2604.01457).

- `v_acc` — unit direction of a probe predicting whether the model's answer is **correct**.
- `v_verb` — unit direction of a probe predicting the model's **verbalized** confidence.
- `J_ℓ = E[∂h_final/∂h_ℓ]`; `J-space(k)` = span of the top-`k` right singular vectors.
- `ρ(v) = ‖P_{J(k)} v‖² / ‖v‖²`.

### 4.1 The inferential asymmetry — read before designing anything

The workspace evidence is **sufficiency**: injecting content into J-space makes the model report
it. Concluding "not in J-space ⇒ not reportable" requires the **converse** (reportable ⇒ in
J-space), which no published work establishes.

Therefore **a low `ρ(v_acc)` is a null on an instrument, not a finding.** At least three things
predict it besides non-verbalizability:

- `J_ℓ` is an expectation over corpus text. Averaging is the lens's *deliberate* mechanism for
  separating "verbalizable in general" from "verbalized in this context." Confidence in *this
  particular answer* is contextual — exactly what that average is built to discard.
- `k` truncation: the direction may sit in a functional but low-variance part of the spectrum.
- `v_acc` is a noisy probe estimate, and ρ inherits the noise.

The covariance-matched baseline (§5.4) shows ρ is below chance. It cannot say why.

### 4.2 The four worlds

| | Outcome | Mechanism | Inference strength |
|---|---|---|---|
| **W1** | `v_verb` in, `v_acc` **out** | availability failure — absent from reportable subspace | **weak** — null on an instrument; needs §5.4 controls *and* §5.5 injection test |
| **W2** | **both in** | routing failure — available but unused; disconnected circuits | **strong** — a positive detection |
| **W3** | both out | lens not tracking anything relevant here | uninterpretable; check setup |
| **W4** | `v_verb` out | setup is broken | should not occur |

**Prior leans W2.** Steering repairs calibration without retraining; joint prompting *inverts*
alignment rather than weakening it; and the existing fix (§2.1) demonstrates both endpoints are
functional. All indicate a pathway that exists and is modulated, not information that is missing.

**Secondary test, independent of the above.** Under joint prompting — where alignment inverts to
−0.63 — ρ should change measurably relative to separate prompting. This test does not share W1's
null-inference problem.

**These are corners, not categories.** ρ is continuous, and the registered prediction (§7.1) is
explicitly a graded one: `v_acc` above the null but below `v_verb`. Report a position on the
continuum with error bars, not a world label. The four-world table is for structuring
interpretation, not for binning the result.

**ρ is a property of the representation under a given elicitation**, not a fixed property of the
model — which is why §5.1 sweeps five prompt conditions rather than two.

**What falsifies the framing:** W3, or both ρ values indistinguishable from the covariance-matched
null. Still a real result: a major reportability failure that J-space does not account for, which
constrains the global workspace story.

---

## 5. Method

### 5.0 Setup
- `gemma-3-4b-it`, bf16, ~8 GB. Pre-fitted lens from
  [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens/tree/main)
  — **use the `-it` lens, not the base one.**
- Debug the whole pipeline on `gemma-3-270m-it` before renting anything.

### 5.1 Behavioral data
- **MMLU** primary: 4-way MC, exact-match grading, no ambiguity.
- **TriviaQA** secondary, for cross-dataset probe transfer.
- ~5000 items.
- Record answer, correctness, verbalized confidence, residual activations across a layer sweep at
  the answer-adjacent position.

**Prompt conditions — fixed in advance, never extended after seeing ρ (R9).** The
prompt-dependence clause in §7.1 predicts ρ varies with elicitation, so this is a sweep rather
than two arms. Exactly these five, no more:

| # | Condition | Purpose |
|---|---|---|
| P1 | **Separate** — answer, then confidence in a fresh turn | the 2603.25052 baseline |
| P2 | **Joint** — answer and rate confidence in one pass | the reasoning-contamination arm |
| P3 | **Introspective** — "how confident are you, and how do you know?" | invites self-report explicitly |
| P4 | **Third-person** — "how likely is a model to get this right?" | decouples report from self-model |
| P5 | **Deferred** — answer, then confidence after an unrelated filler turn | tests whether the cached answer-adjacent signal (2604.01457) persists |

Five conditions × two datasets is the full grid. **Adding a sixth after seeing results
invalidates §7.** If a post-hoc condition looks irresistible, it is a follow-up experiment on
fresh data, reported as exploratory.

### 5.2 Probes (replication floor)
- Logistic probe → `v_acc`; ridge probe → `v_verb`. Split by item.
- **Reproduce the orthogonality**: `cos(v_acc, v_verb)` per layer, both conditions.
- Probe AUROC vs. verbalized-confidence AUROC.
- *Complete, defensible result on its own. This is the safety net.*

### 5.3 J-space projection
- SVD of `J_ℓ`. Pick `k` to match the ~6–10% variance figure, then **sweep `k` and report a curve,
  never a single point.**
- `ρ(v_acc)`, `ρ(v_verb)` across layers and both conditions.

### 5.4 Controls that make a null interpretable ← **load-bearing, not calibration**

Without these, W1 is uninterpretable and not worth spending hours on. Run at **identical layer,
`k`, and probe methodology** as the main measurement.

- **Known-verbalizable positive control** (sentiment, or topic). If the lens picks this up at high
  ρ under matched methodology, a low ρ for `v_acc` becomes a *discriminative* measurement rather
  than an instrument failure. **This comparison is what licenses any W1 claim.**
- **Known-non-verbalizable negative control** (token position, sequence length). Sets the floor.
- **Covariance-matched random directions.** Activation space is strongly anisotropic; an isotropic
  null would inflate significance and a reviewer will say so. Draw from the empirical activation
  covariance.
- **Confound check on `v_acc`:** within-topic analysis and MMLU → TriviaQA transfer. If it only
  encodes difficulty or topic, say so.

### 5.5 Causal injection test — internal bridge vs. external scaffold

Defined against §2.1. The existing pipeline builds an **external** bridge from `v_acc` to `v_verb`
(probe → isotonic → inverse transfer → α). This asks whether an **internal** one exists:

> Inject along `v_acc` into J-space. Does verbalized confidence move — *without* the
> probe/isotonic/α scaffold?

- **Moves** → the wiring is present and unused. W1 collapses toward W2; the external scaffold is
  substituting for a path the model already has.
- **Doesn't move** → causal evidence for genuine absence rather than an instrument null, and the
  scaffold is doing work no internal path can.

This applies the workspace paper's own sufficiency method to our own direction, converting the
weak arm of the experiment into a positive one, and reuses machinery §5.3 already requires.

### 5.6 Optional extension (only if everything above finishes)
Gemma Scope 2 has SAEs for Gemma 3 4B. Ask whether accuracy-relevant **SAE features** sit
similarly relative to J-space. **The core project needs no SAEs.**

---

## 6. Verification gates

- **V1** Lens loads; jlens readout on a known prompt gives sensible top tokens.
- **V2** Verbalized confidence has **non-degenerate spread** at 4B. *Hard gate — see R1.*
- **V3** Probe AUROC on correctness meaningfully exceeds chance.
- **V4** `cos(v_acc, v_verb)` small, replicating 2603.25052 qualitatively.
- **V5** Covariance-matched null for ρ computed **before** looking at `ρ(v_acc)` — *and* the
  effect size that counts as "meaningfully above null" fixed at the same moment, in units of the
  null's standard deviation (R10).
- **V6** **Positive control separates from negative control under matched methodology.** If not,
  the lens cannot discriminate at this layer/`k`, no null is interpretable — stop and reconsider
  rather than reporting W1.
- **V7** ρ stable across the `k` sweep and across layers.
- **V8** Cross-dataset probe transfer holds.
- **V9** (if run) Injection produces a monotone dose-response in verbalized confidence.

---

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

---

## 8. Compute and budget

| Item | Estimate |
|---|---|
| Model | `gemma-3-4b-it`, bf16, ~8 GB — comfortable on a 4090 |
| Lens fitting | **$0 — pre-fitted, downloaded** |
| Generation, ~5000 items × 5 prompt conditions × 2 datasets | ~1–2 h (short generations, batched) |
| Probes, SVD, projections, injection sweep | minutes |
| Total GPU across all sessions incl. re-runs | ~5–7 hours ≈ **$2** |

The budget risk remains idle instances, not the experiment. `just down`.

---

## 9. Time budget (~20 h)

| Hours | Work | Risk |
|---|---|---|
| 1–3 | Setup, lens loads, V1 | low |
| 4–8 | Behavioral data, both conditions, V2 | **R1 lives here** |
| 9–12 | Probes, reproduce orthogonality, V3–V4 | low — **replication floor reached** |
| 13–15 | Controls first (V5–V6), *then* projections | gates the rest |
| 16–18 | ρ across layers and `k`; injection test (§5.5) | the real result |
| 19–20 | Writeup and plots | — |

Controls run **before** the main measurement, so a failed V6 redirects the remaining hours instead
of producing an uninterpretable null at hour 19.

---

## 10. Risks

- **R1 — verbalized confidence may be degenerate at 4B.** Reference literature uses 7–8B models.
  If Gemma 3 4B says "90%" to everything, the phenomenon does not exist at this scale. **Check at
  hour ~6 (V2).** Escalation: `gemma-3-12b-it` (lens exists; ~24 GB bf16 — rent an A6000 at
  ~$0.50/hr). Comparison point: [arXiv 2604.24070](https://arxiv.org/pdf/2604.24070) works on
  Gemma 3 4B.
- **R2 — J-space definition sensitivity.** Sweep `k`, report the curve.
- **R3 — anisotropy.** Covariance-matched nulls, never isotropic.
- **R4 — lens/model mismatch.** Base vs instruct lenses are different files. Verify the pairing.
- **R5 — probe confounds.** Within-topic analysis and cross-dataset transfer.
- **R6 — `ρ(v_verb)` being high is near-tautological.** `v_verb` is by construction the direction
  that gets spoken. Never present it as a standalone finding; it earns its place only as the
  positive half of the W1/W2 contrast.
- **R7 — unmaintained reference implementation.** `jacobian-lens` is explicitly not maintained.
  Budget time for rough edges.
- **R8 — inference from absence.** The central methodological risk. Necessity is unestablished, so
  a low ρ has several benign explanations. Mitigations: V6 positive control, and §5.5.
- **R9 — forking paths in the prompt sweep.** §7.1 predicts *some* prompt format raises
  `ρ(v_acc)`. With enough formats, one will, by chance. This is the easiest way to manufacture a
  false positive in this design. Mitigations: the five conditions in §5.1 are fixed before data
  collection and never extended; correct for five comparisons when testing the prompt-dependence
  claim; report ρ for **all five** conditions whatever they show, not just the winner. A single
  condition clearing the bar after correction is weak evidence and should be described that way.
- **R10 — the graded prediction is harder to falsify than a binary one.** §7.1 predicts
  `ρ(v_acc)` sits *above the null but below* `ρ(v_verb)`, which is a wide target. Mitigation: fix
  the effect size that counts as "meaningfully above null" — in units of the covariance-matched
  null's standard deviation — at the same time the null is computed (V5), before looking at
  `ρ(v_acc)`.

---

## 11. Language discipline

A probe direction is not a thought. Write "the accuracy-predictive linear direction at
answer-adjacent positions", not "what the model really believes."

**Do not write "architectural limit."** An earlier draft claimed miscalibration was architectural
rather than a training artifact. That was wrong: both the availability of a signal and its routing
into speech are learned. Nothing here separates "cannot" from "did not learn to." The defensible
claim is about **location** — whether the accuracy signal is present in the subspace the report
pathway reads from — not about permanence.

---

## 12. Deliverable

- Short research writeup: motivation (§3), outcome space, method, controls, which world we landed
  in, honest limitations — with the necessity/sufficiency asymmetry stated explicitly rather than
  buried.
- Plots: ρ vs layer with null band and both control directions; ρ vs `k` sweep;
  `cos(v_acc, v_verb)` per condition; injection dose-response if run.
- This repo, reproducible from `just up` to figures.

A negative result is publishable here and should be written up with the same care as a positive
one.
