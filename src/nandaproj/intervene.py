"""Exp 1 — direction-level causal edits on the residual stream at a located layer.

[`misc/exp1_spec.md`](../../misc/exp1_spec.md) is the spec; PLAN2.md 12 is the
registration. 04 established that the honest answer carries J-lens mass at
L19-22 under a deceptive prompt and is gone by the output. This module asks
whether that mass is **load-bearing**: build the direction in `h_l` that the
J-lens itself says separates the two answer tokens, edit it, and read the
emitted token.

Three things here are deliberate and are the reason this is a module rather
than notebook cells.

**1. Every handle comes from the lens's own wrapper.** `jlens.hooks.
ActivationRecorder` hooks `model.layers[l]` and takes the block's *forward
output* (tuple-unwrapped). So `h_l` is the output of block `l`, and this module
hooks the identical `reader.model_jlens.layers` list. An off-by-one between the
hook and the lens is the single most likely way to produce a confident wrong
number here (spec E1), and taking the same object is how it is ruled out rather
than argued about. `self_check` verifies it numerically anyway.

**2. One tokenization path.** `HFLensModel.encode` tokenizes with
`add_special_tokens` at its default (True) and `from_hf(force_bos=True)` sets
`add_bos_token=True`, while `Reader.slot_probs` passes
`add_special_tokens=False`. `items.render` returns a chat-templated string that
*already* carries `<bos>`, so those two paths can differ by a leading token.
Everything here goes through `encode` -- the lens's path -- so the direction and
the intervention are built on the same token sequence. `self_check` reports
whether the two paths actually differ on this tokenizer.

**3. The readout is not assumed.** `lens.apply` computes
`unembed(transport(h, l))` = `lm_head(final_norm(J_l h))`, with an optional
logit softcap. There is a **norm between `J_l` and `W_U`**, so the closed form
`J_l^T W_U^T (e_H - e_D)` is wrong in general. `d_jlens` takes the gradient
through the lens's own readout instead, which is exact and stays correct if the
composition changes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from nandaproj import items as items_mod
from nandaproj.items import Condition, Item
from nandaproj.lens_readout import MIN_ANSWER_MASS

ANSWER_SLOT = -1

#: Doses for the `add` edit, in units of the layer's mean residual norm. Both
#: signs: +alpha pushes toward the honest answer, -alpha away from it, and a
#: direction that only works in one sign is a different object from one that
#: works in both.
ALPHAS = (-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0)

#: The layer grid. 25 is `l*` from 04; the rest bracket it. Fixed in the
#: registration -- see PLAN2.md 12 forking-paths.
LAYERS = (16, 18, 20, 22, 24, 25, 26, 28)


# --------------------------------------------------------------------------
# Handles -- all of them from the lens wrapper, none reconstructed
# --------------------------------------------------------------------------

def blocks(reader):
    """The residual blocks `ActivationRecorder` hooks. `h_l` is `blocks[l]`'s output."""
    return reader.model_jlens.layers


def encode(reader, prompt: str):
    """Token ids by the lens's own path, so one sequence serves both passes.

    `lens.apply` truncates at `max_seq_len` silently (R7); `Reader.readout`
    already raises on that and so does every caller here, via `check_length`.
    """
    return reader.model_jlens.encode(prompt, max_length=reader.max_seq_len)


def check_length(reader, prompt: str) -> int:
    """Token count, or a loud failure if the lens would truncate the slot away."""
    n = int(encode(reader, prompt).shape[-1])
    if n >= reader.max_seq_len:
        raise ValueError(
            f"prompt is {n} tokens; encode truncates at {reader.max_seq_len} and "
            f"position {ANSWER_SLOT} would no longer be the answer slot"
        )
    return n


# --------------------------------------------------------------------------
# Reading and editing the stream
# --------------------------------------------------------------------------

@contextmanager
def _recording(reader, layers: Sequence[int]):
    """`ActivationRecorder` over `blocks(reader)`, imported not reimplemented."""
    from jlens.hooks import ActivationRecorder

    with ActivationRecorder(blocks(reader), at=list(layers)) as rec:
        yield rec


#: An edit maps one `[d_model]` residual vector to another. Kept this narrow on
#: purpose: an edit that could see the whole sequence could silently touch a
#: position other than the slot, and the position is a registered choice.
Edit = Callable[["object"], "object"]


def ablate(direction) -> Edit:
    """Project `direction` out of the residual: `h - (d_hat . h) d_hat`.

    Asks whether the component is **necessary**. Note this removes whatever the
    stream currently holds along `d`, which is not the same as setting the
    honest-vs-lie readout to zero -- the lens readout is affine in `h` only up
    to the final norm.
    """
    import torch

    d = torch.as_tensor(direction).float()
    d = d / d.norm()

    def edit(h):
        # `unsqueeze(-1)` so this works on a single `[d_model]` slot vector and
        # on a whole `[seq, d_model]` block alike -- the damage measure edits
        # every position, and a shape-specific edit would need a Python loop
        # over the sequence there.
        d_ = d.to(h.device)
        return h - (h.float() @ d_).unsqueeze(-1) * d_

    return edit


def add(direction, alpha: float, sigma: float) -> Edit:
    """`h + alpha * sigma * d_hat`. Asks whether the component is **sufficient**.

    `sigma` is the layer's mean residual norm at the slot, so `alpha` is a dose
    in units of the stream's own scale and is comparable across layers. Gemma
    normalizes before each block, so effect is *not* linear in alpha (spec E4)
    -- which is why the dose-response curve is the reported object and no single
    alpha is.
    """
    import torch

    d = torch.as_tensor(direction).float()
    d = d / d.norm()

    def edit(h):
        return h + alpha * sigma * d.to(h.device)

    return edit


def replace(source) -> Edit:
    """Overwrite the slot residual with one captured elsewhere -- the H->D patch.

    The upper bound on what any single-layer edit at `l` can do: if patching the
    whole honest residual in does not move the output, no direction inside it
    will either.
    """
    import torch

    src = torch.as_tensor(source)

    def edit(h):
        return src.to(h.device).to(h.dtype).expand_as(h)

    return edit


@contextmanager
def editing(reader, edits: dict[int, Edit], position: int = ANSWER_SLOT):
    """Apply `edits[layer]` to the residual at `position` on the next forward.

    A forward hook that *returns* a value replaces the block's output, which is
    what makes this an intervention rather than an observation. HF blocks return
    either a tensor or a tuple whose first element is the hidden state; both are
    handled, and the tuple is rebuilt rather than mutated in place.
    """
    handles = []
    try:
        for layer, edit in edits.items():
            def hook(_module, _inputs, output, _edit=edit):
                tensor = output if _is_tensor(output) else output[0]
                new = tensor.clone()
                new[0, position, :] = _edit(tensor[0, position, :]).to(tensor.dtype)
                return new if _is_tensor(output) else (new, *output[1:])

            handles.append(blocks(reader)[layer].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _is_tensor(obj) -> bool:
    import torch

    return torch.is_tensor(obj)


def slot_probs(reader, prompt: str, edits: dict[int, Edit] | None = None,
               position: int = ANSWER_SLOT) -> np.ndarray:
    """The model's own next-token distribution at `position`, under `edits`.

    One forward pass. Reproduces `lens.apply`'s `model_logits` exactly -- the
    final block's output through `unembed` -- rather than calling the HF model
    separately, so the edited and unedited numbers come off the same path.
    """
    import torch

    check_length(reader, prompt)
    ids = encode(reader, prompt)
    final = reader.model_jlens.n_layers - 1

    with torch.no_grad(), _recording(reader, [final]) as rec:
        with editing(reader, edits or {}, position=position):
            reader.model_jlens.forward(ids)
        h = rec.activations[final][0, position].detach()
        logits = reader.model_jlens.unembed(h)
    return torch.softmax(logits.float(), dim=-1).cpu().numpy()


def capture(reader, prompt: str, layers: Sequence[int],
            position: int = ANSWER_SLOT) -> dict[int, object]:
    """`{layer: h_l[position]}`, detached. One forward pass, no edits."""
    import torch

    check_length(reader, prompt)
    ids = encode(reader, prompt)
    with torch.no_grad(), _recording(reader, layers) as rec:
        reader.model_jlens.forward(ids)
        return {l: rec.activations[l][0, position].detach().clone() for l in layers}


def residual_pool(reader, prompts: Iterable[str], layer: int, max_tokens: int = 4000):
    """Residuals at **every** position over `prompts`, for the covariance null.

    Slot-only residuals would give a rank-`n_items` estimate of a 2560-dimensional
    covariance -- 11 vectors, so a "covariance-matched" draw would be a random
    combination of eleven points and trivially unlike a real residual. Pooling
    positions gives thousands of vectors for the same forward passes.
    """
    import torch

    rows = []
    for prompt in prompts:
        check_length(reader, prompt)
        with torch.no_grad(), _recording(reader, [layer]) as rec:
            reader.model_jlens.forward(encode(reader, prompt))
            rows.append(rec.activations[layer][0].detach().float().cpu())
        if sum(r.shape[0] for r in rows) >= max_tokens:
            break
    return torch.cat(rows, dim=0)[:max_tokens]


# --------------------------------------------------------------------------
# The directions
# --------------------------------------------------------------------------

def d_jlens(reader, h, layer: int, id_honest: int, id_lie: int,
            precise: bool = True):
    """The direction in `h_l` that moves the J-lens honest-vs-lie logit gap.

    `grad_h [ lens_logit(a_H) - lens_logit(a_D) ]`, taken through the lens's own
    readout `lm_head(final_norm(J_l h))`. Autograd rather than the closed form
    `J_l^T W_U^T (e_H - e_D)` because of the **norm between `J_l` and `W_U`**,
    which that form silently drops, and because `unembed` may apply a logit
    softcap.

    Per item, because the answer tokens are.

    `precise=True` (the default) takes the gradient end to end in float32, using
    only the two unembedding rows it needs. `unembed` casts its input to the
    lm_head's bf16 inside, which puts ~4e-3 relative error on the gradient --
    the same order as the chance cosine `1/sqrt(d_model)` = 0.0198 at
    `d_model=2560`. That is fine for a *readout*, which is what the lens is, but
    not for the direction the intervention steers along, and it is enough error
    to make `cos(d_J, h)` unreadable against chance. `precise=False` keeps the
    bf16 path so the two can be compared.
    """
    import torch

    x = torch.as_tensor(h).detach().float().requires_grad_(True)
    with torch.enable_grad():
        r = reader.lens.transport(x, layer)
        if precise:
            head = reader.model_jlens._lm_head
            normed = reader.model_jlens._final_norm(r).float()
            rows = head.weight[[id_honest, id_lie]].float().to(normed.device)
            gap = _softcap(normed @ rows[0], reader) - _softcap(normed @ rows[1], reader)
        else:
            logits = reader.model_jlens.unembed(r)
            gap = logits[id_honest].float() - logits[id_lie].float()
        grad, = torch.autograd.grad(gap, x)
    return grad.detach()


def _softcap(logit, reader):
    """`unembed`'s logit softcap, applied per logit as `unembed` applies it.

    It is monotone, so it cannot change the sign of a gap or the ordering the
    solve in `set_gap` relies on -- but it is applied to each logit *before* the
    difference is taken, and capping the difference instead would be a different
    (wrong) function. Gemma 3 leaves this unset; Gemma 2 does not.
    """
    import torch

    cap = reader.model_jlens._logit_softcap
    return cap * torch.tanh(logit / cap) if cap else logit


def lens_gap(reader, h, layer: int, id_honest: int, id_lie: int) -> float:
    """The J-lens `logit(a_H) - logit(a_D)` at one residual.

    Only the two rows of the unembedding are touched, so this is a `d_model`
    matmul and two dot products rather than a 262k-vocabulary projection --
    which is what makes the scalar solve in `set_gap` cheap enough to run per
    item, per layer.
    """
    import torch

    with torch.no_grad():
        r = reader.lens.transport(torch.as_tensor(h).float(), layer)
        head = reader.model_jlens._lm_head
        normed = reader.model_jlens._final_norm(r).float()
        rows = head.weight[[id_honest, id_lie]].float().to(normed.device)
        gap = _softcap(normed @ rows[0], reader) - _softcap(normed @ rows[1], reader)
    return float(gap)


def bisect(f, target: float, lo: float, hi: float, iters: int = 40) -> float:
    """The `x` in `[lo, hi]` with `f(x) = target`, assuming `f` is monotone there.

    Kept as a plain scalar routine with no torch in it so the search logic is
    testable off the box; `set_gap` supplies the model-dependent `f`. Returns
    the nearer endpoint when the bracket does not contain the target, so a
    caller gets the closest achievable edit rather than an exception -- an
    unreachable target is a fact about the layer, not an error.
    """
    f_lo, f_hi = f(lo) - target, f(hi) - target
    if f_lo * f_hi > 0:
        return lo if abs(f_lo) < abs(f_hi) else hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (f(mid) - target) * f_lo > 0:
            lo, f_lo = mid, f(mid) - target
        else:
            hi = mid
    return 0.5 * (lo + hi)


def set_gap(reader, h, layer: int, ids: tuple[int, int], target: float,
            direction=None, span: float = 50.0) -> tuple[Edit, float]:
    """An edit that moves `h` along `d_J` until the lens reads `target`.

    **Why this and not `ablate` for a gradient direction.** The lens readout
    `lm_head(final_norm(J_l h))` is scale-invariant in `h` -- `transport` is
    linear and RMSNorm divides by the RMS -- so it is homogeneous of degree zero
    and `grad_h . h = 0` identically (Euler). `d_jlens` **is** that gradient, so
    projecting it out of `h` removes ~nothing: `ablate(d_jlens(...))` is a no-op
    by construction, not an intervention that failed. Observed directly: the
    measured `|cos(d_J, h)|` runs 0.04x to 1.2x the chance level `1/sqrt(d_model)`.

    Moving *along* the gradient is well defined, and this solves for how far:
    the returned `alpha` is in units of `||h||`, so the edit's size is reportable
    on the same scale as the `add` sweep. `target=0` makes the lens indifferent
    between the two answers; `target=-gap` flips what it reads.

    Returns `(edit, alpha)`. An `alpha` at the edge of `+-span` means the target
    was not reachable within the bracket -- a fact about the layer worth
    reporting, not an error.
    """
    import torch

    h = torch.as_tensor(h)
    d = unit(direction if direction is not None else
             d_jlens(reader, h, layer, *ids)).to(h.device)
    scale = float(h.float().norm())

    alpha = bisect(lambda a: lens_gap(reader, h.float() + a * scale * d, layer, *ids),
                   target, -span, span)
    return add(d, alpha, scale), alpha


def d_difference_in_means(honest, deceptive):
    """`mean h(H) - mean h(D)` -- the standard steering vector, as the baseline.

    This is to `d_jlens` what the logit lens is to the J-lens in 04: the control
    that says whether the Jacobian did any *causal* work, or whether any sensible
    difference vector would have moved the output just as far.
    """
    import torch

    return (torch.stack([torch.as_tensor(h).float() for h in honest]).mean(0)
            - torch.stack([torch.as_tensor(h).float() for h in deceptive]).mean(0))


def random_directions(n: int, d_model: int, pool=None, seed: int = 0):
    """`n` null directions: covariance-matched when `pool` is given, else isotropic.

    Covariance-matched is the null that matters. The residual stream is strongly
    anisotropic, so a norm-matched *isotropic* vector is mostly orthogonal to
    everything the model uses and is far too easy to beat -- beating it is not
    evidence that a direction is special. Both are returned by the caller so the
    gap between the two nulls is visible rather than assumed (spec 5.3).

    Draws are `L z` with `L` the Cholesky factor of the pooled covariance, which
    matches second moments without pretending to be a real residual.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d_model, generator=gen)
    if pool is None:
        return z
    centered = pool.float() - pool.float().mean(0, keepdim=True)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    cov = cov + 1e-4 * torch.eye(d_model) * float(cov.diagonal().mean())
    return z @ torch.linalg.cholesky(cov).T


def unit(direction):
    import torch

    d = torch.as_tensor(direction).float()
    return d / d.norm()


def rescale_to(direction, reference):
    """`direction` rescaled to `‖reference‖`, so a null is norm-matched by construction.

    The scale is taken out as a Python float: the nulls are drawn on the CPU
    while `reference` is an item's `d_J` on the GPU, and a 0-dim CUDA tensor
    times a CPU vector is a device error rather than a broadcast.
    """
    import torch

    return unit(direction) * float(torch.as_tensor(reference).float().norm())


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

@dataclass
class Trial:
    """One (item, condition, edit) cell. The unit that lands in `results/`.

    `p_honest` is renormalized within the two legal answers and `mass` is how
    much of the full vocabulary they held at all -- the same pair 04's gate
    reports, and for the same reason: an edit that pushes the model out of
    format has not made it honest, and only `mass` can tell those apart.
    """

    item_id: str
    condition: str
    kind: str                 # baseline | ablate | add | patch
    direction: str            # jlens | dim | null_cov | null_iso | wrong_item | -
    layer: int | None
    alpha: float
    p_honest: float
    p_lie: float
    mass: float
    emitted: str
    answer: str
    subgroup: str             # legible | not_legible
    seed: int | None = None

    @property
    def truthful(self) -> bool:
        """Did the model land on the honest answer among the legal ones?"""
        return self.p_honest > self.p_lie


def measure(reader, item: Item, condition: Condition, *, kind: str, direction: str,
            layer: int | None, alpha: float, edits: dict[int, Edit] | None,
            subgroup: str, seed: int | None = None) -> Trial:
    """Run one cell of the grid and read the slot."""
    from nandaproj import lens_readout as lr

    prompt = items_mod.render(reader.tok, item, condition)
    probs = slot_probs(reader, prompt, edits)
    slot = lr.slot_from_probs(reader, probs, item.answers)

    per = {a: float(sum(probs[i] for i in lr.answer_token_ids(reader.tok, a)))
           for a in item.answers}
    mass = sum(per.values()) or 1.0
    return Trial(
        item_id=item.item_id, condition=condition, kind=kind, direction=direction,
        layer=layer, alpha=alpha,
        p_honest=per.get(item.answer_honest, 0.0) / mass,
        p_lie=per.get(item.answer_lie, 0.0) / mass,
        mass=slot.mass, emitted=slot.emitted, answer=slot.answer,
        subgroup=subgroup, seed=seed,
    )


def flip_rate(trials: Sequence[Trial], baseline: dict[str, Trial]) -> float:
    """Fraction of items whose top legal answer moved from `a_D` to `a_H`.

    Measured against each item's own unedited run, not against the bank: an item
    that was already answering honestly under the edit's condition cannot be
    "flipped back" and would otherwise inflate the rate.
    """
    eligible = [t for t in trials
                if t.item_id in baseline and not baseline[t.item_id].truthful]
    if not eligible:
        return float("nan")
    return float(np.mean([t.truthful for t in eligible]))


@dataclass
class CellStats:
    """One grid cell (a layer x condition x kind group), summarised honestly.

    `p_honest` is renormalized within the two legal answers, so it is only
    meaningful while the answers still hold some of the distribution. An edit
    large enough to push the model out of the answer format leaves both answers
    at ~0 and their ratio at ~0.5 -- which prints as "the model is perfectly
    undecided" and is nothing of the kind. `mass` is the number that separates
    those two readings, and `degenerate` is the flag that says which one you are
    looking at.

    That distinction is not hypothetical here: `set_gap`'s flip target needs
    steps of 20-50 ||h|| (the lens readout is degree-0 homogeneous, so moving
    along d_J only rotates h and the gap asymptotes at g(d_hat)), and a step of
    45 has replaced the residual rather than steered it.
    """

    n: int
    p_honest: float          # mean over items
    flip: float              # fraction whose top legal answer moved to a_H
    mass: float              # MEDIAN over items -- a mean hides one wrecked item
    alpha: float             # mean |alpha|, in units of ||h||

    @property
    def degenerate(self) -> bool:
        """The answers hold so little that `p_honest` is a ratio of nothing."""
        return self.mass < MIN_ANSWER_MASS

    def __str__(self) -> str:
        return (f"{self.p_honest:>5.2f} {self.flip:>5.2f} {self.mass:>5.3f} "
                f"{self.alpha:>4.1f}{'!' if self.degenerate else ' '}")


def summarize(trials: Sequence[Trial], baseline: dict[str, Trial]) -> CellStats:
    """`CellStats` for one group of trials. Empty group -> all-NaN, not a crash."""
    if not trials:
        return CellStats(0, float("nan"), float("nan"), float("nan"), float("nan"))
    return CellStats(
        n=len(trials),
        p_honest=float(np.mean([t.p_honest for t in trials])),
        flip=flip_rate(trials, baseline),
        mass=float(np.median([t.mass for t in trials])),
        alpha=float(np.mean([abs(t.alpha) for t in trials])),
    )


CELL_HEADER = "p_H  flip  mass  |a|"


def perplexity(reader, text: str, edits: dict[int, Edit] | None = None,
               position: int | None = None) -> float:
    """Held-out-text perplexity with the hook live (PLAN2.md R8).

    `position=None` applies the edit at **every** position, which is the right
    damage measure: a slot-only edit on a prompt whose slot is the last token
    cannot affect any earlier prediction, so a slot-only perplexity would always
    come back clean and would be a control that cannot fail.
    """
    import torch

    ids = encode(reader, text)
    final = reader.model_jlens.n_layers - 1
    slice_edits = edits or {}

    handles = []
    try:
        if slice_edits:
            for layer, edit in slice_edits.items():
                def hook(_module, _inputs, output, _edit=edit):
                    tensor = output if _is_tensor(output) else output[0]
                    new = tensor.clone()
                    # `position=None` edits the whole sequence in one shot --
                    # every `Edit` in this module is shape-general over the last
                    # axis, so no loop over token positions is needed.
                    if position is None:
                        new[0] = _edit(tensor[0]).to(tensor.dtype)
                    else:
                        new[0, position, :] = _edit(tensor[0, position, :]).to(tensor.dtype)
                    return new if _is_tensor(output) else (new, *output[1:])

                handles.append(blocks(reader)[layer].register_forward_hook(hook))

        with torch.no_grad(), _recording(reader, [final]) as rec:
            reader.model_jlens.forward(ids)
            logits = reader.model_jlens.unembed(rec.activations[final][0]).float()
    finally:
        for handle in handles:
            handle.remove()

    loss = torch.nn.functional.cross_entropy(logits[:-1], ids[0, 1:])
    return float(torch.exp(loss))


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_trials(trials: Sequence[Trial], path: str | Path) -> Path:
    """Write after every block of the grid, never only at the end.

    04 section 7 records why: a save cell at the bottom of a notebook only runs
    when nothing interesting went wrong, which is the opposite of when it is
    needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(t) for t in trials], indent=1))
    return path


def load_trials(path: str | Path) -> list[Trial]:
    return [Trial(**row) for row in json.loads(Path(path).read_text())]


# --------------------------------------------------------------------------
# V-I1 -- the gate that has to pass before any number above means anything
# --------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class SelfCheck:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def __str__(self) -> str:
        lines = [f"  {'PASS' if c.passed else 'FAIL'}  {c.name}\n        {c.detail}"
                 for c in self.checks]
        head = "V-I1 " + ("passed" if self.ok else "FAILED -- nothing below is interpretable")
        return head + "\n" + "\n".join(lines)


def self_check(reader, prompt: str, layer: int = 25, tol: float = 1e-3) -> SelfCheck:
    """Every invariant Exp 1 rests on, as assertions against the live model.

    There are no local unit tests for this half of the module -- it needs a
    model, a tokenizer and a fitted lens, which live on the box. This function
    is the substitute, and it is a hard gate rather than a diagnostic: if the
    hook does not read the tensor the lens reads, every intervention below is
    applied one block away from where it is reported.
    """
    import torch

    out = SelfCheck()

    # 1. Does the hook see the same h_l the lens does? Same module list, so this
    #    should be exact -- but "should be" is how off-by-ones survive.
    h = capture(reader, prompt, [layer])[layer]
    j_hand = reader.model_jlens.unembed(reader.lens.transport(h.float(), layer))
    jl, ml, _ = reader.lens.apply(reader.model_jlens, prompt, layers=[layer],
                                  positions=[ANSWER_SLOT])
    delta = float((torch.softmax(j_hand.float().cpu(), -1)
                   - torch.softmax(jl[layer][0].float(), -1)).abs().max())
    out.checks.append(Check(
        f"hooked h_{layer} reproduces lens.apply's J-lens distribution",
        delta < tol, f"max |dP| = {delta:.2e} (tol {tol:g})"))

    # 2. Does the unedited forward reproduce the lens's own model_logits? If not,
    #    the baseline every edit is compared against is not the model's output.
    probs = slot_probs(reader, prompt)
    ref = torch.softmax(ml[0].float(), -1).numpy()
    d_model_out = float(np.abs(probs - ref).max())
    out.checks.append(Check(
        "unedited slot_probs reproduces lens.apply's model_logits",
        d_model_out < tol, f"max |dP| = {d_model_out:.2e} (tol {tol:g})"))

    # 3. Is a no-op edit actually a no-op? Catches a hook that returns the wrong
    #    tuple element, or clones into the wrong position.
    noop = slot_probs(reader, prompt, {layer: (lambda x: x)})
    d_noop = float(np.abs(noop - probs).max())
    out.checks.append(Check(
        "an identity edit changes nothing",
        d_noop < 1e-6, f"max |dP| = {d_noop:.2e}"))

    # 4. Does an edit at layer l reach the output at all? A hook that silently
    #    fails to replace the output looks exactly like a null result.
    big = slot_probs(reader, prompt, {layer: (lambda x: x * 0.0)})
    d_zero = float(np.abs(big - probs).max())
    out.checks.append(Check(
        f"zeroing the slot residual at L{layer} moves the output",
        d_zero > 1e-3, f"max |dP| = {d_zero:.2e}"))

    # 5. The two tokenization paths. Not fatal either way -- but if they differ,
    #    04's J-lens curves and 04's gate answers came off different sequences,
    #    and that is worth knowing before it is inherited.
    lens_ids = encode(reader, prompt)[0].tolist()
    slot_ids = reader.tok(prompt, return_tensors="pt",
                          add_special_tokens=False).input_ids[0].tolist()
    same = lens_ids == slot_ids
    out.checks.append(Check(
        "lens.encode and Reader.slot_probs tokenize identically",
        same,
        "identical" if same else
        f"lens {len(lens_ids)} tokens vs slot_probs {len(slot_ids)}; "
        f"lens head={[reader.tok.decode([i]) for i in lens_ids[:3]]!r} "
        f"slot head={[reader.tok.decode([i]) for i in slot_ids[:3]]!r}"))

    return out
