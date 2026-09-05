#!/usr/bin/env python3
"""Which interpreter is the kernel, and does nnsight actually work on it?

Run on the box, from any python:

    python3 /workspace/NandaProj/infra/probe_env.py           # imports only, no GPU
    python3 /workspace/NandaProj/infra/probe_env.py --model   # + a real nnsight trace

The interpreter you launch it with does not matter. Stage 0 finds the python
that Jupyter actually starts kernels with -- by reading the registered
kernelspec, not by guessing a venv path -- and re-execs itself under it. So the
versions printed below are the versions the notebook sees, which is the only
thing worth knowing.

`--model` is the part that decides the 05 architecture. Importing nnsight
proves nothing: what 05 needs is to *read* one attention head's contribution and
to *write* a different one back in. So this loads the 270m debug model (~0.6 GB,
seconds) and does exactly that, then checks the numbers:

  - the per-head split is exact         sum_h z_h @ W_O[h] == o_proj(z)
  - a no-op trace is the identity        logits unchanged, bit for bit
  - an intervention actually lands       zeroing a head moves the logits

If those three pass, hooks would be rebuilding a wheel. If any fails, the
failure is printed with its exception and we fall back -- TL, then raw hooks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

RULE = "-" * 72
REEXEC_FLAG = "NANDAPROJ_PROBE_REEXEC"


# -- stage 0: run under the kernel's interpreter ---------------------------

def kernel_python() -> tuple[str, str]:
    """(path, how we found it) for the python Jupyter starts kernels with."""
    import json

    roots = [
        Path("/usr/local/share/jupyter/kernels"),
        Path("/usr/share/jupyter/kernels"),
        Path.home() / ".local/share/jupyter/kernels",
        Path("/venv/main/share/jupyter/kernels"),
        Path("/opt/conda/share/jupyter/kernels"),
    ]
    # `nandaproj` is the name provision.sh registers; prefer it, then any other.
    specs = []
    for root in roots:
        if root.is_dir():
            specs += sorted(root.glob("*/kernel.json"),
                            key=lambda p: (p.parent.name != "nandaproj", str(p)))
    for spec in specs:
        try:
            argv = json.loads(spec.read_text()).get("argv") or []
        except (OSError, ValueError):
            continue
        if argv and Path(argv[0]).exists():
            return argv[0], f"kernelspec {spec.parent.name} ({spec})"

    for cand in ("/venv/main/bin/python", "/opt/conda/bin/python", "/usr/bin/python3"):
        if Path(cand).exists():
            return cand, "fallback: provision.sh candidate list"
    return sys.executable, "fallback: the interpreter you launched"


def reexec_if_needed() -> None:
    if os.environ.get(REEXEC_FLAG):
        return
    target, how = kernel_python()
    print(f"launched with : {sys.executable}")
    print(f"kernel python : {target}\n  found via   : {how}")
    if Path(target).resolve() != Path(sys.executable).resolve():
        print(f"\n>> re-exec under the kernel python\n{RULE}")
        os.environ[REEXEC_FLAG] = "1"
        # execv replaces this process image without flushing -- everything above
        # is still sitting in the stdout buffer and would be lost, which is the
        # one thing this stage exists to report.
        sys.stdout.flush()
        os.execv(target, [target, *sys.argv])
    print(f"\n>> already the kernel python\n{RULE}")
    os.environ[REEXEC_FLAG] = "1"


# -- the HF token, which a non-login shell does not have -------------------

def ensure_hf_token() -> str:
    """Put HF_TOKEN in the environment without ever printing it.

    provision.sh writes the token into /etc/profile.d/nandaproj.sh, which is
    sourced by *login* shells. `just ssh` runs a non-login shell, so a gated
    model 401s here while working fine in the notebook -- which reads as
    "nnsight is broken" if you are not looking for it.

    The value is never printed. CLAUDE.md: two tokens have already leaked into
    transcripts by cat-ing exactly this file. Match the one line, take group 1,
    say only that it worked -- the value is group 2 and never leaves this function.
    """
    import re

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(var):
            return f"{var} already in the environment"

    profile = Path("/etc/profile.d/nandaproj.sh")
    if not profile.is_file():
        return "no token: /etc/profile.d/nandaproj.sh not found"
    for line in profile.read_text().splitlines():
        m = re.match(r"\s*export\s+(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=[\"\']?([^\"\'\s]+)", line)
        if m:
            os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = m.group(2)
            return f"HF_TOKEN loaded from profile.d (len {len(m.group(2))}, not printed)"
    return "no token: no export line in profile.d"


# -- stage 1: what is installed -------------------------------------------

def version(mod: str) -> str:
    import importlib
    import importlib.metadata as md
    try:
        m = importlib.import_module(mod)
    except Exception as exc:                     # noqa: BLE001 -- report, do not raise
        return f"MISSING/BROKEN  {type(exc).__name__}: {exc}"[:120]
    for attr in ("__version__", "VERSION"):
        if hasattr(m, attr):
            return str(getattr(m, attr))
    try:
        return md.version(mod.replace("_", "-"))
    except md.PackageNotFoundError:
        return "installed (no version attr)"


def stage1() -> None:
    print("\n== imports, as the kernel sees them ==")
    for mod in ("torch", "transformers", "nnsight", "transformer_lens", "jlens",
                "huggingface_hub", "numpy"):
        print(f"  {mod:<18} {version(mod)}")

    try:
        import torch
        print(f"\n  cuda      : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"  gpu       : {torch.cuda.get_device_name(0)}")
            print(f"  vram free : {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
    except Exception as exc:                     # noqa: BLE001
        print(f"  torch unusable: {exc}")

    # TL is the second choice, and it does not wrap arbitrary HF models -- each
    # architecture needs an explicit port. So the question is not "does it
    # import" but "is gemma-3 in its list of ported models".
    try:
        from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES
        g3 = [m for m in OFFICIAL_MODEL_NAMES if "gemma-3" in m]
        print(f"\n  transformer_lens gemma-3 ports: {g3 or 'NONE -- TL is out'}")
    except Exception as exc:                     # noqa: BLE001
        print(f"\n  transformer_lens unusable: {type(exc).__name__}: {exc}"[:200])


# -- stage 2: does nnsight read and write a head on this model? ------------

def find_layer_path(model) -> tuple[str, str]:
    """Locate the decoder layer list without assuming the wrapper's shape.

    gemma-3-4b loads as the *multimodal* wrapper, so the text layers are nested
    somewhere under it and the depth differs between transformers versions.
    Hardcoding `model.model.layers` is how 05 would break on a version bump.
    """
    for name, _ in model.named_modules():
        if name.endswith("layers.0.self_attn.o_proj"):
            prefix = name[: -len(".0.self_attn.o_proj")]      # ...<x>.layers
            return prefix, name
    raise RuntimeError("no `layers.0.self_attn.o_proj` in this model")


def load_hf(preset: str):
    """The HF model both stages measure against -- loaded once, shared."""
    import torch

    sys.path.insert(0, "/workspace/NandaProj/src")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nandaproj import config

    cfg = config.get_model_config(preset)
    print(f"\n== loading {cfg.name} ==")
    tok = AutoTokenizer.from_pretrained(cfg.name, cache_dir=str(config.HF_CACHE))
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name, cache_dir=str(config.HF_CACHE),
        dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tok, cfg


PROMPT = "The capital of France is"


def hf_logits(model, tok):
    import torch
    with torch.no_grad():
        return model(**tok(PROMPT, return_tensors="pt").to(model.device)).logits[0, -1]


def stage2(model, tok, cfg) -> None:
    import torch

    print(f"\n== stage 2: nnsight on {cfg.name} ==")

    prefix, o_proj_name = find_layer_path(model)
    print(f"  decoder layers at : model.{prefix}")
    print(f"  o_proj at         : model.{o_proj_name}")

    text_cfg = getattr(model.config, "text_config", model.config)
    n_heads = text_cfg.num_attention_heads
    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // n_heads)
    print(f"  layers={text_cfg.num_hidden_layers}  d_model={text_cfg.hidden_size}  "
          f"q_heads={n_heads}  kv_heads={getattr(text_cfg, 'num_key_value_heads', '?')}  "
          f"head_dim={head_dim}")

    # R4: local vs global attention must never be pooled. Print how this
    # transformers version *names* the distinction, so 05 can split on it.
    for attr in ("layer_types", "sliding_window", "sliding_window_pattern",
                 "interleaved_sliding_window"):
        if hasattr(text_cfg, attr):
            val = getattr(text_cfg, attr)
            val = val[:8] if isinstance(val, list) else val
            print(f"  {attr:<26} {val}")

    # The residual-add subtlety that decides whether per-head DLA is linear:
    # Gemma 3 norms the attention output *before* adding it back.
    layer0 = dict(model.named_modules())[f"{prefix}.0"]
    print(f"  submodules of layer 0: {[n for n, _ in layer0.named_children()]}")

    import nnsight
    from nnsight import LanguageModel

    print(f"\n  nnsight {getattr(nnsight, '__version__', '?')}: wrapping the loaded model")
    lm = LanguageModel(model, tokenizer=tok)

    L = text_cfg.num_hidden_layers // 2

    def envoy(path: str):
        obj = lm
        for part in path.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        return obj

    o_proj = envoy(f"{o_proj_name.replace('.0.', f'.{L}.', 1)}")

    base = hf_logits(model, tok)

    # (a) read: capture the o_proj input, which is z for every head concatenated
    with lm.trace(PROMPT):
        z = o_proj.input.save()
        noop = lm.output.logits.save()
    z = z if not isinstance(z, tuple) else z[0]
    print(f"  read  z shape      : {tuple(z.shape)}   (expect [1, T, {n_heads * head_dim}])")

    same = torch.allclose(noop[0, -1].float(), base.float(), atol=0, rtol=0)
    print(f"  no-op trace is the identity : {same}"
          f"{'' if same else '  <- interventions cannot be trusted if this is False'}")

    # (b) the per-head split, which is what DLA rests on. o_proj has no bias in
    # Gemma, so head h contributes exactly z_h @ W_O[h].
    W_O = dict(model.named_modules())[o_proj_name.replace(".0.", f".{L}.", 1)].weight
    zl = z[0, -1].float()
    full = (W_O.float() @ zl)
    per_head = sum(W_O.float()[:, h * head_dim:(h + 1) * head_dim] @ zl[h * head_dim:(h + 1) * head_dim]
                   for h in range(n_heads))
    err = (full - per_head).abs().max().item()
    print(f"  per-head split exact        : max|sum_h - full| = {err:.2e}")

    # (c) write: zero one head at the last position and confirm it lands.
    with lm.trace(PROMPT):
        zin = o_proj.input
        zin = zin if not isinstance(zin, tuple) else zin[0]
        zin[0, -1, 0:head_dim] = 0
        patched = lm.output.logits.save()
    delta = (patched[0, -1].float() - base.float()).abs().max().item()
    print(f"  intervention lands          : max|delta logit| = {delta:.4f}"
          f"{'  <- ZERO: the write did not take' if delta == 0 else ''}")

    print("\n  verdict: nnsight is usable for 05 if the split is ~1e-2 or better, "
          "the no-op is the identity, and the intervention delta is non-zero.")



# -- stage 3: does TransformerLens agree with the model 04 measured? -------

def stage3(model, tok, cfg) -> None:
    """The same three checks, plus the one that is specific to TL.

    TL does not wrap the HF model -- it re-implements the architecture, converts
    the weights and folds the layernorms. That is what buys the clean `hook_z`
    and the patching utilities, and it is also the risk: 04 located l* = 25 on
    the HF object, so if TL's forward pass disagrees, every component 05 finds
    is a claim about a *different* model. Hence check (0), which nnsight does
    not need because it runs the real module.

    `hf_model=` reuses the weights already in memory rather than pulling 8 GB
    again, and `dtype` must be passed explicitly: TL defaults to float32, which
    on the 4b would be ~17 GB on top of whatever 04's kernel is holding.
    """
    import torch
    from transformer_lens import HookedTransformer

    print(f"\n== stage 3: transformer_lens on {cfg.name} ==")
    tl = HookedTransformer.from_pretrained(
        cfg.name, hf_model=model, tokenizer=tok,
        dtype=torch.bfloat16, device="cuda",
    )
    print(f"  n_layers={tl.cfg.n_layers}  d_model={tl.cfg.d_model}  "
          f"n_heads={tl.cfg.n_heads}  d_head={tl.cfg.d_head}  "
          f"n_kv_heads={getattr(tl.cfg, 'n_key_value_heads', '?')}")

    # R4 again: whatever TL calls the local/global alternation, 05 must be able
    # to read it off cfg rather than re-deriving the 5:1 pattern by hand.
    for attr in ("use_local_attn", "window_size", "attn_types",
                 "use_NTK_by_parts_rope", "final_logit_softcap"):
        if hasattr(tl.cfg, attr):
            val = getattr(tl.cfg, attr)
            val = val[:8] if isinstance(val, list) else val
            print(f"  cfg.{attr:<24} {val}")

    # (0) does TL reproduce the HF forward pass at the answer slot?
    base = hf_logits(model, tok).float()
    with torch.no_grad():
        tl_logits = tl(PROMPT, return_type="logits")[0, -1].float()
    lp_hf = torch.log_softmax(base, -1)
    lp_tl = torch.log_softmax(tl_logits, -1)
    print(f"\n  agrees with HF   : max|dlogprob| = {(lp_hf - lp_tl).abs().max():.4f}  "
          f"top-1 same = {int(lp_hf.argmax()) == int(lp_tl.argmax())}")
    print(f"    HF top-1 {tok.decode([int(lp_hf.argmax())])!r}  "
          f"TL top-1 {tok.decode([int(lp_tl.argmax())])!r}")

    # (a) read, (b) the per-head split -- free in TL: hook_z IS per head, and
    # hook_result is z @ W_O already split. That is the whole argument for TL.
    L = tl.cfg.n_layers // 2
    _, cache = tl.run_with_cache(PROMPT, names_filter=lambda n: n.endswith(
        (f"blocks.{L}.attn.hook_z", f"blocks.{L}.hook_mlp_out")))
    z = cache["z", L]
    print(f"\n  cache['z', {L}]   : {tuple(z.shape)}   (expect [1, T, n_heads, d_head])")
    print("  per-head split   : structural -- hook_z is already [.., head, d_head]")

    # (c) write: zero one head's z and confirm it lands.
    def zero_head(value, hook):
        value[:, -1, 0, :] = 0
        return value

    with torch.no_grad():
        patched = tl.run_with_hooks(
            PROMPT, return_type="logits",
            fwd_hooks=[(f"blocks.{L}.attn.hook_z", zero_head)])[0, -1].float()
    delta = (patched - tl_logits).abs().max().item()
    print(f"  intervention lands: max|delta logit| = {delta:.4f}"
          f"{'  <- ZERO: the write did not take' if delta == 0 else ''}")

    print("\n  verdict: TL is usable if it AGREES WITH HF at the slot. A clean "
          "hook API on a model that answers differently than the one 04 measured "
          "is worse than a rough API on the right model.")


if __name__ == "__main__":
    reexec_if_needed()
    print(f"hf token      : {ensure_hf_token()}")
    stage1()

    argv = sys.argv
    preset = argv[argv.index("--preset") + 1] if "--preset" in argv else "debug"
    want_nnsight = "--model" in argv or "--all" in argv
    want_tl = "--tl" in argv or "--all" in argv

    if want_nnsight or want_tl:
        model, tok, cfg = load_hf(preset)
        for name, fn in (("stage 2 (nnsight)", stage2 if want_nnsight else None),
                         ("stage 3 (transformer_lens)", stage3 if want_tl else None)):
            if fn is None:
                continue
            try:
                fn(model, tok, cfg)
            except Exception:                    # noqa: BLE001 -- the point is the traceback
                import traceback
                print(f"\n!! {name} failed -- this is the answer, not an error to fix:\n")
                traceback.print_exc()
