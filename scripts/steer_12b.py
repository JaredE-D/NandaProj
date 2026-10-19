"""Steering stretch for `misc/prereg_12b_same_token.md`: is the lie direction causal?

Direction: D-row mean(lied) - mean(honest) residual at one (layer, position),
cross-fitted -- computed on half the pairs, applied to the other half -- so no
item is steered by a direction computed from its own activations. Control: a
random direction of the same norm.

Under D, the direction is subtracted on lied items (does the lie rate fall?) and
added on honest items (does it rise?), at every position from `position` onward,
at the output of block `layer`.

  direction  locally:  uv run python scripts/steer_12b.py direction --layer L --position slot
  run        on box:   just ssh "bash -lc 'cd /workspace/NandaProj && /venv/main/bin/python scripts/steer_12b.py run'"
  report     locally:  uv run python scripts/steer_12b.py report
Local dry run on the 270m capture (XPU):
  python scripts/steer_12b.py direction --layer 5 --capture /tmp/cap_dry --directions /tmp/steer_dry
  python scripts/steer_12b.py run --capture /tmp/cap_dry --directions /tmp/steer_dry --out /tmp/steer_dry.json \
      --model-name unsloth/gemma-3-270m-it --device xpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nandaproj import config, pilot

LENS = "gemma-3-12b-it"
CAPTURE = config.RESULTS / f"capture_12b_{LENS}"
DIRECTIONS = config.RESULTS / f"steer_directions_{LENS}"
OUT = config.RESULTS / f"steer_12b_{LENS}.json"
POSITIONS = ("pre", "slot")
DOSES = (0.0, 1.0, 2.0, 4.0)     # multiples of the mean-difference vector's norm
SEED = 20261018


# --------------------------------------------------------------------------
# direction (local, from the capture)
# --------------------------------------------------------------------------

def direction(args) -> None:
    meta = json.loads(args.capture.with_suffix(".json").read_text())
    rows = meta["rows"]
    d_idx = np.array([i for i, r in enumerate(rows) if r["condition"] == "D"])
    X = np.load(args.capture.with_suffix(".npz"))["resid"][d_idx, POSITIONS.index(args.position), args.layer]
    X = X.astype(np.float32)
    X[:, ~np.isfinite(X).all(0)] = 0.0     # the float16-overflowed massive-activation channel: not steered
    lied = np.array([rows[i]["label"] == "lied" for i in d_idx])
    pairs = sorted({rows[i]["pair_id"] for i in d_idx})
    half = set(np.random.default_rng(SEED).permutation(pairs)[: len(pairs) // 2])
    fold = np.array([rows[i]["pair_id"] in half for i in d_idx])      # True = fold 0

    dirs, folds = [], {}
    for f in (0, 1):
        fit = fold == (f == 0)
        dirs.append(X[fit & lied].mean(0) - X[fit & ~lied].mean(0))
        # The other fold's items are the ones this direction steers: every lied
        # item, and as many honest ones (seeded sample) to keep the box run short.
        held = d_idx[~fit]
        lied_items = sorted({rows[i]["item_id"] for i in held if rows[i]["label"] == "lied"})
        honest_items = sorted({rows[i]["item_id"] for i in held if rows[i]["label"] == "honest"})
        sample = np.random.default_rng(SEED + f).choice(honest_items, min(len(lied_items), len(honest_items)), replace=False)
        folds[f] = lied_items + sorted(sample.tolist())
    rng = np.random.default_rng(SEED + 1)
    rand = [rng.normal(size=d.shape).astype(np.float32) for d in dirs]
    rand = [r / np.linalg.norm(r) * np.linalg.norm(d) for r, d in zip(rand, dirs)]
    np.savez(args.directions.with_suffix(".npz"), probe=np.stack(dirs), random=np.stack(rand),
             layer=args.layer, position=args.position)
    args.directions.with_suffix(".json").write_text(json.dumps(
        {"layer": args.layer, "position": args.position, "steered_items": folds,
         "norms": [float(np.linalg.norm(d)) for d in dirs],
         "cos_between_folds": float(dirs[0] @ dirs[1] / np.linalg.norm(dirs[0]) / np.linalg.norm(dirs[1]))},
        indent=1))
    print(f"-> {args.directions}  norms {[round(float(np.linalg.norm(d)), 1) for d in dirs]}")


# --------------------------------------------------------------------------
# run (box)
# --------------------------------------------------------------------------

def decoder_layers(model):
    """The decoder's block list, wherever the (multimodal) wrapper keeps it."""
    n = model.config.get_text_config().num_hidden_layers
    for name, mod in model.named_modules():
        if name.endswith("layers") and hasattr(mod, "__len__") and len(mod) == n:
            return mod
    raise RuntimeError("no decoder layer list found")


def run(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nandaproj import deception, items as items_mod
    from nandaproj.lens_readout import _progress

    D = np.load(args.directions.with_suffix(".npz"))
    spec = json.loads(args.directions.with_suffix(".json").read_text())
    layer, position = int(D["layer"]), str(D["position"])
    labels = {r["item_id"]: r["label"] for r in json.loads(args.capture.with_suffix(".json").read_text())["rows"]}
    alleged = {it.item.item_id: it for it in pilot.alleged_items()}

    name = args.model_name or config.get_model_config("escalate").name
    tok = AutoTokenizer.from_pretrained(name, cache_dir=str(config.HF_CACHE))
    load = dict(cache_dir=str(config.HF_CACHE), dtype=torch.bfloat16)
    if args.device == "auto":
        model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", **load).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(name, **load).to(args.device).eval()
    block = decoder_layers(model)[layer]
    letter_ids = {"A": items_mod.token_id(tok, " A"), "B": items_mod.token_id(tok, " B")}

    state = {"vec": None, "start": None}

    def hook(_mod, _inp, out):
        if state["vec"] is None:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs[:, state["start"]:] += state["vec"].to(hs.dtype)
        return out

    block.register_forward_hook(hook)

    jobs = []
    for f in ("0", "1"):
        for item_id in spec["steered_items"][f]:
            lab = labels[item_id]
            sign = -1.0 if lab == "lied" else 1.0         # push lies to honest, honest to lies
            for kind in ("probe", "random"):
                for dose in args.doses:
                    if dose == 0.0 and kind == "random":
                        continue                            # dose 0 is the same pass for both
                    for fmt in ("ab_yesA", "ab_yesB"):
                        jobs.append((item_id, lab, int(f), kind, dose, sign, fmt))

    rows = []
    for item_id, lab, f, kind, dose, sign, fmt in _progress(jobs, total=len(jobs), desc="steer"):
        it = alleged[item_id]
        prompt = pilot.render(tok, it.item, "D", fmt)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        head = tok(prompt[: -len(deception.ANSWER_PREFIX)], add_special_tokens=False).input_ids
        start_at = args.apply_from or position
        state["start"] = {"all": 0, "pre": len(head) - 1, "slot": ids.shape[1] - 1}[start_at]
        vec = torch.from_numpy(D[kind][f]).to(model.device)
        state["vec"] = sign * dose * vec if dose else None
        with torch.no_grad():
            logits = model(ids.to(model.device)).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        p_a, p_b = float(probs[letter_ids["A"]]), float(probs[letter_ids["B"]])
        yes_letter = pilot.candidates(fmt)[0].strip()
        emitted_yes = (p_a > p_b) == (yes_letter == "A")
        rows.append({"item_id": item_id, "label": lab, "kind": kind, "dose": dose, "format": fmt,
                     "p_a": p_a, "p_b": p_b, "mass": p_a + p_b,
                     "lied_now": emitted_yes != it.truth_yes})
    args.out.write_text(json.dumps({"layer": layer, "position": position,
                                    "apply_from": args.apply_from or position, "doses": args.doses, "rows": rows}, indent=1))
    print(f"-> {args.out}")


# --------------------------------------------------------------------------
# report (local)
# --------------------------------------------------------------------------

def report(args) -> None:
    r = json.loads(args.out.read_text())
    rows = r["rows"]
    print(f"layer {r['layer']} direction from {r['position']}, applied from "
          f"{r.get('apply_from', r['position'])}; lie rate per row (both mappings pooled)")
    print(f"{'items':<8}{'kind':<8}" + "".join(f"x{d:<12g}" for d in r["doses"]) + "  min mass")
    for lab in ("lied", "honest"):
        for kind in ("probe", "random"):
            cells = []
            for dose in r["doses"]:
                sel = [x for x in rows if x["label"] == lab and x["dose"] == dose
                       and (x["kind"] == kind or dose == 0.0)]
                k, n = sum(x["lied_now"] for x in sel), len(sel)
                cells.append(f"{k / n:.2f} ({k}/{n})".ljust(13))
            mass = min(x["mass"] for x in rows if x["label"] == lab and x["kind"] == kind)
            print(f"{lab:<8}{kind:<8}" + "".join(cells) + f"  {mass:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("direction", "run", "report"))
    ap.add_argument("--layer", type=int)
    ap.add_argument("--position", choices=POSITIONS, default="slot")
    ap.add_argument("--capture", type=Path, default=CAPTURE, help="capture path without extension")
    ap.add_argument("--directions", type=Path, default=DIRECTIONS, help="directions path without extension")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--model-name", help="override the HF model id (local dry runs)")
    ap.add_argument("--device", default="auto", help="'auto' (device_map), or e.g. 'xpu'")
    ap.add_argument("--doses", type=float, nargs="+", default=list(DOSES),
                    help="run: multiples of the mean-difference norm (include 0)")
    ap.add_argument("--apply-from", choices=("all",) + POSITIONS,
                    help="run: first steered position (default: the direction's own position)")
    args = ap.parse_args()
    {"direction": direction, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
