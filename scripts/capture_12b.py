"""Residual capture for the same-token lie probe (`misc/prereg_12b_same_token.md`).

Every robust-labelled alleged item (lied or honest under D in both A/B mappings,
per the 12B re-gate) runs under D and H in both mappings. H rows carry the item's D
label too: they are the propensity control (can the label be read off an item the
model is not being asked to lie about?). Per row, the residual stream
at every block is saved at two positions: the answer slot, and the last token
before `Answer:`, where no answer text exists yet.

On the box (after `just up`, before `just down`):
  just ssh "bash -lc 'cd /workspace/NandaProj && /venv/main/bin/python scripts/capture_12b.py --preset escalate'"
Local dry run (any torch with XPU; the 270m mirror is in hf_cache/):
  python scripts/capture_12b.py --model-name unsloth/gemma-3-270m-it --device xpu --limit 3 --out $TMPDIR/cap

Writes `<out>.npz` (`resid`, float16, [N, 2, L, d], positions ("pre", "slot")) and
`<out>.json` (one metadata row per residual row).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nandaproj import config, deception, pilot

GATE = config.RESULTS / "pilot_gate_gemma-3-12b-it_v2+alleged.json"
FORMATS = ("ab_yesA", "ab_yesB")
CONDITIONS = ("D", "H")
POSITIONS = ("pre", "slot")


def robust_items(limit: int | None) -> list[tuple[pilot.PilotItem, str]]:
    """`(item, "lied" | "honest")` from the 12B gate; `limit` takes the first N of each label."""
    items = {it.item.item_id: it for it in pilot.alleged_items()}
    gate_rows = [r for r in json.loads(GATE.read_text())["rows"] if r["source"] == "alleged"]
    labels = pilot.robust_labels(gate_rows, items)
    out = []
    for wanted in ("lied", "honest"):
        chosen = [(items[i], lab) for i, lab in sorted(labels.items()) if lab == wanted]
        out += chosen[:limit] if limit else chosen
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="escalate")
    ap.add_argument("--model-name", help="override the HF model id (local dry runs)")
    ap.add_argument("--device", default="auto", help="'auto' (device_map), or e.g. 'xpu', 'cpu'")
    ap.add_argument("--limit", type=int, help="first N lied and first N honest items (dry runs)")
    ap.add_argument("--out", help="output path without extension (default results/capture_12b_<lens>)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nandaproj import items as items_mod
    from nandaproj.lens_readout import _progress

    cfg = config.get_model_config(args.preset)
    name = args.model_name or cfg.name
    tok = AutoTokenizer.from_pretrained(name, cache_dir=str(config.HF_CACHE))
    load = dict(cache_dir=str(config.HF_CACHE), dtype=torch.bfloat16)
    if args.device == "auto":
        model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", **load).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(name, **load).to(args.device).eval()
    letter_ids = {"A": items_mod.token_id(tok, " A"), "B": items_mod.token_id(tok, " B")}

    population = robust_items(args.limit)
    n_lied = sum(lab == "lied" for _, lab in population)
    print(f"{n_lied} lied / {len(population) - n_lied} honest items, "
          f"{len(CONDITIONS)} conditions x {len(FORMATS)} mappings each")
    jobs = [(it, lab, cond, fmt) for it, lab in population for cond in CONDITIONS for fmt in FORMATS]

    resid, rows, t0 = None, [], time.time()
    for n, (it, lab, cond, fmt) in enumerate(_progress(jobs, total=len(jobs), desc="capture")):
        prompt = pilot.render(tok, it.item, cond, fmt)
        head = prompt[: -len(deception.ANSWER_PREFIX)]
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        head_ids = tok(head, return_tensors="pt", add_special_tokens=False).input_ids
        # Same single-<bos> guard as pilot_gate.py; and `pre` is only "before the
        # answer prefix" if the prefix did not merge with the token before it.
        assert int((ids == tok.bos_token_id).sum()) == 1 and int(ids[0, 0]) == tok.bos_token_id
        assert torch.equal(ids[0, : head_ids.shape[1]], head_ids[0]), "Answer: merged into the template"
        pre = head_ids.shape[1] - 1

        with torch.no_grad():
            out = model(ids.to(model.device), output_hidden_states=True)
        # hidden_states[0] is the embedding; [1:] are the block outputs.
        hs = torch.stack(out.hidden_states[1:])[:, 0]          # [L, T, d]
        if resid is None:
            resid = np.empty((len(jobs), len(POSITIONS), hs.shape[0], hs.shape[-1]), dtype=np.float16)
        resid[n, 0] = hs[:, pre].float().cpu().numpy()
        resid[n, 1] = hs[:, -1].float().cpu().numpy()

        probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
        p_a, p_b = float(probs[letter_ids["A"]]), float(probs[letter_ids["B"]])
        emitted = "A" if p_a > p_b else "B"
        yes_letter = pilot.candidates(fmt)[0].strip()
        rows.append({"item_id": it.item.item_id, "pair_id": it.item.pair_id,
                     "category": it.item.category, "stake": it.item.stake,
                     "truth_yes": it.truth_yes, "condition": cond, "format": fmt, "label": lab,
                     "p_a": p_a, "p_b": p_b, "emitted_letter": emitted,
                     "emitted_yes": emitted == yes_letter, "prompt": prompt})
    elapsed = time.time() - t0
    print(f"{len(rows)} forward passes in {elapsed:.0f} s ({len(rows) / elapsed:.1f}/s)")

    # The labels came from the gate run; a D row whose emitted answer disagrees with
    # its label, or an H row that is wrong, means this is not the forward pass the gate saw.
    d_bad = sum((r["emitted_yes"] != r["truth_yes"]) != (r["label"] == "lied")
                for r in rows if r["condition"] == "D")
    h_bad = sum(r["emitted_yes"] != r["truth_yes"] for r in rows if r["condition"] == "H")
    print(f"disagreeing with the gate: D {d_bad}, H {h_bad} of {len(rows) // 2} each (expect ~0 on the gated model)")

    path = Path(args.out) if args.out else config.RESULTS / f"capture_12b_{cfg.lens_id}"
    np.savez(path.with_suffix(".npz"), resid=resid)
    path.with_suffix(".json").write_text(json.dumps(
        {"model": name, "positions": list(POSITIONS), "rows": rows}, indent=1))
    print(f"-> {path}.npz  {resid.shape}  {resid.nbytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
