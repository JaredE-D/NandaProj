"""Answers-only pilot gate: Yes/No and both A/B mappings, final-token probabilities only.

Pre-registrations: `misc/prereg_pilot_ab_cities.md` (sources v2, cities) and
`misc/prereg_model_size_regate.md` (source alleged, 4B vs 12B). Subcommands:

  run      on the box: one forward pass per (item, condition, format).
           just ssh "bash -lc 'cd /workspace/NandaProj && /venv/main/bin/python scripts/pilot_gate.py run --preset target'"
  report   locally, after `just sync`: lie rates per arm, robust labels, pass/fail.
           uv run python scripts/pilot_gate.py report --preset target
  compare  locally: the pre-registered 4B-vs-12B criteria from two report files.
           uv run python scripts/pilot_gate.py compare

Results land in results/pilot_gate_<lens>_<sources>.json. The Geometry of Truth
CSVs are fetched at a pinned commit into results/ (gitignored): the repo carries
no license, so they are not committed here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nandaproj import config, pilot

SEED = 20261018
N_CITIES = 30
GOT_SHA = "5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb"
GOT_URL = f"https://raw.githubusercontent.com/saprmarks/geometry-of-truth/{GOT_SHA}/datasets/{{}}.csv"
GOT_DIR = config.RESULTS / "external_geometry_of_truth"
SOURCES = ("v2", "cities", "alleged")


def out_path(lens_id: str, sources: list[str]) -> Path:
    return config.RESULTS / f"pilot_gate_{lens_id}_{'+'.join(sources)}.json"


def fetch_got() -> tuple[Path, Path]:
    GOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("cities", "neg_cities"):
        path = GOT_DIR / f"{name}.csv"
        if not path.exists():
            urllib.request.urlretrieve(GOT_URL.format(name), path)
        paths.append(path)
    return paths[0], paths[1]


def load_items(sources: list[str]) -> list[pilot.PilotItem]:
    out = []
    if "v2" in sources:
        out += pilot.v2_items()
    if "cities" in sources:
        out += pilot.city_items(*fetch_got(), N_CITIES, np.random.default_rng(SEED))
    if "alleged" in sources:
        out += pilot.alleged_items()
    return out


# --------------------------------------------------------------------------
# run (box)
# --------------------------------------------------------------------------

def run(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nandaproj import items as items_mod
    from nandaproj.lens_readout import _progress

    cfg = config.get_model_config(args.preset)
    name = args.model_name or cfg.name
    tok = AutoTokenizer.from_pretrained(name, cache_dir=str(config.HF_CACHE))
    model = AutoModelForCausalLM.from_pretrained(
        name, cache_dir=str(config.HF_CACHE), dtype=torch.bfloat16, device_map="auto").eval()
    cand_ids = {fmt: tuple(items_mod.token_id(tok, t) for t in pilot.candidates(fmt))
                for fmt in pilot.FORMATS}

    pitems = load_items(args.sources)
    if args.limit:      # dry runs: the first N of each source, so every code path runs
        pitems = [it for src in args.sources
                  for it in [p for p in pitems if p.source == src][: args.limit]]
    jobs = [(it, c, f) for it in pitems for c in pilot.SOURCE_CONDITIONS[it.source]
            for f in pilot.FORMATS]
    rows, t0 = [], time.time()
    for it, cond, fmt in _progress(jobs, total=len(jobs), desc="pilot gate"):
        prompt = pilot.render(tok, it.item, cond, fmt)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        # One tokenization for everything, exactly one <bos> (the plan-2 seam).
        # Qwen has no <bos>; there the check is that the template added none either.
        if tok.bos_token_id is not None:
            assert int((ids == tok.bos_token_id).sum()) == 1 and int(ids[0, 0]) == tok.bos_token_id, \
                f"{it.item.item_id}/{cond}/{fmt}: expected exactly one leading <bos>"
        with torch.no_grad():
            logits = model(ids.to(model.device)).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        yes_id, no_id = cand_ids[fmt]
        top = np.argsort(probs)[-3:][::-1]
        rows.append({"item_id": it.item.item_id, "source": it.source, "condition": cond,
                     "format": fmt, "p_yes": float(probs[yes_id]), "p_no": float(probs[no_id]),
                     "mass": float(probs[yes_id] + probs[no_id]),
                     "top3": [(tok.decode([int(i)]), float(probs[i])) for i in top]})
    elapsed = time.time() - t0
    print(f"{len(rows)} forward passes in {elapsed:.0f} s ({len(rows) / elapsed:.1f}/s)")

    out = {"model": name, "seed": SEED, "n_cities": N_CITIES, "got_sha": GOT_SHA,
           "sources": args.sources,
           "example_prompt_D_ab_yesA": pilot.render(tok, pitems[0].item, "D", "ab_yesA"),
           "items": [{"item_id": it.item.item_id, "source": it.source,
                      "truth": it.item.answer_true, "negated": it.negated, "target": it.target,
                      "context": it.item.context, "question": it.item.question}
                     for it in pitems],
           "rows": rows}
    path = Path(args.out) if args.out else out_path(cfg.lens_id, args.sources)
    path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}")


# --------------------------------------------------------------------------
# report (local)
# --------------------------------------------------------------------------

def _fmt(c: dict) -> str:
    lo, hi = c["ci"]
    return f"{c['lied']:>3}/{c['n']:<3} = {c['rate']:.2f} [{lo:.2f}, {hi:.2f}]"


def report_source(source: str, rows: list[dict], items: dict[str, pilot.PilotItem]) -> dict:
    rows = [r for r in rows if r["source"] == source]
    items = {i: it for i, it in items.items() if it.source == source}
    entry = {"n_items": len(items)}
    print(f"\n=== {source}: {len(items)} items, {len(rows)} rows")

    print("says-Yes rate per condition and format")
    for fmt in pilot.FORMATS:
        cells = []
        for cond in pilot.SOURCE_CONDITIONS[source]:
            sel = [r for r in rows if r["format"] == fmt and r["condition"] == cond]
            cells.append(f"{cond} {np.mean([pilot.said_yes(r) for r in sel]):.2f}")
        print(f"  {fmt:<9}" + "   ".join(cells))

    d_ab = [r for r in rows if r["condition"] == "D" and r["format"] != "yesno"]
    entry["p_answer_A_under_D"] = float(np.mean(
        [pilot.said_yes(r) == (r["format"] == "ab_yesA") for r in d_ab]))
    print(f"letter bias: P(D answer = A) = {entry['p_answer_A_under_D']:.2f}  (n={len(d_ab)})")

    print("D lie rate among H-correct conflict items, by truth arm  [Wilson 95% CI]")
    print("  (ab_pooled counts each item once per mapping, so its n is up to 2x and its CI optimistic)")
    groups = {"yesno": ("yesno",), "ab_pooled": ("ab_yesA", "ab_yesB"),
              "ab_yesA": ("ab_yesA",), "ab_yesB": ("ab_yesB",)}
    entry["groups"] = {}
    for gname, fmts in groups.items():
        outs = list(pilot.outcomes(rows, items, fmts))
        arms = {"Yes-true": pilot.cell([lied for it, _, lied in outs if it.truth_yes]),
                "No-true": pilot.cell([lied for it, _, lied in outs if not it.truth_yes])}
        gap = arms["No-true"]["rate"] - arms["Yes-true"]["rate"]
        ok = pilot.passes(arms)
        print(f"  {gname:<10} Yes-true {_fmt(arms['Yes-true'])}   No-true {_fmt(arms['No-true'])}"
              f"   gap {gap:+.2f}   {'PASS' if ok else 'fail'}")
        g = {"arms": arms, "gap_no_minus_yes": gap, "passes": ok}
        if source == "cities":
            g["by_negation_x_truth"] = {
                f"{'neg' if neg else 'aff'}_{'T' if t else 'F'}": pilot.cell(
                    [lied for it, _, lied in outs if it.negated == neg and it.truth_yes == t])
                for neg in (False, True) for t in (True, False)}
        entry["groups"][gname] = g

    labels = pilot.robust_labels(rows, items)
    robust = {}
    print("robust labels (H-correct under both A/B mappings): lied / flips / honest")
    for arm, truth in (("Yes-true", True), ("No-true", False)):
        counts = {k: sum(1 for i, v in labels.items() if v == k and items[i].truth_yes == truth)
                  for k in ("lied", "flips", "honest")}
        decided = counts["lied"] + counts["honest"]
        counts["robust_lie_rate"] = counts["lied"] / decided if decided else float("nan")
        robust[arm] = counts
        print(f"  {arm:<9} {counts['lied']:>4} / {counts['flips']:>4} / {counts['honest']:>4}"
              f"   lie rate among robust {counts['robust_lie_rate']:.2f}")
    entry["robust"] = robust
    entry["robust_lied"] = sum(r["lied"] for r in robust.values())
    entry["robust_honest"] = sum(r["honest"] for r in robust.values())
    entry["robust_arms_balanced"] = all(0.25 <= r["robust_lie_rate"] <= 0.75 for r in robust.values())
    entry["robust_labels"] = labels
    return entry


def report(args) -> None:
    lens_id = config.get_model_config(args.preset).lens_id
    path = Path(args.out) if args.out else out_path(lens_id, args.sources)
    data = json.loads(path.read_text())
    rows = data["rows"]
    by_id = {it.item.item_id: it for it in load_items(data["sources"])}
    assert {r["item_id"] for r in rows} <= set(by_id), "results were made from a different sample"
    items = {i: by_id[i] for i in {r["item_id"] for r in rows}}

    print(f"{data['model']}: {len(rows)} rows")
    low_mass = [r for r in rows if r["mass"] < 0.5]
    print(f"format compliance: {len(low_mass)} rows put < 0.5 on the two candidates")
    for r in low_mass[:5]:
        print(f"  {r['item_id']}/{r['condition']}/{r['format']}: mass {r['mass']:.2f} top3 {r['top3']}")

    summary = {"source_file": path.name, "model": data["model"],
               "n_low_mass_rows": len(low_mass), "sources": {}}
    for source in data["sources"]:
        summary["sources"][source] = report_source(source, rows, items)

    if "v2" in summary["sources"]:
        g = summary["sources"]["v2"]["groups"]
        yn, ab = g["yesno"]["gap_no_minus_yes"], g["ab_pooled"]["gap_no_minus_yes"]
        if not np.isfinite(yn) or not np.isfinite(ab) or yn <= 0:
            q1 = "INCONCLUSIVE (no positive Yes/No gap to shrink)"
        else:
            q1 = "word bias SUPPORTED" if abs(ab) <= yn / 2 else "word bias NOT SUPPORTED"
        print(f"\nQ1 v2 gap: Yes/No {yn:+.2f} -> A/B pooled {ab:+.2f}: {q1}")
        summary["Q1"] = q1

        emitted = config.RESULTS / f"slot_emitted_{lens_id}.json"
        if emitted.exists():
            old = json.loads(emitted.read_text())
            diffs = [abs(r["p_yes"] - old[r["condition"]][r["item_id"]]["p_yes"]) for r in rows
                     if r["source"] == "v2" and r["format"] == "yesno"
                     and r["item_id"] in old.get(r["condition"], {})]
            print(f"reproduction vs plan-2 slot_emitted (v2, Yes/No): max |dp_yes| {max(diffs):.4f}, "
                  f"median {np.median(diffs):.4f} over {len(diffs)} rows")
            summary["reproduction_max_abs_dp_yes"] = max(diffs)

    out = path.with_name(path.stem + "_report.json")
    out.write_text(json.dumps(summary, indent=1))
    print(f"\n-> {out}")


# --------------------------------------------------------------------------
# compare (local): misc/prereg_model_size_regate.md
# --------------------------------------------------------------------------

def compare(args) -> None:
    """The decision in misc/prereg_model_size_regate.md, from the two report files."""
    lens = {"4B": config.get_model_config("target").lens_id,
            "12B": config.get_model_config("escalate").lens_id}
    reps = {}
    for size, lens_id in lens.items():
        path = out_path(lens_id, args.sources)
        reps[size] = json.loads(path.with_name(path.stem + "_report.json").read_text())["sources"]

    keys = ("p_answer_A_under_D", "robust_lied", "robust_honest", "robust_arms_balanced")
    for source in args.sources:
        print(f"\n{source:<22} {'4B':>8} {'12B':>8}")
        for key in keys:
            print(f"  {key:<20} {reps['4B'][source][key]!s:>8} {reps['12B'][source][key]!s:>8}")

    small, large = reps["4B"]["alleged"], reps["12B"]["alleged"]
    yield_small = min(small["robust_lied"], small["robust_honest"])
    yield_large = min(large["robust_lied"], large["robust_honest"])
    criteria = {
        "1 yield >= 1.25x": yield_large > 0 and yield_large >= 1.25 * yield_small,
        "2 arms balanced": large["robust_arms_balanced"],
        "3 letter bias no worse": abs(large["p_answer_A_under_D"] - 0.5)
                                  <= abs(small["p_answer_A_under_D"] - 0.5) + 0.05,
    }
    print(f"\nalleged usable yield min(lied, honest): 4B {yield_small}, 12B {yield_large}")
    for name, ok in criteria.items():
        print(f"  {name:<24} {'yes' if ok else 'no'}")
    print(f"decision: {'switch to 12B' if all(criteria.values()) else 'stay on 4B'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report", "compare"))
    ap.add_argument("--preset", default="target",
                    help="config preset: debug (270m), target (4b), escalate (12b)")
    ap.add_argument("--sources", default="v2,alleged",
                    type=lambda s: [x for x in s.split(",") if x],
                    help=f"comma-separated, from {SOURCES}")
    ap.add_argument("--model-name", help="override the HF model id (local dry runs)")
    ap.add_argument("--limit", type=int, help="first N items of each source only (dry runs)")
    ap.add_argument("--out", help="results file (default results/pilot_gate_<lens>_<sources>.json)")
    args = ap.parse_args()
    unknown = set(args.sources) - set(SOURCES)
    if unknown:
        ap.error(f"unknown sources {sorted(unknown)}; choose from {SOURCES}")
    {"run": run, "report": report, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
