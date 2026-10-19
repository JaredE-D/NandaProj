"""Zero-shot J-lens lie score: probability mass on a fixed "incorrect" word list, no fitting.

Exploratory addendum to `misc/prereg_12b_same_token.md` ("Zero-shot J-lens addendum").
Reads the top-50 store written by `jlens_12b.py run`; numpy only, runs locally.

  uv run python scripts/jlens_zeroshot_12b.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jlens_12b as jl
from nandaproj import baselines, config

WORDS = ("error", "errors", "wrong", "incorrect", "false", "mistake", "mistaken",
         "doubt", "untrue", "inaccurate")
N_NULL = 1000
OUT = config.RESULTS / "jlens_zeroshot_12b_gemma-3-12b-it.json"


def word_ids() -> np.ndarray:
    """Every vocab id decoding (stripped, lower-cased) to a list word.

    Gemma 3 shares one tokenizer across sizes; the 270m copy is cached locally.
    """
    path = glob.glob(str(ROOT / "hf_cache/models--unsloth--gemma-3-270m-it/snapshots/*/tokenizer.json"))[0]
    vocab = json.loads(Path(path).read_text())["model"]["vocab"]
    pieces = vocab.items() if isinstance(vocab, dict) else ((t[0], i) for i, t in enumerate(vocab))
    return np.array(sorted(i for t, i in pieces if t.replace("▁", " ").strip().lower() in WORDS))


def mass(ids: np.ndarray, lp: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per row: summed probability of `targets` among the stored top-k (0 where none appear)."""
    return np.where(np.isin(ids, targets), np.exp(lp), 0.0).sum(-1)


def main() -> None:
    import lie_probe_12b as lp

    rows_all = json.loads(jl.CAPTURE.with_suffix(".json").read_text())["rows"]
    T = np.load(jl.TOPK)
    idx, layers = T["rows"], [int(l) for l in T["layers"]]
    rows = [rows_all[i] for i in idx]
    y = np.array([r["label"] == "lied" for r in rows])
    truth = np.array([r["truth_yes"] for r in rows])
    letter_a = np.array([r["emitted_letter"] == "A" for r in rows])
    items = np.array([r["item_id"] for r in rows])
    targets = word_ids()
    print(f"{len(targets)} vocab ids for {len(WORDS)} words; {len(rows)} rows")

    rng = np.random.default_rng(jl.SEED)
    perms = [lp.permute_items_within_arms(y, items, truth, rng) for _ in range(N_NULL)]
    out = {"words": WORDS, "n_ids": int(len(targets)), "layers": layers}
    for lens in ("j", "l"):
        scores = np.stack([mass(T[f"{lens}_ids"][:, li], T[f"{lens}_lp"][:, li], targets)
                           for li in range(len(layers))])                 # [L, n]
        auc = np.array([baselines.auroc(s, y) for s in scores])
        null_max = np.array([max(baselines.auroc(s, p) for s in scores) for p in perms])
        best = int(np.argmax(auc))
        s = scores[best]
        cells = {f"truth_{'yes' if t else 'no'}_emit_{'A' if a else 'B'}":
                 baselines.auroc(s[(truth == t) & (letter_a == a)], y[(truth == t) & (letter_a == a)])
                 for t in (True, False) for a in (True, False)}
        # Balanced accuracy needs a threshold; the median score is the label-free choice.
        bal = baselines.balanced(s > np.median(s), y)
        nonzero = float((s > 0).mean())
        out[lens] = {"auroc": dict(zip(map(str, layers), auc.round(4).tolist())),
                     "null_p95_max_auroc": float(np.percentile(null_max, 95)),
                     "best_layer": layers[best], "best_auroc": float(auc[best]),
                     "balanced_at_median": bal, "cells_auroc": cells,
                     "frac_rows_with_any_list_token": nonzero}
        name = "J-lens" if lens == "j" else "logit lens"
        print(f"\n{name}: AUROC per layer " + " ".join(f"L{l}:{a:.2f}" for l, a in zip(layers, auc)))
        print(f"  best L{layers[best]} AUROC {auc[best]:.3f}; null p95 (max over layers) "
              f"{out[lens]['null_p95_max_auroc']:.3f}; balanced at median {bal:.3f}; "
              f"rows with any list token {nonzero:.2f}")
        print(f"  cells: { {k: round(v, 3) for k, v in cells.items()} }")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
