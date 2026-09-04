"""Single place to switch model scale, paths, and device.

Imported by notebooks on the *remote* kernel and by tests locally, so nothing
here may import torch at module level -- the local venv has no torch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# On the remote box WORKSPACE is /workspace on the *instance disk*, which dies
# with the instance -- there is no persistent volume, so weights re-download
# each rental (~8 GB, ~5 min). Locally it falls back to the repo root.
WORKSPACE = Path(os.environ.get("WORKSPACE", Path(__file__).resolve().parents[2]))
HF_CACHE = WORKSPACE / "hf_cache"
RESULTS = WORKSPACE / "results"

# The repo itself, which is **not** WORKSPACE on the box: provision.sh sets
# WORKSPACE=/workspace and `just push` rsyncs the repo to /workspace/NandaProj.
# So caches and results sit *beside* the checkout, deliberately -- results/ is
# excluded from the push and synced back separately, and neither survives the
# instance. Anything that ships *with* the code (an item bank, a fixture) must
# be found relative to REPO; deriving it from WORKSPACE resolves to
# /workspace/data on the box and works fine locally, which is the worst kind of
# path bug -- it only fails once a GPU is already billing.
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"


@dataclass(frozen=True)
class ModelConfig:
    """A model we might run. `name` is the TransformerLens identifier."""

    name: str
    n_params: str
    dtype: str = "float32"
    gated: bool = False
    lens_id: str | None = None  # subdir in neuronpedia/jacobian-lens, if fitted

    @property
    def needs_hf_token(self) -> bool:
        return self.gated

    @property
    def has_lens(self) -> bool:
        return self.lens_id is not None


# Ordered smallest first. Start at the top, move down only when the science
# demands it -- iteration speed is the scarce resource in a 20-hour project.
# PLAN2.md 4.0: debug on 270m-it, run the experiment on 4b-it, escalate to
# 12b-it only if V2 fails (the model will not lie convincingly at 4B -- R1).
# Instruction-tuned throughout: the -it lens, never the base one.
PRESETS: dict[str, ModelConfig] = {
    "debug": ModelConfig(
        "google/gemma-3-270m-it", "270M",
        dtype="bfloat16", gated=True, lens_id="gemma-3-270m-it",
    ),
    "main": ModelConfig(
        "google/gemma-3-1b-it", "1B",
        dtype="bfloat16", gated=True, lens_id="gemma-3-1b-it",
    ),
    "target": ModelConfig(
        "google/gemma-3-4b-it", "4B",
        dtype="bfloat16", gated=True, lens_id="gemma-3-4b-it",
    ),
    "escalate": ModelConfig(
        "google/gemma-3-12b-it", "12B",
        dtype="bfloat16", gated=True, lens_id="gemma-3-12b-it",
    ),
}

# Debug is the default so an accidental run costs seconds, not GPU-minutes.
DEFAULT_PRESET = "debug"

LENS_REPO = "neuronpedia/jacobian-lens"


def get_model_config(preset: str | None = None) -> ModelConfig:
    """Look up a preset, defaulting to the env var then to `tiny`."""
    key = preset or os.environ.get("NANDA_PRESET", DEFAULT_PRESET)
    if key not in PRESETS:
        raise KeyError(f"unknown preset {key!r}; choose from {sorted(PRESETS)}")
    return PRESETS[key]


def get_device() -> str:
    """Return the torch device string, or 'cpu' where torch/CUDA is absent."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dirs() -> None:
    """Create the cache and results directories if they don't exist."""
    for path in (HF_CACHE, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
