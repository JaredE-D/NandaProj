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
# On the remote box WORKSPACE is the persistent vast.ai volume (/workspace),
# so the HF cache survives `just down`. Locally it falls back to the repo root.
WORKSPACE = Path(os.environ.get("WORKSPACE", Path(__file__).resolve().parents[2]))
HF_CACHE = WORKSPACE / "hf_cache"
RESULTS = WORKSPACE / "results"


@dataclass(frozen=True)
class ModelConfig:
    """A model we might run. `name` is the TransformerLens identifier."""

    name: str
    n_params: str
    dtype: str = "float32"
    gated: bool = False

    @property
    def needs_hf_token(self) -> bool:
        return self.gated


# Ordered smallest first. Start at the top, move down only when the science
# demands it -- iteration speed is the scarce resource in a 20-hour project.
PRESETS: dict[str, ModelConfig] = {
    "tiny": ModelConfig("gpt2-small", "124M"),
    "small": ModelConfig("gpt2-medium", "355M"),
    "pythia": ModelConfig("pythia-160m", "160M"),
    "gemma": ModelConfig("gemma-2-2b", "2.6B", dtype="bfloat16", gated=True),
}

DEFAULT_PRESET = "tiny"


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
