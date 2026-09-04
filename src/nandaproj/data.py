"""Dataset loading and cache I/O for behavioral data collection.

Cache schema:
- One .npz file per (dataset, condition) pair
- File: results/behavioral_{dataset}_{condition}.npz
- Keys:
    - item_ids: (N,) str array, unique identifiers for cross-reference
    - answers: (N,) int array, predicted answer index
    - correct: (N,) bool array, whether prediction matches ground truth
    - confidence: (N,) float32 array, verbalized confidence
    - activations_L{12,14,16,18,20,22,...}: (N, 3072) float32, residual activations
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np

# Add to PYTHONPATH at module level so imports work locally too
if os.path.exists("../../../"):
    import sys
    sys.path.insert(0, "../../../")

from nandaproj import config


# Dataset names
DATASETS = Literal["mmlu", "triviaqa"]
CONDITIONS = ["P1_separate", "P2_joint", "P3_introspective", "P4_third_person", "P5_deferred"]

# Layers to capture: last 12 layers, every other (initially)
# gemma-3-4b-it has 24 layers total (0-23), so last 12 are 12-23
LAYER_INDICES = [12, 14, 16, 18, 20, 22]  # Can be extended to all 24


def load_dataset_mmlu(n_items: int | None = None) -> dict:
    """Load MMLU from HF datasets.
    
    Args:
        n_items: If set, take first n_items from the validation split.
        
    Returns:
        dict with keys:
        - item_ids: list of identifiers
        - questions: list of question texts
        - options: list of (A, B, C, D) tuples
        - answers: list of correct answer indices (0-3)
        - splits: list of subject names
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets' library: pip install datasets")
    
    # Same trap as TriviaQA, smaller: the "all" config carries an
    # `auxiliary_train` split of ~100k rows that comes down whether or not we
    # ask for it. Streaming a bounded `n_items` skips it entirely.
    if n_items is not None:
        stream = load_dataset(
            "cais/mmlu", "all", split="validation", streaming=True,
        )
        rows = list(stream.take(n_items))
    else:
        rows = load_dataset("cais/mmlu", "all", split="validation")

    # MMLU format: question, subject, choices (A, B, C, D), answer (0-3)
    data = {
        "item_ids": [f"mmlu_{i:05d}" for i in range(len(rows))],
        "questions": [r["question"] for r in rows],
        "options": [r["choices"] for r in rows],  # [A, B, C, D] per item
        "answers": [r["answer"] for r in rows],  # Ground truth indices
        "splits": [r["subject"] for r in rows],  # Which MMLU subset
    }
    return data


# TriviaQA's default "rc" config ships the full reading-comprehension evidence
# documents -- 26 parquet shards, ~10 GB, downloaded in full before `split=` is
# ever applied. We use two fields, question and answer, so every byte of that is
# waste: billed GPU time spent filling an instance disk that dies with the box.
# "rc.nocontext" is the same questions and answers with the documents dropped.
TRIVIAQA_CONFIG = "rc.nocontext"


def load_dataset_triviaqa(n_items: int | None = None) -> dict:
    """Load TriviaQA from HF datasets.

    Args:
        n_items: If set, take first n_items from the validation split. Small
            values stream, so nothing is written to the HF cache at all.

    Returns:
        dict with keys:
        - item_ids: list of identifiers
        - questions: list of question texts
        - answers_text: list of correct answer texts (variable length)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets' library: pip install datasets")

    if n_items is not None:
        # Streaming pulls only the rows we ask for. For the 50-item baseline in
        # 01b this is the difference between seconds and a multi-GB download.
        stream = load_dataset(
            "mandarjoshi/trivia_qa", TRIVIAQA_CONFIG,
            split="validation", streaming=True,
        )
        rows = list(stream.take(n_items))
    else:
        rows = load_dataset(
            "mandarjoshi/trivia_qa", TRIVIAQA_CONFIG, split="validation",
        )

    # TriviaQA format: question, answer (with 'value' and aliases)
    data = {
        "item_ids": [f"tqa_{i:05d}" for i in range(len(rows))],
        "questions": [r["question"] for r in rows],
        "answers_text": [r["answer"]["value"] for r in rows],  # Primary answer
        "answer_aliases": [r["answer"]["aliases"] for r in rows],  # Alternatives
    }
    return data


def create_cache_file(
    dataset: str,
    condition: str,
    n_items: int = 5000,
    layer_indices: list[int] | None = None,
) -> Path:
    """Create and initialize an empty cache .npz file.
    
    Allocates storage but does not fill it. Call save_cache to write data.
    
    Args:
        dataset: "mmlu" or "triviaqa"
        condition: one of CONDITIONS
        n_items: number of items to allocate space for
        layer_indices: which layers to include; defaults to LAYER_INDICES
        
    Returns:
        Path to the created .npz file
    """
    if layer_indices is None:
        layer_indices = LAYER_INDICES
    
    config.ensure_dirs()
    cache_path = config.RESULTS / f"behavioral_{dataset}_{condition}.npz"
    
    # Pre-allocate arrays
    arrays = {
        "item_ids": np.empty(n_items, dtype=object),
        "answers": np.zeros(n_items, dtype=np.int32),
        "correct": np.zeros(n_items, dtype=np.bool_),
        "confidence": np.zeros(n_items, dtype=np.float32),
    }
    
    # Add activations for each layer
    hidden_dim = 3072  # gemma-3-4b-it
    for layer_idx in layer_indices:
        arrays[f"activations_L{layer_idx}"] = np.zeros((n_items, hidden_dim), dtype=np.float32)
    
    np.savez(cache_path, **arrays)
    return cache_path


def save_cache(
    dataset: str,
    condition: str,
    item_ids: np.ndarray,
    answers: np.ndarray,
    correct: np.ndarray,
    confidence: np.ndarray,
    activations: dict[int, np.ndarray],  # {layer_idx: (N, 3072) array}
) -> Path:
    """Save behavioral data to cache file.
    
    Args:
        dataset: "mmlu" or "triviaqa"
        condition: one of CONDITIONS
        item_ids: (N,) array of item identifiers
        answers: (N,) int array, predicted answer indices
        correct: (N,) bool array
        confidence: (N,) float32 array
        activations: dict mapping layer index → (N, 3072) activation array
        
    Returns:
        Path to saved .npz file
    """
    config.ensure_dirs()
    cache_path = config.RESULTS / f"behavioral_{dataset}_{condition}.npz"
    
    arrays = {
        "item_ids": item_ids,
        "answers": answers,
        "correct": correct,
        "confidence": confidence,
    }
    
    for layer_idx, acts in activations.items():
        arrays[f"activations_L{layer_idx}"] = acts
    
    np.savez(cache_path, **arrays)
    print(f"✓ Saved {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")
    return cache_path


def load_cache(dataset: str, condition: str) -> dict:
    """Load cached behavioral data.
    
    Args:
        dataset: "mmlu" or "triviaqa"
        condition: one of CONDITIONS
        
    Returns:
        dict with keys: item_ids, answers, correct, confidence,
        and activations_L{layer_idx} for each saved layer
    """
    cache_path = config.RESULTS / f"behavioral_{dataset}_{condition}.npz"
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    
    with np.load(cache_path, allow_pickle=True) as f:
        return {key: f[key] for key in f.files}


def cache_exists(dataset: str, condition: str) -> bool:
    """Check if a cache file exists for (dataset, condition)."""
    cache_path = config.RESULTS / f"behavioral_{dataset}_{condition}.npz"
    return cache_path.exists()


def get_cache_size(dataset: str, condition: str) -> float:
    """Return cache file size in MB, or 0 if not found."""
    cache_path = config.RESULTS / f"behavioral_{dataset}_{condition}.npz"
    if cache_path.exists():
        return cache_path.stat().st_size / 1e6
    return 0.0
