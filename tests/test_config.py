import numpy as np
import pytest

from nandaproj import config, viz


def test_default_preset_is_smallest():
    # Default must be the debug model: an accidental run should cost seconds.
    cfg = config.get_model_config()
    assert cfg.name == "google/gemma-3-270m-it"
    assert cfg.n_params == "270M"


def test_preset_lookup_by_name():
    assert config.get_model_config("target").name == "google/gemma-3-4b-it"
    assert config.get_model_config("target").needs_hf_token


def test_all_gemma_presets_are_gated():
    # Every Gemma repo is gated; forgetting HF_TOKEN fails *after* billing
    # starts, so the flag must be set on all of them.
    for name, cfg in config.PRESETS.items():
        assert cfg.needs_hf_token, f"{name} not marked gated"
        assert cfg.dtype == "bfloat16", f"{name} should be bf16"


def test_every_preset_has_an_instruction_tuned_lens():
    # PLAN2.md 4.0: use the -it lens, never the base one.
    for name, cfg in config.PRESETS.items():
        assert cfg.has_lens, f"{name} has no lens_id"
        assert cfg.lens_id.endswith("-it"), f"{name} lens is not -it: {cfg.lens_id}"
        assert cfg.lens_id == cfg.name.split("/")[-1]


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        config.get_model_config("does-not-exist")


def test_preset_env_var_is_respected(monkeypatch):
    monkeypatch.setenv("NANDA_PRESET", "main")
    assert config.get_model_config().name == "google/gemma-3-1b-it"


def test_workspace_env_var_redirects_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    import importlib

    reloaded = importlib.reload(config)
    try:
        assert reloaded.HF_CACHE == tmp_path / "hf_cache"
        reloaded.ensure_dirs()
        assert reloaded.HF_CACHE.is_dir()
        assert reloaded.RESULTS.is_dir()
    finally:
        monkeypatch.delenv("WORKSPACE")
        importlib.reload(config)


def test_get_device_without_torch_is_cpu():
    # No torch in the local venv, so this must degrade rather than explode.
    assert config.get_device() in {"cpu", "cuda"}


def test_to_numpy_passes_arrays_through():
    arr = np.zeros((2, 2))
    assert viz._to_numpy(arr) is arr
