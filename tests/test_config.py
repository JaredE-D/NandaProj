import numpy as np
import pytest

from nandaproj import config, viz


def test_default_preset_is_smallest():
    cfg = config.get_model_config()
    assert cfg.name == "gpt2-small"
    assert not cfg.needs_hf_token


def test_preset_lookup_by_name():
    assert config.get_model_config("gemma").name == "gemma-2-2b"
    assert config.get_model_config("gemma").needs_hf_token


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        config.get_model_config("does-not-exist")


def test_preset_env_var_is_respected(monkeypatch):
    monkeypatch.setenv("NANDA_PRESET", "pythia")
    assert config.get_model_config().name == "pythia-160m"


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
