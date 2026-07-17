import json
from types import SimpleNamespace

import pytest

from coreai_models.export import bundle


def test_bundle_preserves_model_generation_config(tmp_path, monkeypatch):
    source = tmp_path / "source-generation-config.json"
    source.write_text(
        json.dumps(
            {
                "do_sample": True,
                "temperature": 0.7,
                "top_k": 20,
                "top_p": 0.8,
            }
        )
    )
    monkeypatch.setattr(bundle, "hf_hub_download", lambda **_: str(source))
    monkeypatch.setattr(bundle, "_write_tokenizer", lambda *_: None)

    bundle.bundle_llm_asset(
        bundle_path=tmp_path / "bundle",
        hf_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        hf_config=SimpleNamespace(vocab_size=100, max_position_embeddings=4096),
        compression="none",
        name="test-model",
    )

    preserved = json.loads((tmp_path / "bundle/generation_config.json").read_text())
    assert preserved["temperature"] == 0.7
    assert preserved["top_k"] == 20
    assert preserved["top_p"] == 0.8


def test_bundle_requires_model_generation_config(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "hf_hub_download", lambda **_: (_ for _ in ()).throw(OSError("missing")))
    monkeypatch.setattr(bundle, "_write_tokenizer", lambda *_: None)

    with pytest.raises(bundle.GenerationConfigurationError, match="missing required"):
        bundle.bundle_llm_asset(
            bundle_path=tmp_path / "bundle",
            hf_model_id="Qwen/Qwen3-0.6B",
            hf_config=SimpleNamespace(vocab_size=100, max_position_embeddings=4096),
            compression="none",
            name="test-model",
        )


def test_bundle_rejects_invalid_model_generation_config(tmp_path, monkeypatch):
    source = tmp_path / "source-generation-config.json"
    source.write_text(json.dumps({"temperature": 0.6}))
    monkeypatch.setattr(bundle, "hf_hub_download", lambda **_: str(source))
    monkeypatch.setattr(bundle, "_write_tokenizer", lambda *_: None)

    with pytest.raises(bundle.GenerationConfigurationError, match="boolean do_sample"):
        bundle.bundle_llm_asset(
            bundle_path=tmp_path / "bundle",
            hf_model_id="Qwen/Qwen3-0.6B",
            hf_config=SimpleNamespace(vocab_size=100, max_position_embeddings=4096),
            compression="none",
            name="test-model",
        )
