"""Tests for llmbeans.models.scanner."""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from llmbeans.models.scanner import (
    ModelFormat,
    _count_params_from_index,
    _estimate_model_size_gb,
    _estimate_vram_gb,
    _get_quant_from_filename,
    _infer_architecture,
    _infer_param_count,
    _scan_gguf,
    _scan_hf_repo,
    _scan_safetensors,
    detect_format,
    scan,
    scan,
)


def test_detect_format_gguf_file(tmp_path):
    model = tmp_path / "model-Q4_K_M.gguf"
    model.write_bytes(b"gguf")
    assert detect_format(str(model)) == ModelFormat.GGUF


def test_detect_format_safetensors_file(tmp_path):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"st")
    assert detect_format(str(model)) == ModelFormat.SAFETENSORS


def test_detect_format_safetensors_dir(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"st")
    assert detect_format(str(tmp_path)) == ModelFormat.SAFETENSORS


def test_detect_format_safetensors_index_dir(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    assert detect_format(str(tmp_path)) == ModelFormat.SAFETENSORS


def test_detect_format_nested_safetensors_dir(tmp_path):
    nested = tmp_path / "nested-model"
    nested.mkdir()
    (nested / "model.safetensors").write_bytes(b"st")
    assert detect_format(str(tmp_path)) == ModelFormat.SAFETENSORS


def test_detect_format_hf_repo():
    assert detect_format("org/model-name") == ModelFormat.HF_REPO


def test_detect_format_unknown_raises():
    with pytest.raises(ValueError, match="Cannot detect"):
        detect_format("definitely-not-a-model-path")


def test_infer_architecture_mappings():
    assert _infer_architecture({"architectures": ["LlamaForCausalLM"]}) == "llama"
    assert _infer_architecture({"model_type": "custom_arch"}) == "custom_arch"
    assert _infer_architecture({}) == "unknown"


def test_infer_param_count_direct_fields():
    assert _infer_param_count({"num_parameters": 7_000_000_000}) == pytest.approx(7.0)
    assert _infer_param_count({"n_params": 3_000_000_000}) == pytest.approx(3.0)


def test_infer_param_count_from_architecture():
    config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    assert _infer_param_count(config) > 0


def test_infer_param_count_returns_zero():
    assert _infer_param_count({}) == 0.0


def test_estimate_model_size_gb_file_and_dir(tmp_path):
    file_path = tmp_path / "model.gguf"
    file_path.write_bytes(b"x" * 1024)
    assert _estimate_model_size_gb(str(file_path), ModelFormat.GGUF) > 0

    nested = tmp_path / "dir"
    nested.mkdir()
    (nested / "a.bin").write_bytes(b"x" * 2048)
    assert _estimate_model_size_gb(str(nested), ModelFormat.SAFETENSORS) > 0
    assert _estimate_model_size_gb("missing/repo", ModelFormat.HF_REPO) == 0.0


def test_estimate_vram_gb():
    assert _estimate_vram_gb(7.0, 4.0) > 0
    assert _estimate_vram_gb(7.0, None) > 0


def test_get_quant_from_filename():
    method, bits = _get_quant_from_filename("llama-7b-Q4_K_M.gguf")
    assert method == "Q4_K_M"
    assert bits == 4.5
    assert _get_quant_from_filename("plain.gguf") == (None, None)


def test_scan_safetensors_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "intermediate_size": 11008,
        "max_position_embeddings": 8192,
        "quantization_config": {"quant_method": "awq", "bits": 4},
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"x" * 4096)

    info = _scan_safetensors(str(model_dir))
    assert info.architecture == "llama"
    assert info.quant_method.startswith("AWQ")


def test_scan_safetensors_with_text_config(tmp_path):
    model_dir = tmp_path / "gemma"
    model_dir.mkdir()
    config = {
        "text_config": {
            "architectures": ["Gemma4ForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 8,
            "vocab_size": 256000,
            "max_position_embeddings": 8192,
        }
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"x" * 1024)
    info = _scan_safetensors(str(model_dir))
    assert info.architecture == "gemma4"


def test_scan_safetensors_missing_config_raises(tmp_path):
    model_dir = tmp_path / "empty"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="config.json not found"):
        _scan_safetensors(str(model_dir))


def test_count_params_from_index_single_file(tmp_path, monkeypatch):
    sf = tmp_path / "model.safetensors"
    sf.write_bytes(b"x" * 16)

    class FakeST:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def keys(self):
            return ["weight"]

        def get_tensor(self, name):
            return np.zeros((10, 10))

    monkeypatch.setattr("safetensors.safe_open", FakeST)
    assert _count_params_from_index(tmp_path) == 100


def test_count_params_from_index_sharded(tmp_path, monkeypatch):
    model_dir = tmp_path / "sharded"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x")
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.weight": "model-00001-of-00002.safetensors"}
    }))

    class FakeST:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_tensor(self, name):
            return np.zeros((4, 4))

    monkeypatch.setattr("safetensors.safe_open", FakeST)
    assert _count_params_from_index(model_dir) == 16


def test_scan_gguf_mocked(tmp_path):
    gguf = tmp_path / "llama-Q4_K_M.gguf"
    gguf.write_bytes(b"gguf")

    tensor = MagicMock()
    tensor.name = "blk.0.attn.weight"
    tensor.n_elements = 1000

    class FakeField:
        def __init__(self, value, type_name="UINT32"):
            self.parts = [None, None, MagicMock(name=type_name), np.array([value])]
            self.types = [MagicMock(name=type_name)]
            self.data = []

    reader = MagicMock()
    reader.fields = {
        "general.architecture": FakeField("llama", "STRING"),
        "general.parameter_count": FakeField(7.0, "FLOAT32"),
        "llama.embedding_length": FakeField(4096),
        "llama.attention.head_count": FakeField(32),
        "llama.attention.head_count_kv": FakeField(8),
        "llama.feed_forward_length": FakeField(11008),
        "tokenizer.ggml.tokens": FakeField(32000),
        "llama.context_length": FakeField(8192),
    }
    reader.fields["general.architecture"].parts = [0, b"name", MagicMock(name="STRING"), 5, b"llama\x00"]
    reader.fields["general.architecture"].types = [MagicMock(name="STRING")]
    reader.tensors = [tensor]

    with patch("gguf.GGUFReader", return_value=reader):
        info = _scan_gguf(str(gguf))

    assert info.format == ModelFormat.GGUF
    assert info.num_layers == 1
    assert info.context_length == 8192


def test_scan_hf_repo_mocked():
    fake_info = MagicMock()
    fake_info.id = "org/model"
    fake_info.siblings = [
        MagicMock(rfilename="model-Q4_K_M.gguf", size=1024**3),
        MagicMock(rfilename="model.safetensors", size=2048),
    ]

    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 8192,
        "quantization_config": {"quant_method": "gptq", "bits": 4},
    }

    with patch("huggingface_hub.HfApi") as mock_api_cls, \
         patch("huggingface_hub.hf_hub_download", return_value="/tmp/config.json"), \
         patch("builtins.open", mock_open(read_data=json.dumps(config))):
        mock_api_cls.return_value.model_info.return_value = fake_info
        info = _scan_hf_repo("org/model")

    assert info.is_remote is True
    assert info.quant_method == "Q4_K_M"


def test_scan_hf_repo_failure():
    with patch("huggingface_hub.HfApi") as mock_api_cls:
        mock_api_cls.return_value.model_info.side_effect = RuntimeError("offline")
        with pytest.raises(ValueError, match="Cannot access HF repo"):
            _scan_hf_repo("org/model")


def test_scan_routes_formats(tmp_path, monkeypatch):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x")
    fake_info = MagicMock(name="gguf-model", architecture="llama")
    monkeypatch.setattr("llmbeans.models.scanner._scan_gguf", lambda source: fake_info)
    assert scan(str(gguf)) is fake_info


def test_scan_unsupported_format(monkeypatch):
    class FakeFormat:
        pass

    monkeypatch.setattr("llmbeans.models.scanner.detect_format", lambda source: FakeFormat())
    with pytest.raises(ValueError, match="Unsupported format"):
        scan("anything")
