"""Additional tests for edge cases and platform-specific branches."""

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from llmbeans.cli import (
    _prompt_custom_path,
    _prompt_search_folder,
    generate_summary,
    prompt_hardware_selection,
    prompt_model_selection,
    write_scripts,
)
from llmbeans.hardware.detector import _detect_disk_is_ssd, detect_hardware
from llmbeans.hardware.estimator import estimate_max_context
from llmbeans.hardware.profiles import (
    HardwareProfileEntry,
    _score_profile_match,
    from_detection,
    get_profile_by_id,
    lookup_specs_for_detection,
)
from llmbeans.models.scanner import (
    ModelFormat,
    _count_params_from_index,
    _estimate_model_size_gb,
    _scan_gguf,
    _scan_hf_repo,
    _scan_safetensors,
    detect_format,
    scan,
)
from llmbeans.recommenders.tools.mlx import generate_flags as mlx_flags
from llmbeans.recommenders.tools.vllm import generate_flags as vllm_flags


class IgnoreSourceDir(dict):
    """Dict that ignores source_dir assignment for CLI display-path tests."""

    def __setitem__(self, key, value):
        if key == "source_dir":
            return
        super().__setitem__(key, value)


@dataclass
class FakeDetectedHardware:
    os: str = "linux"
    cpu_cores: int = 8
    ram_total_gb: float = 16
    ram_free_gb: float = 8
    gpu_vendor: str | None = "nvidia"
    gpu_vram_gb: float | None = 8.0
    gpu_name: str | None = "NVIDIA GeForce RTX 4060 Laptop GPU"
    is_apple_silicon: bool = False
    unified_memory: bool = False
    metal_supported: bool = False
    disk_is_ssd: bool = True
    laptop_model: str | None = None
    memory_bandwidth_gbps: float | None = None


def test_prompt_model_selection_relative_display(tmp_path):
    src = tmp_path / "models"
    src.mkdir()
    model = src / "nested" / "model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")
    with patch("llmbeans.cli._scan_models_in_dir", return_value=[{
        "path": str(model),
        "format": "GGUF",
        "size_gb": 0.1,
        "source_dir": str(src),
    }]), patch("llmbeans.cli.IntPrompt.ask", return_value=1), \
         patch("llmbeans.cli._resolve_model_dir", return_value=model):
        assert prompt_model_selection("llamacpp") == str(model)


def test_prompt_model_selection_display_without_source_dir(tmp_path):
    model = tmp_path / "orphan.gguf"
    model.write_bytes(b"x")
    entry = IgnoreSourceDir({
        "path": str(model),
        "format": "GGUF",
        "size_gb": 0.1,
    })
    with patch("llmbeans.cli.TOOL_MODEL_DIRS", {"llamacpp": [str(tmp_path)]}), \
         patch("llmbeans.cli._scan_models_in_dir", return_value=[entry]), \
         patch("llmbeans.cli.IntPrompt.ask", return_value=1), \
         patch("llmbeans.cli._resolve_model_dir", return_value=model):
        assert prompt_model_selection("llamacpp") == str(model)


def test_prompt_model_selection_invalid_choice_then_valid(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    with patch("llmbeans.cli.TOOL_MODEL_DIRS", {"llamacpp": [str(tmp_path)]}), \
         patch("llmbeans.cli._scan_models_in_dir", return_value=[{
             "path": str(model),
             "format": "GGUF",
             "size_gb": 0.1,
         }]), patch("llmbeans.cli.IntPrompt.ask", side_effect=[99, 1]), \
         patch("llmbeans.cli._resolve_model_dir", return_value=model):
        assert prompt_model_selection("llamacpp") == str(model)


def test_prompt_custom_path_display_outside_cwd(tmp_path):
    cwd = tmp_path / "work"
    cwd.mkdir()
    model = tmp_path / "elsewhere" / "remote.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")
    with patch("llmbeans.cli._working_directory", return_value=cwd), \
         patch("llmbeans.cli._scan_models_in_dir", return_value=[{
             "path": str(model),
             "format": "GGUF",
             "size_gb": 0.1,
         }]), patch("llmbeans.cli.IntPrompt.ask", return_value=1), \
         patch("llmbeans.cli._resolve_model_dir", return_value=model):
        assert _prompt_custom_path() == str(model)


def test_prompt_search_folder_not_a_directory(tmp_path):
    file_path = tmp_path / "file.gguf"
    file_path.write_bytes(b"x")
    model = tmp_path / "found.gguf"
    model.write_bytes(b"x")
    with patch("llmbeans.cli.Prompt.ask", side_effect=[str(file_path), str(tmp_path)]), \
         patch("llmbeans.cli._scan_models_in_dir", return_value=[{
             "path": str(model),
             "format": "GGUF",
             "size_gb": 0.1,
         }]), patch("llmbeans.cli.IntPrompt.ask", return_value=1), \
         patch("llmbeans.cli._resolve_model_dir", return_value=model):
        assert _prompt_search_folder() == str(model)


def test_prompt_search_folder_invalid_model_choice(tmp_path):
    model = tmp_path / "found.gguf"
    model.write_bytes(b"x")
    with patch("llmbeans.cli.Prompt.ask", return_value=str(tmp_path)), \
         patch("llmbeans.cli._scan_models_in_dir", return_value=[{
             "path": str(model),
             "format": "GGUF",
             "size_gb": 0.1,
         }]), patch("llmbeans.cli.IntPrompt.ask", side_effect=[99, 1]), \
         patch("llmbeans.cli._resolve_model_dir", return_value=model):
        assert _prompt_search_folder() == str(model)


def test_scan_models_index_json_only(tmp_path):
    shard = tmp_path / "index-only"
    shard.mkdir()
    (shard / "model.safetensors.index.json").write_text("{}")
    (shard / "weights.bin").write_bytes(b"x" * 32)
    from llmbeans.cli import _scan_models_in_dir
    assert _scan_models_in_dir(str(shard))


def test_prompt_search_folder_empty_then_valid(tmp_path):
    model = tmp_path / "found.gguf"
    model.write_bytes(b"x")
    with patch("llmbeans.cli.Prompt.ask", side_effect=["", str(tmp_path)]), \
         patch("llmbeans.cli._scan_models_in_dir", return_value=[{"path": str(model), "format": "GGUF", "size_gb": 0.1}]), \
         patch("llmbeans.cli.IntPrompt.ask", return_value=1):
        assert _prompt_search_folder() == str(model)


def test_prompt_hardware_selection_vram_profile():
    profile = get_profile_by_id("generic-rtx-4060-laptop")
    with patch("llmbeans.cli.IntPrompt.ask", return_value=1):
        assert prompt_hardware_selection([profile], auto_detected=None) == profile


def test_generate_summary_gpu_vram_and_no_usage(sample_model, nvidia_hardware, sample_recommendation):
    nvidia_hardware.unified_memory = False
    sample_recommendation.estimated_vram_usage_gb = 0
    sample_recommendation.estimated_ram_usage_gb = 0
    summary = generate_summary(sample_model, nvidia_hardware, sample_recommendation)
    assert "GB VRAM" in summary
    assert "VRAM usage" not in summary


def test_generate_summary_unified_memory(sample_model, apple_hardware, sample_recommendation):
    apple_hardware.unified_memory = True
    summary = generate_summary(sample_model, apple_hardware, sample_recommendation)
    assert "Unified memory" in summary


def test_write_scripts_ollama_and_lmstudio(sample_model, nvidia_hardware, sample_recommendation):
    sample_recommendation.hosting_tool = "ollama"
    sample_recommendation.extra_config = "FROM model"
    with tempfile.TemporaryDirectory() as tmpdir:
        assert "modelfile" in write_scripts(sample_recommendation, sample_model, nvidia_hardware, "s", tmpdir)
    sample_recommendation.hosting_tool = "lmstudio"
    sample_recommendation.extra_config = "{}"
    with tempfile.TemporaryDirectory() as tmpdir:
        assert "config" in write_scripts(sample_recommendation, sample_model, nvidia_hardware, "s", tmpdir)


def test_detect_hardware_platform_paths():
    with patch("llmbeans.hardware.detector.platform.system", return_value="Darwin"), \
         patch("llmbeans.hardware.detector.psutil", create=True) as mock_psutil, \
         patch("llmbeans.hardware.detector.subprocess.run", side_effect=OSError("fail")), \
         patch("llmbeans.hardware.detector._detect_disk_is_ssd", return_value=True), \
         patch("llmbeans.hardware.detector.lookup_specs_for_detection", return_value={"memory_bandwidth_gbps": 0}):
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3)
        assert detect_hardware().is_apple_silicon is False

    with patch("llmbeans.hardware.detector.platform.system", return_value="Windows"), \
         patch("llmbeans.hardware.detector.psutil", create=True) as mock_psutil, \
         patch("llmbeans.hardware.detector._detect_windows_gpu", side_effect=RuntimeError("fail")), \
         patch("llmbeans.hardware.detector.lookup_specs_for_detection", return_value={"memory_bandwidth_gbps": 0}):
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3)
        assert detect_hardware().gpu_name is None


def test_detect_disk_is_ssd_linux_and_windows():
    with patch("llmbeans.hardware.detector.platform.system", return_value="Linux"), \
         patch("llmbeans.hardware.detector.os.listdir", return_value=["sda"]), \
         patch("builtins.open", mock_open(read_data="0\n")):
        assert _detect_disk_is_ssd() is True

    with patch("llmbeans.hardware.detector.platform.system", return_value="Windows"), \
         patch("llmbeans.hardware.detector.subprocess.run", side_effect=OSError("fail")):
        assert _detect_disk_is_ssd() is True


def test_detect_disk_is_ssd_darwin_subprocess_failure():
    with patch("llmbeans.hardware.detector.platform.system", return_value="Darwin"), \
         patch("llmbeans.hardware.detector.subprocess.run", side_effect=OSError("fail")):
        assert _detect_disk_is_ssd() is True


def test_from_detection_profiler_branches():
    hw = FakeDetectedHardware(
        laptop_model="Mac16,1",
        is_apple_silicon=True,
        unified_memory=True,
        metal_supported=True,
        gpu_vendor="apple",
        gpu_name="Apple M4",
    )
    with patch("llmbeans.hardware.profiles.platform.system", return_value="Darwin"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = '{"SPHardwareDataType":[{"chip_type":"Apple M4"}]}'
        assert from_detection(hw).name == "Apple M4"

    with patch("llmbeans.hardware.profiles.platform.system", return_value="Darwin"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = '{"SPHardwareDataType":[{"machine_name":"MBP"}]}'
        assert from_detection(hw).name == "MBP"


def test_lookup_specs_without_profile_match():
    hw = FakeDetectedHardware(gpu_vendor="nvidia", gpu_name="RTX", ram_total_gb=1)
    with patch("llmbeans.hardware.profiles.match_profile_for_detection", return_value=None):
        assert lookup_specs_for_detection(hw)["ram_type"] == "DDR5"


def test_score_profile_match_gpu_substring_and_ram_near_miss():
    profile = HardwareProfileEntry(
        id="test-gpu",
        name="Test GPU",
        year=2024,
        ram_gb=20,
        ram_type="DDR5",
        cpu_cores=8,
        cpu_threads=16,
        gpu_type="discrete",
        gpu_name="RTX 4060 Laptop",
        gpu_vram_gb=8,
        gpu_cores=3072,
        unified_memory=False,
        memory_bandwidth_gbps=51.2,
        metal=False,
        cuda=True,
        cuda_compute="8.9",
        vram_bandwidth_gbps=84.96,
        ssd_recommended=True,
        category="windows_nvidia",
    )
    hw = FakeDetectedHardware(
        gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
        ram_total_gb=16,
    )
    score = _score_profile_match(hw, profile)
    assert score >= 12

    near_hw = FakeDetectedHardware(gpu_name=None, gpu_vendor=None, ram_total_gb=16)
    near_score = _score_profile_match(near_hw, profile)
    assert near_score >= 2


def test_lookup_specs_apple_m1_fallback_ram_type():
    hw = FakeDetectedHardware(
        gpu_vendor="apple",
        gpu_name="Apple M1",
        is_apple_silicon=True,
        unified_memory=True,
        metal_supported=True,
        ram_total_gb=16,
    )
    with patch("llmbeans.hardware.profiles.match_profile_for_detection", return_value=None):
        assert lookup_specs_for_detection(hw)["ram_type"] == "LPDDR4X"


def test_lookup_specs_apple_m2_fallback_ram_type():
    hw = FakeDetectedHardware(
        gpu_vendor="apple",
        gpu_name="Apple M2",
        is_apple_silicon=True,
        unified_memory=True,
        metal_supported=True,
        ram_total_gb=16,
    )
    with patch("llmbeans.hardware.profiles.match_profile_for_detection", return_value=None):
        assert lookup_specs_for_detection(hw)["ram_type"] == "LPDDR5"


def test_estimate_max_context_returns_512(apple_hardware, sample_model):
    apple_hardware.ram_gb = 4
    sample_model.num_layers = 400
    sample_model.hidden_size = 8192
    sample_model.num_attention_heads = 32
    sample_model.num_key_value_heads = 8
    assert estimate_max_context(sample_model, apple_hardware, gpu_offload_layers=32) >= 512


def test_scan_detect_format_variants(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    sub = root / "child"
    sub.mkdir()
    (sub / "weights.safetensors").write_bytes(b"x")
    assert detect_format(str(root)) == ModelFormat.SAFETENSORS

    only_index = tmp_path / "idx"
    only_index.mkdir()
    (only_index / "model.safetensors.index.json").write_text("{}")
    assert detect_format(str(only_index)) == ModelFormat.SAFETENSORS

    nested_index = tmp_path / "nested-index"
    nested_index.mkdir()
    nested_sub = nested_index / "shard"
    nested_sub.mkdir()
    (nested_sub / "model.safetensors.index.json").write_text("{}")
    assert detect_format(str(nested_index)) == ModelFormat.SAFETENSORS


def test_estimate_model_size_gb_missing_local_path():
    assert _estimate_model_size_gb("/definitely/missing/model.gguf", ModelFormat.GGUF) == 0.0


def test_count_params_from_index_empty_and_missing_shard(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _count_params_from_index(empty) == 0

    sharded = tmp_path / "sharded"
    sharded.mkdir()
    (sharded / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.weight": "missing.safetensors"},
    }))
    assert _count_params_from_index(sharded) == 0


def test_scan_safetensors_quant_and_index(tmp_path, monkeypatch):
    model_dir = tmp_path / "indexed"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "vocab_size": 1000}))
    (model_dir / "model.safetensors").write_bytes(b"x" * 64)
    monkeypatch.setattr("llmbeans.models.scanner._count_params_from_index", lambda path: 2_000_000_000)
    assert _scan_safetensors(str(model_dir)).parameter_count == 2.0

    bits_dir = tmp_path / "bits"
    bits_dir.mkdir()
    (bits_dir / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "vocab_size": 1000,
        "quantization_config": {"bits": 4},
    }))
    (bits_dir / "model.safetensors").write_bytes(b"x" * 64)
    assert _scan_safetensors(str(bits_dir)).quant_bits == 4.0


def test_scan_gguf_default_vocab_and_file_type_quant(tmp_path):
    gguf = tmp_path / "plain.gguf"
    gguf.write_bytes(b"x")

    class StringField:
        def __init__(self, value):
            encoded = value.encode() + b"\x00"
            self.parts = [0, b"n", type("T", (), {"name": "STRING"})(), len(encoded), encoded]
            self.types = [type("T", (), {"name": "STRING"})()]

    class IntField:
        def __init__(self, value):
            self.parts = [0, 0, type("T", (), {"name": "UINT32"})(), np.array([value])]
            self.types = [type("T", (), {"name": "UINT32"})()]
            self.data = []

    reader = MagicMock()
    reader.fields = {
        "general.architecture": StringField("llama"),
        "general.file_type": StringField("Q4_K_M"),
        "llama.context_length": IntField(8192),
        "llama.embedding_length": IntField(4096),
        "llama.attention.head_count": IntField(32),
        "llama.attention.head_count_kv": IntField(8),
        "llama.feed_forward_length": IntField(11008),
    }
    reader.tensors = [MagicMock(name="blk.0.weight", n_elements=100)]
    reader.tensors[0].name = "blk.0.weight"

    with patch("gguf.GGUFReader", return_value=reader):
        info = _scan_gguf(str(gguf))
    assert info.vocab_size == 32000
    assert info.quant_method == "Q4_K_M"


def test_scan_gguf_with_array_vocab(tmp_path):
    gguf = tmp_path / "meta.gguf"
    gguf.write_bytes(b"x")

    class ArrayField:
        parts = []
        types = [type("T", (), {"name": "ARRAY"})()]
        data = [1, 2, 3, 4, 5]

    class StringField:
        parts = [0, b"n", type("T", (), {"name": "STRING"})(), 5, b"llama\x00"]
        types = [type("T", (), {"name": "STRING"})()]

    class IntField:
        def __init__(self, value):
            self.parts = [0, 0, type("T", (), {"name": "UINT32"})(), np.array([value])]
            self.types = [type("T", (), {"name": "UINT32"})()]
            self.data = []

    reader = MagicMock()
    reader.fields = {
        "general.architecture": StringField(),
        "llama.context_length": IntField(8192),
        "llama.embedding_length": IntField(4096),
        "llama.attention.head_count": IntField(32),
        "llama.attention.head_count_kv": IntField(8),
        "llama.feed_forward_length": IntField(11008),
        "tokenizer.ggml.tokens": ArrayField(),
    }
    reader.tensors = [MagicMock(name="blk.0.weight", n_elements=100)]
    reader.tensors[0].name = "blk.0.weight"

    with patch("gguf.GGUFReader", return_value=reader):
        info = _scan_gguf(str(gguf))
    assert info.vocab_size == 5


def test_scan_hf_repo_remote_paths():
    fake_info = MagicMock(
        id="org/model",
        siblings=[SimpleNamespace(rfilename="model.safetensors", size=2048 * 1024 * 1024)],
    )
    with patch("huggingface_hub.HfApi") as mock_api_cls, \
         patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("no config")):
        mock_api_cls.return_value.model_info.return_value = fake_info
        assert _scan_hf_repo("org/model").model_size_gb > 0


def test_scan_hf_repo_quant_config_from_config_only():
    fake_info = MagicMock(
        id="org/quant-model",
        siblings=[SimpleNamespace(rfilename="model.safetensors", size=1024 * 1024 * 1024)],
    )
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
        info = _scan_hf_repo("org/quant-model")
    assert info.quant_method == "GPTQ_4bit"
    assert info.quant_bits == 4.0


def test_scan_routing(monkeypatch, tmp_path):
    fake = MagicMock()
    monkeypatch.setattr("llmbeans.models.scanner._scan_safetensors", lambda s: fake)
    monkeypatch.setattr("llmbeans.models.scanner.detect_format", lambda s: ModelFormat.SAFETENSORS)
    assert scan(str(tmp_path)) is fake
    monkeypatch.setattr("llmbeans.models.scanner.detect_format", lambda s: ModelFormat.HF_REPO)
    monkeypatch.setattr("llmbeans.models.scanner._scan_hf_repo", lambda s: fake)
    assert scan("org/model") is fake


def test_output_generate_summary_unified(apple_hardware, sample_model, sample_recommendation):
    from llmbeans.output.script_gen import generate_summary as output_summary
    apple_hardware.gpu_vram_gb = None
    text = output_summary(sample_model, apple_hardware, sample_recommendation)
    assert "Unified memory" in text


def test_mlx_safetensors_and_vllm_branches(sample_model, apple_hardware, nvidia_hardware):
    sample_model.format = ModelFormat.SAFETENSORS
    sample_model.is_remote = False
    assert sample_model.source_path in mlx_flags(
        model=sample_model, hardware=apple_hardware, gpu_offload_layers=32,
        context_length=4096, batch_size=512, thread_count=8, quality_mode="balanced",
    )["command"]

    sample_model.quant_method = "Q4_0"
    result = vllm_flags(
        model=sample_model, hardware=nvidia_hardware, gpu_offload_layers=32,
        context_length=4096, batch_size=512, thread_count=8, quality_mode="balanced",
    )
    assert result["flags"]["--quantization"] == "q4_0"
