"""Tests for llmbeans.recommenders.engine and registry."""

import pytest
from dataclasses import replace

from llmbeans.models.scanner import ModelFormat, ModelInfo
from llmbeans.recommenders.engine import recommend
from llmbeans.recommenders.registry import get_available_tools, get_tool_generator, register_tool


def test_registry_helpers():
    tools = get_available_tools()
    assert "llamacpp" in tools
    assert get_tool_generator("llamacpp") is not None
    assert get_tool_generator("missing-tool") is None

    @register_tool("test-tool-temp")
    def _temp_tool(**kwargs):
        return {"flags": {}, "command": "test", "extra_config": None}

    assert "test-tool-temp" in get_available_tools()


def test_recommend_llamacpp_balanced(sample_model, nvidia_hardware):
    rec = recommend(sample_model, nvidia_hardware, "llamacpp", "balanced")
    assert rec.hosting_tool == "llamacpp"
    assert rec.context_length >= 512
    assert rec.command.startswith("llama-cli")
    assert rec.flags["-ngl"] != "0"


def test_recommend_quality_and_speed_modes(sample_model, nvidia_hardware):
    quality = recommend(sample_model, nvidia_hardware, "llamacpp", "quality")
    speed = recommend(sample_model, nvidia_hardware, "llamacpp", "speed")
    assert quality.batch_size == 1024
    assert speed.batch_size == 256


def test_recommend_apple_unified_offload(sample_model, apple_hardware):
    rec = recommend(sample_model, apple_hardware, "llamacpp", "balanced")
    assert rec.gpu_offload_layers == sample_model.num_layers
    assert rec.estimated_ram_usage_gb == 0.0


def test_recommend_partial_gpu_offload(sample_model, nvidia_hardware):
    sample_model.estimated_vram_gb = 20
    rec = recommend(sample_model, nvidia_hardware, "llamacpp", "balanced")
    assert rec.gpu_offload_layers < sample_model.num_layers
    assert any("Only" in warning for warning in rec.warnings)


def test_recommend_no_gpu(sample_model, nvidia_hardware):
    hw = replace(nvidia_hardware, gpu_vram_gb=None, unified_memory=False, cuda=False)
    rec = recommend(sample_model, hw, "llamacpp", "balanced")
    assert rec.gpu_offload_layers == 0


def test_recommend_low_quant_warning(sample_model, nvidia_hardware):
    sample_model.quant_bits = 2.5
    sample_model.quant_method = "Q2_K"
    rec = recommend(sample_model, nvidia_hardware, "llamacpp", "balanced")
    assert any("Low quantization" in w for w in rec.warnings)


def test_recommend_slow_speed_warning(sample_model, nvidia_hardware):
    sample_model.parameter_count = 100
    rec = recommend(sample_model, nvidia_hardware, "llamacpp", "balanced")
    assert any("very slow" in w for w in rec.warnings)


def test_recommend_unified_memory_utilization_warning(sample_model, apple_hardware):
    sample_model.estimated_vram_gb = 1.0
    rec = recommend(sample_model, apple_hardware, "llamacpp", "balanced")
    assert any("smaller than available unified memory" in w for w in rec.warnings)


def test_recommend_ram_pressure_warning(sample_model, nvidia_hardware):
    sample_model.estimated_vram_gb = 100
    sample_model.parameter_count = 100
    rec = recommend(sample_model, nvidia_hardware, "llamacpp", "balanced")
    assert any("exceed available RAM" in w for w in rec.warnings)


def test_recommend_unknown_tool_raises(sample_model, nvidia_hardware):
    with pytest.raises(ValueError, match="Unknown hosting tool"):
        recommend(sample_model, nvidia_hardware, "not-a-tool", "balanced")


def test_recommend_context_rounding_to_512(sample_model, apple_hardware):
    apple_hardware.ram_gb = 4
    rec = recommend(sample_model, apple_hardware, "llamacpp", "speed")
    assert rec.context_length in {512, 1024, 2048, 4096}
