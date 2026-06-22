"""Tests for llmbeans.hardware.estimator."""

from types import SimpleNamespace

import pytest

from llmbeans.hardware.estimator import (
    estimate_kv_cache_gb,
    estimate_max_context,
    estimate_tokens_per_sec,
    estimate_total_memory_gb,
)
from llmbeans.models.scanner import ModelFormat, ModelInfo


def test_estimate_kv_cache_gb_zero_heads_or_dim():
    assert estimate_kv_cache_gb(32, 0, 128, 4096) == 0.0
    assert estimate_kv_cache_gb(32, 8, 0, 4096) == 0.0


def test_estimate_kv_cache_gb_positive(sample_model):
    result = estimate_kv_cache_gb(
        num_layers=sample_model.num_layers,
        num_kv_heads=8,
        head_dim=128,
        context_length=4096,
    )
    assert result > 0


def test_estimate_total_memory_without_kv_cache(sample_model):
    result = estimate_total_memory_gb(sample_model, 4096, include_kv_cache=False)
    assert result["kv_cache_gb"] == 0.0
    assert result["total_gb"] == pytest.approx(result["weights_gb"] + result["overhead_gb"])


def test_estimate_total_memory_with_zero_heads(sample_model):
    sample_model.num_attention_heads = 0
    result = estimate_total_memory_gb(sample_model, 4096)
    assert result["kv_cache_gb"] == 0.0


def test_estimate_tokens_per_sec_zero_when_missing_data(sample_model, nvidia_hardware):
    sample_model.parameter_count = 0
    assert estimate_tokens_per_sec(sample_model, nvidia_hardware) == 0.0

    sample_model.parameter_count = 7.0
    nvidia_hardware.memory_bandwidth_gbps = 0
    assert estimate_tokens_per_sec(sample_model, nvidia_hardware) == 0.0


def test_estimate_tokens_per_sec_unified_memory(apple_hardware, sample_model):
    speed = estimate_tokens_per_sec(sample_model, apple_hardware, gpu_offload_ratio=1.0)
    assert speed > 0


def test_estimate_tokens_per_sec_discrete_gpu(nvidia_hardware, sample_model):
    speed = estimate_tokens_per_sec(sample_model, nvidia_hardware, gpu_offload_ratio=0.5)
    assert speed > 0


def test_estimate_tokens_per_sec_discrete_gpu_full_offload(nvidia_hardware, sample_model):
    speed = estimate_tokens_per_sec(sample_model, nvidia_hardware, gpu_offload_ratio=1.0)
    assert speed > 0


def test_estimate_max_context_unified(apple_hardware, sample_model):
    ctx = estimate_max_context(sample_model, apple_hardware, gpu_offload_layers=32)
    assert ctx >= 512


def test_estimate_max_context_no_gpu(sample_model):
    hw = SimpleNamespace(
        unified_memory=False,
        gpu_vram_gb=None,
        ram_gb=16,
        ram_total_gb=16,
    )
    ctx = estimate_max_context(sample_model, hw, gpu_offload_layers=0)
    assert ctx >= 512


def test_estimate_max_context_discrete_gpu_fits(nvidia_hardware, sample_model):
    ctx = estimate_max_context(sample_model, nvidia_hardware, gpu_offload_layers=32)
    assert ctx >= 512


def test_estimate_max_context_discrete_gpu_too_large(nvidia_hardware, sample_model):
    sample_model.estimated_vram_gb = 1000
    assert estimate_max_context(sample_model, nvidia_hardware, gpu_offload_layers=32) == 0


def test_estimate_max_context_discrete_gpu_ram_too_small(nvidia_hardware, sample_model):
    sample_model.estimated_vram_gb = 100
    nvidia_hardware.ram_gb = 8
    assert estimate_max_context(sample_model, nvidia_hardware, gpu_offload_layers=0) == 0


def test_estimate_max_context_no_available_memory(apple_hardware, sample_model):
    apple_hardware.ram_gb = 1
    assert estimate_max_context(sample_model, apple_hardware, gpu_offload_layers=32) == 0


def test_estimate_max_context_defaults_without_head_info(apple_hardware):
    model = ModelInfo(
        name="tiny",
        format=ModelFormat.GGUF,
        architecture="llama",
        parameter_count=1.0,
        quant_method="Q4_K_M",
        quant_bits=4.5,
        context_length=4096,
        hidden_size=0,
        intermediate_size=None,
        num_layers=1,
        num_attention_heads=0,
        num_key_value_heads=None,
        vocab_size=32000,
        model_size_gb=1.0,
        estimated_vram_gb=1.0,
        source_path="/tmp/tiny.gguf",
    )
    assert estimate_max_context(model, apple_hardware, gpu_offload_layers=1) == 4096


def test_estimate_max_context_small_memory_returns_512(apple_hardware, sample_model):
    apple_hardware.ram_gb = 4
    ctx = estimate_max_context(sample_model, apple_hardware, gpu_offload_layers=32)
    assert ctx >= 0
