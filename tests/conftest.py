"""Shared pytest fixtures for llmbeans tests."""

from __future__ import annotations

import pytest

from llmbeans.hardware.profiles import get_profile_by_id
from llmbeans.models.scanner import ModelFormat, ModelInfo
from llmbeans.recommenders.engine import Recommendation


@pytest.fixture
def sample_model() -> ModelInfo:
    return ModelInfo(
        name="test-model",
        format=ModelFormat.GGUF,
        architecture="llama",
        parameter_count=7.0,
        quant_method="Q4_K_M",
        quant_bits=4.5,
        context_length=32768,
        hidden_size=4096,
        intermediate_size=11008,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=32000,
        model_size_gb=4.5,
        estimated_vram_gb=4.5,
        source_path="/tmp/test-model.gguf",
        is_remote=False,
    )


@pytest.fixture
def nvidia_hardware():
    profile = get_profile_by_id("generic-rtx-4060-laptop")
    assert profile is not None
    return profile


@pytest.fixture
def apple_hardware():
    profile = get_profile_by_id("macbook-pro-m4-2024")
    assert profile is not None
    return profile


@pytest.fixture
def sample_recommendation(sample_model) -> Recommendation:
    return Recommendation(
        hosting_tool="llamacpp",
        context_length=4096,
        batch_size=512,
        thread_count=8,
        gpu_offload_layers=32,
        estimated_tok_per_sec=25.0,
        estimated_vram_usage_gb=4.0,
        estimated_ram_usage_gb=2.0,
        memory_breakdown={
            "weights_gb": 4.5,
            "kv_cache_gb": 0.5,
            "overhead_gb": 1.5,
            "total_gb": 6.5,
        },
        flags={"-ngl": "32", "-c": "4096"},
        command="llama-cli -m /tmp/test-model.gguf -ngl 32",
        extra_config=None,
        warnings=["test warning"],
    )
