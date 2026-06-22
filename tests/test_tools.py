"""Tests for hosting tool flag generators."""

import json
from dataclasses import replace

import pytest

from llmbeans.models.scanner import ModelFormat
from llmbeans.recommenders.tools.llamacpp import generate_flags as llamacpp_flags
from llmbeans.recommenders.tools.lmstudio import generate_flags as lmstudio_flags
from llmbeans.recommenders.tools.mlx import generate_flags as mlx_flags
from llmbeans.recommenders.tools.ollama import generate_flags as ollama_flags
from llmbeans.recommenders.tools.vllm import generate_flags as vllm_flags


def test_llamacpp_cuda_partial_offload(sample_model, nvidia_hardware):
    result = llamacpp_flags(
        model=sample_model,
        hardware=nvidia_hardware,
        gpu_offload_layers=16,
        context_length=16384,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert result["flags"]["-ngl"] == "16"
    assert result["flags"]["-sm"] == "layer"
    assert "-fa" in result["flags"]
    assert "--rope-scaling" in result["flags"]


def test_llamacpp_no_cuda(sample_model, nvidia_hardware):
    hw = replace(nvidia_hardware, cuda=False, metal=False)
    result = llamacpp_flags(
        model=sample_model,
        hardware=hw,
        gpu_offload_layers=0,
        context_length=4096,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert result["flags"]["-ngl"] == "0"


def test_llamacpp_tight_memory_uses_q8(sample_model, nvidia_hardware):
    sample_model.estimated_vram_gb = 100
    result = llamacpp_flags(
        model=sample_model,
        hardware=nvidia_hardware,
        gpu_offload_layers=32,
        context_length=4096,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert result["flags"]["-ctk"] == "q8_0"


def test_llamacpp_mlock_when_model_fits(sample_model, nvidia_hardware):
    sample_model.estimated_vram_gb = 2.0
    result = llamacpp_flags(
        model=sample_model,
        hardware=nvidia_hardware,
        gpu_offload_layers=32,
        context_length=4096,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert "-mlock" in result["flags"]


def test_lmstudio_cuda_and_remote(sample_model, nvidia_hardware):
    result = lmstudio_flags(
        model=sample_model,
        hardware=nvidia_hardware,
        gpu_offload_layers=32,
        context_length=8192,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert "CUDA_VISIBLE_DEVICES" in result["flags"]
    assert json.loads(result["extra_config"])["load"]["gpu_layers"] == 32

    sample_model.is_remote = True
    remote = lmstudio_flags(
        model=sample_model,
        hardware=nvidia_hardware,
        gpu_offload_layers=32,
        context_length=8192,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert "Load" in remote["command"]


def test_lmstudio_metal_and_large_vram(sample_model, apple_hardware):
    apple_hardware = replace(apple_hardware, gpu_vram_gb=24.0, unified_memory=False, cuda=False)
    result = lmstudio_flags(
        model=sample_model,
        hardware=apple_hardware,
        gpu_offload_layers=32,
        context_length=8192,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert result["flags"]["kv_cache_type"] == "f16"


def test_ollama_cuda_and_metal(sample_model, nvidia_hardware, apple_hardware):
    nvidia = ollama_flags(
        model=sample_model,
        hardware=nvidia_hardware,
        gpu_offload_layers=20,
        context_length=4096,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert "OLLAMA_FLASH_ATTENTION" in nvidia["flags"]
    assert "num_gpu_layers" in nvidia["flags"]

    apple = ollama_flags(
        model=sample_model,
        hardware=apple_hardware,
        gpu_offload_layers=32,
        context_length=4096,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert "OLLAMA_METAL" in apple["flags"]


def test_mlx_requires_metal(sample_model, nvidia_hardware):
    with pytest.raises(ValueError, match="Apple Silicon only"):
        mlx_flags(
            model=sample_model,
            hardware=nvidia_hardware,
            gpu_offload_layers=32,
            context_length=4096,
            batch_size=512,
            thread_count=8,
            quality_mode="balanced",
        )


def test_mlx_local_and_remote(sample_model, apple_hardware):
    sample_model.format = ModelFormat.GGUF
    sample_model.quant_bits = 4.0
    sample_model.quant_method = "Q4_K_M"
    local = mlx_flags(
        model=sample_model,
        hardware=apple_hardware,
        gpu_offload_layers=32,
        context_length=4096,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert "mlx-community" in local["command"]

    sample_model.is_remote = True
    remote = mlx_flags(
        model=sample_model,
        hardware=apple_hardware,
        gpu_offload_layers=32,
        context_length=4096,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert sample_model.source_path in remote["command"]


def test_vllm_requires_cuda(sample_model, apple_hardware):
    with pytest.raises(ValueError, match="requires NVIDIA GPU"):
        vllm_flags(
            model=sample_model,
            hardware=apple_hardware,
            gpu_offload_layers=32,
            context_length=4096,
            batch_size=512,
            thread_count=8,
            quality_mode="balanced",
        )


def test_vllm_quantization_paths(sample_model, nvidia_hardware):
    sample_model.quant_method = "AWQ"
    awq = vllm_flags(
        model=sample_model,
        hardware=nvidia_hardware,
        gpu_offload_layers=16,
        context_length=4096,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert awq["flags"]["--quantization"] == "awq"
    assert "--swap-space" in awq["flags"]

    sample_model.quant_method = "GPTQ"
    gptq = vllm_flags(
        model=sample_model,
        hardware=nvidia_hardware,
        gpu_offload_layers=32,
        context_length=4096,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert gptq["flags"]["--quantization"] == "gptq"

    sample_model.quant_method = "GGUF"
    sample_model.format = ModelFormat.GGUF
    gguf = vllm_flags(
        model=sample_model,
        hardware=nvidia_hardware,
        gpu_offload_layers=32,
        context_length=4096,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert gguf["flags"]["--quantization"] == "gguf"


def test_vllm_large_vram_uses_auto_kv(sample_model, nvidia_hardware):
    nvidia_hardware = replace(nvidia_hardware, gpu_vram_gb=32.0)
    result = vllm_flags(
        model=sample_model,
        hardware=nvidia_hardware,
        gpu_offload_layers=32,
        context_length=4096,
        batch_size=512,
        thread_count=8,
        quality_mode="balanced",
    )
    assert result["flags"]["--kv-cache-dtype"] == "auto"
