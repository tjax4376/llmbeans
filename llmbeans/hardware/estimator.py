# llmbeans/hardware/estimator.py
"""Performance estimation formula.

Estimates tokens/sec and memory usage based on model architecture,
quantization, context length, and hardware specs.

Core model:
- Inference is memory-bandwidth bound (not compute bound) for most local LLM use
- Each token requires reading the full model weights once from memory
- tokens/sec ≈ memory_bandwidth / model_size_in_memory
- GPU offloading changes the bottleneck: partial weights on fast VRAM, rest on slower RAM

References:
- M1 Max: ~400 GB/s unified memory bandwidth
- DDR5-5600 dual channel: ~89.6 GB/s
- RTX 4090: ~1008 GB/s (GDDR6X)
- RTX 4090 laptop: ~576 GB/s
"""

from llmbeans.hardware.profiles import HardwareProfileEntry
from llmbeans.models.scanner import ModelInfo


# ── Memory Estimation ─────────────────────────────────────────

def estimate_kv_cache_gb(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_length: int,
    bytes_per_element: float = 2.0,  # bf16 default
) -> float:
    """Estimate KV cache size for a given context length.

    KV cache shape per layer: [2, batch, num_kv_heads, seq_len, head_dim]
    We store K and V for each token in the cache.

    Formula: 2 * layers * kv_heads * head_dim * seq_len * bytes_per_element
    """
    if num_kv_heads == 0:
        return 0.0
    if head_dim == 0:
        return 0.0
    cache_bytes = 2 * num_layers * num_kv_heads * head_dim * context_length * bytes_per_element
    return cache_bytes / (1024**3)


def estimate_total_memory_gb(
    model: ModelInfo,
    context_length: int,
    include_kv_cache: bool = True,
) -> dict[str, float]:
    """Return estimated memory breakdown in GB:
    - weights_gb: model weights in memory
    - kv_cache_gb: KV cache for given context length
    - overhead_gb: OS + runtime overhead
    - total_gb: sum of all components
    """
    weights_gb = model.estimated_vram_gb
    kv_cache_gb = 0.0

    if include_kv_cache:
        # Calculate head_dim from hidden_size / attention_heads
        head_dim = 0
        if model.num_attention_heads > 0:
            head_dim = model.hidden_size // model.num_attention_heads

        kv_heads = model.num_key_value_heads or model.num_attention_heads
        if head_dim > 0 and kv_heads > 0:
            bytes_per = (model.quant_bits or 16.0) / 8
            kv_cache_gb = estimate_kv_cache_gb(
                num_layers=model.num_layers,
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                context_length=context_length,
                bytes_per_element=bytes_per if context_length > 0 else 2.0,
            )

    overhead_gb = 1.5  # OS + runtime baseline
    total_gb = weights_gb + kv_cache_gb + overhead_gb

    return {
        "weights_gb": round(weights_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "overhead_gb": overhead_gb,
        "total_gb": round(total_gb, 2),
    }


# ── Performance Estimation ────────────────────────────────────

def estimate_tokens_per_sec(
    model: ModelInfo,
    hardware: HardwareProfileEntry,
    gpu_offload_ratio: float = 0.0,
) -> float:
    """Estimate inference tokens/sec.

    Model: each token requires one full pass through model weights.
    Bottleneck is whichever memory the weights sit on.

    For unified memory (Apple Silicon): bandwidth is the single pool.
    For discrete GPU: offloaded layers use VRAM bandwidth, rest use RAM bandwidth.

    Args:
        model: ModelInfo from scanner
        hardware: HardwareProfileEntry from laptop database
        gpu_offload_ratio: 0.0 = all CPU/RAM, 1.0 = all GPU/VRAM
    """
    if model.parameter_count <= 0 or hardware.memory_bandwidth_gbps <= 0:
        return 0.0

    # Model weight size in GB (in memory at inference precision)
    quant_bytes = (model.quant_bits or 16.0) / 8
    model_size_gb = model.parameter_count * 1e9 * quant_bytes / (1024**3)

    if model_size_gb <= 0:
        return 0.0  # pragma: no cover - defensive guard

    if hardware.unified_memory or hardware.gpu_vram_gb is None:
        # Apple Silicon / integrated GPU: single memory pool
        # Bandwidth-limited: tokens/s = bandwidth / model_size
        bandwidth = hardware.memory_bandwidth_gbps
        # Apply efficiency factor (not all bandwidth goes to inference)
        efficiency = 0.6  # ~60% effective bandwidth for inference
        return round((bandwidth * efficiency) / model_size_gb, 1)
    else:
        # Discrete GPU + system RAM
        vram_bandwidth = hardware.vram_bandwidth_gbps or 500.0
        ram_bandwidth = hardware.memory_bandwidth_gbps

        offload_ratio = max(0.0, min(1.0, gpu_offload_ratio))

        # Weighted bandwidth: portion on GPU uses VRAM bandwidth, rest uses RAM
        vram_model_gb = model_size_gb * offload_ratio
        ram_model_gb = model_size_gb * (1.0 - offload_ratio)

        # Time for each portion (sequential bottleneck)
        vram_time = vram_model_gb / (vram_bandwidth * 0.6) if offload_ratio > 0 else 0
        ram_time = ram_model_gb / (ram_bandwidth * 0.6) if offload_ratio < 1 else 0

        total_time = vram_time + ram_time
        if total_time <= 0:
            return 0.0  # pragma: no cover - defensive guard

        return round(1.0 / total_time, 1)


def estimate_max_context(
    model: ModelInfo,
    hardware: HardwareProfileEntry,
    gpu_offload_layers: int = 0,
) -> int:
    """Estimate maximum context length that fits in available memory.

    Returns context length in tokens.
    """
    # Available memory for model + cache
    if hardware.unified_memory:
        # Unified memory: total RAM is shared, leave 2-4GB for OS
        ram_gb = getattr(hardware, 'ram_gb', getattr(hardware, 'ram_total_gb', 0.0))
        available_gb = ram_gb - 3.0
    elif hardware.gpu_vram_gb:
        # Discrete GPU: model portion on VRAM + system RAM for rest
        vram_for_model = hardware.gpu_vram_gb * 0.9  # leave 10% headroom
        ram_for_model = getattr(hardware, 'ram_gb', getattr(hardware, 'ram_total_gb', 0.0)) * 0.7  # leave 30% for OS + misc
        # The limiting factor depends on offload ratio
        gpu_ratio = gpu_offload_layers / max(model.num_layers, 1)
        model_on_vram = model.estimated_vram_gb * gpu_ratio
        model_on_ram = model.estimated_vram_gb * (1 - gpu_ratio)

        if model_on_vram > vram_for_model:
            return 0  # doesn't even fit
        if model_on_ram > ram_for_model:
            return 0  # doesn't fit in RAM either
        available_gb = min(vram_for_model - model_on_vram, ram_for_model - model_on_ram)
    else:
        # No GPU: system RAM only
        available_gb = getattr(hardware, 'ram_gb', getattr(hardware, 'ram_total_gb', 0.0)) * 0.7

    if available_gb <= 0:
        return 0

    # Calculate KV cache per token
    head_dim = 0
    if model.num_attention_heads > 0:
        head_dim = model.hidden_size // model.num_attention_heads

    kv_heads = model.num_key_value_heads or model.num_attention_heads
    if head_dim <= 0 or kv_heads <= 0:
        # Default: assume we can fit at least 4096
        return 4096

    bytes_per_element = (model.quant_bits or 16.0) / 8
    gb_per_token = (2 * model.num_layers * kv_heads * head_dim * bytes_per_element) / (1024**3)

    if gb_per_token <= 0:
        return 4096  # pragma: no cover - defensive guard

    # Context = available memory / memory per token
    max_context = int(available_gb / gb_per_token)

    # Round down to nearest power of 2-ish increment (common config values)
    common_contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    for ctx in reversed(common_contexts):
        if max_context >= ctx:
            return ctx

    return 512  # pragma: no cover