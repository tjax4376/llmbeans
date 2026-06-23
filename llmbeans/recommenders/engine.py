# llmbeans/recommenders/engine.py
"""Core recommendation engine.

Combines model info + hardware profile + hosting tool to produce
an optimal Recommendation with flags, performance estimates, and warnings.
"""

from dataclasses import dataclass, field

from llmbeans.hardware.profiles import HardwareProfileEntry
from llmbeans.hardware.estimator import (
    estimate_tokens_per_sec,
    estimate_total_memory_gb,
    estimate_max_context,
)
from llmbeans.models.scanner import ModelInfo

# Import registry to get register_tool and get_available_tools
from llmbeans.recommenders.registry import register_tool, get_available_tools

# Import tool modules to register them with the engine
from llmbeans.recommenders import tools  # noqa: F401


@dataclass
class Recommendation:
    """A complete recommendation for running a model on given hardware."""
    hosting_tool: str
    context_length: int
    batch_size: int
    thread_count: int
    gpu_offload_layers: int
    estimated_tok_per_sec: float
    estimated_vram_usage_gb: float = 0.0
    estimated_ram_usage_gb: float = 0.0
    memory_breakdown: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    command: str = ""
    extra_config: str | None = None
    warnings: list[str] = field(default_factory=list)


def recommend(
    model: ModelInfo,
    hardware: HardwareProfileEntry,
    hosting_tool: str,
    quality_mode: str = "balanced",
) -> Recommendation:
    """Generate an optimal recommendation for running *model* on *hardware*
    using *hosting_tool* with the given *quality_mode*.

    Quality modes:
        balanced — good default, moderate context
        quality  — max context, slower
        speed    — shorter context, faster
    """
    from llmbeans.recommenders.registry import get_tool_generator

    # ── Determine GPU offload ──────────────────────────────────
    if hardware.unified_memory:
        # Apple Silicon: offload all layers to GPU
        gpu_offload_layers = model.num_layers
    elif hardware.gpu_vram_gb and hardware.gpu_vram_gb > 0:
        # Discrete GPU: figure out how many layers fit
        vram_available = hardware.gpu_vram_gb * 0.9  # 10% headroom
        model_size_gb = model.estimated_vram_gb

        if model_size_gb <= vram_available:
            gpu_offload_layers = model.num_layers
        else:
            # Each layer costs roughly model_size / num_layers
            layer_cost = model_size_gb / max(model.num_layers, 1)
            gpu_offload_layers = max(0, int(vram_available / layer_cost))
    else:
        gpu_offload_layers = 0

    # ── Determine context length ───────────────────────────────
    max_ctx = estimate_max_context(model, hardware, gpu_offload_layers)

    # Apply quality mode multiplier
    if quality_mode == "quality":
        context_length = min(max_ctx, model.context_length)
    elif quality_mode == "speed":
        # Cap at 4096 for speed mode
        context_length = min(max_ctx, 4096)
    else:
        # Balanced: use 75% of max or model default, whichever is smaller
        context_length = min(int(max_ctx * 0.75), model.context_length)

    # Round to common values
    common_contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    for ctx in reversed(common_contexts):
        if context_length >= ctx:
            context_length = ctx
            break
    else:
        context_length = 512

    # ── Determine thread count ─────────────────────────────────
    if hardware.unified_memory:
        # Apple Silicon: use physical cores
        thread_count = hardware.cpu_cores
    else:
        # Use 75% of threads to leave headroom
        thread_count = max(1, int(hardware.cpu_threads * 0.75))

    # ── Determine batch size ───────────────────────────────────
    if quality_mode == "speed":
        batch_size = 256
    elif quality_mode == "quality":
        batch_size = 1024
    else:
        batch_size = 512

    # ── Performance estimate ───────────────────────────────────
    gpu_ratio = gpu_offload_layers / max(model.num_layers, 1)
    estimated_tok_per_sec = estimate_tokens_per_sec(model, hardware, gpu_ratio)

    # ── Memory breakdown ───────────────────────────────────────
    memory_breakdown = estimate_total_memory_gb(model, context_length)

    # ── VRAM / RAM usage estimates ─────────────────────────────
    if hardware.unified_memory:
        estimated_vram_usage_gb = memory_breakdown["total_gb"]
        estimated_ram_usage_gb = 0.0  # unified, so same pool
    elif hardware.gpu_vram_gb:
        vram_ratio = gpu_offload_layers / max(model.num_layers, 1)
        estimated_vram_usage_gb = model.estimated_vram_gb * vram_ratio
        estimated_ram_usage_gb = memory_breakdown["total_gb"] - estimated_vram_usage_gb
    else:
        estimated_vram_usage_gb = 0.0
        estimated_ram_usage_gb = memory_breakdown["total_gb"]

    # ── Generate tool-specific flags ───────────────────────────
    generator = get_tool_generator(hosting_tool)
    if generator is None:
        raise ValueError(f"Unknown hosting tool: {hosting_tool}")

    tool_result = generator(
        model=model,
        hardware=hardware,
        gpu_offload_layers=gpu_offload_layers,
        context_length=context_length,
        batch_size=batch_size,
        thread_count=thread_count,
        quality_mode=quality_mode,
    )

    flags = tool_result.get("flags", {})
    command = tool_result.get("command", "")
    extra_config = tool_result.get("extra_config", None)

    # ── Warnings ───────────────────────────────────────────────
    warnings = list(tool_result.get("warnings", []))

    if memory_breakdown["total_gb"] > hardware.ram_gb * 0.9:
        warnings.append(
            f"Model + context may exceed available RAM "
            f"({memory_breakdown['total_gb']:.1f}GB needed vs {hardware.ram_gb}GB available). "
            f"Consider reducing context length or using a smaller quant."
        )

    if gpu_offload_layers < model.num_layers and hardware.gpu_vram_gb:
        warnings.append(
            f"Only {gpu_offload_layers}/{model.num_layers} layers offloaded to GPU. "
            f"Remaining layers will run on CPU, reducing speed."
        )

    if model.quant_bits and model.quant_bits <= 3.0:
        warnings.append(
            f"Low quantization ({model.quant_method}, {model.quant_bits}-bit) "
            f"may produce noticeably lower quality output."
        )

    if estimated_tok_per_sec > 0 and estimated_tok_per_sec < 5:
        warnings.append(
            f"Estimated speed is very slow (~{estimated_tok_per_sec} tok/s). "
            f"Consider a smaller model or lower quantization."
        )

    if hardware.unified_memory and model.estimated_vram_gb < hardware.ram_gb * 0.3:
        warnings.append(
            "Model is much smaller than available unified memory. "
            "Consider increasing context length for better utilization."
        )

    return Recommendation(
        hosting_tool=hosting_tool,
        context_length=context_length,
        batch_size=batch_size,
        thread_count=thread_count,
        gpu_offload_layers=gpu_offload_layers,
        estimated_tok_per_sec=estimated_tok_per_sec,
        estimated_vram_usage_gb=round(estimated_vram_usage_gb, 2),
        estimated_ram_usage_gb=round(estimated_ram_usage_gb, 2),
        memory_breakdown=memory_breakdown,
        flags=flags,
        command=command,
        extra_config=extra_config,
        warnings=warnings,
    )
