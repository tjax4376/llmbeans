# llmbeans/recommenders/tools/lmstudio.py
"""LM Studio flag generator.

LM Studio uses a GUI but supports environment variables and launch flags
for advanced configuration. Also outputs llama.cpp compatible flags
since LM Studio uses llama.cpp under the hood.
"""

import json

from llmbeans.recommenders.registry import register_tool


@register_tool("lmstudio")
def generate_flags(model, hardware, gpu_offload_layers, context_length,
                   batch_size, thread_count, quality_mode):
    flags = {}
    total_layers = model.num_layers

    # LM Studio uses llama.cpp backend — these env vars are respected
    if hardware.cuda:
        flags["CUDA_VISIBLE_DEVICES"] = "0"
    if hardware.metal:
        # LM Studio auto-detects Metal on macOS
        pass

    # GPU layers hint (LM Studio's "GPU Offload" slider maps to -ngl)
    flags["gpu_layers"] = str(gpu_offload_layers)
    flags["ctx_size"] = str(context_length)
    flags["n_threads"] = str(thread_count)
    flags["batch_size"] = str(batch_size)

    # KV cache quantization (LM Studio setting)
    if hardware.unified_memory or (hardware.gpu_vram_gb and hardware.gpu_vram_gb < 16):
        flags["kv_cache_type"] = "q8_0"
    else:
        flags["kv_cache_type"] = "f16"

    # LM Studio config JSON
    config = {
        "gpu_layers": gpu_offload_layers,
        "ctx_size": context_length,
        "n_threads": thread_count,
        "batch_size": batch_size,
        "flash_attention": bool(hardware.cuda),
        "cache_type_k": flags["kv_cache_type"],
        "cache_type_v": flags["kv_cache_type"],
    }

    config_json = json.dumps({"load": config}, indent=2)

    # Command: LM Studio is GUI, but lms CLI exists
    model_path = model.source_path if not model.is_remote else model.name
    env_vars = " ".join(f'{k}="{v}"' for k, v in flags.items())
    command = f"{env_vars} lms server start" if not model.is_remote else \
              f"# Load {model.source_path} in LM Studio's model browser"

    return {
        "flags": flags,
        "command": command,
        "extra_config": config_json,
    }
