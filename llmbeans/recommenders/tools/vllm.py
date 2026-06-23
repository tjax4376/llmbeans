# llmbeans/recommenders/tools/vllm.py
"""vLLM flag generator."""

from llmbeans.recommenders.registry import register_tool


@register_tool("vllm")
def generate_flags(model, hardware, gpu_offload_layers, context_length,
                   batch_size, thread_count, quality_mode):
    if not hardware.cuda:
        raise ValueError("vLLM requires NVIDIA GPU with CUDA support")

    flags = {}

    # Model
    model_ref = model.source_path if not model.is_remote else model.source_path
    flags["model"] = model_ref

    # GPU memory utilization
    total_layers = model.num_layers
    if total_layers > 0 and gpu_offload_layers < total_layers:
        # Partial offload: set gpu-memory-utilization to fit
        vram = hardware.gpu_vram_gb or 24
        model_gb = model.estimated_vram_gb
        ratio = gpu_offload_layers / total_layers
        needed_vram = model_gb * ratio + 2.0  # +2GB overhead
        gpu_util = min(0.95, needed_vram / vram)
        flags["--gpu-memory-utilization"] = f"{gpu_util:.2f}"
    else:
        flags["--gpu-memory-utilization"] = "0.90"

    # Max model length (context)
    flags["--max-model-len"] = str(context_length)

    # Tensor parallelism (single GPU = 1)
    flags["--tensor-parallel-size"] = "1"

    # Quantization
    if model.quant_method:
        q = model.quant_method.upper()
        if q.startswith("Q") or "AWQ" in q or "GPTQ" in q or "GGUF" in q:
            flags["--quantization"] = model.quant_method.lower()
        if "AWQ" in q:
            flags["--quantization"] = "awq"
        elif "GPTQ" in q:
            flags["--quantization"] = "gptq"
        elif "GGUF" in q or model.format and model.format.value == "gguf":
            flags["--quantization"] = "gguf"

    # KV cache dtype
    if hardware.gpu_vram_gb and hardware.gpu_vram_gb < 24:
        flags["--kv-cache-dtype"] = "fp8"
    else:
        flags["--kv-cache-dtype"] = "auto"

    # Swap space for CPU offload
    if gpu_offload_layers < total_layers:
        flags["--swap-space"] = "4"

    # Disable custom all-reduce for single GPU
    flags["--disable-custom-all-reduce"] = ""

    # Build command
    cmd_parts = ["python -m vllm.entrypoints.openai.api_server"]
    for k, v in flags.items():
        if v == "":
            cmd_parts.append(k)
        else:
            cmd_parts.append(f"{k} {v}")
    command = " \\\n    ".join(cmd_parts)

    return {"flags": flags, "command": command, "extra_config": None}
