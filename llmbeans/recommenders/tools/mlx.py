# llmbeans/recommenders/tools/mlx.py
"""MLX flag generator for Apple Silicon."""

from llmbeans.recommenders.registry import register_tool


@register_tool("mlx")
def generate_flags(model, hardware, gpu_offload_layers, context_length,
                   batch_size, thread_count, quality_mode):
    if not hardware.metal:
        raise ValueError("MLX is Apple Silicon only (requires Metal)")

    flags = {}

    # MLX-LM specific options
    flags["--temp"] = "0.7"
    flags["--max-tokens"] = str(min(context_length, 4096))

    warnings: list[str] = []

    # Quantization
    if model.quant_method and model.quant_bits and model.quant_bits <= 4:
        # MLX supports static quantization natively
        bits = int(model.quant_bits)
        if bits in (4, 8):
            flags["--quantize"] = ""
            flags["--q-bits"] = str(bits)
            flags["--q-group-size"] = "64"

    # Memory limit — MLX uses unified memory, set soft limit
    available_gb = hardware.ram_gb - 3.0
    flags["--max-kv-size"] = str(min(context_length, 8192))

    # Model reference
    if model.is_remote:
        model_ref = model.source_path
    elif model.format and model.format.value == "gguf":
        # MLX prefers native safetensors or mlx-community converted repos
        model_ref = f"mlx-community/{model.name}" if model.quant_bits and model.quant_bits <= 8 else model.source_path
        if model_ref.startswith("mlx-community/"):
            warnings.append(
                f"Guessed MLX repo '{model_ref}' from local filename; verify on Hugging Face "
                "or pass an explicit mlx-community repo ID."
            )
    else:
        model_ref = model.source_path

    command = f"python -m mlx_lm.generate --model {model_ref}"
    for k, v in flags.items():
        if v == "":
            command += f" {k}"
        else:
            command += f" {k} {v}"

    return {"flags": flags, "command": command, "extra_config": None, "warnings": warnings}
