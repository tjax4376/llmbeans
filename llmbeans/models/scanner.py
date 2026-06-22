# llmbeans/models/scanner.py
"""Model file format detection and metadata extraction.

Supports:
- GGUF files (single file with embedded metadata)
- Safetensors directories (config.json + .safetensors files)
- HuggingFace repo IDs (remote, via huggingface_hub)
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class ModelFormat(Enum):
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    HF_REPO = "hf_repo"


@dataclass
class ModelInfo:
    name: str
    format: ModelFormat
    architecture: str            # e.g. "llama", "mistral", "gemma", "qwen2"
    parameter_count: float       # in billions (B)
    quant_method: str | None     # e.g. "Q4_K_M", "Q6_K", "F16", "BF16", "Q4_0"
    quant_bits: float | None     # effective bits per weight
    context_length: int          # max context from config (tokens)
    hidden_size: int             # hidden dimension
    intermediate_size: int | None # FFN intermediate size
    num_layers: int              # total transformer layers
    num_attention_heads: int
    num_key_value_heads: int | None  # GQA/MQA: if None, equals num_attention_heads
    vocab_size: int
    # Memory estimates
    model_size_gb: float         # on-disk size
    estimated_vram_gb: float     # memory to load weights into RAM/VRAM
    # Source
    source_path: str
    is_remote: bool = False
    # Extra metadata
    metadata: dict | None = None


# Architecture mapping from config json architecture field
ARCH_MAP = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "mistral",
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma2",
    "Gemma3ForCausalLM": "gemma3",
    "Gemma3ForConditionalGeneration": "gemma3",
    "Gemma4ForCausalLM": "gemma4",
    "Gemma4ForConditionalGeneration": "gemma4",
    "Qwen2ForCausalLM": "qwen2",
    "Qwen3ForCausalLM": "qwen3",
    "Qwen2MoeForCausalLM": "qwen2_moe",
    "Qwen3MoeForCausalLM": "qwen3_moe",
    "PhiForCausalLM": "phi",
    "Phi3ForCausalLM": "phi3",
    "DeepseekForCausalLM": "deepseek",
    "DeepseekV2ForCausalLM": "deepseek_v2",
    "DeepseekV3ForCausalLM": "deepseek_v3",
    "MixtralForCausalLM": "mixtral",
    "Starcoder2ForCausalLM": "starcoder2",
    "CohereForCausalLM": "command-r",
    "OlmoForCausalLM": "olmo",
    "Olmo2ForCausalLM": "olmo2",
    "InternLM2ForCausalLM": "internlm2",
    "MiniCPMForCausalLM": "minicpm",
    "MiniCPM3ForCausalLM": "minicpm3",
    "ChatGLMForCausalLM": "chatglm",
    "BaichuanForCausalLM": "baichuan",
    "FalconForCausalLM": "falcon",
    "MPTForCausalLM": "mpt",
    "StableLMEpochForCausalLM": "stablelm",
    "StableLmForCausalLM": "stablelm",
}


# Known quant methods and their effective bits
QUANT_BITS = {
    "F16": 16.0, "BF16": 16.0, "FP16": 16.0,
    "Q8_0": 8.0, "Q8_K": 8.125,  # K quirks: Q8_K is slightly above 8
    "Q6_K": 6.5625,
    "Q5_0": 5.0, "Q5_1": 5.0, "Q5_K": 5.5, "Q5_K_M": 5.5, "Q5_K_S": 5.5,
    "Q4_0": 4.0, "Q4_1": 4.0,
    "Q4_K": 4.5, "Q4_K_M": 4.5, "Q4_K_S": 4.5,
    "Q3_K": 3.5, "Q3_K_M": 3.5, "Q3_K_S": 3.5, "Q3_K_L": 3.5,
    "Q2_K": 2.5,
    "IQ1_S": 1.5, "IQ1_M": 1.5,
    "IQ2_XXS": 2.0, "IQ2_XS": 2.0, "IQ2_S": 2.0,
    "IQ3_XXS": 3.0, "IQ3_S": 3.0,
    "IQ4_NL": 4.5, "IQ4_XS": 4.5,
}


def detect_format(source: str) -> ModelFormat:
    """Detect model format from file path or identifier."""
    source = source.strip()

    # Local path checks first (before HF repo heuristic)
    path = Path(source)

    if path.is_file() and path.suffix.lower() == ".gguf":
        return ModelFormat.GGUF

    if path.is_file() and path.name in ("model.safetensors",):
        return ModelFormat.SAFETENSORS

    if path.is_dir():
        if any(path.glob("*.safetensors")):
            return ModelFormat.SAFETENSORS
        if (path / "model.safetensors.index.json").exists():
            return ModelFormat.SAFETENSORS
        # Check subdirectories one level deep for safetensors models
        for sub in path.iterdir():
            if sub.is_dir():
                if any(sub.glob("*.safetensors")):
                    return ModelFormat.SAFETENSORS
                if (sub / "model.safetensors.index.json").exists():
                    return ModelFormat.SAFETENSORS

    # HF repo ID: contains "/" and doesn't exist as local path
    if "/" in source:
        return ModelFormat.HF_REPO

    raise ValueError(f"Cannot detect model format for: {source}")


def _infer_architecture(config: dict) -> str:
    """Extract architecture name from config."""
    # Try direct architecture field
    arch_field = config.get("architectures", [None])[0] or config.get("model_type", "")
    # Map known class names
    if arch_field in ARCH_MAP:
        return ARCH_MAP[arch_field]
    # Use model_type if it's a clean string
    if arch_field and isinstance(arch_field, str):
        return arch_field.lower().replace(" ", "_")
    # Fallback
    return arch_field or "unknown"


def _infer_param_count(config: dict, tensors_info: list | None = None) -> float:
    """Estimate parameter count from config fields."""
    # Direct field
    if "num_parameters" in config:
        return config["num_parameters"] / 1e9
    if "n_params" in config:
        return config["n_params"] / 1e9

    # Calculate from architecture
    hidden = config.get("hidden_size", 0)
    layers = config.get("num_hidden_layers", 0)
    vocab = config.get("vocab_size", 0)
    intermediate = config.get("intermediate_size", 0)

    if hidden and layers and vocab:
        # Rough estimate: embedding + transformer layers
        embedding_params = vocab * hidden * 2  # input + output embeddings
        # Per layer: attention + FFN
        head_dim = config.get("head_dim", hidden // max(config.get("num_attention_heads", 1), 1))
        kv_heads = config.get("num_key_value_heads", config.get("num_attention_heads", 1))
        attn_params = layers * (
            4 * hidden * hidden  # Q,K,V,O projections (simplified)
        )
        ffn_params = layers * (3 * hidden * intermediate if intermediate else 8/3 * hidden * hidden)
        total = embedding_params + attn_params + ffn_params
        return total / 1e9

    return 0.0


def _estimate_model_size_gb(source: str, fmt: ModelFormat) -> float:
    """Calculate on-disk model size in GB."""
    if fmt == ModelFormat.HF_REPO:
        # Will be populated from remote metadata
        return 0.0
    path = Path(source)
    if path.is_file():
        return path.stat().st_size / (1024**3)
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024**3)
    return 0.0


def _estimate_vram_gb(param_count_b: float, quant_bits: float | None) -> float:
    """Estimate VRAM needed to load model weights."""
    if quant_bits and quant_bits > 0:
        bytes_per_weight = quant_bits / 8
    else:
        bytes_per_weight = 2.0  # default bf16
    return (param_count_b * 1e9 * bytes_per_weight) / (1024**3)


def _get_quant_from_filename(filename: str) -> tuple[str | None, float | None]:
    """Extract quantization method from GGUF filename."""
    name = Path(filename).stem.upper()
    # Sort by length descending so Q4_K_M matches before Q4_K
    for method in sorted(QUANT_BITS.keys(), key=len, reverse=True):
        if method.upper() in name:
            return method, QUANT_BITS[method]
    return None, None


# ── GGUF Scanner ──────────────────────────────────────────────

def _scan_gguf(source: str) -> ModelInfo:
    """Scan a GGUF file and extract metadata."""
    from gguf import GGUFReader

    path = Path(source)
    reader = GGUFReader(str(path))

    # Extract metadata fields
    # ReaderField.parts layout: [offset, name_bytes, type_code, value_or_length, string_bytes?]
    # For scalars: parts[3] holds the decoded value (numpy scalar)
    # For strings: parts[3] = length (uint64), parts[4] = raw bytes
    # For arrays: parts[3:] hold the elements
    def get_str(field: str, default: str | None = None) -> str | None:
        rf = reader.fields.get(field)
        if rf is None:
            return default
        try:
            if rf.types[0].name == "STRING" and len(rf.parts) >= 5:
                return bytes(rf.parts[4]).decode("utf-8", errors="replace").strip("\x00")
            # Fallback: try parts[3] as bytes
            if len(rf.parts) >= 4 and rf.parts[3].dtype == np.uint8:  # pragma: no cover
                return bytes(rf.parts[3]).decode("utf-8", errors="replace").strip("\x00")
        except Exception:  # pragma: no cover
            pass
        return default

    def get_int(field: str, default: int = 0) -> int:
        rf = reader.fields.get(field)
        if rf is None:
            return default
        try:
            if rf.types[0].name == "ARRAY":
                # For array fields (e.g. tokenizer.ggml.tokens), return element count
                return len(rf.data)
            return int(rf.parts[3][0])
        except Exception:  # pragma: no cover
            return default

    def get_float(field: str, default: float = 0.0) -> float:
        rf = reader.fields.get(field)
        if rf is None:
            return default
        try:
            return float(rf.parts[3][0])
        except Exception:  # pragma: no cover
            return default

    # Architecture
    arch = get_str("general.architecture", "unknown")
    arch_lower = arch.lower()

    # Count layers from tensor names
    layer_indices = set()
    total_elements = 0
    for tensor in reader.tensors:
        m = re.search(r"\.(\d+)\.", tensor.name)
        if m:
            layer_indices.add(int(m.group(1)))
        total_elements += tensor.n_elements

    num_layers = max(layer_indices) + 1 if layer_indices else 0

    # Extract context length — GGUF stores this differently per arch
    context_length = 0
    context_keys = [
        f"{arch_lower}.context_length",
        "llama.context_length",
        "context_length",
        f"{context_length}.attention.context_length",
    ]
    for key in context_keys:
        val = get_int(key)
        if val > 0:
            context_length = val
            break

    # Hidden size
    hidden_size = get_int(f"{arch_lower}.embedding_length")
    if not hidden_size:
        hidden_size = get_int("embedding_length")
    if not hidden_size:
        hidden_size = get_int("hidden_size")

    # Attention heads
    n_heads = get_int(f"{arch_lower}.attention.head_count")
    if not n_heads:
        n_heads = get_int("attention.head_count")
    n_kv_heads = get_int(f"{arch_lower}.attention.head_count_kv")
    if not n_kv_heads:
        n_kv_heads = get_int("attention.head_count_kv")

    # Feed-forward
    ffn_size = get_int(f"{arch_lower}.feed_forward_length")
    if not ffn_size:
        ffn_size = get_int("feed_forward_length")

    #vocab
    vocab_size = get_int("tokenizer.ggml.tokens")
    if not vocab_size:
        vocab_size = get_int("vocab_size", 32000)

    # Parameter count
    param_count = get_float("general.parameter_count", 0.0)
    if param_count == 0.0:
        # Estimate from tensor elements
        param_count = total_elements / 1e9

    # Quantization
    quant_type = get_str("general.quantization_version")
    quant_method, quant_bits = _get_quant_from_filename(path.name)
    # Also check if GGUF has quant metadata
    if not quant_method:
        file_type = get_str("general.file_type")
        if file_type:
            quant_method = file_type
            quant_bits = QUANT_BITS.get(file_type.upper(), None)  # pragma: no cover

    # Size
    size_gb = path.stat().st_size / (1024**3)
    vram_gb = _estimate_vram_gb(param_count, quant_bits)

    return ModelInfo(
        name=path.stem,
        format=ModelFormat.GGUF,
        architecture=arch_lower,
        parameter_count=round(param_count, 1),
        quant_method=quant_method,
        quant_bits=quant_bits,
        context_length=context_length if context_length > 0 else 4096,
        hidden_size=hidden_size,
        intermediate_size=ffn_size if ffn_size > 0 else None,
        num_layers=num_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads if n_kv_heads > 0 else None,
        vocab_size=vocab_size,
        model_size_gb=round(size_gb, 2),
        estimated_vram_gb=round(vram_gb, 2),
        source_path=str(path.resolve()),
        is_remote=False,
        metadata={
            "file_type": get_str("general.file_type"),
            "name": get_str("general.name"),
            "author": get_str("general.author"),
            "version": get_str("general.version"),
            "total_tensors": len(reader.tensors),
            "total_elements": total_elements,
        },
    )


# ── Safetensors Scanner ───────────────────────────────────────

def _scan_safetensors(source: str) -> ModelInfo:
    """Scan a safetensors model directory or file."""
    path = Path(source)

    # Load config
    config_path = path / "config.json" if path.is_dir() else None
    if config_path and not config_path.exists():
        raise ValueError(f"config.json not found in {source}")

    with open(config_path) as f:
        config = json.load(f)

    # Merge nested configs (e.g. Gemma4 multimodal: text_config, vision_config)
    text_config = config.get("text_config", {})
    if text_config:
        # Text config has the LLM fields; merge them into top-level for scanning
        merged = dict(config)
        merged.update(text_config)
        config = merged

    arch = _infer_architecture(config)
    param_count = _infer_param_count(config)

    # Detect quantization
    quant_method = None
    quant_bits = None

    # Standard quantization_config (AWQ, GPTQ, etc.)
    quant_config = config.get("quantization_config", config.get("quantization", {}))
    if quant_config:
        q_method = quant_config.get("quant_method", "")
        bits = quant_config.get("bits", 0)
        if q_method:
            quant_method = f"{q_method.upper()}_{bits}bit" if bits else q_method
            quant_bits = float(bits) if bits else None
        elif bits:
            # Some configs just have bits (e.g. OptiQ affine format)
            mode = quant_config.get("mode", "unknown")
            quant_method = f"{mode.upper()}_{bits}bit"
            quant_bits = float(bits)

    # Count layers from config
    num_layers = config.get("num_hidden_layers", config.get("n_layers", 0))

    # Count parameters from safetensors index if we couldn't infer from config
    if param_count == 0.0:
        total_params = _count_params_from_index(path)
        if total_params > 0:
            param_count = total_params / 1e9

    size_gb = _estimate_model_size_gb(source, ModelFormat.SAFETENSORS)
    vram_gb = _estimate_vram_gb(param_count, quant_bits or 16.0)

    return ModelInfo(
        name=path.name,
        format=ModelFormat.SAFETENSORS,
        architecture=arch,
        parameter_count=round(param_count, 1),
        quant_method=quant_method or "BF16",
        quant_bits=quant_bits or 16.0,
        context_length=config.get("max_position_embeddings", config.get("max_sequence_length", 4096)),
        hidden_size=config.get("hidden_size", 0),
        intermediate_size=config.get("intermediate_size", None),
        num_layers=num_layers,
        num_attention_heads=config.get("num_attention_heads", 0),
        num_key_value_heads=config.get("num_key_value_heads", None),
        vocab_size=config.get("vocab_size", 32000),
        model_size_gb=round(size_gb, 2),
        estimated_vram_gb=round(vram_gb, 2),
        source_path=str(path.resolve()),
        is_remote=False,
    )


def _count_params_from_index(path: Path) -> int:
    """Count total parameters from safetensors index + headers (fast, no data loaded)."""
    from safetensors import safe_open

    index_file = path / "model.safetensors.index.json"
    if not index_file.exists():
        # Single file: read shapes directly
        for sf in path.glob("*.safetensors"):
            total = 0
            try:
                with safe_open(sf, framework="numpy") as st:
                    for name in st.keys():
                        n = 1
                        for dim in st.get_tensor(name).shape:
                            n *= dim
                        total += n
            except Exception:  # pragma: no cover
                pass
            return total
        return 0

    with open(index_file) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})

    # For each unique safetensors file, open header and sum shapes
    tensors_per_file: dict[str, list[str]] = {}
    for tensor_name, filename in weight_map.items():
        tensors_per_file.setdefault(filename, []).append(tensor_name)

    total = 0
    for filename, names in tensors_per_file.items():
        sf_path = path / filename
        if not sf_path.exists():
            continue
        try:
            with safe_open(sf_path, framework="numpy") as st:
                for name in names:
                    n = 1
                    for dim in st.get_tensor(name).shape:
                        n *= dim
                    total += n
        except Exception:  # pragma: no cover
            pass
    return total


# ── HuggingFace Repo Scanner ────────────────────────────────────

def _scan_hf_repo(repo_id: str) -> ModelInfo:
    """Scan a HuggingFace model repo remotely via API."""
    from huggingface_hub import HfApi, model_info

    api = HfApi()

    try:
        info = api.model_info(repo_id)
    except Exception as e:
        raise ValueError(f"Cannot access HF repo {repo_id}: {e}")

    # Download config.json
    from huggingface_hub import hf_hub_download
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
    except Exception:
        config = {}

    arch = _infer_architecture(config)
    param_count = 0.0

    # Try to get param count from model card / siblings
    size_gb = 0.0
    quant_method = None
    quant_bits = None

    # Check siblings for quant method
    for sibling in (info.siblings or []):
        fname = sibling.rfilename
        if fname.endswith(".gguf"):
            qm, qb = _get_quant_from_filename(fname)
            if qm:
                quant_method = qm
                quant_bits = qb
                size_gb += getattr(sibling, "size", 0) / (1024**3)

    if size_gb == 0.0:
        # Sum all safetensors
        for sibling in (info.siblings or []):
            sfname = sibling.rfilename
            if sfname.endswith(".safetensors"):
                size_gb += getattr(sibling, "size", 0) / (1024**3)

    # Use config to determine if quantized
    quant_config = config.get("quantization_config", {})
    if quant_config and not quant_method:
        q_method = quant_config.get("quant_method", "")
        bits = quant_config.get("bits", 0)
        quant_method = f"{q_method.upper()}_{bits}bit" if q_method else None
        quant_bits = float(bits) if bits else None

    num_layers = config.get("num_hidden_layers", config.get("n_layers", 0))
    param_count = _infer_param_count(config)

    vram_gb = _estimate_vram_gb(param_count, quant_bits or 16.0)

    return ModelInfo(
        name=info.id.split("/")[-1] if info.id else repo_id.split("/")[-1],
        format=ModelFormat.HF_REPO,
        architecture=arch,
        parameter_count=round(param_count, 1),
        quant_method=quant_method or "BF16",
        quant_bits=quant_bits or 16.0,
        context_length=config.get("max_position_embeddings", 4096),
        hidden_size=config.get("hidden_size", 0),
        intermediate_size=config.get("intermediate_size", None),
        num_layers=num_layers,
        num_attention_heads=config.get("num_attention_heads", 0),
        num_key_value_heads=config.get("num_key_value_heads", None),
        vocab_size=config.get("vocab_size", 32000),
        model_size_gb=round(size_gb, 2),
        estimated_vram_gb=round(vram_gb, 2),
        source_path=repo_id,
        is_remote=True,
    )


# ── Main Entry Point ──────────────────────────────────────────

def scan(source: str) -> ModelInfo:
    """Scan a model file/directory/repo and return ModelInfo."""
    fmt = detect_format(source)

    if fmt == ModelFormat.GGUF:
        return _scan_gguf(source)
    elif fmt == ModelFormat.SAFETENSORS:
        return _scan_safetensors(source)
    elif fmt == ModelFormat.HF_REPO:
        return _scan_hf_repo(source)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


# Alias for backwards compatibility
scan_model = scan
