# llmbeans/cli.py
"""Interactive CLI for llmbeans — model scanner + config recommender.

Guided terminal workflow:
  1. Pick a hosting tool (llama-cli, lmstudio, omlx, ollama)
  2. Select a model from the tool's default model directory
  3. Auto-detect or select hardware profile
  4. Choose quality mode (balanced / quality / speed)
  5. Get recommendation with flags, estimates, and warnings
  6. Optionally write launch scripts
"""

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, Confirm
from rich import box

from llmbeans.models.scanner import scan_model, ModelInfo
from llmbeans.hardware.detector import detect_hardware
from llmbeans.hardware.profiles import load_profiles, HardwareProfileEntry, from_detection
from llmbeans.recommenders.engine import recommend, Recommendation
from llmbeans.recommenders.registry import get_available_tools

console = Console()


# ── Tool → model directory mapping ────────────────────────────

TOOL_MODEL_DIRS = {
    "llamacpp": [
        "~/.local/share/llama.cpp/models",
        "~/.cache/llama.cpp/models",
    ],
    "ollama": [
        "~/.ollama/models",
    ],
    "mlx": [
        "~/.mlx/models",
        "~/.cache/mlx/models",
        "~/.lmstudio/models/mlx-community",
    ],
    "lmstudio": [
        "~/.lmstudio/models",
    ],
    "vllm": [
        "~/.cache/huggingface/hub",
    ],
}


def _tool_description(tool: str) -> str:
    descriptions = {
        "llamacpp": "llama.cpp — fast, flexible, CLI-based",
        "ollama": "Ollama — simple, Docker-like experience",
        "mlx": "MLX — Apple Silicon optimized (safetensors)",
        "vllm": "vLLM — high-throughput server (CUDA)",
        "lmstudio": "LM Studio — GUI with llama.cpp backend",
    }
    return descriptions.get(tool, "")


# ── Step 1: Tool selection ────────────────────────────────────

def prompt_tool_selection(available: list[str], hardware: HardwareProfileEntry | None = None) -> str:
    """Let user pick a hosting tool, filtered by hardware compatibility."""
    console.print()
    console.print("[bold cyan]Step 1: Choose your hosting tool[/bold cyan]")

    # Filter tools by hardware
    compatible = []
    for tool in available:
        if tool == "mlx" and hardware and not hardware.metal:
            continue
        if tool == "vllm" and hardware and not hardware.cuda:
            continue
        compatible.append(tool)

    if not compatible:
        # No hardware info yet — show all
        compatible = available[:]

    for i, tool in enumerate(compatible, 1):
        desc = _tool_description(tool)
        console.print(f"  [bold]{i}.[/bold] [green]{tool}[/green] — {desc}")

    while True:
        choice = IntPrompt.ask("\nSelect tool (number)", default=1)
        if 1 <= choice <= len(compatible):
            return compatible[choice - 1]
        console.print("[red]Invalid selection.[/red]")


# ── Step 2: Model selection ───────────────────────────────────

def _working_directory() -> Path:
    """Return the process current working directory."""
    return Path.cwd()


def _scan_models_in_dir(directory: str) -> list[dict]:
    """Scan a directory for GGUF files and safetensors directories.

    Returns list of dicts with keys: path, format, size_gb.
    """
    path = Path(directory).expanduser()
    if not path.is_dir():
        return []

    results = []

    for gguf in sorted(path.rglob("*.gguf")):
        size_gb = gguf.stat().st_size / (1024 ** 3)
        results.append({"path": str(gguf), "format": "GGUF", "size_gb": size_gb})

    for st_file in sorted(path.rglob("*.safetensors")):
        parent = st_file.parent
        if not any(r["path"] == str(parent) for r in results):
            total = sum(f.stat().st_size for f in parent.rglob("*") if f.is_file())
            results.append({
                "path": str(parent),
                "format": "safetensors",
                "size_gb": total / (1024 ** 3),
            })

    # Also check for model.safetensors.index.json (sharded)
    for idx in sorted(path.rglob("model.safetensors.index.json")):
        parent = idx.parent
        if not any(r["path"] == str(parent) for r in results):
            total = sum(f.stat().st_size for f in parent.rglob("*") if f.is_file())
            results.append({
                "path": str(parent),
                "format": "safetensors",
                "size_gb": total / (1024 ** 3),
            })

    return results


def _resolve_model_dir(path: Path) -> Path:
    """If a directory contains exactly one model subdirectory, auto-descend."""
    if not path.is_dir():
        return path
    sub = _find_model_subdirs(path)
    if len(sub) == 1:
        return sub[0]
    return path


def _find_model_subdirs(path: Path) -> list[Path]:
    """Find subdirectories that contain model files."""
    results = []
    for sub in sorted(path.iterdir()):
        if sub.is_dir():
            if any(sub.glob("*.safetensors")):
                results.append(sub)
            elif (sub / "model.safetensors.index.json").exists():
                results.append(sub)
            elif any(sub.glob("*.gguf")):
                results.append(sub)
    return results


def prompt_model_selection(tool: str) -> str:
    """Present tool's known model dirs for the user to pick from."""
    console.print()
    console.print(f"[bold cyan]Step 2: Select a model (tool: {tool})[/bold cyan]")

    known_dirs = TOOL_MODEL_DIRS.get(tool, [])

    all_models: list[dict] = []
    for directory in known_dirs:
        expanded = str(Path(directory).expanduser().resolve())
        models = _scan_models_in_dir(expanded)
        for m in models:
            m["source_dir"] = expanded
        all_models.extend(models)

    if all_models:
        console.print(f"\n[green]Found {len(all_models)} model(s) in {tool}'s known directories:[/green]")
        for i, m in enumerate(all_models, 1):
            # Show path relative to the source dir for readability
            src = m.get("source_dir", "")
            model_path = Path(m["path"])
            if src:
                try:
                    display_name = str(model_path.resolve().relative_to(Path(src).resolve()))
                except ValueError:
                    display_name = model_path.name
            else:
                display_name = model_path.name

            console.print(
                f"  [bold]{i:3d}.[/bold] {display_name} "
                f"[dim]({m['format']}, {m['size_gb']:.1f} GB)[/dim]"
            )

        console.print(f"  [bold]{len(all_models) + 1:3d}.[/bold] [yellow]Enter a custom path[/yellow]")
        console.print(f"  [bold]{len(all_models) + 2:3d}.[/bold] [yellow]Search a folder[/yellow]")
    else:
        console.print(f"\n[yellow]No models found in {tool}'s default directories.[/yellow]")
        console.print("You'll need to enter a custom path or search a folder.")
        all_models = []

    max_option = len(all_models) + 2

    while True:
        if all_models:
            choice = IntPrompt.ask(f"\nSelect model (1-{max_option})", default=1)
            if 1 <= choice <= len(all_models):
                selected = all_models[choice - 1]["path"]
                resolved = _resolve_model_dir(Path(selected))
                return str(resolved)
            if choice == len(all_models) + 1:
                return _prompt_custom_path()
            if choice == len(all_models) + 2:
                return _prompt_search_folder()
            console.print("[red]Invalid selection.[/red]")
        else:
            choice = Prompt.ask(
                "Options: [c]ustom path / [s]earch folder",
                default="c",
            ).strip().lower()
            if choice.startswith("c"):
                return _prompt_custom_path()
            elif choice.startswith("s"):
                return _prompt_search_folder()
            console.print("[red]Invalid choice. Type 'c' or 's'.[/red]")


def _prompt_custom_path() -> str:
    """Prompt for a model path, starting with models found in the current directory."""
    cwd = _working_directory()
    cwd_models = _scan_models_in_dir(str(cwd))
    if cwd_models:
        console.print(
            f"\n[green]Found {len(cwd_models)} model(s) in {cwd}:[/green]"
        )
        for i, m in enumerate(cwd_models, 1):
            model_path = Path(m["path"])
            try:
                display_name = str(model_path.relative_to(cwd))
            except ValueError:
                display_name = model_path.name
            console.print(
                f"  [bold]{i}.[/bold] {display_name} "
                f"[dim]({m['format']}, {m['size_gb']:.1f} GB)[/dim]"
            )
        console.print(
            f"  [bold]{len(cwd_models) + 1}.[/bold] "
            "[yellow]Enter a path manually[/yellow]"
        )

        while True:
            choice = IntPrompt.ask(
                f"Select model (1-{len(cwd_models) + 1})",
                default=1,
            )
            if 1 <= choice <= len(cwd_models):
                resolved = _resolve_model_dir(Path(cwd_models[choice - 1]["path"]))
                return str(resolved)
            if choice == len(cwd_models) + 1:
                break
            console.print("[red]Invalid selection.[/red]")

    while True:
        raw = Prompt.ask("[bold]Model path[/bold]").strip()
        if not raw:
            console.print("[red]Please enter a path.[/red]")
            continue

        # HF repo ID
        if "/" in raw and not os.path.exists(raw):
            return raw

        path = Path(raw).expanduser()
        if path.exists():
            resolved = _resolve_model_dir(path)
            return str(resolved)

        console.print(f"[red]Path not found: {raw}[/red]")


def _prompt_search_folder() -> str:
    """Prompt for a folder to search for models."""
    while True:
        raw = Prompt.ask("[bold]Folder to search[/bold]").strip()
        if not raw:
            console.print("[red]Please enter a folder path.[/red]")
            continue

        path = Path(raw).expanduser()
        if not path.is_dir():
            console.print(f"[red]Not a directory: {raw}[/red]")
            continue

        models = _scan_models_in_dir(str(path))
        if not models:
            console.print(f"[yellow]No models found in {raw}[/yellow]")
            continue

        console.print(f"\n[green]Found {len(models)} model(s):[/green]")
        for i, m in enumerate(models, 1):
            display_name = Path(m["path"]).name
            console.print(
                f"  [bold]{i}.[/bold] {display_name} "
                f"[dim]({m['format']}, {m['size_gb']:.1f} GB)[/dim]"
            )

        choice = IntPrompt.ask("Select a model (number)", default=1)
        if 1 <= choice <= len(models):
            resolved = _resolve_model_dir(Path(models[choice - 1]["path"]))
            return str(resolved)
        console.print("[red]Invalid selection.[/red]")


# ── Step 3: Hardware ──────────────────────────────────────────

def get_hardware_profiles() -> list[HardwareProfileEntry]:
    """Load all hardware profiles from the database."""
    return load_profiles()


def prompt_hardware_selection(
    profiles: list[HardwareProfileEntry],
    auto_detected: HardwareProfileEntry | None = None,
) -> HardwareProfileEntry:
    """Let user pick a hardware profile or use auto-detected."""
    console.print()
    console.print("[bold cyan]Step 3: Hardware[/bold cyan]")

    if auto_detected:
        console.print(
            f"[green]Auto-detected:[/green] {auto_detected.name} "
            f"({auto_detected.ram_gb}GB RAM, {auto_detected.gpu_name})"
        )
        if Confirm.ask("Use auto-detected hardware?", default=True):
            return auto_detected

    if not profiles:
        console.print("[yellow]No hardware profiles available. Using auto-detected.[/yellow]")
        return auto_detected

    # Group by category
    categories: dict[str, list[HardwareProfileEntry]] = {}
    for p in profiles:
        categories.setdefault(p.category, []).append(p)

    console.print("\n[bold]Available hardware profiles:[/bold]")
    idx = 1
    indexed: dict[int, HardwareProfileEntry] = {}
    for cat, entries in sorted(categories.items()):
        console.print(f"\n  [bold yellow]{cat.upper()}[/bold yellow]")
        for entry in entries:
            gpu_info = f"{entry.gpu_name}"
            if entry.unified_memory:
                gpu_info += " (unified)"
            elif entry.gpu_vram_gb:
                gpu_info += f" ({entry.gpu_vram_gb}GB VRAM)"
            console.print(
                f"    [bold]{idx:3d}.[/bold] {entry.name} "
                f"[dim]— {entry.ram_gb}GB RAM, {gpu_info}[/dim]"
            )
            indexed[idx] = entry
            idx += 1

    while True:
        choice = IntPrompt.ask("\nSelect hardware (number)", default=1)
        if choice in indexed:
            return indexed[choice]
        console.print("[red]Invalid selection.[/red]")


# ── Step 4: Quality mode ──────────────────────────────────────

def prompt_quality_mode() -> str:
    """Let user pick a quality mode."""
    console.print()
    console.print("[bold cyan]Step 4: Quality mode[/bold cyan]")

    modes = [
        ("balanced", "Good default — moderate context, solid speed"),
        ("quality", "Maximum context length — slower but better for long conversations"),
        ("speed", "Shorter context — fastest response time"),
    ]

    for i, (mode, desc) in enumerate(modes, 1):
        console.print(f"  [bold]{i}.[/bold] [green]{mode}[/green] — {desc}")

    while True:
        choice = IntPrompt.ask("\nSelect mode (number)", default=1)
        if 1 <= choice <= len(modes):
            return modes[choice - 1][0]
        console.print("[red]Invalid selection.[/red]")


# ── Step 5: Generate recommendation ────────────────────────────

def generate_summary(
    model: ModelInfo,
    hardware: HardwareProfileEntry,
    rec: Recommendation,
) -> str:
    """Generate a human-readable summary of the recommendation."""
    lines = []

    # Model info
    lines.append(f"Model: {model.name} ({model.architecture})")
    lines.append(f"Architecture: {model.architecture}")
    if model.parameter_count:
        lines.append(f"Parameters: ~{model.parameter_count}B")
    if model.quant_method:
        lines.append(f"Quantization: {model.quant_method} ({model.quant_bits or '?'} bits)")
    lines.append(f"Size: {model.model_size_gb} GB")
    lines.append(f"Context length: {model.context_length} tokens")
    lines.append(f"Layers: {model.num_layers}")
    lines.append("")

    # Hardware info
    lines.append(f"Hardware: {hardware.name}")
    ram_line = f"RAM: {hardware.ram_gb} GB"
    if hardware.ram_type:
        ram_line += f" ({hardware.ram_type})"
    lines.append(ram_line)
    gpu_str = hardware.gpu_name
    if hardware.unified_memory:
        gpu_str += " (Unified memory)"
    elif hardware.gpu_vram_gb:
        gpu_str += f" ({hardware.gpu_vram_gb} GB VRAM)"
    lines.append(f"GPU: {gpu_str}")
    lines.append(f"Memory bandwidth: {hardware.memory_bandwidth_gbps} GB/s")
    lines.append("")

    # Tool + mode
    lines.append(f"Hosting Tool: {rec.hosting_tool}")
    lines.append("")

    # Recommended configuration
    lines.append("Recommended Configuration:")
    for flag, value in rec.flags.items():
        if value == "":
            lines.append(f"  {flag}")
        else:
            lines.append(f"  {flag}: {value}")

    lines.append("")

    # Performance estimates
    lines.append("Estimated Performance:")
    lines.append(f"  Tokens/second: ~{rec.estimated_tok_per_sec}")
    if rec.estimated_vram_usage_gb:
        lines.append(f"  VRAM usage: ~{rec.estimated_vram_usage_gb} GB")
    if rec.estimated_ram_usage_gb:
        lines.append(f"  RAM usage: ~{rec.estimated_ram_usage_gb} GB")
    lines.append(f"  GPU offload: {rec.gpu_offload_layers}/{model.num_layers} layers")
    lines.append(f"  Context: {rec.context_length} tokens")
    lines.append(f"  Threads: {rec.thread_count}")
    lines.append(f"  Batch size: {rec.batch_size}")

    # Command
    if rec.command:
        lines.append("")
        lines.append("Command:")
        lines.append(f"  {rec.command}")

    # Extra config (e.g. Modelfile, LM Studio JSON)
    if rec.extra_config:
        lines.append("")
        lines.append("Extra config:")
        for cfg_line in rec.extra_config.split("\n"):
            lines.append(f"  {cfg_line}")

    # Warnings
    if rec.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in rec.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


# ── Step 6: Write scripts ─────────────────────────────────────

def write_scripts(
    rec: Recommendation,
    model: ModelInfo,
    hardware: HardwareProfileEntry,
    summary: str,
    output_dir: str | None = None,
) -> dict[str, str]:
    """Write launch scripts to disk. Returns dict of {type: path}."""
    if output_dir is None:
        output_dir = str(_working_directory() / "llmbeans-output")

    os.makedirs(output_dir, exist_ok=True)
    result = {}

    # Shell script
    shell_path = os.path.join(output_dir, "run.sh")
    with open(shell_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# llmbeans generated script for {model.name}\n")
        f.write(f"# Tool: {rec.hosting_tool}\n")
        f.write(f"# Context: {rec.context_length} | Threads: {rec.thread_count}\n")
        f.write("\n")
        f.write(rec.command + "\n")
    os.chmod(shell_path, 0o755)
    result["shell"] = shell_path

    # Batch script (Windows)
    batch_path = os.path.join(output_dir, "run.bat")
    with open(batch_path, "w") as f:
        f.write("@echo off\n")
        f.write(f"REM llmbeans generated script for {model.name}\n")
        f.write(f"REM Tool: {rec.hosting_tool}\n")
        f.write("\n")
        cmd = rec.command.replace("llama-cli", "llama-cli.exe")
        f.write(cmd + "\n")
    result["batch"] = batch_path

    # Extra config file (Modelfile, etc.)
    if rec.extra_config:
        if rec.hosting_tool == "ollama":
            modelfile_path = os.path.join(output_dir, "Modelfile")
            with open(modelfile_path, "w") as f:
                f.write(rec.extra_config)
            result["modelfile"] = modelfile_path
        elif rec.hosting_tool == "lmstudio":
            config_path = os.path.join(output_dir, "lmstudio-config.json")
            with open(config_path, "w") as f:
                f.write(rec.extra_config)
            result["config"] = config_path

    # Summary file
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    result["summary"] = summary_path

    return result


# ── Display helpers ───────────────────────────────────────────

def _safe_get(obj, attr, default="unknown"):
    """Safely get an attribute, returning default if missing or non-primitive."""
    if not hasattr(obj, attr):
        return default
    val = getattr(obj, attr)
    if val is None:
        return default
    if isinstance(val, (str, int, float, bool, list, dict)):
        return val
    if hasattr(val, 'value') and isinstance(val.value, (str, int, float)):
        return val.value
    return default


def display_model_info(model: ModelInfo):
    """Display scanned model info."""
    table = Table(title=f"Model: {_safe_get(model, 'name', 'unknown')}", box=box.ROUNDED)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value")

    fmt = _safe_get(model, 'format')
    table.add_row("Format", fmt or "unknown")
    table.add_row("Architecture", _safe_get(model, 'architecture'))
    param_count = _safe_get(model, 'parameter_count', 0)
    table.add_row("Parameters", f"~{param_count}B" if param_count else "unknown")
    quant_method = _safe_get(model, 'quant_method')
    quant_bits = _safe_get(model, 'quant_bits')
    table.add_row("Quantization", f"{quant_method} ({quant_bits or '?'} bits)" if quant_method else "none")
    table.add_row("Size", f"{_safe_get(model, 'model_size_gb', 0)} GB")
    table.add_row("Est. VRAM", f"{_safe_get(model, 'estimated_vram_gb', 0)} GB")
    table.add_row("Context", f"{_safe_get(model, 'context_length', 0)} tokens")
    table.add_row("Layers", str(_safe_get(model, 'num_layers', 0)))
    table.add_row("Source", _safe_get(model, 'source_path'))

    console.print()
    console.print(table)


def display_recommendation(rec: Recommendation, model: ModelInfo):
    """Display the recommendation results."""
    console.print()

    num_layers = _safe_get(model, 'num_layers', 0)

    # Performance table
    perf = Table(title="Performance Estimate", box=box.ROUNDED)
    perf.add_column("Metric", style="bold cyan")
    perf.add_column("Value", style="green")
    perf.add_row("Tokens/sec", f"~{_safe_get(rec, 'estimated_tok_per_sec', 0)}")
    gpu_off = _safe_get(rec, 'gpu_offload_layers', 0)
    perf.add_row("GPU offload", f"{gpu_off}/{num_layers} layers")
    perf.add_row("Context length", f"{_safe_get(rec, 'context_length', 0)} tokens")
    perf.add_row("Threads", str(_safe_get(rec, 'thread_count', 0)))
    perf.add_row("Batch size", str(_safe_get(rec, 'batch_size', 0)))
    vram = _safe_get(rec, 'estimated_vram_usage_gb', 0)
    if vram:
        perf.add_row("VRAM usage", f"~{vram} GB")
    ram = _safe_get(rec, 'estimated_ram_usage_gb', 0)
    if ram:
        perf.add_row("RAM usage", f"~{ram} GB")
    console.print(perf)

    # Flags table
    flags = _safe_get(rec, 'flags', {})
    if flags:
        flags_table = Table(title="Flags", box=box.ROUNDED)
        flags_table.add_column("Flag", style="bold yellow")
        flags_table.add_column("Value")
        for flag, value in flags.items():
            flags_table.add_row(flag, value if value else "(flag-only)")
        console.print()
        console.print(flags_table)

    # Command
    command = _safe_get(rec, 'command', '')
    if command:
        console.print()
        console.print(Panel(
            command,
            title="Command",
            border_style="green",
            expand=False,
        ))

    # Warnings
    warnings = _safe_get(rec, 'warnings', [])
    if warnings:
        console.print()
        for w in warnings:
            console.print(f"  [bold yellow]WARNING:[/bold yellow] {w}")


# ── Main flow ─────────────────────────────────────────────────

def main():
    """Run the llmbeans CLI."""
    console.print()
    console.print(Panel.fit(
        "[bold green]llmbeans[/bold green] — Find the optimal way to run any LLM locally",
        border_style="cyan",
    ))

    # Step 1: Tool selection
    available_tools = get_available_tools()
    tool = prompt_tool_selection(available_tools)

    # Step 2: Model selection (from tool's known dirs)
    model_path = prompt_model_selection(tool)
    console.print(f"\n[dim]Scanning {model_path}...[/dim]")

    try:
        model = scan_model(model_path)
    except Exception as e:
        console.print(f"[red]Error scanning model: {e}[/red]")
        sys.exit(1)

    display_model_info(model)

    # Step 3: Hardware
    profiles = get_hardware_profiles()
    auto_hw = None
    try:
        detected = detect_hardware()
        if detected is not None:
            auto_hw = from_detection(detected)
    except Exception:
        pass

    hardware = prompt_hardware_selection(profiles, auto_hw)

    # Step 4: Quality mode
    quality_mode = prompt_quality_mode()

    # Step 5: Generate recommendation
    console.print(f"\n[dim]Generating recommendation...[/dim]")
    try:
        rec = recommend(model, hardware, tool, quality_mode)
    except Exception as e:
        console.print(f"[red]Error generating recommendation: {e}[/red]")
        sys.exit(1)

    display_recommendation(rec, model)

    # Step 6: Summary + scripts
    summary = generate_summary(model, hardware, rec)
    console.print()
    console.print(Panel(summary, title="Summary", border_style="blue"))

    if Confirm.ask("\nWrite launch scripts to disk?", default=True):
        output_dir = "llmbeans-output"
        try:
            written = write_scripts(rec, model, hardware, summary, output_dir)
            console.print()
            for ftype, fpath in written.items():
                console.print(f"  [green]{ftype} script[/green] saved to: {fpath}")
        except Exception as e:
            console.print(f"[red]Error writing scripts: {e}[/red]")

    console.print()
    console.print("[bold green]Done![/bold green] Happy inferencing.")
    sys.exit(0)


if __name__ == "__main__":
    main()
