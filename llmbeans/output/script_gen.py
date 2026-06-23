# llmbeans/output/script_gen.py
"""Generate startup scripts from Recommendation objects.

Canonical summary/script writers live in llmbeans.cli; re-exported here
for backwards-compatible imports. This module keeps richer shell/batch
generators used by tests and external callers.
"""

from datetime import datetime

from llmbeans.cli import generate_summary, write_scripts
from llmbeans.recommenders.engine import Recommendation
from llmbeans.models.scanner import ModelInfo
from llmbeans.hardware.profiles import HardwareProfileEntry

__all__ = [
    "generate_summary",
    "write_scripts",
    "generate_shell_script",
    "generate_batch_script",
]


def generate_shell_script(
    model: ModelInfo,
    hardware: HardwareProfileEntry,
    rec: Recommendation,
) -> str:
    """Generate a bash startup script."""
    lines = [
        "#!/usr/bin/env bash",
        f"# llmbeans generated startup script for {model.name}",
        f"# Hosting tool: {rec.hosting_tool}",
        f"# Hardware: {hardware.name}",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"# Estimated speed: ~{rec.estimated_tok_per_sec} tok/s",
        "",
        "set -euo pipefail",
        "",
    ]

    for key, val in rec.flags.items():
        if key.startswith("OLLAMA_") or key.startswith("CUDA_") or key.startswith("GGML_"):
            lines.append(f'export {key}="{val}"')

    if any(k.startswith("OLLAMA_") or k.startswith("CUDA_") for k in rec.flags):
        lines.append("")

    if rec.command:
        for cmd_line in rec.command.split("\n"):
            lines.append(cmd_line)

    lines.append("")
    return "\n".join(lines)


def generate_batch_script(
    model: ModelInfo,
    hardware: HardwareProfileEntry,
    rec: Recommendation,
) -> str:
    """Generate a Windows batch startup script."""
    lines = [
        "@echo off",
        f"REM llmbeans generated startup script for {model.name}",
        f"REM Hosting tool: {rec.hosting_tool}",
        f"REM Hardware: {hardware.name}",
        f"REM Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"REM Estimated speed: ~{rec.estimated_tok_per_sec} tok/s",
        "",
    ]

    for key, val in rec.flags.items():
        if key.startswith("CUDA_") or key.startswith("OLLAMA_") or key.startswith("GGML_"):
            lines.append(f'set "{key}={val}"')

    if any(k.startswith("CUDA_") or k.startswith("OLLAMA_") for k in rec.flags):
        lines.append("")

    if rec.command:
        for cmd_line in rec.command.split("\n"):
            lines.append(cmd_line)

    lines.append("")
    lines.append("pause")
    return "\n".join(lines)
