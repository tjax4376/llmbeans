#!/usr/bin/env python3
"""Demo llmbeans CLI with mocked model scan and hardware detection."""

import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_cli_demo():
    print("Running llmbeans CLI demo with mocked inputs...")
    print("=" * 60)

    mock_inputs = iter(["1", "1", "1", "1", "n"])

    def mock_input(_prompt=""):
        try:
            return next(mock_inputs)
        except StopIteration:
            return ""

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        from llmbeans.cli import main
        from llmbeans.hardware.detector import HardwareProfile
        from llmbeans.hardware.profiles import get_profile_by_id
        from llmbeans.models.scanner import ModelInfo
        from llmbeans.recommenders.engine import Recommendation

        profile = get_profile_by_id("macbook-pro-m4-2024")
        model = ModelInfo(
            name="demo-model",
            architecture="llama",
            parameter_count=7.0,
            quant_method="Q4_K_M",
            quant_bits=4.0,
            model_size_gb=4.5,
            estimated_vram_gb=4.5,
            context_length=32768,
            hidden_size=4096,
            intermediate_size=11008,
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=32000,
            source_path="/fake/path/to/model.gguf",
            is_remote=False,
            format=None,
        )
        detected = HardwareProfile(
            os="darwin",
            cpu_cores=12,
            ram_total_gb=24.0,
            ram_free_gb=12.0,
            gpu_vendor="apple",
            gpu_vram_gb=None,
            gpu_name="Apple M4 Pro",
            is_apple_silicon=True,
            unified_memory=True,
            metal_supported=True,
            disk_is_ssd=True,
            laptop_model="Mac16,1",
            memory_bandwidth_gbps=273.0,
        )
        rec = Recommendation(
            hosting_tool="llamacpp",
            context_length=8192,
            batch_size=512,
            thread_count=12,
            gpu_offload_layers=32,
            estimated_tok_per_sec=45.0,
            estimated_vram_usage_gb=3.6,
            estimated_ram_usage_gb=5.2,
            command="llama-cli -m /fake/path/to/model.gguf",
            warnings=[
                "Model is much smaller than available unified memory. "
                "Consider increasing context length for better utilization.",
            ],
        )

        with patch("builtins.input", side_effect=mock_input), \
             patch("llmbeans.cli._scan_models_in_dir", return_value=[{
                 "path": "/fake/path/to/model.gguf", "format": "GGUF", "size_gb": 4.5,
             }]), \
             patch("llmbeans.cli.scan", return_value=model), \
             patch("llmbeans.cli.detect_hardware", return_value=detected), \
             patch("llmbeans.cli.from_detection", return_value=profile), \
             patch("llmbeans.cli.get_available_tools", return_value=["llamacpp"]), \
             patch("llmbeans.cli.recommend", return_value=rec):
            try:
                main()
            except SystemExit:
                pass
    finally:
        output = captured_output.getvalue()
        sys.stdout = old_stdout

    return output


if __name__ == "__main__":
    output = run_cli_demo()
    print("\n" + "=" * 60)
    print("DEMO OUTPUT:")
    print("=" * 60)
    print(output)
