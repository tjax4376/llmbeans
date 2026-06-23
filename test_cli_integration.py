#!/usr/bin/env python3
"""Integration smoke test for llmbeans CLI with mocked inputs."""

import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_cli_with_mocked_inputs():
    print("Testing llmbeans CLI with mocked inputs...")

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
            context_length=8192,
            hidden_size=4096,
            intermediate_size=11008,
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=32000,
            source_path="/tmp/demo.gguf",
            is_remote=False,
            format=None,
        )
        rec = Recommendation(
            hosting_tool="llamacpp",
            context_length=8192,
            batch_size=512,
            thread_count=8,
            gpu_offload_layers=32,
            estimated_tok_per_sec=45.0,
            command="llama-cli -m /tmp/demo.gguf",
        )

        with patch("builtins.input", side_effect=mock_input), \
             patch("llmbeans.cli._scan_models_in_dir", return_value=[{
                 "path": "/tmp/demo.gguf", "format": "GGUF", "size_gb": 4.5,
             }]), \
             patch("llmbeans.cli.scan", return_value=model), \
             patch("llmbeans.cli.detect_hardware", return_value=MagicMock()), \
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

    print(output)
    assert "llmbeans" in output.lower()
    print("✓ CLI smoke test passed")
    return output


if __name__ == "__main__":
    test_cli_with_mocked_inputs()
