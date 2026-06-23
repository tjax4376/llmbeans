"""Tests for canonical write_scripts in llmbeans.cli."""

import os
import tempfile

from llmbeans.cli import generate_summary, write_scripts


def test_cli_write_scripts(sample_model, nvidia_hardware, sample_recommendation):
    sample_recommendation.command = "llama-cli -m /tmp/model.gguf"
    summary = generate_summary(sample_model, nvidia_hardware, sample_recommendation)
    with tempfile.TemporaryDirectory() as tmpdir:
        written = write_scripts(
            sample_recommendation,
            sample_model,
            nvidia_hardware,
            summary,
            output_dir=tmpdir,
        )
        assert os.path.exists(written["shell"])
        assert os.path.exists(written["batch"])
        assert os.path.exists(written["summary"])

        with open(written["shell"]) as f:
            content = f.read()
        assert "llama-cli" in content

        with open(written["summary"]) as f:
            summary_text = f.read()
        assert sample_model.name in summary_text
