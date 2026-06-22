"""Tests for llmbeans.recommenders.script_gen."""

import os
import tempfile

from llmbeans.recommenders.script_gen import write_scripts


def test_recommenders_write_scripts(sample_model, sample_recommendation):
    sample_recommendation.command = "llama-cli -m /tmp/model.gguf"
    with tempfile.TemporaryDirectory() as tmpdir:
        written = write_scripts(sample_recommendation, sample_model, output_dir=tmpdir)
        assert os.path.exists(written["shell"])
        assert os.path.exists(written["batch"])
        assert os.path.exists(written["summary"])

        with open(written["shell"]) as f:
            content = f.read()
        assert "llama-cli" in content

        with open(written["summary"]) as f:
            summary = f.read()
        assert sample_model.name in summary
