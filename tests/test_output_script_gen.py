"""Tests for llmbeans.output.script_gen."""

import os
import tempfile

from llmbeans.output.script_gen import (
    generate_batch_script,
    generate_shell_script,
    generate_summary,
    write_scripts,
)


def test_generate_summary_and_scripts(sample_model, nvidia_hardware, sample_recommendation):
    summary = generate_summary(sample_model, nvidia_hardware, sample_recommendation)
    assert "llmbeans" in summary
    assert "Warnings" in summary
    assert sample_recommendation.command in summary

    shell = generate_shell_script(sample_model, nvidia_hardware, sample_recommendation)
    assert shell.startswith("#!/usr/bin/env bash")
    assert "export" not in shell  # no OLLAMA/CUDA flags in sample

    batch = generate_batch_script(sample_model, nvidia_hardware, sample_recommendation)
    assert batch.startswith("@echo off")
    assert "pause" in batch


def test_generate_shell_and_batch_with_env_flags(sample_model, nvidia_hardware, sample_recommendation):
    sample_recommendation.flags = {
        "CUDA_VISIBLE_DEVICES": "0",
        "OLLAMA_METAL": "1",
        "GGML_CUDA": "1",
        "-ngl": "32",
    }
    shell = generate_shell_script(sample_model, nvidia_hardware, sample_recommendation)
    assert 'export CUDA_VISIBLE_DEVICES="0"' in shell
    assert 'export OLLAMA_METAL="1"' in shell

    batch = generate_batch_script(sample_model, nvidia_hardware, sample_recommendation)
    assert 'set "CUDA_VISIBLE_DEVICES=0"' in batch


def test_write_scripts_creates_files(sample_model, nvidia_hardware, sample_recommendation):
    sample_recommendation.extra_config = '{"load": {"gpu_layers": 32}}'
    with tempfile.TemporaryDirectory() as tmpdir:
        files = write_scripts(sample_model, nvidia_hardware, sample_recommendation, output_dir=tmpdir)
        assert os.path.exists(files["summary"])
        assert os.path.exists(files["shell"])
        assert os.path.exists(files["batch"])
        assert os.path.exists(files["config"])
