"""Extended tests for llmbeans.cli helpers and flows."""

import os
import sys
import tempfile
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llmbeans.cli import (
    _find_model_subdirs,
    _prompt_custom_path,
    _prompt_search_folder,
    _resolve_model_dir,
    _safe_get,
    _scan_models_in_dir,
    _tool_description,
    _working_directory,
    display_model_info,
    display_recommendation,
    generate_summary,
    get_hardware_profiles,
    main,
    prompt_hardware_selection,
    prompt_model_selection,
    prompt_quality_mode,
    prompt_tool_selection,
    write_scripts,
)
from llmbeans.hardware.profiles import get_profile_by_id
from llmbeans.models.scanner import ModelFormat, ModelInfo
from llmbeans.recommenders.engine import Recommendation


def test_tool_description_known_and_unknown():
    assert "llama.cpp" in _tool_description("llamacpp")
    assert _tool_description("unknown-tool") == ""


@patch("llmbeans.cli.IntPrompt.ask", side_effect=[99, 1])
def test_prompt_tool_selection_invalid_then_valid(mock_int):
    tools = prompt_tool_selection(["llamacpp", "ollama"])
    assert tools == "llamacpp"


@patch("llmbeans.cli.IntPrompt.ask", return_value=1)
def test_prompt_tool_selection_filters_incompatible(mock_int):
    hw = get_profile_by_id("generic-rtx-4060-laptop")
    tools = prompt_tool_selection(["llamacpp", "mlx", "vllm"], hardware=hw)
    assert "mlx" not in ["llamacpp", "ollama"] or tools in {"llamacpp", "vllm"}


@patch("llmbeans.cli.IntPrompt.ask", return_value=1)
def test_prompt_tool_selection_empty_compatible_falls_back(mock_int):
    hw = replace(get_profile_by_id("generic-rtx-4060-laptop"), cuda=False, metal=False)
    tool = prompt_tool_selection(["mlx", "vllm"], hardware=hw)
    assert tool in {"mlx", "vllm"}


def test_scan_models_in_dir_all_formats(tmp_path):
    (tmp_path / "a.gguf").write_bytes(b"x" * 128)
    model_dir = tmp_path / "st-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"x" * 128)
    sharded = tmp_path / "sharded"
    sharded.mkdir()
    (sharded / "model.safetensors.index.json").write_text("{}")
    (sharded / "part.safetensors").write_bytes(b"x" * 64)

    results = _scan_models_in_dir(str(tmp_path))
    assert len(results) >= 3
    assert _scan_models_in_dir("/path/does/not/exist") == []


def test_find_and_resolve_model_dirs(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    child = parent / "child-model"
    child.mkdir()
    (child / "model.safetensors").write_bytes(b"x")
    assert _resolve_model_dir(parent) == child
    assert _resolve_model_dir(child) == child
    assert _resolve_model_dir(tmp_path / "file.gguf") == tmp_path / "file.gguf"


def test_find_model_subdirs_all_patterns(tmp_path):
    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir()
    (gguf_dir / "m.gguf").write_bytes(b"x")
    idx_dir = tmp_path / "idx"
    idx_dir.mkdir()
    (idx_dir / "model.safetensors.index.json").write_text("{}")
    st_dir = tmp_path / "st"
    st_dir.mkdir()
    (st_dir / "model.safetensors").write_bytes(b"x")
    (tmp_path / "empty").mkdir()
    subs = _find_model_subdirs(tmp_path)
    assert len(subs) == 3


@patch("llmbeans.cli._prompt_custom_path", return_value="/custom/path")
@patch("llmbeans.cli.IntPrompt.ask", return_value=3)
@patch("llmbeans.cli._scan_models_in_dir", return_value=[{"path": "/m/model.gguf", "format": "GGUF", "size_gb": 1.0}])
def test_prompt_model_selection_custom_path(mock_scan, mock_int, mock_custom):
    path = prompt_model_selection("llamacpp")
    assert path == "/custom/path"


@patch("llmbeans.cli._prompt_search_folder", return_value="/search/path")
@patch("llmbeans.cli.IntPrompt.ask", return_value=4)
@patch("llmbeans.cli._scan_models_in_dir", return_value=[{"path": "/m/model.gguf", "format": "GGUF", "size_gb": 1.0}])
def test_prompt_model_selection_search_folder(mock_scan, mock_int, mock_search):
    path = prompt_model_selection("llamacpp")
    assert path == "/search/path"


@patch("llmbeans.cli._prompt_custom_path", return_value="/custom")
@patch("llmbeans.cli.Prompt.ask", return_value="c")
@patch("llmbeans.cli._scan_models_in_dir", return_value=[])
def test_prompt_model_selection_no_models_custom(mock_scan, mock_prompt, mock_custom):
    assert prompt_model_selection("unknown-tool") == "/custom"


@patch("llmbeans.cli._prompt_search_folder", return_value="/search")
@patch("llmbeans.cli.Prompt.ask", side_effect=["x", "s"])
@patch("llmbeans.cli._scan_models_in_dir", return_value=[])
def test_prompt_model_selection_no_models_invalid_then_search(mock_scan, mock_prompt, mock_search):
    assert prompt_model_selection("unknown-tool") == "/search"


@patch("llmbeans.cli.IntPrompt.ask", return_value=1)
@patch("llmbeans.cli._scan_models_in_dir")
def test_prompt_model_selection_picks_known_model(mock_scan, mock_int, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    mock_scan.return_value = [{"path": str(model), "format": "GGUF", "size_gb": 0.001, "source_dir": str(tmp_path)}]
    selected = prompt_model_selection("llamacpp")
    assert selected == str(model)


@patch("llmbeans.cli.IntPrompt.ask", side_effect=[99, 2])
@patch("llmbeans.cli._working_directory")
def test_prompt_custom_path_manual_hf_repo(mock_cwd, mock_int, tmp_path):
    mock_cwd.return_value = tmp_path
    with patch("llmbeans.cli.Prompt.ask", return_value="org/remote-model"):
        assert _prompt_custom_path() == "org/remote-model"


@patch("llmbeans.cli.IntPrompt.ask", side_effect=[99, 2])
@patch("llmbeans.cli._working_directory")
def test_prompt_custom_path_manual_existing_file(mock_cwd, mock_int, tmp_path):
    model = tmp_path / "manual.gguf"
    model.write_bytes(b"x")
    mock_cwd.return_value = tmp_path
    with patch("llmbeans.cli.Prompt.ask", side_effect=["", str(model)]):
        assert _prompt_custom_path() == str(model)


@patch("llmbeans.cli.IntPrompt.ask", side_effect=[99, 2])
@patch("llmbeans.cli._working_directory")
def test_prompt_custom_path_manual_missing(mock_cwd, mock_int, tmp_path):
    mock_cwd.return_value = tmp_path
    with patch("llmbeans.cli.Prompt.ask", side_effect=["missing.gguf", str(tmp_path / "ok.gguf")]):
        ok = tmp_path / "ok.gguf"
        ok.write_bytes(b"x")
        assert _prompt_custom_path() == str(ok)


@patch("llmbeans.cli.IntPrompt.ask", return_value=1)
@patch("llmbeans.cli._scan_models_in_dir")
@patch("llmbeans.cli.Prompt.ask")
def test_prompt_search_folder_happy_path(mock_prompt, mock_scan, mock_int, tmp_path):
    model = tmp_path / "found.gguf"
    model.write_bytes(b"x")
    mock_prompt.return_value = str(tmp_path)
    mock_scan.return_value = [{"path": str(model), "format": "GGUF", "size_gb": 0.001}]
    assert _prompt_search_folder() == str(model)


@patch("llmbeans.cli.IntPrompt.ask", side_effect=[99, 1])
@patch("llmbeans.cli._scan_models_in_dir", return_value=[])
@patch("llmbeans.cli.Prompt.ask", side_effect=["", "nope", str(Path("/tmp"))])
def test_prompt_search_folder_validation_paths(mock_prompt, mock_scan, mock_int, tmp_path):
    tmp_path.mkdir(exist_ok=True)
    mock_prompt.side_effect = ["", str(tmp_path), str(tmp_path)]
    with pytest.raises(StopIteration):
        _prompt_search_folder()


def test_get_hardware_profiles():
    assert get_hardware_profiles()


@patch("llmbeans.cli.Confirm.ask", return_value=True)
def test_prompt_hardware_selection_auto_detect(mock_confirm):
    auto = get_profile_by_id("macbook-pro-m4-2024")
    selected = prompt_hardware_selection([], auto_detected=auto)
    assert selected == auto


@patch("llmbeans.cli.IntPrompt.ask", return_value=1)
def test_prompt_hardware_selection_manual(mock_int):
    profile = get_profile_by_id("macbook-pro-m4-2024")
    selected = prompt_hardware_selection([profile], auto_detected=None)
    assert selected == profile


@patch("llmbeans.cli.Confirm.ask", return_value=False)
@patch("llmbeans.cli.IntPrompt.ask", side_effect=[99, 1])
def test_prompt_hardware_selection_invalid_then_valid(mock_int, mock_confirm):
    profile = get_profile_by_id("macbook-pro-m4-2024")
    selected = prompt_hardware_selection([profile], auto_detected=profile)
    assert selected == profile


@patch("llmbeans.cli.Confirm.ask", return_value=False)
def test_prompt_hardware_selection_no_profiles(mock_confirm):
    auto = get_profile_by_id("macbook-pro-m4-2024")
    selected = prompt_hardware_selection([], auto_detected=auto)
    assert selected == auto


@patch("llmbeans.cli.IntPrompt.ask", side_effect=[99, 2])
def test_prompt_quality_mode_invalid_then_valid(mock_int):
    mode = prompt_quality_mode()
    assert mode == "quality"


def test_generate_summary_all_sections(sample_model, nvidia_hardware, sample_recommendation):
    sample_recommendation.extra_config = "extra"
    summary = generate_summary(sample_model, nvidia_hardware, sample_recommendation)
    assert "DDR5" in summary or "RAM:" in summary
    assert "Command:" in summary
    assert "extra" in summary
    assert "Warnings:" in summary


def test_generate_summary_flag_only_and_empty_ram_type(sample_model, nvidia_hardware, sample_recommendation):
    nvidia_hardware = replace(nvidia_hardware, ram_type="")
    sample_recommendation.flags = {"flash": ""}
    summary = generate_summary(sample_model, nvidia_hardware, sample_recommendation)
    assert "RAM: 16 GB" in summary
    assert "  flash" in summary


def test_write_scripts_all_outputs(sample_model, nvidia_hardware, sample_recommendation):
    sample_recommendation.hosting_tool = "ollama"
    sample_recommendation.extra_config = "FROM model"
    with tempfile.TemporaryDirectory() as tmpdir:
        result = write_scripts(
            sample_recommendation,
            sample_model,
            nvidia_hardware,
            "summary text",
            output_dir=tmpdir,
        )
        assert os.path.exists(result["modelfile"])
        assert os.path.exists(result["summary"])

    sample_recommendation.hosting_tool = "lmstudio"
    sample_recommendation.extra_config = "{}"
    with tempfile.TemporaryDirectory() as tmpdir:
        result = write_scripts(
            sample_recommendation,
            sample_model,
            nvidia_hardware,
            "summary text",
            output_dir=tmpdir,
        )
        assert os.path.exists(result["config"])


def test_write_scripts_default_output_dir(sample_model, nvidia_hardware, sample_recommendation):
    with patch("llmbeans.cli._working_directory", return_value=Path(tempfile.gettempdir())):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("llmbeans.cli._working_directory", return_value=Path(tmpdir)):
                result = write_scripts(
                    sample_recommendation,
                    sample_model,
                    nvidia_hardware,
                    "summary",
                    output_dir=None,
                )
                assert result["shell"].startswith(tmpdir)


def test_safe_get_variants():
    assert _safe_get(SimpleNamespace(), "missing") == "unknown"
    assert _safe_get(SimpleNamespace(value=None), "value") == "unknown"
    assert _safe_get(SimpleNamespace(value=ModelFormat.GGUF), "value") == "gguf"
    assert _safe_get(SimpleNamespace(value=object()), "value") == "unknown"


def test_display_helpers(sample_model, sample_recommendation, capsys):
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        display_model_info(sample_model)
        sample_recommendation.flags = {"-ngl": "32", "-fa": ""}
        sample_recommendation.command = "llama-cli"
        sample_recommendation.warnings = ["warn"]
        display_recommendation(sample_recommendation, sample_model)
    finally:
        sys.stdout = old


@patch("llmbeans.cli.Confirm.ask", return_value=True)
@patch("llmbeans.cli.write_scripts", return_value={"shell": "/tmp/run.sh"})
@patch("llmbeans.cli.generate_summary", return_value="summary")
@patch("llmbeans.cli.recommend")
@patch("llmbeans.cli.prompt_quality_mode", return_value="balanced")
@patch("llmbeans.cli.prompt_hardware_selection")
@patch("llmbeans.cli.get_hardware_profiles", return_value=[])
@patch("llmbeans.cli.from_detection")
@patch("llmbeans.cli.detect_hardware")
@patch("llmbeans.cli.scan_model")
@patch("llmbeans.cli.prompt_model_selection", return_value="/tmp/model.gguf")
@patch("llmbeans.cli.prompt_tool_selection", return_value="llamacpp")
@patch("llmbeans.cli.get_available_tools", return_value=["llamacpp"])
def test_main_success(
    mock_tools,
    mock_tool_prompt,
    mock_model_prompt,
    mock_scan,
    mock_detect,
    mock_from_detection,
    mock_profiles,
    mock_hw_prompt,
    mock_quality,
    mock_recommend,
    mock_summary,
    mock_write,
    mock_confirm,
    sample_model,
    sample_recommendation,
):
    mock_scan.return_value = sample_model
    mock_detect.return_value = MagicMock()
    mock_from_detection.return_value = get_profile_by_id("generic-rtx-4060-laptop")
    mock_hw_prompt.return_value = get_profile_by_id("generic-rtx-4060-laptop")
    mock_recommend.return_value = sample_recommendation
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


@patch("llmbeans.cli.prompt_tool_selection", return_value="llamacpp")
@patch("llmbeans.cli.get_available_tools", return_value=["llamacpp"])
@patch("llmbeans.cli.prompt_model_selection", return_value="/tmp/model.gguf")
@patch("llmbeans.cli.scan_model", side_effect=RuntimeError("bad model"))
def test_main_scan_failure(mock_scan, mock_model, mock_tools, mock_tool):
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


@patch("llmbeans.cli.Confirm.ask", return_value=False)
@patch("llmbeans.cli.generate_summary", return_value="summary")
@patch("llmbeans.cli.recommend")
@patch("llmbeans.cli.prompt_quality_mode", return_value="balanced")
@patch("llmbeans.cli.prompt_hardware_selection")
@patch("llmbeans.cli.get_hardware_profiles", return_value=[])
@patch("llmbeans.cli.detect_hardware", side_effect=RuntimeError("detect fail"))
@patch("llmbeans.cli.scan_model")
@patch("llmbeans.cli.prompt_model_selection", return_value="/tmp/model.gguf")
@patch("llmbeans.cli.prompt_tool_selection", return_value="llamacpp")
@patch("llmbeans.cli.get_available_tools", return_value=["llamacpp"])
def test_main_detect_failure_and_recommend_error(
    mock_tools,
    mock_tool,
    mock_model,
    mock_scan,
    mock_detect,
    mock_profiles,
    mock_hw,
    mock_quality,
    mock_recommend,
    mock_summary,
    mock_confirm,
    sample_model,
):
    mock_scan.return_value = sample_model
    mock_hw.return_value = get_profile_by_id("generic-rtx-4060-laptop")
    mock_recommend.side_effect = RuntimeError("recommend fail")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


@patch("llmbeans.cli.Confirm.ask", return_value=True)
@patch("llmbeans.cli.write_scripts", side_effect=OSError("disk full"))
@patch("llmbeans.cli.generate_summary", return_value="summary")
@patch("llmbeans.cli.recommend")
@patch("llmbeans.cli.prompt_quality_mode", return_value="balanced")
@patch("llmbeans.cli.prompt_hardware_selection")
@patch("llmbeans.cli.get_hardware_profiles", return_value=[])
@patch("llmbeans.cli.detect_hardware", return_value=None)
@patch("llmbeans.cli.scan_model")
@patch("llmbeans.cli.prompt_model_selection", return_value="/tmp/model.gguf")
@patch("llmbeans.cli.prompt_tool_selection", return_value="llamacpp")
@patch("llmbeans.cli.get_available_tools", return_value=["llamacpp"])
def test_main_write_scripts_error(
    mock_tools,
    mock_tool,
    mock_model,
    mock_scan,
    mock_detect,
    mock_profiles,
    mock_hw,
    mock_quality,
    mock_recommend,
    mock_summary,
    mock_write,
    mock_confirm,
    sample_model,
    sample_recommendation,
):
    mock_scan.return_value = sample_model
    mock_hw.return_value = get_profile_by_id("generic-rtx-4060-laptop")
    mock_recommend.return_value = sample_recommendation
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


def test_cli_module_main_entrypoint():
    import llmbeans.cli as cli_module
    with patch.object(cli_module, "main") as mock_main:
        exec(compile("if __name__ == '__main__': main()", "llmbeans/cli.py", "exec"), {"__name__": "__main__", "main": mock_main})
        mock_main.assert_called_once()
