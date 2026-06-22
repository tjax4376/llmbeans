# llmbeans/tests/test_cli.py
"""
Unit tests for the llmbeans CLI.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmbeans.cli import _prompt_custom_path, _working_directory, main
from llmbeans.models.scanner import ModelInfo
from llmbeans.hardware.profiles import HardwareProfileEntry
from llmbeans.recommenders.engine import Recommendation


class TestCLI(unittest.TestCase):

    def test_working_directory_uses_process_cwd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('llmbeans.cli.Path.cwd', return_value=Path(tmpdir)):
                self.assertEqual(_working_directory(), Path(tmpdir))

    @patch('llmbeans.cli.IntPrompt.ask', return_value=1)
    def test_custom_path_menu_scans_working_directory(self, mock_int_prompt):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "demo-model.gguf"
            model_path.write_bytes(b"gguf")

            with patch('llmbeans.cli._working_directory', return_value=Path(tmpdir)):
                selected = _prompt_custom_path()

            self.assertEqual(selected, str(model_path))

    @patch('llmbeans.cli.Confirm.ask', return_value=True)
    @patch('llmbeans.cli.prompt_tool_selection')
    @patch('llmbeans.cli.get_available_tools')
    @patch('llmbeans.cli.prompt_model_selection')
    @patch('llmbeans.cli.scan_model')
    @patch('llmbeans.cli.detect_hardware')
    @patch('llmbeans.cli.get_hardware_profiles')
    @patch('llmbeans.cli.prompt_hardware_selection')
    @patch('llmbeans.cli.prompt_quality_mode')
    @patch('llmbeans.cli.recommend')
    @patch('llmbeans.cli.generate_summary')
    @patch('llmbeans.cli.write_scripts')
    @patch('builtins.input', side_effect=['y'])
    def test_main_flow(
        self,
        mock_input,
        mock_write_scripts,
        mock_generate_summary,
        mock_recommend,
        mock_prompt_quality_mode,
        mock_prompt_hardware_selection,
        mock_get_hardware_profiles,
        mock_detect_hardware,
        mock_scan_model,
        mock_prompt_model_selection,
        mock_get_available_tools,
        mock_prompt_tool_selection,
        mock_confirm,
    ):
        """Test the main flow of the CLI with mocked inputs and functions."""
        # Setup mocks
        mock_get_available_tools.return_value = ['llamacpp', 'ollama']
        mock_prompt_tool_selection.return_value = 'llamacpp'
        mock_prompt_model_selection.return_value = '/fake/path/to/model.gguf'

        # Mock model info
        mock_model_info = MagicMock(spec=ModelInfo)
        mock_model_info.name = 'test-model'
        mock_model_info.architecture = 'llama'
        mock_model_info.quant_method = 'Q4_K_M'
        mock_model_info.quant_bits = 4.5
        mock_model_info.num_layers = 32
        mock_model_info.context_length = 32768
        mock_model_info.estimated_vram_gb = 4.5
        mock_model_info.source_path = '/fake/path/to/model.gguf'
        mock_model_info.is_remote = False
        mock_scan_model.return_value = mock_model_info

        # Mock hardware info
        mock_hardware_info = MagicMock(spec=HardwareProfileEntry)
        mock_hardware_info.name = 'Test Hardware'
        mock_hardware_info.ram_gb = 16
        mock_hardware_info.ram_type = 'DDR4'
        mock_hardware_info.cpu_cores = 8
        mock_hardware_info.cpu_threads = 16
        mock_hardware_info.gpu_name = 'Test GPU'
        mock_hardware_info.gpu_vram_gb = 8.0
        mock_hardware_info.unified_memory = False
        mock_hardware_info.memory_bandwidth_gbps = 50.0
        mock_hardware_info.metal = False
        mock_hardware_info.cuda = True
        mock_detect_hardware.return_value = mock_hardware_info

        # Mock hardware profiles (empty to force auto-detect path)
        mock_get_hardware_profiles.return_value = []

        # Mock quality mode selection
        mock_prompt_quality_mode.return_value = 'balanced'

        # Mock recommendation
        mock_recommendation = MagicMock(spec=Recommendation)
        mock_recommendation.hosting_tool = 'llamacpp'
        mock_recommendation.estimated_tok_per_sec = 50.0
        mock_recommendation.estimated_vram_usage_gb = 3.5
        mock_recommendation.estimated_ram_usage_gb = 5.0
        mock_recommendation.context_length = 4096
        mock_recommendation.batch_size = 512
        mock_recommendation.thread_count = 8
        mock_recommendation.gpu_offload_layers = 20
        mock_recommendation.warnings = []
        mock_recommend.return_value = mock_recommendation

        # Mock summary generation
        mock_generate_summary.return_value = "Test Summary"

        # Mock script writing
        mock_write_scripts.return_value = {
            'shell': '/fake/path/llmbeans-output/run.sh',
            'batch': '/fake/path/llmbeans-output/run.bat',
            'summary': '/fake/path/llmbeans-output/summary.txt',
        }

        # Capture stdout
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Run the main function
            main()
        except SystemExit:
            pass  # main() calls sys.exit(0) on success
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

        # Assertions
        self.assertIn('Test Summary', output)
        self.assertIn('summary script saved to:', output)
        self.assertIn('shell script saved to:', output)
        self.assertIn('batch script saved to:', output)
        mock_write_scripts.assert_called_once()
        mock_confirm.assert_called_once()

    @patch('llmbeans.cli.Confirm.ask', return_value=False)
    @patch('llmbeans.cli.prompt_tool_selection')
    @patch('llmbeans.cli.get_available_tools')
    @patch('llmbeans.cli.prompt_model_selection')
    @patch('llmbeans.cli.scan_model')
    @patch('llmbeans.cli.detect_hardware')
    @patch('llmbeans.cli.get_hardware_profiles')
    @patch('llmbeans.cli.prompt_hardware_selection')
    @patch('llmbeans.cli.prompt_quality_mode')
    @patch('llmbeans.cli.recommend')
    @patch('llmbeans.cli.generate_summary')
    @patch('llmbeans.cli.write_scripts')
    def test_main_flow_skips_disk_write_when_declined(
        self,
        mock_write_scripts,
        mock_generate_summary,
        mock_recommend,
        mock_prompt_quality_mode,
        mock_prompt_hardware_selection,
        mock_get_hardware_profiles,
        mock_detect_hardware,
        mock_scan_model,
        mock_prompt_model_selection,
        mock_get_available_tools,
        mock_prompt_tool_selection,
        mock_confirm,
    ):
        mock_get_available_tools.return_value = ['llamacpp']
        mock_prompt_tool_selection.return_value = 'llamacpp'
        mock_prompt_model_selection.return_value = '/fake/path/to/model.gguf'

        mock_model_info = MagicMock(spec=ModelInfo)
        mock_model_info.name = 'test-model'
        mock_model_info.architecture = 'llama'
        mock_model_info.quant_method = 'Q4_K_M'
        mock_model_info.quant_bits = 4.5
        mock_model_info.num_layers = 32
        mock_model_info.context_length = 32768
        mock_model_info.estimated_vram_gb = 4.5
        mock_model_info.source_path = '/fake/path/to/model.gguf'
        mock_model_info.is_remote = False
        mock_scan_model.return_value = mock_model_info

        mock_prompt_hardware_selection.return_value = MagicMock(spec=HardwareProfileEntry)
        mock_prompt_quality_mode.return_value = 'balanced'
        mock_recommend.return_value = MagicMock(spec=Recommendation, warnings=[])
        mock_generate_summary.return_value = "Test Summary"

        try:
            main()
        except SystemExit:
            pass

        mock_write_scripts.assert_not_called()
        mock_confirm.assert_called_once()


if __name__ == '__main__':
    unittest.main()
