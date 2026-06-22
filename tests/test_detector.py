"""Tests for llmbeans.hardware.detector."""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from llmbeans.hardware.detector import (
    HardwareProfile,
    _detect_disk_is_ssd,
    _detect_linux_gpu,
    _detect_mac_model,
    _detect_windows_gpu,
    detect_hardware,
)


def test_detect_hardware_darwin_apple_silicon():
    with patch("llmbeans.hardware.detector.platform.system", return_value="Darwin"), \
         patch("llmbeans.hardware.detector.platform.system", return_value="Darwin"), \
         patch("llmbeans.hardware.detector.psutil", create=True) as mock_psutil, \
         patch("llmbeans.hardware.detector.subprocess.run") as mock_run, \
         patch("llmbeans.hardware.detector._detect_disk_is_ssd", return_value=True), \
         patch("llmbeans.hardware.detector.lookup_specs_for_detection", return_value={"memory_bandwidth_gbps": 120}):
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3)
        mock_run.side_effect = [
            MagicMock(stdout="Apple M4\n", returncode=0),
            MagicMock(stdout="Mac16,1\n", returncode=0),
        ]
        with patch("llmbeans.hardware.detector.platform.system", side_effect=lambda: "Darwin"):
            with patch("llmbeans.hardware.detector.os.cpu_count", return_value=10):
                profile = detect_hardware()

    assert profile is not None
    assert profile.is_apple_silicon is True
    assert profile.gpu_vendor == "apple"
    assert profile.memory_bandwidth_gbps == 120


def test_detect_hardware_without_psutil():
    import builtins
    real_import = builtins.__import__

    def import_without_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_psutil), \
         patch("llmbeans.hardware.detector.platform.system", return_value="Linux"), \
         patch("llmbeans.hardware.detector.os.cpu_count", return_value=4), \
         patch("llmbeans.hardware.detector._detect_disk_is_ssd", return_value=True), \
         patch("llmbeans.hardware.detector._detect_linux_gpu", return_value=(None, None, None)), \
         patch("llmbeans.hardware.detector.lookup_specs_for_detection", return_value={"memory_bandwidth_gbps": 0}), \
         patch("builtins.open", mock_open(read_data="MemTotal:       16384000 kB\n")):
        profile = detect_hardware()

    assert profile.ram_total_gb == pytest.approx(15.6, abs=0.2)


def test_detect_hardware_linux_nvidia():
    with patch("llmbeans.hardware.detector.platform.system", return_value="Linux"), \
         patch("llmbeans.hardware.detector.psutil", create=True) as mock_psutil, \
         patch("llmbeans.hardware.detector.os.cpu_count", return_value=8), \
         patch("llmbeans.hardware.detector._detect_disk_is_ssd", return_value=True), \
         patch("llmbeans.hardware.detector._detect_linux_gpu", return_value=("nvidia", 8.0, "RTX 4060")), \
         patch("llmbeans.hardware.detector.lookup_specs_for_detection", return_value={"memory_bandwidth_gbps": 51.2}), \
         patch("builtins.open", mock_open(read_data="MemTotal:       16777216 kB\n")):
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3)
        profile = detect_hardware()

    assert profile.gpu_vendor == "nvidia"
    assert profile.gpu_vram_gb == 8.0


def test_detect_hardware_windows():
    with patch("llmbeans.hardware.detector.platform.system", return_value="Windows"), \
         patch("llmbeans.hardware.detector.psutil", create=True) as mock_psutil, \
         patch("llmbeans.hardware.detector.os.cpu_count", return_value=8), \
         patch("llmbeans.hardware.detector._detect_disk_is_ssd", return_value=True), \
         patch("llmbeans.hardware.detector._detect_windows_gpu", return_value=("nvidia", None, "NVIDIA RTX 4060")), \
         patch("llmbeans.hardware.detector.lookup_specs_for_detection", return_value={"memory_bandwidth_gbps": 51.2}):
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3)
        profile = detect_hardware()

    assert profile.gpu_name == "NVIDIA RTX 4060"


def test_detect_hardware_preserves_existing_bandwidth():
    with patch("llmbeans.hardware.detector.platform.system", return_value="Linux"), \
         patch("llmbeans.hardware.detector.psutil", create=True) as mock_psutil, \
         patch("llmbeans.hardware.detector.os.cpu_count", return_value=8), \
         patch("llmbeans.hardware.detector._detect_disk_is_ssd", return_value=True), \
         patch("llmbeans.hardware.detector._detect_linux_gpu", return_value=(None, None, None)), \
         patch("llmbeans.hardware.detector.lookup_specs_for_detection") as mock_lookup:
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3)
        mock_lookup.return_value = {"memory_bandwidth_gbps": 99.0}
        with patch("llmbeans.hardware.detector.HardwareProfile") as mock_profile_cls:
            mock_profile_cls.return_value = HardwareProfile(
                os="linux", cpu_cores=8, ram_total_gb=16, ram_free_gb=8,
                gpu_vendor=None, gpu_vram_gb=None, gpu_name=None,
                is_apple_silicon=False, unified_memory=False, metal_supported=False,
                disk_is_ssd=True, laptop_model=None, memory_bandwidth_gbps=42.0,
            )
            profile = detect_hardware()
        assert profile.memory_bandwidth_gbps == 42.0


def test_detect_disk_is_ssd_darwin_yes():
    with patch("llmbeans.hardware.detector.platform.system", return_value="Darwin"), \
         patch("llmbeans.hardware.detector.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Device /dev/disk3\nSolid State:               Yes\n")
        assert _detect_disk_is_ssd() is True


def test_detect_disk_is_ssd_darwin_no():
    with patch("llmbeans.hardware.detector.platform.system", return_value="Darwin"), \
         patch("llmbeans.hardware.detector.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Solid State:               No\n")
        assert _detect_disk_is_ssd() is False


def test_detect_disk_is_ssd_linux_fallback():
    with patch("llmbeans.hardware.detector.platform.system", return_value="Linux"), \
         patch("llmbeans.hardware.detector.os.listdir", return_value=["nvme0n1", "sda"]), \
         patch("builtins.open", mock_open(read_data="0\n")):
        assert _detect_disk_is_ssd() is True


def test_detect_disk_is_ssd_windows():
    with patch("llmbeans.hardware.detector.platform.system", return_value="Windows"), \
         patch("llmbeans.hardware.detector.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="SSD\n")
        assert _detect_disk_is_ssd() is True


def test_detect_disk_is_ssd_default_true():
    with patch("llmbeans.hardware.detector.platform.system", return_value="OpenBSD"), \
         patch("llmbeans.hardware.detector.subprocess.run", side_effect=OSError("nope")):
        assert _detect_disk_is_ssd() is True


def test_detect_mac_model_success():
    with patch("llmbeans.hardware.detector.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="Mac16,1\n", returncode=0)
        assert _detect_mac_model() == "Mac16,1"


def test_detect_mac_model_failure():
    with patch("llmbeans.hardware.detector.subprocess.run", side_effect=OSError("fail")):
        assert _detect_mac_model() is None


def test_detect_linux_gpu_success():
    with patch("llmbeans.hardware.detector.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA RTX 4060, 8192 MiB\n",
        )
        assert _detect_linux_gpu() == ("nvidia", 8.0, "NVIDIA RTX 4060")


def test_detect_linux_gpu_failure():
    with patch("llmbeans.hardware.detector.subprocess.run", side_effect=OSError("fail")):
        assert _detect_linux_gpu() == (None, None, None)


def test_detect_windows_gpu_nvidia():
    with patch("llmbeans.hardware.detector.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Name  AdapterRAM\nNVIDIA RTX 4060  4294967296\n",
        )
        result = _detect_windows_gpu()
        assert result[0] == "nvidia"


def test_detect_windows_gpu_failure():
    with patch("llmbeans.hardware.detector.subprocess.run", side_effect=OSError("fail")):
        assert _detect_windows_gpu() is None
