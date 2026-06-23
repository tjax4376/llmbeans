"""Tests for llmbeans.hardware.profiles."""

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from llmbeans.hardware.profiles import (
    _gpu_match_tokens,
    _score_profile_match,
    from_detection,
    get_categories,
    get_profile_by_id,
    get_profiles_by_category,
    load_profiles,
    lookup_specs_for_detection,
    match_profile_for_detection,
)


@dataclass
class FakeDetectedHardware:
    os: str = "linux"
    cpu_cores: int = 8
    ram_total_gb: float = 16
    ram_free_gb: float = 8
    gpu_vendor: str | None = "nvidia"
    gpu_vram_gb: float | None = 8.0
    gpu_name: str | None = "NVIDIA GeForce RTX 4060 Laptop GPU"
    is_apple_silicon: bool = False
    unified_memory: bool = False
    metal_supported: bool = False
    disk_is_ssd: bool = True
    laptop_model: str | None = None
    memory_bandwidth_gbps: float | None = None


def test_load_profiles_and_lookup_helpers():
    profiles = load_profiles()
    assert profiles
    assert get_profile_by_id("macbook-pro-m4-2024") is not None
    assert get_profile_by_id("missing-id") is None
    assert get_profiles_by_category("apple")
    categories = get_categories()
    assert "apple" in categories
    assert "meta" not in categories


def test_gpu_match_tokens():
    assert "rtx 4060" in _gpu_match_tokens("NVIDIA GeForce RTX 4060 Laptop GPU")
    assert "m4 pro" in _gpu_match_tokens("Apple M4 Pro")
    assert _gpu_match_tokens("") == set()


def test_score_profile_match_amd():
    hw = FakeDetectedHardware(gpu_vendor="amd", gpu_name="AMD Radeon")
    profile = get_profile_by_id("generic-rtx-4060-laptop")
    score = _score_profile_match(hw, profile)
    assert score >= 0


def test_match_profile_for_detection_no_gpu_match():
    hw = FakeDetectedHardware(
        gpu_vendor=None,
        gpu_name=None,
        gpu_vram_gb=None,
        ram_total_gb=1,
    )
    assert match_profile_for_detection(hw) is None


def test_lookup_specs_for_detection_nvidia():
    specs = lookup_specs_for_detection(FakeDetectedHardware())
    assert specs["cuda"] is True
    assert specs["memory_bandwidth_gbps"] > 0
    assert specs["ram_type"] == "DDR5"


def test_lookup_specs_for_detection_apple():
    hw = FakeDetectedHardware(
        gpu_vendor="apple",
        gpu_name="Apple M4",
        gpu_vram_gb=None,
        is_apple_silicon=True,
        unified_memory=True,
        metal_supported=True,
        ram_total_gb=16,
    )
    specs = lookup_specs_for_detection(hw)
    assert specs["cuda"] is False
    assert specs["ram_type"] == "LPDDR5X"


def test_lookup_specs_apple_ram_type_fallbacks():
    cases = [
        ("Apple M3", "LPDDR5X"),
        ("Apple M2", "LPDDR5"),
        ("Apple M1", "LPDDR4X"),
        ("Apple A18", "LPDDR5"),
    ]
    for gpu_name, expected in cases:
        hw = FakeDetectedHardware(
            gpu_vendor="apple",
            gpu_name=gpu_name,
            gpu_vram_gb=None,
            is_apple_silicon=True,
            unified_memory=True,
            metal_supported=True,
            ram_total_gb=16,
        )
        with patch("llmbeans.hardware.profiles.match_profile_for_detection", return_value=None):
            assert lookup_specs_for_detection(hw)["ram_type"] == expected


def test_from_detection_nvidia():
    profile = from_detection(FakeDetectedHardware())
    assert profile.cuda is True
    assert profile.memory_bandwidth_gbps > 0


def test_from_detection_apple_cpu_threads():
    hw = FakeDetectedHardware(
        gpu_vendor="apple",
        gpu_name="Apple M4",
        gpu_vram_gb=None,
        is_apple_silicon=True,
        unified_memory=True,
        metal_supported=True,
        cpu_cores=10,
        ram_total_gb=16,
    )
    profile = from_detection(hw)
    assert profile.cpu_threads >= 10


def test_from_detection_uses_matched_vram():
    hw = FakeDetectedHardware(gpu_vram_gb=None, gpu_name="NVIDIA RTX 4060 Laptop GPU")
    profile = from_detection(hw)
    assert profile.gpu_vram_gb is not None


def test_from_detection_mac_name_from_system_profiler():
    hw = FakeDetectedHardware(
        os="darwin",
        gpu_vendor="apple",
        gpu_name="Apple M4",
        gpu_vram_gb=None,
        is_apple_silicon=True,
        unified_memory=True,
        metal_supported=True,
        laptop_model="Mac16,1",
        ram_total_gb=16,
    )
    profiler_json = '{"SPHardwareDataType":[{"machine_name":"MacBook Pro","chip_type":"Apple M4"}]}'
    with patch("llmbeans.hardware.profiles.platform.system", return_value="Darwin"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = profiler_json
        mock_run.return_value.returncode = 0
        profile = from_detection(hw)
    assert profile.name == "MacBook Pro (Apple M4)"


def test_from_detection_mac_name_chip_only():
    hw = FakeDetectedHardware(
        os="darwin",
        gpu_vendor="apple",
        gpu_name="Apple M4",
        is_apple_silicon=True,
        unified_memory=True,
        metal_supported=True,
        laptop_model="Mac16,1",
    )
    profiler_json = '{"SPHardwareDataType":[{"chip_type":"Apple M4"}]}'
    with patch("llmbeans.hardware.profiles.platform.system", return_value="Darwin"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = profiler_json
        profile = from_detection(hw)
    assert profile.name == "Apple M4"


def test_from_detection_mac_name_machine_only():
    hw = FakeDetectedHardware(
        os="darwin",
        gpu_vendor="apple",
        gpu_name="Apple M4",
        is_apple_silicon=True,
        unified_memory=True,
        metal_supported=True,
        laptop_model="Mac16,1",
    )
    profiler_json = '{"SPHardwareDataType":[{"machine_name":"MacBook Pro"}]}'
    with patch("llmbeans.hardware.profiles.platform.system", return_value="Darwin"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = profiler_json
        profile = from_detection(hw)
    assert profile.name == "MacBook Pro"


def test_from_detection_system_profiler_failure():
    hw = FakeDetectedHardware(laptop_model="Mac16,1")
    with patch("llmbeans.hardware.profiles.platform.system", return_value="Darwin"), \
         patch("subprocess.run", side_effect=OSError("fail")):
        profile = from_detection(hw)
    assert profile.name == "Mac16,1"
