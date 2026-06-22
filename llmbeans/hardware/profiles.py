# llmbeans/hardware/profiles.py
"""Laptop and desktop hardware profile database.

Loads profiles from profiles.json and provides lookup helpers.
"""

import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROFILES_PATH = Path(__file__).parent / "profiles.json"


@dataclass
class HardwareProfileEntry:
    id: str
    name: str
    year: int
    ram_gb: int
    ram_type: str
    cpu_cores: int
    cpu_threads: int
    gpu_type: str          # "integrated" | "discrete" | "apple_silicon"
    gpu_name: str
    gpu_vram_gb: Optional[float]
    gpu_cores: int
    unified_memory: bool
    memory_bandwidth_gbps: float
    metal: bool
    cuda: bool
    cuda_compute: Optional[str]
    vram_bandwidth_gbps: Optional[float]
    ssd_recommended: bool
    category: str


def load_profiles() -> list[HardwareProfileEntry]:
    """Load all hardware profiles from profiles.json."""
    with open(PROFILES_PATH) as f:
        data = json.load(f)

    profiles = []
    for category, entries in data.items():
        if category == "meta":
            continue
        for entry in entries:
            profiles.append(HardwareProfileEntry(
                id=entry["id"],
                name=entry["name"],
                year=entry["year"],
                ram_gb=entry["ram_gb"],
                ram_type=entry["ram_type"],
                cpu_cores=entry["cpu_cores"],
                cpu_threads=entry["cpu_threads"],
                gpu_type=entry["gpu_type"],
                gpu_name=entry["gpu_name"],
                gpu_vram_gb=entry.get("gpu_vram_gb"),
                gpu_cores=entry["gpu_cores"],
                unified_memory=entry["unified_memory"],
                memory_bandwidth_gbps=entry["memory_bandwidth_gbps"],
                metal=entry.get("metal", False),
                cuda=entry.get("cuda", False),
                cuda_compute=entry.get("cuda_compute"),
                vram_bandwidth_gbps=entry.get("vram_bandwidth_gbps"),
                ssd_recommended=entry.get("ssd_recommended", True),
                category=category,
            ))
    return profiles


def get_profiles_by_category(category: str) -> list[HardwareProfileEntry]:
    """Get all profiles in a category (e.g. 'apple', 'windows_nvidia')."""
    return [p for p in load_profiles() if p.category == category]


def get_profile_by_id(profile_id: str) -> Optional[HardwareProfileEntry]:
    """Look up a single profile by its ID."""
    for p in load_profiles():
        if p.id == profile_id:
            return p
    return None


def get_categories() -> list[str]:
    """Return available profile categories."""
    with open(PROFILES_PATH) as f:
        data = json.load(f)
    return [k for k in data if k != "meta"]


def _gpu_match_tokens(name: str) -> set[str]:
    """Extract comparable GPU tokens from a human-readable GPU name."""
    if not name:
        return set()
    normalized = name.lower()
    tokens: set[str] = set()
    for match in re.finditer(
        r"\b(m\d+(?:\s+(?:pro|max|ultra))?|rtx\s*\d{4}|gtx\s*\d{4})\b",
        normalized,
    ):
        tokens.add(re.sub(r"\s+", " ", match.group(1).strip()))
    for match in re.finditer(r"\b(\d{4})\b", normalized):
        token = match.group(1)
        if token.startswith(("30", "40", "50")):
            tokens.add(f"rtx {token}")
    return tokens


def _score_profile_match(hw, profile: HardwareProfileEntry) -> int:
    """Score how well a profile matches auto-detected hardware (higher = better)."""
    score = 0
    ram_gb = int(getattr(hw, "ram_total_gb", getattr(hw, "ram_gb", 0)) or 0)
    gpu_name = (getattr(hw, "gpu_name", None) or "").lower()
    gpu_vendor = (getattr(hw, "gpu_vendor", None) or "").lower()
    is_apple = getattr(hw, "is_apple_silicon", False)

    if is_apple and profile.category == "apple":
        score += 3
    elif gpu_vendor == "nvidia" and profile.cuda:
        score += 3
    elif gpu_vendor == "amd" and profile.category.startswith("windows"):
        score += 2

    if gpu_name and profile.gpu_name.lower() in gpu_name:
        score += 12
    elif gpu_name and gpu_name in profile.gpu_name.lower():
        score += 10

    detected_tokens = _gpu_match_tokens(gpu_name)
    profile_tokens = _gpu_match_tokens(profile.gpu_name)
    shared_tokens = detected_tokens & profile_tokens
    if shared_tokens:
        score += 10 + len(shared_tokens)

    if ram_gb and abs(profile.ram_gb - ram_gb) <= max(2, int(ram_gb * 0.15)):
        score += 6
    elif ram_gb and abs(profile.ram_gb - ram_gb) <= 4:
        score += 2

    if getattr(hw, "gpu_vram_gb", None) and profile.gpu_vram_gb:
        detected_vram = round(float(hw.gpu_vram_gb))
        profile_vram = round(float(profile.gpu_vram_gb))
        if detected_vram == profile_vram:
            score += 4

    return score


def match_profile_for_detection(hw) -> Optional[HardwareProfileEntry]:
    """Find the best matching profile entry for auto-detected hardware."""
    candidates: list[tuple[int, HardwareProfileEntry]] = []
    for profile in load_profiles():
        score = _score_profile_match(hw, profile)
        if score > 0:
            candidates.append((score, profile))

    if not candidates:
        return None

    ram_gb = int(getattr(hw, "ram_total_gb", getattr(hw, "ram_gb", 0)) or 0)
    candidates.sort(key=lambda item: (-item[0], abs(item[1].ram_gb - ram_gb)))
    return candidates[0][1]


def lookup_specs_for_detection(hw) -> dict:
    """Resolve bandwidth, RAM type, and CUDA flags for auto-detected hardware."""
    matched = match_profile_for_detection(hw)
    gpu_vendor = (getattr(hw, "gpu_vendor", None) or "").lower()
    gpu_name = (getattr(hw, "gpu_name", None) or "").lower()
    is_apple = getattr(hw, "is_apple_silicon", False)

    cuda = bool(
        matched and matched.cuda
    ) or gpu_vendor == "nvidia" or "nvidia" in gpu_name

    ram_type = matched.ram_type if matched else ""
    memory_bandwidth = (
        getattr(hw, "memory_bandwidth_gbps", None)
        or (matched.memory_bandwidth_gbps if matched else None)
        or 0
    )
    vram_bandwidth = matched.vram_bandwidth_gbps if matched else None
    cuda_compute = matched.cuda_compute if matched else None

    if is_apple and not ram_type:
        ram_type = "LPDDR5X" if "m4" in gpu_name or "m3" in gpu_name else "LPDDR5"
    elif not ram_type and cuda:
        ram_type = "DDR5"

    return {
        "matched_profile": matched,
        "ram_type": ram_type,
        "memory_bandwidth_gbps": memory_bandwidth,
        "cuda": cuda,
        "cuda_compute": cuda_compute,
        "vram_bandwidth_gbps": vram_bandwidth,
    }


def from_detection(hw) -> HardwareProfileEntry:
    """Convert a HardwareProfile (from detector) into a HardwareProfileEntry.

    The detector and the profile database use different dataclasses with
    overlapping but not identical fields.  This bridges the gap so the
    auto-detected hardware can be used everywhere a profile entry is
    expected.
    """
    # Derive cpu_threads: on Apple Silicon physical ≈ logical, otherwise
    # assume 2× cores when we can't detect.
    cpu_threads = getattr(hw, "cpu_threads", None)
    if cpu_threads is None:
        cpu_cores = hw.cpu_cores
        # Apple Silicon: no hyperthreading on efficiency cores, but
        # performance cores have 2 threads.  Use cores * 1.5 as estimate.
        if getattr(hw, "is_apple_silicon", False):
            cpu_threads = max(cpu_cores, int(cpu_cores * 1.5))
        else:
            cpu_threads = cpu_cores * 2

    ram_gb = getattr(hw, "ram_total_gb", getattr(hw, "ram_gb", 0))

    # Build a human-readable name
    name = hw.laptop_model or "Auto-detected System"
    try:
        # Only run system_profiler on macOS
        if platform.system() == "Darwin":
            import subprocess as _sp
            result = _sp.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                capture_output=True, text=True, timeout=5,
            )
            import json as _json
            data = _json.loads(result.stdout)
            hw_data = data.get("SPHardwareDataType", [{}])[0]
            machine_name = hw_data.get("machine_name", "")
            chip_type = hw_data.get("chip_type", "")
            if machine_name and chip_type:
                name = f"{machine_name} ({chip_type})"
            elif chip_type:
                name = chip_type
            elif machine_name:
                name = machine_name
    except Exception:
        pass

    specs = lookup_specs_for_detection(hw)
    matched = specs["matched_profile"]
    gpu_vram_gb = hw.gpu_vram_gb
    if gpu_vram_gb is None and matched and matched.gpu_vram_gb is not None:
        gpu_vram_gb = matched.gpu_vram_gb

    return HardwareProfileEntry(
        id="auto-detected",
        name=name,
        year=matched.year if matched else 0,
        ram_gb=int(ram_gb),
        ram_type=specs["ram_type"],
        cpu_cores=hw.cpu_cores,
        cpu_threads=cpu_threads,
        gpu_type="apple_silicon" if getattr(hw, "is_apple_silicon", False)
                else "discrete" if gpu_vram_gb else "integrated",
        gpu_name=hw.gpu_name or (matched.gpu_name if matched else "Unknown"),
        gpu_vram_gb=gpu_vram_gb,
        gpu_cores=matched.gpu_cores if matched else 0,
        unified_memory=hw.unified_memory,
        memory_bandwidth_gbps=specs["memory_bandwidth_gbps"],
        metal=getattr(hw, "metal_supported", getattr(hw, "metal", False)),
        cuda=specs["cuda"],
        cuda_compute=specs["cuda_compute"],
        vram_bandwidth_gbps=specs["vram_bandwidth_gbps"],
        ssd_recommended=getattr(hw, "disk_is_ssd", True),
        category="auto",
    )
