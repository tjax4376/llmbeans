# llmbeans/hardware/detector.py
"""System hardware detection — OS, RAM, GPU, disk."""

import os
import platform
import subprocess
from dataclasses import dataclass

from llmbeans.hardware.profiles import lookup_specs_for_detection


@dataclass
class HardwareProfile:
    os: str
    cpu_cores: int
    ram_total_gb: float
    ram_free_gb: float
    gpu_vendor: str | None
    gpu_vram_gb: float | None
    gpu_name: str | None
    is_apple_silicon: bool
    unified_memory: bool
    metal_supported: bool
    disk_is_ssd: bool
    laptop_model: str | None
    memory_bandwidth_gbps: float | None


def detect_hardware() -> HardwareProfile | None:
    """Attempt to auto-detect system hardware. Returns None if detection fails."""
    try:
        import psutil
        ram = psutil.virtual_memory()
        ram_total_gb = round(ram.total / (1024**3), 1)
        ram_free_gb = round(ram.available / (1024**3), 1)
    except ImportError:
        ram_total_gb = 0.0
        ram_free_gb = 0.0

    os_type = platform.system().lower()
    cpu_cores = os.cpu_count() or 0
    is_apple = False
    metal = False
    unified = False
    gpu_vendor = None
    gpu_vram = None
    gpu_name = None
    laptop_model = None
    bandwidth = None
    disk_is_ssd = _detect_disk_is_ssd()  # Actually detect if disk is SSD

    if os_type == "darwin":
        unified = True
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            chip = result.stdout.strip()
            if "Apple" in chip:
                is_apple = True
                gpu_name = chip
                gpu_vendor = "apple"
                metal = True
            laptop_model = _detect_mac_model()
        except Exception:
            pass

    elif os_type == "linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        ram_total_gb = round(int(line.split()[1]) / (1024**2), 1)
                        break
        except Exception:
            pass
        gpu_vendor, gpu_vram, gpu_name = _detect_linux_gpu()

    elif os_type == "windows":
        try:
            import psutil
            gpu_info = _detect_windows_gpu()
            if gpu_info:
                gpu_vendor, gpu_vram, gpu_name = gpu_info
        except Exception:
            pass

    profile = HardwareProfile(
        os=os_type,
        cpu_cores=cpu_cores,
        ram_total_gb=ram_total_gb,
        ram_free_gb=ram_free_gb,
        gpu_vendor=gpu_vendor,
        gpu_vram_gb=gpu_vram,
        gpu_name=gpu_name,
        is_apple_silicon=is_apple,
        unified_memory=unified,
        metal_supported=metal,
        disk_is_ssd=disk_is_ssd,
        laptop_model=laptop_model,
        memory_bandwidth_gbps=bandwidth,
    )

    specs = lookup_specs_for_detection(profile)
    if not profile.memory_bandwidth_gbps:
        profile.memory_bandwidth_gbps = specs["memory_bandwidth_gbps"] or None

    return profile


def _detect_disk_is_ssd() -> bool:
    """Detect if the primary disk is an SSD.
    
    Returns:
        bool: True if SSD is detected, False otherwise (defaults to True for safety)
    """
    try:
        if platform.system() == "Darwin":  # macOS
            # Check if root partition is on SSD using diskutil
            result = subprocess.run(
                ["diskutil", "info", "/"], 
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if "Solid State" in line and "Yes" in line:
                        return True
                    if "Solid State" in line and "No" in line:
                        return False
        elif platform.system() == "Linux":  # pragma: no cover
            # Check if root partition is on SSD using rotational flag
            try:
                with open("/sys/block/$(mountpoint -d / | cut -d'/' -f3)/queue/rotational", "r") as f:
                    return f.read().strip() == "0"  # 0 means SSD (non-rotational)
            except Exception:
                # Fallback: check if any block device is non-rotational
                for device in os.listdir("/sys/block/"):
                    if device.startswith(("sd", "nvme", "mmcblk")):
                        try:
                            with open(f"/sys/block/{device}/queue/rotational", "r") as f:
                                if f.read().strip() == "0":
                                    return True
                        except Exception:
                            continue
        elif platform.system() == "Windows":  # pragma: no cover
            # Use PowerShell to check if disk is SSD
            try:
                result = subprocess.run([
                    "powershell", "-command", 
                    "Get-PhysicalMedia | Where-Object {$_.MediaType -eq 'SSD'} | Select-Object -First 1"
                ], capture_output=True, text=True, timeout=10)
                return result.returncode == 0 and result.stdout.strip() != ""
            except Exception:
                pass
    except Exception:
        pass
    
    # Default to True (assume SSD) for safety in recommendations
    # This maintains existing behavior while improving accuracy when possible
    return True


def _detect_mac_model() -> str | None:
    """Detect Mac model identifier."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _detect_linux_gpu() -> tuple:
    """Detect GPU on Linux."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            name = parts[0].strip()
            vram = float(parts[1].strip().replace("MiB", "").replace("MB", "")) / 1024
            return ("nvidia", round(vram, 1), name)
    except Exception:
        pass
    return (None, None, None)


def _detect_windows_gpu() -> tuple | None:
    """Detect GPU on Windows via wmic."""
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_videocontroller", "get", "name,adapterram"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].strip().split()
                if parts:
                    name = " ".join(parts[:-1]) if len(parts) > 1 else parts[0]
                    return ("nvidia" if "nvidia" in name.lower() else "amd", None, name)
    except Exception:
        pass
    return None
