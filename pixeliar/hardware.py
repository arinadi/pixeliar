"""
pixeliar — HardwareDetector (Module 18)
GPU/CPU detection and mode switching.
"""

import subprocess


def detect_hardware():
    """Detect GPU and return hardware info dict."""
    info = {"mode": "CPU", "name": None, "vram_gb": 0, "cuda_available": False}

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            name, vram_mb = lines[0].split(", ")
            info["mode"] = "GPU"
            info["name"] = name.strip()
            info["vram_gb"] = round(int(vram_mb.strip()) / 1024, 1)
            info["cuda_available"] = True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    if info["mode"] == "CPU":
        try:
            import torch
            if torch.cuda.is_available():
                info["mode"] = "GPU"
                info["name"] = torch.cuda.get_device_name(0)
                info["vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
                info["cuda_available"] = True
        except ImportError:
            pass

    return info
