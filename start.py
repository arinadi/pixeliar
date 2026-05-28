#!/usr/bin/env python3
"""
pixeliar — Smart Runner (Module 1)
Detect GPU, set PHOTON_MODE, launch main.py.
Adapted from TTB/start.py pattern.
"""

import subprocess
import sys
import os
import signal
import time


def detect_gpu():
    """Detect GPU availability and return info dict."""
    info = {"mode": "CPU", "name": None, "vram_gb": 0, "cuda_available": False}

    # Method 1: nvidia-smi (works even if torch not loaded yet)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
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

    # Method 2: torch.cuda as fallback
    if info["mode"] == "CPU":
        try:
            import torch
            if torch.cuda.is_available():
                info["mode"] = "GPU"
                info["name"] = torch.cuda.get_device_name(0)
                info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
                info["cuda_available"] = True
        except ImportError:
            pass

    return info


def setup_cleanup():
    """Register signal handler for graceful VRAM cleanup."""
    def cleanup(signum, frame):
        print("\n🧹 Cleaning up VRAM...", flush=True)
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)


def main():
    t0 = time.time()

    print("=" * 50)
    print("  pixeliar — Smart Runner")
    print("=" * 50)

    # Detect hardware
    hw = detect_gpu()
    os.environ["PHOTON_MODE"] = hw["mode"]

    print(f"\n  🔍 Mode     : {hw['mode']}", flush=True)
    if hw["name"]:
        print(f"  🎮 GPU      : {hw['name']}", flush=True)
        print(f"  💾 VRAM     : {hw['vram_gb']} GB", flush=True)
    else:
        print("  ⚠️  No GPU detected", flush=True)
        print("     → Limited pipeline (DeepWB only)", flush=True)
        print("     → For full pipeline, use Colab with T4/GPU runtime", flush=True)

    # Setup cleanup
    setup_cleanup()

    # Launch main.py
    elapsed = time.time() - t0
    print(f"\n  🚀 Ready ({elapsed:.1f}s)")
    print("=" * 50)
    print(flush=True)

    main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
    if not os.path.exists(main_path):
        print(f"  ❌ {main_path} not found!", flush=True)
        sys.exit(1)

    sys.exit(subprocess.call([sys.executable, main_path]))


if __name__ == "__main__":
    main()
