# Module 1: start.py — Smart Runner

| | |
|---|---|
| **Estimated Complexity** | S |
| **Estimated Files** | 1 |
| **Key Risks** | nvidia-smi not found on CPU-only runtime |

## Requirements
- Detect GPU via nvidia-smi
- Set PHOTON_MODE environment variable (GPU / CPU)
- Launch main.py as subprocess
- Handle graceful shutdown (SIGTERM → cleanup VRAM)

## Data & API
```python
# Input: env vars from runner.py
# Output: PHOTON_MODE = "GPU" | "CPU"

# Mode effects:
# GPU  → full pipeline (IAT + Retinex + DeepWB + CSRNet + NAFNet)
# CPU  → limited (DeepWB only, skip ML-heavy models)
```

## Technical Implementation
```python
import subprocess, sys, os

def detect_gpu() -> str:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "NVIDIA" in result.stdout:
            return "GPU"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "CPU"

if __name__ == "__main__":
    mode = detect_gpu()
    os.environ["PHOTON_MODE"] = mode
    print(f"🔧 PHOTON_MODE = {mode}")

    if mode == "CPU":
        print("⚠️  No GPU — limited pipeline (DeepWB only)")

    main_path = os.path.join(os.path.dirname(__file__), "main.py")
    sys.exit(subprocess.call([sys.executable, main_path]))
```

## Testing
- [ ] On GPU runtime: `PHOTON_MODE=GPU`
- [ ] On CPU runtime: `PHOTON_MODE=CPU`
- [ ] main.py launched as subprocess with correct env
- [ ] Exit code propagated from main.py
