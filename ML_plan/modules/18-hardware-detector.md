# Module 18: HardwareDetector — GPU/CPU Mode

| | |
|---|---|
| **Estimated Complexity** | S |
| **Estimated Files** | 1 |
| **Key Risks** | False GPU detection; wrong mode selection |

## Requirements
- Detect GPU via nvidia-smi
- Set mode: GPU (full pipeline) or CPU (limited)
- Log detected hardware info

## Data & API
```python
def detect_hardware() -> dict:
    return {
        "mode": "GPU" | "CPU",
        "gpu_name": "Tesla T4" | None,
        "vram_gb": 16.0 | 0,
        "cuda_available": True | False
    }
```

## Technical Implementation
- nvidia-smi subprocess call with 5s timeout
- torch.cuda.is_available() as secondary check
- GPU name from torch.cuda.get_device_name()
- VRAM from torch.cuda.get_device_properties().total_memory

## Testing
- [ ] GPU runtime: mode=GPU, vram > 0
- [ ] CPU runtime: mode=CPU, vram = 0
- [ ] Detection completes in < 2s
- [ ] Logged correctly
