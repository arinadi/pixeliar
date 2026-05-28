# Module 6: ModelLoader — VRAM Management

| | |
|---|---|
| **Estimated Complexity** | M |
| **Estimated Files** | 1 |
| **Key Risks** | OOM during load; missing weights; wrong architecture |

## Requirements
- Load all 6 ML models to GPU VRAM
- Log param count and VRAM per model
- Total VRAM usage summary
- Warm-up dummy forward pass per model
- Optional load for Restormer (opt-in)
- Handle missing weights gracefully

## Data & API
```python
models = load_all_models(config)
# Returns dict:
{
    "iat": model_90K_params,
    "retinex": model_1.6M_params,
    "deepwb": model_14M_params,
    "csrnet": model_37K_params,
    "naf_denoise": processor_sidd,
    "naf_deblur": processor_reds,
    "restormer": model_26M_params  # if load_restormer=True
}

# Log output:
# ✅ IAT           loaded   (90K params,   0.4 MB)
# ✅ Retinexformer loaded   (1.6M params,  6.4 MB)
# 📊 Total VRAM used: 191 MB / 16384 MB (1.2%)
```

## Technical Implementation
- Sequential load: IAT → Retinex → DeepWB → CSRNet → NAFNet×2 → Restormer(opt)
- Each model: instantiate → load_state_dict → .cuda().eval() → warm-up pass
- VRAM tracking via `torch.cuda.memory_allocated()`
- NAFNet uses nafnetlib (auto-download weights)
- Error per model: log warning, skip, continue (don't crash)

## Testing
- [ ] All models load without OOM
- [ ] VRAM usage logged correctly
- [ ] Warm-up passes complete
- [ ] Restormer skipped when load_restormer=False
- [ ] Missing weights → graceful skip with warning
