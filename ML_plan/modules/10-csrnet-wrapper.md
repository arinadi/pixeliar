# Module 10: CSRNetWrapper — Professional Retouch

| | |
|---|---|
| **Estimated Complexity** | S |
| **Estimated Files** | 1 |
| **Key Risks** | Over-retouch on portraits; runs before exposure fix |

## Requirements
- Global professional retouching (color harmony + tonal balance)
- Always runs after IAT/Retinex + DeepWB
- Alpha blend for strength control (default 0.85)
- Input: full resolution (1×1 conv, no resize needed)

## Data & API
```python
# Inference
with torch.no_grad():
    retouched = csrnet_model(img_tensor)

# Blend
result = alpha * retouched + (1 - alpha) * original
# alpha = config["csrnet_strength"]  # default 0.85
```

## Technical Implementation
- CSRNet from `github.com/hejingwenhejingwen/CSRNet`
- Weight: `experiments/pretrain_models/csrnet.pth`
- Architecture: BaseNet (6 conv 1×1) + ConditionNet
- 1×1 convolution = pixel-independent = full resolution OK
- Log before/after mean_lum and contrast_std

## Testing
- [ ] Retouched image has improved contrast
- [ ] Alpha=0.85 produces subtle (not aggressive) result
- [ ] Runs AFTER exposure fix (hard-coded in orchestrator)
- [ ] Inference < 20ms
