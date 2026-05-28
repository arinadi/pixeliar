# Module 8: RetinexWrapper — Severe Low-Light

| | |
|---|---|
| **Estimated Complexity** | M |
| **Estimated Files** | 1 |
| **Key Risks** | OOM on 4K images; padding misalignment |

## Requirements
- Severe low-light enhancement (mean_lum < 40)
- Replaces IAT entirely (not additive)
- Padding to multiple of 4
- Tiling for images > 1500px
- Gaussian feathering for tile blending

## Data & API
```python
# Padding
h, w = img_tensor.shape[2], img_tensor.shape[3]
pad_h, pad_w = (4 - h % 4) % 4, (4 - w % 4) % 4
padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")

# Inference
with torch.no_grad():
    out = retinex_model(padded)
out = out[:, :, :h, :w]

# Tiling (for large images)
# tiles: 512x512 with 64px overlap
# blend: gaussian feathering at seams
```

## Technical Implementation
- Retinexformer from `github.com/caiyuanhao1998/Retinexformer`
- Weight: LOL_v1.pth (HuggingFace or Google Drive)
- Tiling: split into 512×512 tiles with 64px overlap
- Blend: gaussian feathering in overlap region
- Peak VRAM: ~2GB for 4K image (within T4 budget)

## Testing
- [ ] Image with lum=30: output lum > 70
- [ ] Padding correctly applied (divisible by 4)
- [ ] Tiling works for 4032×3024 images
- [ ] No visible seam artifacts in tiled output
- [ ] VRAM stays < 3GB
