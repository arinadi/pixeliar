# Module 7: IATWrapper — Exposure Correction

| | |
|---|---|
| **Estimated Complexity** | S |
| **Estimated Files** | 1 |
| **Key Risks** | Wrong weight file; input normalization error |

## Requirements
- Exposure correction for under/over-exposed images
- Input: [1,3,H,W] float32 tensor, range [0,1]
- Output: same shape, range [0,1]
- Trigger: `40 <= mean_lum < 60` or `mean_lum > 190`

## Data & API
```python
# Input
img_tensor = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().cuda()

# Inference
with torch.no_grad():
    enhanced, _, _ = iat_model(img_tensor)

# Output
result = enhanced.squeeze().permute(1,2,0).cpu().numpy()
```

## Technical Implementation
- IAT model from `github.com/cuiziteng/Illumination-Adaptive-Transformer`
- Weight: `IAT_enhance/weights/exposure.pth`
- Warm-up: `torch.zeros(1,3,64,64).cuda()` after load
- Clamp output to [0,1] (model output should already be in range)

## Testing
- [ ] Dark image (lum=38): output lum > 80
- [ ] Bright image (lum=200): output lum < 180
- [ ] Normal image: minimal change
- [ ] Inference time < 200ms per image
