# Module 12: RestormerWrapper — Heavy Degradation

| | |
|---|---|
| **Estimated Complexity** | M |
| **Estimated Files** | 1 |
| **Key Risks** | OOM (182MB load + 3GB peak); opt-in only |

## Requirements
- Heavy combined degradation (noisy AND blurry simultaneously)
- Optional load (load_restormer=True)
- Patch-based inference for large images
- VRAM budget: ~3GB peak

## Data & API
```python
from basicsr.models.archs.restormer_arch import Restormer

model = Restormer(
    inp_channels=3, out_channels=3,
    dim=48, num_blocks=[4,6,6,8],
    num_refinement_blocks=4,
    heads=[1,2,4,8],
    ffn_expansion_factor=2.66, bias=False,
    LayerNorm_type="BiasFree",
    dual_pixel_task=False
).cuda().eval()

with torch.no_grad():
    restored = model(img_tensor)
```

## Technical Implementation
- Restormer from `github.com/swz30/Restormer`
- Weight: `real_denoising.pth` (Google Drive link)
- Patch-based: sliding window 512×512, overlap 128px
- Load ONLY when load_restormer=True (default False)
- VRAM: 182MB load + ~3GB inference peak

## Testing
- [ ] Load only when config flag = True
- [ ] Noisy+blurry image: both degraded corrected
- [ ] Patch-based: no visible seams
- [ ] VRAM stays < 4GB
- [ ] Skip when load_restormer=False
