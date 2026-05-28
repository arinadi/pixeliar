# Module 11: NAFNetWrapper — Denoise + Deblur

| | |
|---|---|
| **Estimated Complexity** | M |
| **Estimated Files** | 1 |
| **Key Risks** | Two models in VRAM; wrong model selection |

## Requirements
- Denoising (NAFNet-SIDD) for noisy images
- Deblurring (NAFNet-REDS) for blurry images
- Also removes JPEG compression artifacts
- Mutually exclusive: noisy→SIDD, blurry→REDS

## Data & API
```python
from nafnetlib import DenoiseProcessor, DeblurProcessor
from PIL import Image

pil_input = Image.fromarray((img_np * 255).astype(np.uint8))

if noise_flag:
    pil_output = naf_denoise.process(pil_input)
elif blur_flag:
    pil_output = naf_deblur.process(pil_input)

img_out = np.array(pil_output).astype(np.float32) / 255.0
```

## Technical Implementation
- `pip install nafnetlib` — easiest install
- SIDD weights for denoising, REDS weights for deblurring
- Auto-download to `nafnet_model_dir` (persistent)
- nafnetlib works with PIL.Image directly
- Input AFTER exposure and WB fix (denoising on dark images amplifies noise)

## Testing
- [ ] Noisy image: noise_flag=True → SIDD applied, noise reduced
- [ ] Blurry image: blur_flag=True → REDS applied, sharper
- [ ] JPEG artifacts removed with REDS
- [ ] Weights cached in nafnet_model_dir
- [ ] Inference < 500ms per model
