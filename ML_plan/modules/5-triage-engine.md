# Module 5: TriageEngine — Classic CV Metrics

| | |
|---|---|
| **Estimated Complexity** | M |
| **Estimated Files** | 1 |
| **Key Risks** | Incorrect threshold triggers; wrong model selection |

## Requirements
- Compute luminance (mean brightness)
- Compute contrast (std dev)
- Detect white balance cast (Gray World deviation)
- Detect noise (local variance in smooth patches)
- Detect blur (Laplacian variance)
- Detect blown highlights (clipped pixel percentage)
- All metrics < 5ms per image

## Data & API
```python
metrics = triage_engine.analyze(img_np)
# Returns:
{
    "mean_lum": 38.2,        # 0-255
    "contrast_std": 22.1,    # 0-128
    "wb_dev": 18.4,          # Gray World deviation
    "wb_cast": "cool",       # warm/cool/green/magenta/neutral
    "noise_flag": False,     # local variance > threshold
    "blur_flag": False,      # Laplacian var < threshold
    "blown_pct": 0.2,        # % of clipped highlights
}
```

## Technical Implementation
```python
import cv2
import numpy as np

def analyze(img_np: np.ndarray, config: dict) -> dict:
    # Luminance
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mean_lum = float(np.mean(gray))

    # Contrast
    contrast_std = float(np.std(gray))

    # White Balance (Gray World)
    r, g, b = img_np[:,:,0].mean(), img_np[:,:,1].mean(), img_np[:,:,2].mean()
    wb_dev = abs(r-g) + abs(g-b) + abs(r-b)
    # Cast detection from channel dominance

    # Noise (local variance in smooth patches)
    # Use Laplacian variance for blur detection
    # Blown highlights: count pixels > 250
```

## Testing
- [ ] Dark image (lum < 40): mean_lum < 40
- [ ] Bright image (lum > 190): mean_lum > 190
- [ ] Cool cast: wb_dev > 8, wb_cast = "cool"
- [ ] Noisy image: noise_flag = True
- [ ] Blurry image: blur_flag = True
- [ ] All metrics computed in < 5ms
