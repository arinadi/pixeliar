# Module 13: ClassicFinisher — Post-ML Micro-Polish

| | |
|---|---|
| **Estimated Complexity** | S |
| **Estimated Files** | 1 |
| **Key Risks** | Over-sharpening; over-contrast |

## Requirements
- CLAHE on L channel (only if contrast still flat post-ML)
- Unsharp Mask (only if blur still detected post-ML)
- Not active by default — conditional on post-ML triage
- Strength: clipLimit=2.0, sigma=2.0, strength=1.5

## Data & API
```python
def classic_finisher(img_np, triage_post_ml, config):
    result = img_np.copy()
    steps = []

    if triage_post_ml["contrast_std"] < config["thresholds"]["flat_contrast_std"]:
        # CLAHE on L channel
        lab = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        steps.append("CLAHE")

    if triage_post_ml["sharpness_var"] < config["thresholds"]["blur_threshold"]:
        # Unsharp Mask
        blurred = cv2.GaussianBlur((result*255).astype(np.uint8), (0,0), 2.0)
        sharpened = cv2.addWeighted((result*255).astype(np.uint8), 1.5, blurred, -0.5, 0)
        result = sharpened.astype(np.float32) / 255.0
        steps.append("Unsharp Mask")

    return result, steps
```

## Testing
- [ ] Flat image post-ML: CLAHE applied
- [ ] Sharp image post-ML: both skipped
- [ ] Output not over-processed
- [ ] Inference < 50ms
