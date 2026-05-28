# Module 14: ResolutionPreserver — Delta Upsample

| | |
|---|---|
| **Estimated Complexity** | L |
| **Estimated Files** | 1 |
| **Key Risks** | Upsample artifacts; resolution mismatch |

## Requirements
- Preserve original resolution (100% — no exceptions)
- ML runs at inference_resolution (512px) — not at full res
- Compute delta (difference) between input and ML output at low-res
- Upsample delta to full resolution using guided filter
- Apply upsampled delta to original full-res image

## Data & API
```python
# 1. Downscale input to inference resolution
input_small = resize(input_full, inference_resolution)

# 2. Run ML pipeline on small image
output_small = run_pipeline(input_small, config)

# 3. Compute delta at small resolution
delta_small = output_small - input_small

# 4. Upsample delta using guided filter (edge-preserving)
delta_full = cv2.ximgproc.guidedFilter(
    guide=input_full_gray,
    src=delta_small,
    radius=8,
    eps=0.01
)
delta_full = resize(delta_full, input_full.shape[:2])

# 5. Apply delta to original
output_full = np.clip(input_full + delta_full, 0, 1)
```

## Technical Implementation
- Input: full-res image (e.g., 4032×3024)
- ML inference at 512px → fast
- Delta = output - input (the "enhancement map")
- Guided filter: edge-aware upsample (preserves edges, smooths noise)
- cv2.ximgproc.guidedFilter requires opencv-contrib-python
- Fallback: bilinear resize if ximgproc unavailable

## Testing
- [ ] Output resolution = input resolution (100%)
- [ ] No visible blockiness from upsample
- [ ] Edges preserved (no halo artifacts)
- [ ] Works with all input resolutions (1080p to 4K)
