# Module 9: DeepWBWrapper — White Balance

| | |
|---|---|
| **Estimated Complexity** | S |
| **Estimated Files** | 1 |
| **Key Risks** | Wrong WB selection; input resize artifacts |

## Requirements
- White balance correction for sRGB images
- Output: 3 results (AWB, indoor, shade)
- Select most neutral (smallest Gray World deviation)
- Trigger: wb_dev > 8.0

## Data & API
```python
with torch.no_grad():
    out_awb, out_indoor, out_shade = wb_model(img_tensor)

scores = {
    "AWB":    neutrality_score(out_awb),
    "indoor": neutrality_score(out_indoor),
    "shade":  neutrality_score(out_shade),
}
best_key = min(scores, key=scores.get)
best_out = {"AWB": out_awb, "indoor": out_indoor, "shade": out_shade}[best_key]
```

## Technical Implementation
- DeepWB from `github.com/mahmoudnafifi/Deep_White_Balance`
- Weight: `models/net_G.pth` (in repo)
- Neutrality score: `|R-G| + |G-B| + |R-B|`
- Input resize to 256×256 for inference, upsample output via delta
- Log all 3 scores for debugging

## Testing
- [ ] Cool-cast image: selected = AWB (most neutral)
- [ ] All 3 scores logged
- [ ] Output R/G/B means within 2.0 of each other
- [ ] Inference < 100ms
