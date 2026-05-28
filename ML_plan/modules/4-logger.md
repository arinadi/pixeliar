# Module 4: PhotonLogger — Structured Print Log

| | |
|---|---|
| **Estimated Complexity** | S |
| **Estimated Files** | 1 |
| **Key Risks** | None |

## Requirements
- Real-time print to stdout with timestamps (HH:MM:SS.mmm)
- Icon-based message types (START, FOLDER, DOWNLOAD, MODEL, VRAM, IMAGE, TRIAGE, ISSUE, SKIP, STEP, UPSAMPLE, SAVE, LOG, DONE, FAIL, COPY, SUMMARY, RESUME, DRIVE, IDLE)
- Indent support (up to 3 levels)
- VRAM usage tracking
- Model load logging
- Triage result logging
- Pipeline step logging (before/after metrics + timing)

## Data & API
```python
logger = PhotonLogger()
logger.p("START", "PHOTON v4.0 starting")
logger.sep()
logger.model_loaded("IAT", 90000, 0.4)
logger.triage(metrics_dict)
logger.step(1, "IAT", "underexposed lum=38.2", before, after, 143)
logger.skip("NAFNet", "noise=False, blur=False")
logger.resume("IMG_0042.jpg", "abc123def456")
logger.drive_op("download", "IMG_0042.jpg (8.7MB)")
logger.idle("Batch complete — auto-shutdown in 5 min")
```

## Technical Implementation
- Class with `_ts()` for timestamp, `_vram()` for GPU memory
- `p(icon_key, msg, indent, **kv)` — main print method
- `model_loaded()`, `triage()`, `step()`, `skip()` — convenience methods
- All output via `print(..., flush=True)` for real-time Colab display

## Testing
- [ ] Timestamps are accurate (HH:MM:SS.mmm)
- [ ] Indent levels render correctly
- [ ] VRAM shows correct values on GPU
- [ ] All icon types render without error
- [ ] flush=True ensures real-time output
