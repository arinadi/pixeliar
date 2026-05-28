# Module 15: Orchestrator — Pipeline Chain Router

| | |
|---|---|
| **Estimated Complexity** | M |
| **Estimated Files** | 1 |
| **Key Risks** | Wrong execution order; CSRNet before exposure fix |

## Requirements
- Route images through pipeline based on triage results
- Hard-coded execution order (NOT configurable):
  1. Retinexformer OR IAT (never both)
  2. DeepWB (conditional)
  3. CSRNet (always)
  4. NAFNet/Restormer (conditional)
  5. ClassicFinisher (conditional)
- Log every skip and every step with reason

## Data & API
```python
def orchestrate(img_np, triage, config, models, logger):
    steps = []
    result = img_np.copy()

    # Step 1: Exposure
    if triage["mean_lum"] < config["thresholds"]["severe_dark_lum"]:
        result = run_retinex(result, models["retinex"], config, logger)
        steps.append("Retinexformer")
    elif triage["mean_lum"] < config["thresholds"]["dark_lum"] or \
         triage["mean_lum"] > config["thresholds"]["bright_lum"]:
        result = run_iat(result, models["iat"], config, logger)
        steps.append("IAT")

    # Step 2: White Balance
    if triage["wb_dev"] > config["thresholds"]["wb_cast_threshold"]:
        result = run_deepwb(result, models["deepwb"], config, logger)
        steps.append("DeepWB")

    # Step 3: Retouch (always)
    result = run_csrnet(result, models["csrnet"], config, logger)
    steps.append("CSRNet")

    # Step 4: Denoise/Deblur
    if triage["noise_flag"] and triage["blur_flag"] and config["load_restormer"]:
        result = run_restormer(result, models["restormer"], config, logger)
        steps.append("Restormer")
    elif triage["noise_flag"]:
        result = run_nafnet_sidd(result, models["naf_denoise"], config, logger)
        steps.append("NAFNet-SIDD")
    elif triage["blur_flag"]:
        result = run_nafnet_reds(result, models["naf_deblur"], config, logger)
        steps.append("NAFNet-REDS")

    # Step 5: Post-ML polish
    post_triage = triage_engine.analyze(result)
    if post_triage["contrast_std"] < config["thresholds"]["flat_contrast_std"] or \
       post_triage["sharpness_var"] < config["thresholds"]["blur_threshold"]:
        result, cf_steps = classic_finisher(result, post_triage, config)
        steps.extend(cf_steps)

    return result, steps
```

## Testing
- [ ] CSRNet ALWAYS runs after exposure correction
- [ ] Retinexformer replaces IAT (never both)
- [ ] DeepWB skipped when wb_dev <= 8.0
- [ ] Restormer only when noisy+blurry AND load_restormer=True
- [ ] ClassicFinisher only when post-ML metrics still below threshold
