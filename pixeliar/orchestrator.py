"""
pixeliar — Orchestrator (Module 15)
Pipeline chain router. Hard-coded execution order (NOT configurable).
"""

import numpy as np
from .wrappers import (
    run_iat, run_retinexformer, run_deepwb, run_csrnet,
    run_nafnet_denoise, run_nafnet_deblur, run_restormer,
    run_classic_finisher,
)
from .resolution import apply_delta_pipeline


def orchestrate(img_np, triage, config, models, logger):
    """Route image through pipeline based on triage results.
    Returns: (result_np, list_of_steps_applied)
    """
    result = img_np.copy()
    steps_applied = []
    thresholds = config.get("thresholds", {})

    # ── Step 1: Exposure Correction ─────────────────────
    severe_dark = thresholds.get("severe_dark_lum", 40)
    dark = thresholds.get("dark_lum", 60)
    bright = thresholds.get("bright_lum", 190)

    if triage["mean_lum"] < severe_dark:
        # Severe dark → Retinexformer replaces IAT
        if "retinex" in models:
            process_fn = lambda img, cfg, log: run_retinexformer(img, models["retinex"], cfg, log)
            result = apply_delta_pipeline(result, process_fn, config, logger)
            steps_applied.append("Retinexformer")
        else:
            logger.p("WARN", "Retinexformer not loaded, skipping", indent=1)
    elif triage["mean_lum"] < dark or triage["mean_lum"] > bright:
        # Under/over exposed → IAT
        if "iat" in models:
            process_fn = lambda img, cfg, log: run_iat(img, models["iat"], cfg, log)
            result = apply_delta_pipeline(result, process_fn, config, logger)
            steps_applied.append("IAT")
        else:
            logger.p("WARN", "IAT not loaded, skipping", indent=1)

    # ── Step 2: White Balance ───────────────────────────
    wb_threshold = thresholds.get("wb_cast_threshold", 8.0)
    # wb_dev is in 0-255 scale, threshold in 0-1 scale → multiply
    if triage["wb_dev"] > wb_threshold * 255:
        if "deepwb" in models:
            result = run_deepwb(result, models["deepwb"], config, logger)
            steps_applied.append("DeepWB")
        else:
            logger.p("WARN", "DeepWB not loaded, skipping", indent=1)

    # ── Step 3: CSRNet — Always ────────────────────────
    if "csrnet" in models:
        result = run_csrnet(result, models["csrnet"], config, logger)
        steps_applied.append("CSRNet")

    # ── Step 4: Denoise / Deblur ───────────────────────
    if triage["noise_flag"] and triage["blur_flag"]:
        if config.get("load_restormer") and "restormer" in models:
            result = run_restormer(result, models["restormer"], config, logger)
            steps_applied.append("Restormer")
        elif "naf_denoise" in models:
            result = run_nafnet_denoise(result, models["naf_denoise"], config, logger)
            steps_applied.append("NAFNet-SIDD")
    elif triage["noise_flag"]:
        if "naf_denoise" in models:
            result = run_nafnet_denoise(result, models["naf_denoise"], config, logger)
            steps_applied.append("NAFNet-SIDD")
    elif triage["blur_flag"]:
        if "naf_deblur" in models:
            result = run_nafnet_deblur(result, models["naf_deblur"], config, logger)
            steps_applied.append("NAFNet-REDS")

    # ── Step 5: Classic Finisher — Post-ML ──────────────
    from .triage import TriageEngine
    post_triage = TriageEngine(config).analyze(result)
    result, cf_steps = run_classic_finisher(result, post_triage, config, logger)
    steps_applied.extend(cf_steps)

    return np.clip(result, 0, 1), steps_applied
