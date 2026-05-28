"""
pixeliar — Orchestrator (Module 15)
Hybrid pipeline: Classical CV (full res) + ML (512px, severe cases only).
"""

import numpy as np
from .triage import TriageEngine
from .colorgrade import grade
from .wrappers import (
    run_retinexformer, run_nafnet_denoise, run_nafnet_deblur,
    run_classic_finisher,
)
from .resolution import apply_delta_pipeline


def orchestrate(img_np, triage, config, models, logger):
    """Route image through hybrid pipeline.
    Returns: (result_np, list_of_steps_applied)
    """
    result = img_np.copy()
    steps_applied = []
    thresholds = config.get("thresholds", {})

    # ── Phase 1: Classical Color Grading (full res) ──────
    triage_engine = TriageEngine(config)
    grade_params = triage_engine.compute_grade_params(triage)
    result, grade_steps = grade(result, grade_params, logger)
    steps_applied.extend(grade_steps)

    # ── Phase 2: ML Refinement — Severe Cases Only ──────
    severe_thresh = thresholds.get("severe_dark_lum", 40)

    # Severe dark → Retinexformer (ML exposure recovery)
    if triage["mean_lum"] < severe_thresh and "retinex" in models:
        process_fn = lambda img, cfg, log: run_retinexformer(img, models["retinex"], cfg, log)
        result = apply_delta_pipeline(result, process_fn, config, logger)
        steps_applied.append("Retinexformer")

    # ── Phase 3: ML Denoise / Deblur (only if flagged) ──
    if triage["noise_flag"] and "naf_denoise" in models:
        process_fn = lambda img, cfg, log: run_nafnet_denoise(img, models["naf_denoise"], cfg, log)
        result = apply_delta_pipeline(result, process_fn, config, logger)
        steps_applied.append("NAFNet-SIDD")

    if triage["blur_flag"] and "naf_deblur" in models:
        process_fn = lambda img, cfg, log: run_nafnet_deblur(img, models["naf_deblur"], cfg, log)
        result = apply_delta_pipeline(result, process_fn, config, logger)
        steps_applied.append("NAFNet-REDS")

    # ── Phase 4: Final Polish (full res) ─────────────────
    post_triage = TriageEngine(config).analyze(result)
    result, cf_steps = run_classic_finisher(result, post_triage, config, logger)
    steps_applied.extend(cf_steps)

    return np.clip(result, 0, 1), steps_applied
