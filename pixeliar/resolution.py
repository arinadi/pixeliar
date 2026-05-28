"""
pixeliar — ResolutionPreserver (Module 14)
Delta-computation + guided filter upsample.
ML runs at low-res, delta is upsampled to full resolution.
"""

import cv2
import numpy as np


def resize_long_edge(img_np, target_px):
    """Resize so longest edge = target_px, preserving aspect ratio."""
    h, w = img_np.shape[:2]
    long = max(h, w)
    if long <= target_px:
        return img_np.copy(), 1.0
    scale = target_px / long
    new_w = int(w * scale)
    new_h = int(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=interp)
    return resized, scale


def upsample_delta(delta_small, guide_gray_full, target_shape):
    """Upsample delta using guided filter (edge-preserving).
    Fallback to bilinear resize if cv2.ximgproc unavailable.
    """
    h, w = target_shape[:2]
    sh, sw = delta_small.shape[:2]

    # Resize delta to full resolution
    delta_resized = cv2.resize(delta_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Try guided filter for edge-aware upsample
    try:
        # Ensure guide is same size as delta
        if guide_gray_full.shape[:2] != (h, w):
            guide_gray_full = cv2.resize(guide_gray_full, (w, h),
                                         interpolation=cv2.INTER_LINEAR)
        if guide_gray_full.ndim == 3:
            guide_gray_full = cv2.cvtColor(guide_gray_full, cv2.COLOR_RGB2GRAY)

        guide_u8 = (guide_gray_full * 255).astype(np.uint8) if guide_gray_full.max() <= 1.0 \
            else guide_gray_full.astype(np.uint8)

        # guidedFilter: guide, src, radius, eps
        # eps controls smoothing — lower = more edge preservation
        if delta_resized.ndim == 3:
            result = np.zeros_like(delta_resized)
            for c in range(delta_resized.shape[2]):
                src = delta_resized[:, :, c]
                if src.dtype != np.float32:
                    src = src.astype(np.float32)
                result[:, :, c] = cv2.ximgproc.guidedFilter(
                    guide=guide_u8, src=src.astype(np.float32),
                    radius=8, eps=0.01
                )
        else:
            result = cv2.ximgproc.guidedFilter(
                guide=guide_u8, src=delta_resized.astype(np.float32),
                radius=8, eps=0.01
            )
        return result
    except (AttributeError, cv2.error):
        # cv2.ximgproc not available — bilinear is fine for most cases
        return delta_resized


def apply_delta_pipeline(input_full, process_fn, config, logger):
    """Run ML at low-res, compute delta, upsample, apply to original.
    process_fn: callable(img_small, config, logger) → img_small_output
    Returns: output at original resolution.
    """
    h, w = input_full.shape[:2]
    long_edge = max(h, w)
    inference_res = config.get("inference_resolution", 512)

    # Step 1: Downscale
    if long_edge > inference_res:
        input_small, scale = resize_long_edge(input_full, inference_res)
        logger.p("UPSAMPLE", f"delta {input_small.shape[1]}x{input_small.shape[0]} "
                  f"→ {w}x{h} (guided filter)", indent=1)
    else:
        # Image already small enough — process directly
        output = process_fn(input_full, config, logger)
        return np.clip(output, 0, 1)

    # Step 2: Run ML on small image
    output_small = process_fn(input_small, config, logger)

    # Step 3: Compute delta at small resolution
    delta_small = output_small.astype(np.float32) - input_small.astype(np.float32)

    # Step 4: Upsample delta
    guide_gray = cv2.cvtColor(input_full, cv2.COLOR_RGB2GRAY) if input_full.ndim == 3 \
        else input_full
    delta_full = upsample_delta(delta_small, guide_gray, input_full.shape)

    # Step 5: Apply delta to original
    result = np.clip(input_full.astype(np.float32) + delta_full, 0, 1)
    return result
