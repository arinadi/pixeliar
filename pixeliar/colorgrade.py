"""
pixeliar — ClassicalColorGrade (Module 17)
5-step professional color grading: WB → Exposure → Tonal → Color → Detail.
All operations at full resolution, detail-preserving.
Parameters follow research.md 1-letter keys with conservative defaults.
"""

import cv2
import numpy as np


DEFAULT_GRADE = {
    "b": 0,      # brightness offset (-80..80)
    "c": 1.0,    # contrast multiplier (0.6..2.0)
    "s": 1.0,    # saturation (0.5..2.0)
    "v": 1.0,    # vibrance (0.8..2.0)
    "h": 0,      # highlights recovery (-80..20)
    "d": 0,      # shadows lift (-20..80)
    "k": 0,      # blacks (-60..15)
    "n": 0,      # whites (0..60)
    "w": 0,      # warmth (-40..40)
    "t": 0,      # tint (-30..30)
    "p": 1.0,    # sharpness (0.5..2.0)
    "l": 0,      # clarity (-20..60)
}


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def fix_white_balance(img, warmth=0, tint=0):
    """Step 1: White Balance correction.

    warmth: positive = warmer (add red/reduce blue), negative = cooler
    tint: positive = magenta, negative = green
    Values in research range: warmth -40..40, tint -30..30
    """
    if warmth == 0 and tint == 0:
        return img

    result = img.copy()
    w_norm = warmth / 255.0
    t_norm = tint / 255.0

    # Warmth: shift red and blue channels
    result[:, :, 0] = np.clip(result[:, :, 0] + w_norm, 0, 1)  # R
    result[:, :, 2] = np.clip(result[:, :, 2] - w_norm, 0, 1)  # B

    # Tint: shift green channel
    result[:, :, 1] = np.clip(result[:, :, 1] + t_norm, 0, 1)  # G

    return result


def fix_exposure(img, brightness=0):
    """Step 2: Exposure — brightness offset.

    brightness: offset in -80..80 range, normalized to [-0.31, 0.31]
    """
    if brightness == 0:
        return img

    offset = brightness / 255.0
    return np.clip(img + offset, 0, 1)


def fix_tonal_range(img, contrast=1.0, highlights=0, shadows=0, blacks=0, whites=0):
    """Step 3: Tonal range — S-curve, highlight/shadow recovery, black/white points.

    Uses parametric curves from research.md:
    - Highlights: norm² (quadratic, broad bright-area targeting)
    - Shadows: (1-norm)² (quadratic, broad dark-area targeting)
    - Blacks: (1-norm)³ (cubic, deepest shadows only)
    - Whites: norm³ (cubic, brightest highlights only)
    """
    result = img.copy()

    # Global contrast S-curve via luminance
    if contrast != 1.0:
        gray = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mean_lum = gray.mean()
        # S-curve: push away from mean
        factor = contrast
        result = np.clip(mean_lum + factor * (result - mean_lum), 0, 1).astype(np.float32)

    # Tonal adjustments via luminance mask
    gray_f = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Highlights recovery: norm² — affects bright areas broadly
    if highlights != 0:
        norm = gray_f
        mask = norm * norm  # norm²
        strength = highlights / 255.0
        result = np.clip(result - mask[:, :, np.newaxis] * strength, 0, 1).astype(np.float32)

    # Shadows lift: (1-norm)² — affects dark areas broadly
    if shadows != 0:
        norm = gray_f
        mask = (1 - norm) * (1 - norm)  # (1-norm)²
        strength = shadows / 255.0
        result = np.clip(result + mask[:, :, np.newaxis] * strength, 0, 1).astype(np.float32)

    # Blacks: (1-norm)³ — deepest shadows only
    if blacks != 0:
        norm = gray_f
        mask = (1 - norm) ** 3  # cubic, steeper
        strength = blacks / 255.0
        result = np.clip(result - mask[:, :, np.newaxis] * abs(strength), 0, 1).astype(np.float32)

    # Whites: norm³ — brightest highlights only
    if whites != 0:
        norm = gray_f
        mask = norm ** 3  # cubic, steeper
        strength = whites / 255.0
        result = np.clip(result + mask[:, :, np.newaxis] * strength, 0, 1).astype(np.float32)

    return result


def fix_color(img, saturation=1.0, vibrance=1.0):
    """Step 4: Color — saturation + vibrance.

    Saturation: linear multiplier on HSV S channel
    Vibrance: non-linear, stronger on low-saturation pixels (skin-safe)
    """
    if saturation == 1.0 and vibrance == 1.0:
        return img

    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # Linear saturation
    if saturation != 1.0:
        s_ch = np.clip(s_ch * saturation, 0, 255)

    # Non-linear vibrance: stronger effect on low-saturation pixels
    if vibrance != 1.0:
        s_norm = s_ch / 255.0  # 0..1
        # Mask: low saturation → high effect, high saturation → low effect
        vig_mask = (1 - s_norm) * (1 - s_norm)  # (1-s_norm)²
        # Apply vibrance proportionally to mask
        boost = (vibrance - 1.0) * vig_mask * s_ch
        s_ch = np.clip(s_ch + boost, 0, 255)

    hsv = cv2.merge([h_ch, s_ch, v_ch])
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return result


def fix_detail(img, sharpness=1.0, clarity=0):
    """Step 5: Detail — sharpen + clarity (midtone contrast).

    Sharpness: unsharp mask via GaussianBlur (sigma=1.5)
    Clarity: midtone contrast via GaussianBlur (sigma=10) × midtone mask
    """
    if sharpness == 1.0 and clarity == 0:
        return img

    u8 = (img * 255).astype(np.uint8)
    result_f32 = img.copy()

    # Unsharp mask
    if sharpness != 1.0:
        blurred = cv2.GaussianBlur(u8, (0, 0), 1.5)
        sharpened = cv2.addWeighted(u8, sharpness, blurred, -(sharpness - 1), 0)
        result_f32 = sharpened.astype(np.float32) / 255.0

    # Clarity (midtone contrast)
    if clarity != 0:
        sigma = max(img.shape[:2]) / 100  # adaptive
        sigma = max(sigma, 5)
        blurred = cv2.GaussianBlur(
            (result_f32 * 255).astype(np.uint8), (0, 0), sigma
        ).astype(np.float32) / 255.0

        # Midtone mask: peaks at 0.5, zero at 0 and 1
        gray = cv2.cvtColor((result_f32 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        mid_mask = 1.0 - 2.0 * np.abs(gray - 0.5)  # triangular: 1 at 0.5, 0 at 0/1

        # Clarity = local contrast boost in midtones
        strength = clarity / 100.0  # normalize
        detail = result_f32 - blurred
        result_f32 = np.clip(
            result_f32 + strength * mid_mask[:, :, np.newaxis] * detail,
            0, 1
        ).astype(np.float32)

    return result_f32


def grade(img, params=None, logger=None):
    """Run full 5-step color grading pipeline.

    params: dict with 1-letter keys (merged with DEFAULT_GRADE)
    Returns: graded image, list of steps applied
    """
    p = {**DEFAULT_GRADE, **(params or {})}
    result = img.copy()
    steps = []

    # Step 1: White Balance
    if p["w"] != 0 or p["t"] != 0:
        result = fix_white_balance(result, p["w"], p["t"])
        steps.append("WB")

    # Step 2: Exposure
    if p["b"] != 0:
        result = fix_exposure(result, p["b"])
        steps.append("Exposure")

    # Step 3: Tonal Range
    tonal_active = (p["c"] != 1.0 or p["h"] != 0 or p["d"] != 0
                    or p["k"] != 0 or p["n"] != 0)
    if tonal_active:
        result = fix_tonal_range(result, p["c"], p["h"], p["d"], p["k"], p["n"])
        steps.append("Tonal")

    # Step 4: Color
    if p["s"] != 1.0 or p["v"] != 1.0:
        result = fix_color(result, p["s"], p["v"])
        steps.append("Color")

    # Step 5: Detail
    if p["p"] != 1.0 or p["l"] != 0:
        result = fix_detail(result, p["p"], p["l"])
        steps.append("Detail")

    if logger and steps:
        logger.step(0, "Grade", " → ".join(steps), {}, {}, 0)

    return np.clip(result, 0, 1).astype(np.float32), steps
