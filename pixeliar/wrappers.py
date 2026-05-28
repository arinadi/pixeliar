"""
pixeliar — Model Wrappers (Modules 7-13)
Thin wrappers around each model. All accept/return float32 numpy [0,1].
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image


def _to_tensor(img_np):
    """numpy (H,W,3) float32 [0,1] → torch (1,3,H,W)"""
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().cuda()


def _from_tensor(tensor):
    """torch (1,3,H,W) → numpy (H,W,3) float32 [0,1]"""
    return tensor.squeeze().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


# ── Module 7: IAT — Exposure Correction ─────────────────

def run_iat(img_np, model, config, logger):
    """Correct under/over-exposure via IAT."""
    t0 = time.time()
    h, w = img_np.shape[:2]
    tensor = _to_tensor(img_np)

    with torch.no_grad():
        enhanced, _, _ = model(tensor)

    result = _from_tensor(enhanced)
    ms = int((time.time() - t0) * 1000)

    before_lum = float(np.mean(cv2.cvtColor(
        (img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)))
    after_lum = float(np.mean(cv2.cvtColor(
        (result * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)))

    logger.step(1, "IAT", f"lum={before_lum:.1f}",
                {"lum": round(before_lum, 1)},
                {"lum": round(after_lum, 1)}, ms)
    return result


# ── Module 8: Retinexformer — Severe Low-Light ──────────

def run_retinexformer(img_np, model, config, logger):
    """Severe low-light enhancement (mean_lum < 40)."""
    t0 = time.time()
    h, w = img_np.shape[:2]
    tensor = _to_tensor(img_np)

    # Pad to multiple of 4
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        out = model(padded)

    out = out[:, :, :h, :w]
    result = _from_tensor(out)
    ms = int((time.time() - t0) * 1000)

    before_lum = float(np.mean(cv2.cvtColor(
        (img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)))
    after_lum = float(np.mean(cv2.cvtColor(
        (result * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)))

    logger.step(1, "Retinexformer", f"lum={before_lum:.1f}",
                {"lum": round(before_lum, 1)},
                {"lum": round(after_lum, 1)}, ms)
    return result


# ── Module 9: DeepWB — White Balance ────────────────────

def _wb_neutrality_score(tensor):
    """Lower = more neutral (better AWB)."""
    arr = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    r, g, b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    return abs(r - g) + abs(g - b) + abs(r - b)


def run_deepwb(img_np, model, config, logger):
    """White balance correction — 3 outputs, select most neutral."""
    t0 = time.time()
    tensor = _to_tensor(img_np)

    with torch.no_grad():
        out_awb, out_indoor, out_shade = model(tensor)

    scores = {
        "AWB": _wb_neutrality_score(out_awb),
        "indoor": _wb_neutrality_score(out_indoor),
        "shade": _wb_neutrality_score(out_shade),
    }
    best_key = min(scores, key=scores.get)
    best_out = {"AWB": out_awb, "indoor": out_indoor, "shade": out_shade}[best_key]
    result = _from_tensor(best_out)
    ms = int((time.time() - t0) * 1000)

    before_r = float(img_np[:, :, 0].mean() * 255)
    before_g = float(img_np[:, :, 1].mean() * 255)
    before_b = float(img_np[:, :, 2].mean() * 255)
    after_r = float(result[:, :, 0].mean() * 255)
    after_g = float(result[:, :, 1].mean() * 255)
    after_b = float(result[:, :, 2].mean() * 255)

    logger.step(2, "DeepWB", f"WB scores: AWB={scores['AWB']:.1f} "
                f"indoor={scores['indoor']:.1f} shade={scores['shade']:.1f} "
                f"→ selected: {best_key}",
                {"R": f"{before_r:.1f}", "G": f"{before_g:.1f}", "B": f"{before_b:.1f}"},
                {"R": f"{after_r:.1f}", "G": f"{after_g:.1f}", "B": f"{after_b:.1f}"},
                ms)
    return result


# ── Module 10: CSRNet — Professional Retouch ────────────

def run_csrnet(img_np, model, config, logger):
    """Professional retouching with alpha blend."""
    t0 = time.time()
    alpha = config.get("csrnet_strength", 0.85)
    tensor = _to_tensor(img_np)

    with torch.no_grad():
        retouched = model(tensor)

    retouched_np = _from_tensor(retouched)
    result = np.clip(alpha * retouched_np + (1 - alpha) * img_np, 0, 1)
    ms = int((time.time() - t0) * 1000)

    before_lum = float(np.mean(cv2.cvtColor(
        (img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)))
    after_lum = float(np.mean(cv2.cvtColor(
        (result * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)))

    logger.step(3, "CSRNet", f"alpha={alpha}",
                {"lum": round(before_lum, 1)},
                {"lum": round(after_lum, 1)}, ms)
    return result


# ── Module 11: NAFNet — Denoise / Deblur ────────────────

def run_nafnet_denoise(img_np, processor, config, logger):
    """Denoise via NAFNet-SIDD."""
    t0 = time.time()
    pil_in = Image.fromarray((img_np * 255).astype(np.uint8))
    pil_out = processor.process(pil_in)
    result = np.array(pil_out).astype(np.float32) / 255.0
    ms = int((time.time() - t0) * 1000)
    logger.step(4, "NAFNet-SIDD", "denoise", {}, {}, ms)
    return result


def run_nafnet_deblur(img_np, processor, config, logger):
    """Deblur + JPEG artifact removal via NAFNet-REDS."""
    t0 = time.time()
    pil_in = Image.fromarray((img_np * 255).astype(np.uint8))
    pil_out = processor.process(pil_in)
    result = np.array(pil_out).astype(np.float32) / 255.0
    ms = int((time.time() - t0) * 1000)
    logger.step(4, "NAFNet-REDS", "deblur + JPEG artifact removal", {}, {}, ms)
    return result


# ── Module 12: Restormer — Heavy Degradation ────────────

def run_restormer(img_np, model, config, logger):
    """Heavy combined degradation (noisy + blurry)."""
    t0 = time.time()
    tensor = _to_tensor(img_np)

    with torch.no_grad():
        restored = model(tensor)

    result = _from_tensor(restored)
    ms = int((time.time() - t0) * 1000)
    logger.step(4, "Restormer", "combined degradation", {}, {}, ms)
    return result


# ── Module 13: ClassicFinisher — Post-ML Micro-Polish ───

def run_classic_finisher(img_np, triage_post, config, logger):
    """CLAHE + Unsharp Mask — only if post-ML metrics still below threshold."""
    result = img_np.copy()
    steps = []

    if triage_post["contrast_std"] < config["thresholds"].get("flat_contrast_std", 45):
        lab = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        steps.append("CLAHE")

    if triage_post.get("sharpness_var", 999) < config["thresholds"].get("blur_threshold", 150):
        blurred = cv2.GaussianBlur(
            (result * 255).astype(np.uint8), (0, 0), 2.0)
        sharpened = cv2.addWeighted(
            (result * 255).astype(np.uint8), 1.5, blurred, -0.5, 0)
        result = sharpened.astype(np.float32) / 255.0
        steps.append("Unsharp Mask")

    if steps:
        logger.step(5, "ClassicFinisher", " + ".join(steps), {}, {}, 0)

    return result, steps
