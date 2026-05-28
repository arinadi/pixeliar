"""
pixeliar — TriageEngine (Module 5)
Classic CV metrics for image analysis. All computations < 5ms.
Determines which ML models to apply.
"""

import cv2
import numpy as np


class TriageEngine:
    """Fast image analysis using only NumPy and OpenCV."""

    def __init__(self, config):
        self.thresholds = config.get("thresholds", {})
        self.noise_thresh = self.thresholds.get("noise_threshold", 15.0)
        self.blur_thresh = self.thresholds.get("blur_threshold", 150.0)

    def analyze(self, img_np):
        """Analyze image and return metrics dict.
        Input: img_np — float32 numpy array, shape (H, W, 3), range [0, 1]
        Returns: dict with all triage metrics.
        """
        # Ensure float32 [0, 1]
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0

        h, w = img_np.shape[:2]
        img_u8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        # ── Grayscale (for lum, contrast, blur) ────────
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        mean_lum = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        # ── White Balance (Gray World) ─────────────────
        r = float(img_np[:, :, 0].mean())
        g = float(img_np[:, :, 1].mean())
        b = float(img_np[:, :, 2].mean())

        # Deviation from neutral (all channels equal)
        wb_dev = abs(r - g) + abs(g - b) + abs(r - b)

        # Classify cast
        wb_cast = self._classify_wb_cast(r, g, b)

        # ── Noise detection ───────────────────────────
        noise_flag = self._detect_noise(gray)

        # ── Blur detection (Laplacian variance) ───────
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_var = float(np.var(laplacian))
        blur_flag = sharpness_var < self.blur_thresh

        # ── Blown highlights ──────────────────────────
        blown_mask = gray > 250
        blown_pct = float(np.count_nonzero(blown_mask)) / (h * w) * 100

        return {
            "mean_lum": round(mean_lum, 1),
            "contrast_std": round(contrast_std, 1),
            "wb_dev": round(wb_dev * 255, 1),  # scale to 0-255 for readability
            "wb_cast": wb_cast,
            "noise_flag": noise_flag,
            "blur_flag": blur_flag,
            "sharpness_var": round(sharpness_var, 1),
            "blown_pct": round(blown_pct, 2),
            "resolution": (h, w),
        }

    def _classify_wb_cast(self, r, g, b):
        """Classify white balance cast from RGB channel means [0, 1]."""
        rg = abs(r - g)
        gb = abs(g - b)
        rb = abs(r - b)

        if max(rg, gb, rb) < 0.01:
            return "neutral"

        if b > g and b > r:
            return "cool"
        if r > g and r > b:
            return "warm"
        if g > r and g > b:
            return "green"
        if r > g and b > r:
            return "magenta"

        return "neutral"

    def _detect_noise(self, gray_u8):
        """Detect noise by measuring local variance in smooth patches.
        Strategy: find low-variance patches, measure variance there.
        High variance in smooth areas = noise.
        """
        h, w = gray_u8.shape

        # Downsample for speed if image is large
        if h > 512 or w > 512:
            scale = min(512 / h, 512 / w)
            small = cv2.resize(gray_u8, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            small = gray_u8

        # Compute local variance in 8x8 patches
        h2, w2 = small.shape
        pad_h = (8 - h2 % 8) % 8
        pad_w = (8 - w2 % 8) % 8
        if pad_h > 0 or pad_w > 0:
            small = cv2.copyMakeBorder(small, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_REFLECT)

        # Split into patches and measure variance
        small_f = small.astype(np.float32)
        patches = []
        for y in range(0, small.shape[0], 8):
            for x in range(0, small.shape[1], 8):
                patch = small_f[y:y + 8, x:x + 8]
                patches.append(float(np.var(patch)))

        if not patches:
            return False

        # Noise = high variance in what should be smooth patches
        # Use median as baseline (smooth patches have low variance)
        # If > 10% of patches have elevated variance, image is noisy
        patches = np.array(patches)
        threshold = self.noise_thresh
        noisy_patches = np.count_nonzero(patches > threshold)
        noise_ratio = noisy_patches / len(patches)

        return noise_ratio > 0.10

    def compute_grade_params(self, metrics):
        """Convert triage metrics to color grading parameters.
        Returns dict with 1-letter keys matching DEFAULT_GRADE.
        """
        from .colorgrade import DEFAULT_GRADE
        params = dict(DEFAULT_GRADE)

        lum = metrics["mean_lum"]
        wb_dev = metrics["wb_dev"]  # 0-255 scale
        contrast = metrics["contrast_std"]

        # Brightness + shadows
        if lum < 60:
            params["b"] = 15
            params["d"] = 20
        elif lum > 180:
            params["b"] = -10
            params["h"] = -15

        # White balance warmth correction
        cast = metrics.get("wb_cast", "neutral")
        if wb_dev > 15:
            if cast == "warm":
                params["w"] = -min(int(wb_dev * 0.5), 40)
            elif cast == "cool":
                params["w"] = min(int(wb_dev * 0.5), 40)
            elif cast == "green":
                params["t"] = 10
            elif cast == "magenta":
                params["t"] = -10

        # Contrast
        if contrast < 40:
            params["c"] = 1.15
            params["k"] = -8
            params["n"] = 5
        elif contrast > 80:
            params["c"] = 0.95

        # Noise: slight desaturation hides noise
        if metrics.get("noise_flag", False):
            params["s"] = 0.95

        # Always mild sharpen + clarity
        params["p"] = 1.2
        params["l"] = 5

        return params

    def get_no_correction_needed(self, metrics):
        """Check if image needs no correction at all."""
        thresholds = self.thresholds
        dark = thresholds.get("dark_lum", 60)
        bright = thresholds.get("bright_lum", 190)
        severe = thresholds.get("severe_dark_lum", 40)
        wb = thresholds.get("wb_cast_threshold", 8.0)

        return (
            severe <= metrics["mean_lum"] < bright
            and metrics["mean_lum"] >= dark
            and metrics["wb_dev"] <= wb * 255  # wb_dev in 0-255 scale
            and not metrics["noise_flag"]
            and not metrics["blur_flag"]
            and metrics["blown_pct"] < 1.0
        )
