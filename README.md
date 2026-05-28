<p align="center">
  <img src="ML_plan/modules/../../docs/simple.md" width="0" height="0" alt="pixeliar"/>
</p>

<h1 align="center">✨ pixeliar</h1>

<p align="center">
  <strong>Local ML Image Enhancement Pipeline</strong><br>
  Batch photo correction using IAT, Retinexformer, DeepWB, CSRNet, NAFNet on Google Colab T4
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/arinadi/pixeliar/blob/main/main.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/GPU-T4%2016GB-green.svg" alt="GPU"/>
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License"/>
</p>

---

## 🎯 The Problem

Your Google Drive is full of photos. Dark ones. Blue-tinted ones from bad indoor lighting. Flat ones from overcast skies. Noisy ones from high ISO. They all need the kind of retouching a professional photographer would do — but you don't have one.

**pixeliar** is one Colab cell. Paste your Drive URL, press Shift+Enter, go grab a coffee — and every photo comes back better.

---

## ✨ What It Does

**6 ML models** work in sequence, each handling what it does best:

| Stage | Model | What It Fixes |
|-------|-------|---------------|
| 1 | **IAT** / **Retinexformer** | Under/over-exposed photos, severe darkness |
| 2 | **DeepWB** | Color casts from mixed/indoor lighting |
| 3 | **CSRNet** | Professional retouching (color harmony, tonal balance) |
| 4 | **NAFNet** | Camera noise, motion blur, JPEG artifacts |
| 5 | **Restormer** | Heavy combined degradation (optional) |
| 6 | **ClassicFinisher** | Final CLAHE + sharpening polish |

**Every model uses official pretrained weights.** No training. No fine-tuning. Just plug and play.

---

## 🚀 Quick Start — One-Click Enhancement

### Step 1: Prepare Your Secrets 🔑

In Colab's **Secrets** tab (🔑 icon on the left), add:

| Secret | Required | Purpose |
|--------|----------|---------|
| `MIMO_API_KEY` | Optional | Cloud fallback if local ML fails |
| `GITHUB_TOKEN` | Optional | Faster downloads from private repos |

> 💡 **No secrets needed** for basic usage — all ML runs locally on T4 GPU.

### Step 2: Choose Your Runtime 🔥

Set runtime to **T4 GPU**:
`Runtime > Change runtime type > T4 GPU`

### Step 3: Run 🛎️

Copy and run this cell. Your personal photo editor will be with you in seconds:

```python
# @title ✨ Start pixeliar
import os, subprocess, sys

# 1. Load Secrets (optional)
try:
    from google.colab import userdata
    for key in ['MIMO_API_KEY', 'GITHUB_TOKEN']:
        try:
            val = userdata.get(key)
            if val: os.environ[key] = str(val)
        except: pass
except: pass

# 2. Launch pipeline
!curl -sL https://raw.githubusercontent.com/arinadi/pixeliar/main/runner.py -o runner.py && python runner.py
```

### Step 4: Edit CONFIG 📝

When prompted, edit the `CONFIG` dict at the top of `main.py`:

```python
CONFIG = {
    "source_gdrive_url": "https://drive.google.com/drive/folders/YOUR_FOLDER_ID",
    "result_folder_name": "Pixeliar Enhanced",
    ...
}
```

### Step 5: Grab Your Coffee ☕

The pipeline will:
1. Clone the repo & load models (~30s)
2. Download all photos from your Drive folder
3. Analyze each photo (triage)
4. Apply the right models per photo
5. Save enhanced versions to a new Drive folder
6. Print a full log of what it did to each photo

---

## 🔧 Architecture

```
runner.py          → Clone, install, load secrets
  └── start.py     → Detect GPU, set mode
        └── main.py → Full pipeline
              ├── Triage (CV metrics)
              ├── IAT / Retinexformer (exposure)
              ├── DeepWB (white balance)
              ├── CSRNet (professional retouch)
              ├── NAFNet (denoise/deblur)
              ├── ResolutionPreserver (delta upsample)
              └── SessionManager (resume support)
```

**Key features:**
- **3-Layer Runner** — clone → detect GPU → run. One cell, one command.
- **Session Resume** — Colab dies? Run again. Already-processed photos are skipped automatically.
- **Drive API v3** — chunk-based downloads, rate limiting, retry with backoff.
- **Idle Monitor** — auto-shutdowns Colab after 5 min idle (save your GPU credits).
- **100% Resolution** — ML runs at 512px, delta is upsampled with guided filter. Output = input resolution. Always.

---

## 📊 VRAM Budget (T4 = 16GB)

| Model | Loaded | Peak (4K) |
|-------|--------|-----------|
| IAT | 0.4 MB | ~200 MB |
| CSRNet | 0.2 MB | ~50 MB |
| DeepWB | 55 MB | ~800 MB |
| NAFNet × 2 | 128 MB | ~1.5 GB |
| Retinexformer | 6.4 MB | ~2.0 GB |
| Restormer (opt) | 182 MB | ~3.0 GB |
| **Total** | **~372 MB** | **~3.0 GB** |

> 🟢 **13 GB headroom.** Safe for 4K images.

---

## 📋 Example Output

```
[10:32:16.774] ══════════════════════════════════════════════════
[10:32:16.774] 🖼️  [01/48] IMG_0042.jpg  (4032×3024, 8.7 MB)
[10:32:16.779]    🔍 TRIAGE  lum=38.2  std=22.1  wb_dev=18.4(cool)
[10:32:16.780]    ⚠️  ISSUE   underexposed (lum=38.2 < 60)
[10:32:16.781]    🔧 STEP 1  IAT — lum=38.2
[10:32:16.924]               lum=38.2 → 91.4   ⏱️ 143ms
[10:32:16.924]    🔧 STEP 2  DeepWB — cool cast dev=18.4
[10:32:17.013]               AWB=2.1 indoor=14.3 shade=31.7 → AWB
[10:32:17.013]    🔧 STEP 3  CSRNet — alpha=0.85
[10:32:17.025]               lum=91.4 → 94.1   ⏱️ 12ms
[10:32:17.025]    ⏩  SKIP    NAFNet  (noise=False, blur=False)
[10:32:17.142]    💾 SAVED   IMG_0042_pixeliar.jpg   ⏱️ 116ms
[10:32:17.143]    ✅ DONE    total=369ms   brightness +53.2
```

---

## 🛠️ Local Development

```bash
git clone https://github.com/arinadi/pixeliar.git
cd pixeliar
pip install -r requirements.txt
python main.py
```

Edit `CONFIG` in `main.py` before running.

---

## 📦 Module Breakdown

| # | Module | Complexity | What |
|---|--------|------------|------|
| 0 | `runner.py` | S | Colab bootstrap |
| 1 | `start.py` | S | GPU detection |
| 2 | `gdrive.py` | M | Drive API v3 |
| 3 | `session.py` | M | Fingerprint resume |
| 4 | `logger.py` | S | Structured print log |
| 5 | `triage.py` | M | CV metrics |
| 6 | `model_loader.py` | M | VRAM management |
| 7-13 | `wrappers.py` | M | 6 ML model wrappers |
| 14 | `resolution.py` | L | Delta upsample |
| 15 | `orchestrator.py` | M | Pipeline chain router |
| 16 | `main.py` | S | BatchRunner entry point |
| 17 | `idle_monitor.py` | S | Auto-shutdown |
| 18 | `hardware.py` | S | GPU/CPU detection |

---

## 🤝 Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Built with ☕ by <a href="https://github.com/arinadi">arinadi</a></em>
</p>
