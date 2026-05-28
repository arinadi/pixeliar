# PHOTON — Local ML Image Enhancement Pipeline
**v4.0 — 3-Layer Runner · Drive API v3 · Background Init · Session Resume**

> *"Every photo deserves the light it was meant to have."*

---

## Project Context

| Field | Value |
|---|---|
| **Project Type** | Greenfield |
| **Project Mode** | `ml-service` + `data-pipeline` |
| **Environment** | Google Colab T4 (16GB VRAM, ~12GB RAM) |
| **Execution Model** | **3-Layer Runner** — `runner.py` → `start.py` → `main.py` — fetch & run via Colab cell |
| **Logging** | **Full real-time print log** ke notebook output — tidak ada yang tersembunyi |
| **External APIs** | **None.** Semua inference lokal di T4. |
| **Google Drive** | **Drive API v3** — OAuth via Colab auth, chunk-based I/O, rate limiting |
| **Secrets** | **Colab Secrets** — API keys & tokens tidak hard-coded |
| **Adapted From** | `simple.md` (Drive patterns) · `TTB` (runner, idle monitor, Colab ops) |

---

## The Story (The Pain)

Folder Google Drive berisi ratusan foto. Ada yang gelap, ada color cast dari lampu, ada yang flat, ada yang noisy dari ISO tinggi. Semua butuh retouching yang seharusnya dilakukan professional photographer. Yang kamu butuhkan adalah satu cell Colab, tempel URL Drive, tekan Shift+Enter, pergi ngopi — dan semua foto sudah lebih baik ketika kembali, dengan log lengkap yang menjelaskan apa yang dilakukan pada setiap foto. Jika session mati di tengah jalan, jalankan ulang — pipeline otomatis skip foto yang sudah selesai.

---

## Execution Model: 3-Layer Runner

Arsitektur 3-layer yang diadaptasi dari `TTB` — setiap layer punya tugas spesifik, robust terhadap Colab session restart.

```
┌─────────────────────────────────────────────────────────────────┐
│  CELL 1 (satu-satunya cell yang perlu dijalankan)               │
│                                                                 │
│  # ── Fetch & run pipeline dari GitHub ────────────────────── │
│  !curl -sL https://raw.githubusercontent.com/USER/REPO/main/  │
│       runner.py | python3                                      │
│                                                                 │
│  Atau: paste CONFIG langsung + jalankan main()                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: runner.py — Colab Entry Point                        │
│                                                                 │
│  • Clone/update repo dari GitHub (git fetch --depth 1)         │
│  • Install lightweight bootstrap deps (requirements.txt)        │
│  • Load Colab Secrets (MIMO_API_KEY, GITHUB_TOKEN, dll)       │
│  • Launch start.py                                             │
│                                                                 │
│  Adapted from TTB/runner.py pattern                            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: start.py — Smart Runner                              │
│                                                                 │
│  • Detect GPU via nvidia-smi (subprocess)                      │
│  • Set PHOTON_MODE = 'GPU' or 'CPU_FALLBACK'                  │
│  • Launch main.py as subprocess                                │
│  • Handle graceful shutdown (SIGTERM → cleanup VRAM)           │
│                                                                 │
│  Adapted from TTB/start.py pattern                             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: main.py — Core Pipeline                              │
│                                                                 │
│  CONFIG = { ... }  ← edit ini sebelum run                      │
│                                                                 │
│  # ── SECTION 1: Imports & Drive Auth ──────────────────────  │
│  # ── SECTION 2: Drive I/O (API v3) ────────────────────────  │
│  # ── SECTION 3: Logger ────────────────────────────────────  │
│  # ── SECTION 4: Session Manager (resume) ──────────────────  │
│  # ── SECTION 5: Triage Engine ─────────────────────────────  │
│  # ── SECTION 6: Model Loader (background) ─────────────────  │
│  # ── SECTION 7: Model Wrappers ────────────────────────────  │
│  # ── SECTION 8: Resolution Preservation ───────────────────  │
│  # ── SECTION 9: Pipeline Orchestrator ─────────────────────  │
│  # ── SECTION 10: Batch Runner (entry point) ───────────────  │
│  # ── SECTION 11: Idle Monitor (optional) ──────────────────  │
│                                                                 │
│  run_photon(CONFIG)  ← eksekusi dimulai di sini               │
└─────────────────────────────────────────────────────────────────┘
```

**Kenapa 3-layer:**
- **runner.py** = Colab-specific bootstrap — clone, install, auth. Tidak perlu edit.
- **start.py** = Hardware detection — GPU → full pipeline, CPU → fallback mode (DeepWB only atau skip).
- **main.py** = Core logic — bisa dijalankan standalone tanpa runner (local dev).

**Colab cell hanya perlu 1 baris:**
```python
!curl -sL https://raw.githubusercontent.com/USER/REPO/main/runner.py | python3
```

**Atau manual mode** (tanpa GitHub): paste isi `main.py` langsung ke cell, edit CONFIG, run.

---

## Google Drive Integration (Drive API v3)

Menggantikan `gdown` dengan **Google Drive API v3** — lebih robust, support retry, rate limiting, dan OAuth yang proven. Pattern diadaptasi dari `simple.md`.

### Authentication

```python
from google.colab import auth
from google.auth import default
from googleapiclient.discovery import build

# OAuth — Colab handles the login popup
auth.authenticate_user()
creds, _ = default()
drive_svc = build("drive", "v3", credentials=creds)
```

### Core Operations

```python
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import re, io

def extract_folder_id(url: str) -> str:
    """Extract folder ID from Google Drive URL."""
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError(f"URL tidak valid: {url}")
    return m.group(1)

def list_images_in_folder(folder_id: str, recursive: bool = False) -> list:
    """List all images in folder. Supports subfolder traversal."""
    IMG_TYPES = "image/jpeg,image/png,image/webp,image/tiff"
    query = f"'{folder_id}' in parents and trashed=false and ("
    query += " or ".join([f"mimeType='{m}'" for m in IMG_TYPES.split(",")])
    query += ")"
    results = drive_svc.files().list(
        q=query,
        fields="files(id, name, mimeType, size, md5Checksum)",
        pageSize=500
    ).execute()
    files = results.get("files", [])

    if recursive:
        # Also list subfolders and traverse them
        folder_query = f"'{folder_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'"
        subfolders = drive_svc.files().list(q=folder_query, fields="files(id, name)").execute().get("files", [])
        for sf in subfolders:
            files.extend(list_images_in_folder(sf["id"], recursive=True))

    return files

def download_drive_file(file_id: str, dest_path: str) -> str:
    """Chunk-based download with progress. Handles large files."""
    request = drive_svc.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return dest_path

def create_drive_folder(name: str, parent_id: str = None) -> tuple:
    """Create folder, return (folder_id, webViewLink)."""
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    folder = drive_svc.files().create(body=meta, fields="id, webViewLink").execute()
    return folder["id"], folder.get("webViewLink", "")

def upload_to_drive(local_path: str, folder_id: str) -> None:
    """Upload file with MIME type detection."""
    import os
    from pathlib import Path
    name = os.path.basename(local_path)
    ext = Path(local_path).suffix.lower()
    mime = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.webp': 'image/webp'}.get(ext, 'image/jpeg')
    drive_svc.files().create(
        body={"name": name, "parents": [folder_id]},
        media_body=MediaFileUpload(local_path, mimetype=mime),
        fields="id"
    ).execute()
```

### Best Practices (dari simple.md)

| Practice | Detail |
|---|---|
| **Chunk-based download** | `MediaIoBaseDownload` + `next_chunk()` — handle file besar tanpa OOM |
| **MIME detection** | Auto-detect dari extension — jangan hardcode |
| **Rate limiting** | `time.sleep(0.5)` antar API calls — Google quota 20K req/100s |
| **Per-file error handling** | `try/except` per file — batch tetap lanjut meski satu gagal |
| **md5Checksum** | Return di `files().list()` — fingerprint untuk resume |
| **Recursive folder** | Traverse subfolder untuk foto yang tersimpan berjenjang |
| **JSON log upload** | Session log di-upload ke Drive result folder |

### Fallback: gdown

Jika Drive API quota habis atau auth gagal, fallback ke `gdown`:
```python
subprocess.run(["pip", "install", "gdown", "-q"])
import gdown
gdown.download_folder(url=source_url, output=str(local_dir), quiet=False)
```

---

## Session Resume (Fingerprint-Based)

Pattern diadaptasi dari TTB's persistence model — setiap foto punya fingerprint unik, processed fotos di-skip otomatis.

### How It Works

```python
import hashlib, json, os

SESSION_FILE = "/content/photon_session.json"

def load_session() -> dict:
    """Load existing session or create new one."""
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE) as f:
            return json.load(f)
    return {"processed": {}, "stats": {"total": 0, "ok": 0, "skip": 0, "fail": 0}}

def save_session(session: dict):
    """Persist session state to disk."""
    with open(SESSION_FILE, "w") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)

def file_fingerprint(file_id: str, md5: str = None) -> str:
    """Generate unique fingerprint for a file.
    Uses Drive md5Checksum if available, else file_id."""
    return md5 if md5 else file_id

def should_process(session: dict, fingerprint: str) -> bool:
    """Check if file already processed in this session."""
    return fingerprint not in session["processed"]

def mark_processed(session: dict, fingerprint: str, status: str, log: dict):
    """Mark file as processed with its log."""
    session["processed"][fingerprint] = {"status": status, **log}
    session["stats"]["total"] += 1
    if status == "ok":
        session["stats"]["ok"] += 1
    elif status == "skip":
        session["stats"]["skip"] += 1
    elif status == "fail":
        session["stats"]["fail"] += 1
    save_session(session)
```

### Resume Flow

```
Session 1: process 30/50 photos → Colab dies
Session 2: run again → session file still at /content/
           → 30 photos skipped instantly (fingerprint match)
           → process remaining 20 photos
```

**Session file location:** `/content/photon_session.json` — persists across cell re-runs within same Colab runtime. For cross-runtime resume, upload session to Drive.

---

## Colab Integration (Adapted from TTB)

### Idle Monitor

Colab billing terus berjalan meski tidak ada yang diproses. Idle monitor auto-shutdown setelah batch selesai.

```python
import asyncio
from datetime import datetime, timedelta

class IdleMonitor:
    """Auto-shutdown Colab after idle period.
    Adapted from TTB/bot_classes.py IdleMonitor."""

    def __init__(self, idle_minutes: int = 5):
        self.idle_minutes = idle_minutes
        self.last_activity = datetime.now()

    def reset(self):
        self.last_activity = datetime.now()

    async def watch(self):
        """Run as background task — shutdown after idle timeout."""
        while True:
            await asyncio.sleep(60)
            elapsed = datetime.now() - self.last_activity
            if elapsed > timedelta(minutes=self.idle_minutes):
                print(f"\n⏰ Idle for {self.idle_minutes} min — shutting down Colab runtime")
                try:
                    from google.colab import runtime
                    runtime.unassign()
                except:
                    print("⚠️  Could not auto-shutdown — shutdown manually")
                break
```

### Colab Secrets

API keys & tokens tidak hard-coded — gunakan Colab Secrets:

```python
# In main.py
from google.colab import userdata

# Access secrets (hidden in notebook)
MIMO_API_KEY = userdata.get("MIMO_API_KEY")  # optional, for cloud fallback
GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")  # for private repos

# In runner.py — inject before launching main.py
os.environ["MIMO_API_KEY"] = MIMO_API_KEY
os.environ["GITHUB_TOKEN"] = GITHUB_TOKEN
```

### GPU Detection (from TTB/start.py)

```python
import subprocess, sys

def detect_gpu() -> str:
    """Detect GPU availability. Returns 'GPU' or 'CPU'."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "NVIDIA" in result.stdout:
            return "GPU"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "CPU"

PHOTON_MODE = detect_gpu()
print(f"🔧 PHOTON_MODE = {PHOTON_MODE}")

if PHOTON_MODE == "CPU":
    print("⚠️  No GPU detected — limited pipeline (DeepWB only)")
    print("   For full pipeline, use Colab with T4/GPU runtime")
```

---

## Full Print Log

Setiap event dicetak ke stdout secara real-time dengan format seragam. Tidak ada silent operation.

```
[10:32:01.042] 🚀 PHOTON v4.0 starting
[10:32:01.043] 📁 Source : https://drive.google.com/drive/folders/ABC123
[10:32:01.044] 📁 Target : MyDrive/enhanced/
[10:32:04.221] 📥 Downloading folder... 48 images found (Drive API v3)
[10:32:08.441] 🤖 Loading models to VRAM...
[10:32:09.102]    ✅ IAT           loaded   (90K params,   0.4 MB)
[10:32:09.580]    ✅ CSRNet        loaded   (37K params,   0.2 MB)
[10:32:11.204]    ✅ DeepWB        loaded   (14M params,  55.1 MB)
[10:32:13.011]    ✅ NAFNet-SIDD   loaded   (67M params,  64.2 MB)
[10:32:13.981]    ✅ NAFNet-REDS   loaded   (67M params,  64.2 MB)
[10:32:16.772]    ✅ Retinexformer loaded   (1.6M params,  6.4 MB)
[10:32:16.773]    📊 Total VRAM used: 191 MB / 16384 MB (1.2%)
[10:32:16.774] ══════════════════════════════════════════════════
[10:32:16.774] 🖼️  [01/48] IMG_0042.jpg  (4032×3024, 8.7 MB)
[10:32:16.779]    🔍 TRIAGE  mean_lum=38.2  std=22.1  wb_dev=18.4(cool)
[10:32:16.779]               noise=False  blur=False  blown=0.2%
[10:32:16.780]    ⚠️  ISSUE   underexposed (lum=38.2 < 60)
[10:32:16.780]    ⚠️  ISSUE   white_balance cast=cool deviation=18.4
[10:32:16.780]    ⏩  SKIP    Retinexformer (lum=38.2 > severe threshold=40)
[10:32:16.781]    🔧 STEP 1  IAT — reason: underexposed lum=38.2
[10:32:16.924]               before: lum=38.2  std=22.1
[10:32:16.924]               after:  lum=91.4  std=38.7   ⏱️ 143ms
[10:32:16.924]    🔧 STEP 2  DeepWB — reason: cool cast dev=18.4 > 8.0
[10:32:16.924]               WB scores: AWB=2.1  indoor=14.3  shade=31.7
[10:32:16.924]               selected: AWB (most neutral)
[10:32:17.013]               before: R=98.2  G=112.4  B=131.0
[10:32:17.013]               after:  R=112.5 G=113.0  B=112.8   ⏱️ 89ms
[10:32:17.013]    🔧 STEP 3  CSRNet — reason: always applied (professional retouch)
[10:32:17.025]               before: lum=91.4  std=38.7
[10:32:17.025]               after:  lum=94.1  std=46.2   ⏱️ 12ms
[10:32:17.025]    ⏩  SKIP    NAFNet  (noise=False, blur=False)
[10:32:17.025]    ⏩  SKIP    ClassicFinisher (std=46.2 > threshold=45)
[10:32:17.026]    📐 UPSAMPLE delta 512px → 4032×3024 (guided filter)
[10:32:17.142]    💾 SAVED   MyDrive/enhanced/IMG_0042_photon.jpg   ⏱️ 116ms
[10:32:17.143]    📝 LOG     MyDrive/enhanced/IMG_0042_photon_log.json
[10:32:17.143]    ✅ DONE    total=400ms   brightness +53.2   wb_fix 73%
[10:32:17.143] ══════════════════════════════════════════════════
[10:32:17.144] 🖼️  [02/48] IMG_0043.jpg  (4032×3024, 9.1 MB)
[10:32:17.148]    ⏩  SKIP    no_correction_needed (all triage normal)
[10:32:17.149]    💾 COPY    original → MyDrive/enhanced/IMG_0043_photon.jpg
[10:32:17.149] ══════════════════════════════════════════════════
...
[10:47:22.001] ══════════════════════════════════════════════════
[10:47:22.002] ✅ SESSION COMPLETE
[10:47:22.002]    Total images    : 48
[10:47:22.002]    Processed       : 44
[10:47:22.002]    No correction   : 3
[10:47:22.002]    Failed          : 1  (see IMG_0029_photon_log.json for error)
[10:47:22.003]    Total time      : 14m 21s
[10:47:22.003]    Avg per image   : 17.9s
[10:47:22.003]    Peak VRAM       : 3.1 GB
[10:47:22.004]    Output folder   : MyDrive/enhanced/  (48 files)
[10:47:22.004] 📊 MODEL USAGE:
[10:47:22.004]    IAT             : 29 images
[10:47:22.004]    DeepWB          : 21 images
[10:47:22.004]    CSRNet          : 44 images
[10:47:22.004]    NAFNet-SIDD     :  8 images
[10:47:22.004]    NAFNet-REDS     :  3 images
[10:47:22.004]    Retinexformer   :  5 images
[10:47:22.004]    ClassicFinisher : 11 images
[10:47:22.004] ☁️  DRIVE    session_log.json uploaded to result folder
[10:47:22.004] ⏰ IDLE     Batch complete — auto-shutdown in 5 min
```

---

## Tech Stack (Research-Validated)

### Prinsip Pemilihan Model

1. **Official pretrained weights tersedia** — tidak ada model yang butuh training ulang
2. **Trained pada dataset yang tepat** — bukan synthetic noise yang tidak relevan dengan foto nyata
3. **Terbukti di benchmark foto nyata** — LOL, SICE, SIDD, MIT-Adobe FiveK
4. **Ada working code** — bukan sekedar paper

### Urutan Pipeline (Urutan Kritis)

```
Gambar input
    │
    ▼
[1] IAT / Retinexformer   ← HARUS PERTAMA: CSRNet gagal pada exposed images
    │
    ▼
[2] DeepWB                ← HARUS KEDUA: WB correction setelah exposure benar
    │
    ▼
[3] CSRNet                ← Professional retouch setelah exposure+WB sudah benar
    │
    ▼
[4] NAFNet / Restormer    ← Denoise/deblur pada gambar yang sudah tone-corrected
    │
    ▼
[5] ClassicFinisher       ← Micro-polish terakhir
```

**Catatan kritis:** Riset (eLIR-Net, WACV 2025) mengkonfirmasi CSRNet menghasilkan hasil kurang memuaskan pada gambar under/over-exposure. CSRNet harus menerima gambar yang exposure-nya sudah diperbaiki terlebih dahulu.

---

### Model A — IAT (Illumination Adaptive Transformer)

**Task:** Exposure correction (under/over-exposed)

| | |
|---|---|
| **Params** | ~90K |
| **VRAM** | ~0.4 MB |
| **Speed** | ~4ms/image batch, ~150ms single image (GPU warm-up overhead) |
| **Dataset** | LOL v1/v2, SICE, MIT-Adobe FiveK |
| **Trigger** | `40 ≤ mean_lum < 60` atau `mean_lum > 190` |
| **Repo** | `github.com/cuiziteng/Illumination-Adaptive-Transformer` |
| **Weights** | `IAT_enhance/weights/exposure.pth` (in repo) |
| **Paper** | BMVC 2022 |

**Installation & Usage:**
```python
# Setup (dalam single cell)
import subprocess
subprocess.run(["git", "clone",
    "https://github.com/cuiziteng/Illumination-Adaptive-Transformer",
    "/content/IAT"], check=True)

import sys
sys.path.insert(0, "/content/IAT")
from IAT_enhance.model.IAT import IAT

# Load model (sekali, simpan di dict)
iat_model = IAT().cuda().eval()
iat_model.load_state_dict(
    torch.load("/content/IAT/IAT_enhance/weights/exposure.pth",
               map_location="cuda")
)

# Inference
# input: torch.Tensor [1, 3, H, W], dtype=float32, range [0, 1]
with torch.no_grad():
    enhanced, _, _ = iat_model(img_tensor)  # output same shape, range [0,1]
```

**Best practice:**
- Input harus di-normalize ke `[0, 1]` — bukan `[0, 255]`
- `enhanced` adalah output langsung, tidak perlu clamp
- Untuk over-exposure: gunakan weight `LOL_v2_synthetic.pth` yang lebih stabil
- Warm-up: jalankan dummy forward pass setelah load (`torch.zeros(1,3,64,64).cuda()`) untuk memastikan GPU ready

---

### Model B — Retinexformer

**Task:** Severe low-light (mean_lum < 40) — IAT alone insufficient

| | |
|---|---|
| **Params** | ~1.6M |
| **VRAM** | ~6.4 MB (inference peak ~2GB untuk 4K image) |
| **Benchmark** | NTIRE 2024 Winner, NTIRE 2025 Winner, NTIRE 2026 Winner |
| **Dataset** | LOL v1/v2, SDSD, MIT-Adobe FiveK |
| **Trigger** | `mean_lum < 40` — menggantikan IAT sepenuhnya |
| **Repo** | `github.com/caiyuanhao1998/Retinexformer` |
| **Weights** | HuggingFace Hub + Google Drive (di README) |
| **Paper** | ICCV 2023 |

**Installation & Usage:**
```python
subprocess.run(["pip", "install", "basicsr", "-q"])
subprocess.run(["git", "clone",
    "https://github.com/caiyuanhao1998/Retinexformer",
    "/content/Retinexformer"], check=True)

sys.path.insert(0, "/content/Retinexformer")
from basicsr.models.archs.Retinexformer_arch import Retinexformer

retinex_model = Retinexformer(
    in_channels=3, out_channels=3, n_feat=40, stage=1,
    num_blocks=[1, 2, 2]
).cuda().eval()
retinex_model.load_state_dict(
    torch.load("/content/weights/LOL_v1.pth",
               map_location="cuda")["params"]
)

# Inference
# PENTING: Retinexformer sensitif terhadap padding
h, w = img_tensor.shape[2], img_tensor.shape[3]
pad_h = (4 - h % 4) % 4  # must be divisible by 4
pad_w = (4 - w % 4) % 4
padded = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
with torch.no_grad():
    out = retinex_model(padded)
out = out[:, :, :h, :w]  # crop padding back
```

**Best practice:**
- Input resolution harus habis dibagi 4 — selalu pad dengan `reflect` mode
- Untuk foto > 1500px sisi terpanjang: tiling dengan overlap 64px, blend dengan gaussian feathering
- Tile size yang aman di T4: `512×512` dengan overlap `64px`

---

### Model C — Deep White Balance (Afifi & Brown)

**Task:** White balance correction pada sRGB images

| | |
|---|---|
| **Params** | ~14M |
| **VRAM** | ~55 MB |
| **Output** | 3 hasil (AWB, indoor, shade) — pilih yang paling mendekati neutral |
| **Dataset** | Rendered WB Dataset (~65K sRGB images, berbagai color temperature) |
| **Trigger** | Gray World deviation > 8.0 |
| **Repo** | `github.com/mahmoudnafifi/Deep_White_Balance` |
| **Weights** | Dalam repo (`models/net_G.pth`) |
| **Paper** | CVPR 2020 |

**Installation & Usage:**
```python
subprocess.run(["git", "clone",
    "https://github.com/mahmoudnafifi/Deep_White_Balance",
    "/content/DeepWB"], check=True)

sys.path.insert(0, "/content/DeepWB")
from arch import deep_wb_model

wb_model = deep_wb_model.DeepWBNet().cuda().eval()
wb_model.load_state_dict(
    torch.load("/content/DeepWB/models/net_G.pth",
               map_location="cuda")
)

# Inference
with torch.no_grad():
    out_awb, out_indoor, out_shade = wb_model(img_tensor)
    # out_* shapes: [1, 3, H, W], range [0, 1]

# Pilih output yang paling mendekati gray-world neutral
def wb_neutrality_score(tensor):
    """Semakin kecil semakin neutral (semakin bagus untuk AWB)"""
    arr = tensor.squeeze().permute(1,2,0).cpu().numpy()
    r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
    return abs(r-g) + abs(g-b) + abs(r-b)

scores = {
    "AWB":    wb_neutrality_score(out_awb),
    "indoor": wb_neutrality_score(out_indoor),
    "shade":  wb_neutrality_score(out_shade),
}
best_key = min(scores, key=scores.get)
best_out = {"AWB": out_awb, "indoor": out_indoor, "shade": out_shade}[best_key]
# → log: f"WB scores: {scores} → selected: {best_key}"
```

**Best practice:**
- Selalu log semua 3 score — membantu debug jika hasil WB tidak sesuai ekspektasi
- Jika semua score hampir sama (< 2.0 perbedaan), WB tidak perlu diaplikasikan (already neutral)
- Input image di-resize ke 256×256 untuk inference, output di-upsample via delta ke resolusi asli

---

### Model D — CSRNet (Conditional Sequential Retouching Network)

**Task:** Global professional photo retouching — color harmony + tonal balance

| | |
|---|---|
| **Params** | **~37K** — lebih kecil dari gambar thumbnail 200×200px |
| **VRAM** | ~0.2 MB |
| **Speed** | ~10–15ms/image |
| **Dataset** | **MIT-Adobe FiveK** — 5000 foto, masing-masing diretouching 5 professional photographer |
| **Trigger** | **Selalu dijalankan** setelah IAT + DeepWB (kecuali `no_correction_needed`) |
| **Repo** | `github.com/hejingwenhejingwen/CSRNet` |
| **Weights** | `experiments/pretrain_models/csrnet.pth` (dalam repo) |
| **Paper** | ECCV 2020 |

**Installation & Usage:**
```python
subprocess.run(["git", "clone",
    "https://github.com/hejingwenhejingwen/CSRNet",
    "/content/CSRNet"], check=True)

# CSRNet menggunakan MMSR framework — butuh basicsr
sys.path.insert(0, "/content/CSRNet")
import torch.nn as nn

# CSRNet architecture: BaseNet (6 conv layers, 1x1) + ConditionNet
# Load via torch.load langsung karena arsitektur simpel
from codes.models.archs.CSRNet_arch import CSRNet as CSRNetArch

csrnet_model = CSRNetArch().cuda().eval()
csrnet_model.load_state_dict(
    torch.load("/content/CSRNet/experiments/pretrain_models/csrnet.pth",
               map_location="cuda")
)

# Inference
# PENTING: CSRNet didesain untuk global retouching
# Input: [1, 3, H, W], range [0, 1]
# CSRNet bekerja pixel-independent (1×1 conv) — tidak perlu resize
with torch.no_grad():
    retouched = csrnet_model(img_tensor)
```

**Best practice:**
- **Harus jalan SETELAH IAT/Retinexformer** — CSRNet menghasilkan hasil buruk pada gambar under/over-exposed
- Karena 1×1 convolution, CSRNet tidak perlu resize — bisa langsung di full resolution
- Strength control: `result = alpha * retouched + (1 - alpha) * original` dengan `alpha=0.85` untuk hasil yang tidak terlalu agresif
- Log sebelum/sesudah dengan mean_lum dan contrast_std untuk audit

---

### Model E — NAFNet (Nonlinear Activation Free Network)

**Task:** Denoising (real camera noise) + Deblurring (motion blur, JPEG artifact)

| | |
|---|---|
| **Params** | ~67M (width64) |
| **VRAM** | ~64 MB per model (2 model: SIDD + REDS = ~128 MB) |
| **Denoising** | **40.30 dB PSNR** pada SIDD — SOTA dengan <50% compute |
| **Deblurring** | **33.69 dB PSNR** pada GoPro — SOTA dengan 8.4% compute |
| **Trigger** | noise_flag=True → SIDD weights; blur_flag=True → REDS weights |
| **Install** | `pip install nafnetlib` — paling mudah |
| **Weights** | Auto-download saat pertama kali dipakai |
| **Paper** | ECCV 2022 (MEGVII) |

**Installation & Usage:**
```python
subprocess.run(["pip", "install", "nafnetlib", "-q"])
from nafnetlib import DenoiseProcessor, DeblurProcessor

# Load kedua processor sekaligus di awal
naf_denoise = DenoiseProcessor(
    model_id="sidd_width64",
    model_dir="/content/nafnet_weights",
    device="cuda"
)
naf_deblur = DeblurProcessor(
    model_id="reds_width64",  # juga menghapus JPEG artifact
    model_dir="/content/nafnet_weights",
    device="cuda"
)

# Inference — nafnetlib bekerja dengan PIL.Image langsung
from PIL import Image
pil_input = Image.fromarray((img_np * 255).astype(np.uint8))

if noise_flag:
    pil_output = naf_denoise.process(pil_input)
    # → log: "NAFNet-SIDD applied (denoise)"
elif blur_flag:
    pil_output = naf_deblur.process(pil_input)
    # → log: "NAFNet-REDS applied (deblur + JPEG artifact removal)"

img_out = np.array(pil_output).astype(np.float32) / 255.0
```

**Best practice:**
- `nafnetlib` auto-downloads weights ke `model_dir` — set path permanen agar tidak re-download tiap session
- Untuk gambar yang NOISY DAN BLURRY sekaligus → gunakan Restormer, bukan NAFNet (lihat Model F)
- NAFNet-REDS secara bonus menghapus JPEG compression artifact — berguna untuk foto yang di-share ulang berkali-kali
- Input ke NAFNet **setelah** exposure dan WB sudah diperbaiki, karena denoising pada gambar gelap akan mengangkat noise bersamaan dengan brightness

---

### Model F — Restormer

**Task:** Heavy combined degradation (noisy + blurry sekaligus)

| | |
|---|---|
| **Params** | ~26M |
| **VRAM** | ~182 MB |
| **Benchmark** | NTIRE 2025 Denoising Challenge backbone |
| **Tasks** | Real denoising + Motion deblurring + Defocus + Deraining |
| **Trigger** | `noise_flag AND blur_flag` — **kombinasi** degradasi |
| **Default** | `load_restormer=False` — opt-in saja |
| **Repo** | `github.com/swz30/Restormer` |
| **Weights** | Google Drive (link di README), task `Real_Denoising` |
| **Paper** | CVPR 2022 Oral |

**Installation & Usage:**
```python
subprocess.run(["git", "clone",
    "https://github.com/swz30/Restormer",
    "/content/Restormer"], check=True)
subprocess.run(["pip", "install", "basicsr", "-q"])

sys.path.insert(0, "/content/Restormer")
from basicsr.models.archs.restormer_arch import Restormer

restormer_model = Restormer(
    inp_channels=3, out_channels=3,
    dim=48, num_blocks=[4,6,6,8],
    num_refinement_blocks=4,
    heads=[1,2,4,8],
    ffn_expansion_factor=2.66, bias=False,
    LayerNorm_type="BiasFree",
    dual_pixel_task=False
).cuda().eval()

checkpoint = torch.load(
    "/content/weights/real_denoising.pth", map_location="cuda"
)
restormer_model.load_state_dict(checkpoint["params"])

# Inference
# Restormer menggunakan demo.py sebagai entry point
# Untuk inline usage:
with torch.no_grad():
    restored = restormer_model(img_tensor)
```

**Best practice:**
- Load Restormer **hanya jika `CONFIG["load_restormer"] = True`** — 182MB VRAM, tidak semua folder foto butuh ini
- Demo API resmi: `python demo.py --task Real_Denoising --input_dir . --result_dir ./out/` — alternatif yang lebih aman
- Untuk foto > 2000px: patch-based inference dengan sliding window, overlap 128px
- Download weights via `gdown` dari link di README, bukan dari clone repo (repo tidak include weights)

---

### Classic Finisher (No ML — Post-processing)

**Task:** Micro-polish terakhir setelah semua ML selesai

**Tidak aktif secara default** — hanya jalan jika metrik tertentu masih di bawah threshold setelah ML:

```python
import cv2
import numpy as np

def classic_finisher(img_np, triage_post_ml, config):
    result = img_np.copy()
    steps = []

    # 1. CLAHE — hanya jika contrast masih flat post-ML
    if triage_post_ml["contrast_std"] < config["thresholds"]["flat_contrast_std"]:
        lab = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        steps.append("CLAHE (L channel, clipLimit=2.0)")

    # 2. Unsharp Mask — hanya jika blur masih terdeteksi post-ML
    if triage_post_ml["sharpness_var"] < config["thresholds"]["blur_threshold"]:
        blurred = cv2.GaussianBlur((result*255).astype(np.uint8), (0,0), 2.0)
        sharpened = cv2.addWeighted(
            (result*255).astype(np.uint8), 1.5, blurred, -0.5, 0
        )
        result = sharpened.astype(np.float32) / 255.0
        steps.append("Unsharp Mask (sigma=2.0, strength=1.5)")

    return result, steps
```

---

## VRAM Budget (T4 = 16GB)

| Model | VRAM Loaded | Inference Peak (4K) |
|---|---|---|
| IAT | 0.4 MB | ~200 MB |
| CSRNet | 0.2 MB | ~50 MB |
| DeepWB | 55 MB | ~800 MB |
| NAFNet × 2 | 128 MB | ~1.5 GB |
| Retinexformer | 6.4 MB | ~2.0 GB |
| Restormer (opt) | 182 MB | ~3.0 GB |
| **Total loaded** | **~372 MB** | |
| **T4 available** | **16,384 MB** | |
| **Worst-case peak** | | ~3.0 GB |
| **Headroom** | | **~13 GB** ✅ |

---

## Konfigurasi (Edit sebelum Run)

```python
# ── PHOTON CONFIG ─────────────────────────────────────────────
# Edit bagian ini, lalu Shift+Enter untuk menjalankan satu cell
# ──────────────────────────────────────────────────────────────
CONFIG = {
    # ── INPUT ──────────────────────────────────────────────────
    "source_gdrive_url": "https://drive.google.com/drive/folders/MASUKKAN_ID_FOLDER",
    "recursive_folder_scan":   False,  # True = traverse subfolders
    # ── OUTPUT ─────────────────────────────────────────────────
    "target_mydrive_path": "MyDrive/photos/enhanced",
    "result_folder_name":  "PHOTON Enhanced",  # nama folder hasil di Drive
    # ── TRIAGE THRESHOLDS ──────────────────────────────────────
    "thresholds": {
        "dark_lum":              60,    # lum < ini → IAT
        "severe_dark_lum":       40,    # lum < ini → Retinexformer (skip IAT)
        "bright_lum":           190,    # lum > ini → IAT
        "flat_contrast_std":     45,    # std < ini → ClassicFinisher CLAHE
        "wb_cast_threshold":     8.0,   # gray-world deviation → DeepWB
        "noise_threshold":       15.0,  # local variance smooth-patch → NAFNet-SIDD
        "blur_threshold":       150.0,  # Laplacian variance → NAFNet-REDS
        "restormer_threshold":   True,  # noisy AND blurry → Restormer
    },
    # ── RESOLUTION ─────────────────────────────────────────────
    "inference_resolution":    512,    # px sisi terpanjang untuk ML inference
    "tiling_threshold":       1500,    # px — tile-based untuk model di atas ini
    "tile_overlap":             64,    # px overlap antar tile
    # ── OUTPUT ─────────────────────────────────────────────────
    "jpeg_output_quality":      95,    # 85–98 direkomendasikan
    "csrnet_strength":        0.85,    # 0.0–1.0 blend strength CSRNet
    "save_thumbnails":        False,   # True = simpan before/after thumbnail 512px
    # ── RUNTIME ────────────────────────────────────────────────
    "load_restormer":         False,   # True hanya jika foto banyak yang noisy+blurry
    "nafnet_model_dir":    "/content/nafnet_weights",  # persistent weight cache
    "weights_dir":         "/content/photon_weights",  # semua model weights
    # ── SESSION ────────────────────────────────────────────────
    "resume_enabled":         True,    # fingerprint-based resume
    "session_upload_to_drive": True,   # upload session JSON ke Drive untuk cross-runtime
    # ── COLAB ──────────────────────────────────────────────────
    "idle_shutdown_minutes":      5,   # auto-shutdown setelah N menit idle
    "rate_limit_sleep":         0.5,   # detik antar Drive API calls
    "max_retries":                3,   # retry download/upload
    # ── MODE ───────────────────────────────────────────────────
    "fallback_on_cpu":         True,   # CPU mode: skip IAT/Retinex/NAFNet, WB only
}
```

---

## User Flow

```mermaid
flowchart TD
    A([User buka Colab]) --> B[Edit CONFIG\nsource_gdrive_url + target_mydrive_path]
    B --> C[Shift+Enter — satu cell\n!curl runner.py | python3]
    C --> D[Layer 1: runner.py\nClone repo + install deps + load secrets]
    D --> E[Layer 2: start.py\nDetect GPU → set PHOTON_MODE]
    E --> F[Layer 3: main.py\nImport + Drive Auth (OAuth)]
    F --> G[Load session file\nResume fingerprint check]
    G --> H[Load semua model ke VRAM\nLog: model sizes + VRAM usage]

    H --> I[Mount Google Drive\nList images via Drive API v3]
    I --> J{Untuk setiap gambar\ndengan progress bar}

    J --> K{Fingerprint match\nsudah diproses?}
    K -- Ya --> L[Log: RESUME skip\n0 detik]
    K -- Tidak --> M[Download via Drive API\nchunk-based + retry]

    M --> N[Classic CV Triage\n< 5ms — print hasil lengkap]
    N --> O{Semua normal?}
    O -- Ya --> P[Log: no_correction_needed\nCopy → target]

    O -- Tidak --> Q{lum < 40?\nsevere dark}
    Q -- Ya --> R[Retinexformer\nLog: trigger reason + before/after]
    Q -- Tidak --> S{lum < 60 atau > 190?}
    S -- Ya --> T[IAT\nLog: trigger reason + before/after]
    S -- Tidak --> U

    R --> U[DeepWB?\nwb_dev > 8.0]
    T --> U
    U -- Ya --> V[DeepWB\nLog: 3 WB scores + selection]
    U -- Tidak --> W

    V --> W[CSRNet\nAlways — Log: before/after metrics]

    W --> X{noisy AND blurry?}
    X -- Ya --> Y[Restormer\nLog: combined degradation]
    X -- Tidak --> Z{noisy only?}
    Z -- Ya --> AA[NAFNet-SIDD\nLog: noise reduction]
    Z -- Tidak --> AB{blurry only?}
    AB -- Ya --> AC[NAFNet-REDS\nLog: deblur + artifact removal]

    Y --> AD
    AA --> AD
    AC --> AD
    AB -- Tidak --> AD{Post-ML triage\nstill flat/blurry?}
    AD -- Ya --> AE[ClassicFinisher\nLog: CLAHE/Unsharp trigger]
    AD -- Tidak --> AF

    AE --> AF[Upsample delta ke resolusi original\nguidance filter]
    AF --> AG[Save _photon.jpg\nLog: filepath + size]
    AG --> AH[Upload to Drive\nWrite _photon_log.json]
    AH --> AI[Mark fingerprint processed\nUpdate session file]
    AI --> AJ[Idle monitor reset]
    AJ --> J

    J -- Selesai --> AK[Print session summary\nWrite session JSON]
    AK --> AL[Upload session to Drive\nUpload log to result folder]
    AL --> AM{Idle monitor active?}
    AM -- Ya --> AN[Auto-shutdown after 5 min idle]
    AM -- Tidak --> AO([Done — semua foto di target folder])
```

---

## Logger Design

```python
import sys
from datetime import datetime
import torch

class PhotonLogger:
    ICONS = {
        "START":  "🚀", "FOLDER": "📁", "DOWNLOAD": "📥",
        "MODEL":  "🤖", "VRAM":   "📊", "SEP":      "══",
        "IMAGE":  "🖼️", "TRIAGE": "🔍", "ISSUE":    "⚠️",
        "SKIP":   "⏩", "STEP":   "🔧", "UPSAMPLE": "📐",
        "SAVE":   "💾", "LOG":    "📝", "DONE":     "✅",
        "FAIL":   "❌", "COPY":   "📋", "SUMMARY":  "📊",
        "WARN":   "⚠️", "RESUME": "🔄", "DRIVE":    "☁️",
        "IDLE":   "⏰",
    }

    def __init__(self):
        self.session_start = datetime.now()

    def _ts(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _vram(self):
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"{used:.1f}GB/{total:.0f}GB"
        return "CPU"

    def p(self, icon_key, msg, indent=0, **kv):
        prefix = "   " * indent
        ts = self._ts()
        icon = self.ICONS.get(icon_key, "  ")
        print(f"[{ts}] {prefix}{icon} {msg}", flush=True)
        for k, v in kv.items():
            print(f"[{ts}]    {prefix}{k}: {v}", flush=True)

    def sep(self):
        print(f"[{self._ts()}] {'═'*50}", flush=True)

    def model_loaded(self, name, params, vram_mb):
        self.p("MODEL", f"{name:<16} loaded  "
               f"({params/1e6:.2f}M params, {vram_mb:.1f} MB)",
               indent=1)

    def triage(self, metrics):
        self.p("TRIAGE",
               f"lum={metrics['mean_lum']:.1f}  "
               f"std={metrics['contrast_std']:.1f}  "
               f"wb_dev={metrics['wb_dev']:.1f}({metrics['wb_cast']})",
               indent=1)
        self.p("TRIAGE",
               f"noise={metrics['noise_flag']}  "
               f"blur={metrics['blur_flag']}  "
               f"blown={metrics['blown_pct']:.1f}%",
               indent=1)

    def step(self, n, model, reason, before, after, ms):
        self.p("STEP", f"STEP {n}  {model} — {reason}", indent=1)
        for k in before:
            b, a = before[k], after.get(k, "?")
            self.p("STEP", f"{k}: {b} → {a}", indent=2)
        self.p("STEP", f"⏱️  {ms}ms", indent=2)

    def skip(self, model, reason):
        self.p("SKIP", f"{model:<16} {reason}", indent=1)

    def resume(self, filename, fingerprint):
        self.p("RESUME", f"SKIP    {filename} (fingerprint={fingerprint[:8]}...)", indent=1)

    def drive_op(self, op, detail):
        self.p("DRIVE", f"{op}  {detail}", indent=1)

    def idle(self, msg):
        self.p("IDLE", msg, indent=1)
```

---

## Non-Functional Requirements

### Performance Targets (T4)

| Skenario | Target |
|---|---|
| Satu gambar — semua model aktif | < 30 detik |
| Satu gambar — IAT + WB + CSRNet | < 5 detik |
| Satu gambar — no correction | < 0.5 detik |
| Batch 50 foto (mixed) | < 20 menit |
| Peak VRAM | < 6 GB |
| Resolusi output = resolusi input | **100% — tanpa eksepsi** |
| Foto yang sudah diproses (resume) | 0 detik (skip via fingerprint) |
| Drive download (10MB file) | < 5 detik (chunk-based) |
| Colab startup (runner.py) | < 30 detik (clone + install + launch) |
| GPU detection | < 2 detik |
| Idle shutdown | < 30 detik setelah timeout |

### Graceful Degradation

Jika satu model gagal pada gambar tertentu:
1. Log error lengkap (traceback) dengan `❌ FAIL` prefix
2. Fallback ke model berikutnya dalam chain
3. Jika semua ML gagal → Classic CV fallback
4. **Selalu output file** — minimal copy original + log error
5. Counter `failed` di session summary dengan link ke log
6. Mark fingerprint sebagai `fail` — tidak di-skip di run berikutnya

**Transient vs Critical error classification** (adapted dari TTB):
```python
TRANSIENT_ERRORS = (ConnectionError, TimeoutError, SSLLError, OSError)
CRITICAL_ERRORS = (MemoryError, RuntimeError, ValueError)

def classify_error(e: Exception) -> str:
    if isinstance(e, TRANSIENT_ERRORS):
        return "transient"  # retry with backoff
    elif isinstance(e, CRITICAL_ERRORS):
        return "critical"   # skip image, log, continue
    return "unknown"        # treat as critical
```

---

## Module Breakdown

| # | Nama | Complexity | Isi |
|---|---|---|---|
| 0 | runner.py | S | Colab bootstrap — clone, install, auth, launch |
| 1 | start.py | S | GPU detection, mode setting, subprocess launch |
| 2 | GDriveI/O | M | Drive API v3 — list, download, upload, create folder, retry |
| 3 | SessionManager | M | Fingerprint resume, session JSON, cross-upload |
| 4 | PhotonLogger | S | Class logger dengan format di atas |
| 5 | TriageEngine | M | Classic CV metrics (NumPy/OpenCV) |
| 6 | ModelLoader | M | Load semua model ke VRAM, log sizes, warm-up |
| 7 | IATWrapper | S | Exposure correction + warm-up |
| 8 | RetinexWrapper | M | Tiling untuk foto besar |
| 9 | DeepWBWrapper | S | 3-output WB scoring |
| 10 | CSRNetWrapper | S | Blend dengan `csrnet_strength` |
| 11 | NAFNetWrapper | M | Dual model (SIDD + REDS), nafnetlib |
| 12 | RestormerWrapper | M | Optional load, patch-based inference |
| 13 | ClassicFinisher | S | CLAHE + Unsharp, post-ML only |
| 14 | ResolutionPreserver | L | Delta computation + guided filter upsample |
| 15 | Orchestrator | M | Chain routing berdasarkan triage |
| 16 | BatchRunner | S | Loop + progress bar + session summary |
| 17 | IdleMonitor | S | Auto-shutdown setelah idle (adapted from TTB) |
| 18 | HardwareDetector | S | GPU/CPU detection, mode switching |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CSRNet dijalankan sebelum exposure fix | **High** | High | **Hard-coded order** dalam orchestrator — bukan configurable |
| T4 OOM pada gambar 4K + Restormer | Medium | High | `load_restormer=False` default; tiled inference |
| DeepWB memilih WB setting yang salah | Medium | Low | Log 3 scores — user bisa inspect manual |
| Drive API quota habis (20K req/100s) | Medium | Medium | Rate limiting `sleep(0.5)`, retry dengan backoff, gdown fallback |
| Drive OAuth expired / auth failure | Low | High | Colab auth auto-refresh; gdown as fallback |
| NAFNet weights tidak ter-cache antar session | Medium | Low | Persistent dir (`/content/nafnet_weights`) + session resume |
| CSRNet agresif pada portrait (skin tone) | Low | Medium | `csrnet_strength=0.85` default (tidak full) |
| Colab session timeout mid-batch | Medium | Low | Fingerprint resume — skip yang sudah diproses |
| Idle billing waste setelah selesai | Medium | Medium | Idle monitor auto-shutdown setelah 5 min idle |
| Subfolder structure tidak ter-preserve | Low | Low | `recursive_folder_scan` option + flatten output |

---

## Out of Scope (v4.0)

- No UI — Colab cell adalah interface (Gradio optional di v5)
- No upscaling — resolusi original adalah ceiling dan floor
- No face restoration (GFPGAN) — v2
- No video
- No RAW files — sRGB only
- No model fine-tuning
- No Telegram bot — dedicated Colab runner (bukan bot seperti TTB)
- No cloud API inference — semua lokal di T4

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-05-28 | Initial PRD |
| 2.0 | 2026-05-28 | Tech stack validated: CSEC→CSRNet, NAFNet+Restormer added |
| 3.0 | 2026-05-28 | Single-cell mode; full print log design; concrete code examples per model; CSRNet ordering constraint validated (must run after exposure fix); NAFNet nafnetlib API; Restormer patch inference; Classic Finisher demoted to post-ML micro-polish |
| 4.0 | 2026-05-28 | **Architecture upgrade**: 3-layer runner (runner.py→start.py→main.py) adapted from TTB; **Drive API v3** replaces gdown (from simple.md) — OAuth, chunk-based I/O, retry, rate limiting, md5 fingerprint; **Session resume** via fingerprint-based skip; **Colab integration**: idle monitor auto-shutdown, GPU detection mode switching, Colab Secrets; **New features**: recursive subfolder scan, before/after thumbnails, cross-runtime session upload; Config expanded with 12 new keys; Module breakdown updated to 19 modules; Risk register updated with Drive API and idle billing risks |