# PRD: Auto Color Grade — Web Edition

## Ringkasan

Porting script Colab `simple.md` ke **SPA client-side** (Vite + vanilla TS). Tidak ada backend/server — semua processing jalan di browser.

## Kelayakan Teknis

| Komponen Colab | Pengganti Browser | Status |
|---|---|---|
| `cv2` (OpenCV Python) | `opencv.js` (WASM, ~8MB) | ✅ Feasible |
| Google Drive API | `<input type="file">` + `<a download>` | ✅ Lebih simpel |
| `google-genai` (Gemini) | `@google/genai` JS SDK | ✅ |
| `Pillow` (resize) | Canvas API / `opencv.js` | ✅ |
| `tqdm` (progress) | Custom progress bar UI | ✅ |

### OpenCV Pipeline — 1:1 Mapping

| Operasi cv2 | opencv.js equivalent |
|---|---|
| `cv2.imread()` | `cv.imread(canvasElement)` |
| `cv2.imwrite()` | `cv.imwrite(canvasElement, mat)` → canvas.toBlob() |
| `cv2.cvtColor(BGR→HSV)` | `cv.cvtColor(src, dst, cv.COLOR_BGR2HSV)` |
| `cv2.cvtColor(HSV→BGR)` | `cv.cvtColor(src, dst, cv.COLOR_HSV2BGR)` |
| `cv2.GaussianBlur()` | `cv.GaussianBlur(src, dst, size, sigmaX)` |
| Pixel math (+, ×, clip) | `mat.data` Uint8Array langsung |

Semua operasi di `apply_corrections()` dapat direproduksi secara identik karena opencv.js adalah kompilasi WASM dari source C++ yang sama.

## Tech Stack

- **Vite** — bundler dev/build
- **TypeScript** — type safety
- **opencv.js** — image processing (WASM)
- **@google/genai** — Gemini API client
- **JSZip** — zip hasil batch
- **Tailwind CSS** — styling (opsional, bisa vanilla CSS)

## Arsitektur

```
Browser (SPA)
├── src/
│   ├── main.ts          — entry, init opencv.js + gemini
│   ├── ui.ts            — DOM: file input, progress, preview
│   ├── analyzer.ts      — panggil Gemini, parse JSON parameter
│   ├── processor.ts     — apply_corrections() via opencv.js
│   └── downloader.ts    — simpan per file atau zip batch
└── public/
    └── opencv.js        — static WASM asset
```

## User Flow

1. Buka halaman → loading opencv.js (sekali, di-cache)
2. Input API key Gemini (disimpan di localStorage)
3. Pilih foto (multiple, drag & drop atau file picker)
4. Preview thumbnail + parameter hasil analisis per foto
5. Klik "Process All" → batch processing dengan progress bar
6. Download hasil per foto atau zip semua

## Perbedaan dengan Colab

| | Colab | Web |
|---|---|---|
| Sumber gambar | Google Drive | Local file system |
| Hasil editan | Upload ke Drive | Download lokal / zip |
| RPM control | `time.sleep(15)` | User-side rate limiter |
| API key | Hardcoded | Input user → localStorage |
| Runtime | Google Colab | Browser tab |

## Risiko

1. **opencv.js load time** — 8MB WASM pertama kali, mitigasi: loading screen + service worker cache
2. **BYOK (Bring Your Own Key)** — user masukkan Gemini API key sendiri, disimpan di localStorage browser. Key tidak pernah dikirim ke server manapun selain API Google Gemini langsung. Tidak ada backend perantara.
3. **Memory untuk batch besar** — 177 foto × OpenCV Mat bisa OOM, mitigasi: proses sekuensial, Mat di-release tiap iterasi

## Estimasi Waktu

| Task | Estimasi |
|---|---|
| Setup Vite + TS + opencv.js | 1 jam |
| Port `apply_corrections()` ke opencv.js | 3 jam |
| Integrasi Gemini JS SDK | 1 jam |
| UI (file input, progress, preview) | 2 jam |
| Batch processing + download | 1 jam |
| Polish & error handling | 1 jam |
| **Total** | **~9 jam** |
