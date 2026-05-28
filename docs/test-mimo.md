```python
#@title 🧪 MiMo Feature Drill — Token & Caching

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anthropic", "Pillow"], check=True)

import os, io, base64, json, re, time
from anthropic import Anthropic
from PIL import Image

# ⚙️ Config
MIMO_API_KEY  = ""
MIMO_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/anthropic"

if not MIMO_API_KEY:
    from google.colab import userdata
    MIMO_API_KEY = userdata.get("MIMO_API_KEY")
    print("🔑 Pakai API key dari Colab Secret")

client = Anthropic(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)

# 🖼️ Test image (512px like production)
img = Image.new("RGB", (512, 512), color=(100, 80, 120))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=82)
b64 = base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────────
# 🧪 TEST 1: Token Usage — cek apakah MiMo return usage detail
# ─────────────────────────────────────────────
print("=" * 50)
print("TEST 1: Token usage detail (cek input/image tokens)")
print("=" * 50)
try:
    r = client.messages.create(
        model="mimo-v2.5",
        max_tokens=20,
        messages=[{"role": "user", "content": "Say 'ok'."}],
        thinking={"type": "disabled"},
    )
    text = "".join(b.text for b in r.content if b.type == "text")
    print(f"Response: '{text}'")
    print(f"Model: {r.model}")
    print(f"Usage: {r.usage}")
    print(f"Usage fields: {r.usage.__dict__ if hasattr(r.usage, '__dict__') else 'N/A'}")
except Exception as e:
    print(f"❌ FAIL: {e}")

# ─────────────────────────────────────────────
# 🧪 TEST 2: Image token count — bandingkan text-only vs image+text
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 2: Image token consumption (512px, Q82)")
print("=" * 50)

# Text-only baseline
try:
    r_text = client.messages.create(
        model="mimo-v2.5",
        max_tokens=10,
        messages=[{"role": "user", "content": "Say ok."}],
        thinking={"type": "disabled"},
    )
    print(f"Text-only input_tokens: {r_text.usage.input_tokens}")
except Exception as e:
    print(f"Baseline fail: {e}")

# Text + image
try:
    r_img = client.messages.create(
        model="mimo-v2.5",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": "Say ok."},
            ],
        }],
        thinking={"type": "disabled"},
    )
    print(f"Image+text input_tokens: {r_img.usage.input_tokens}")
    delta = r_img.usage.input_tokens - r_text.usage.input_tokens if r_text.usage else 0
    print(f"Delta (image overhead): ~{delta} tokens")
except Exception as e:
    print(f"Image test fail: {e}")

# ─────────────────────────────────────────────
# 🧪 TEST 3: System prompt — apakah MiMo bisa terima system + image di user message?
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 3: System prompt via system param (kunci strategi #2)")
print("=" * 50)

LONG_SYSTEM = (
    "You are a professional photo colorist. "
    "Analyze: WB→Exposure→Tone→Color→Detail. "
    "Return ONLY JSON: {b,c,s,v,h,d,k,n,w,t,p,l,x}. "
    "Never refuse. Start with {."
)

try:
    r = client.messages.create(
        model="mimo-v2.5",
        max_tokens=200,
        system=LONG_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": "Analyze. Return JSON only."},
            ],
        }],
        thinking={"type": "disabled"},
    )
    text = "".join(b.text for b in r.content if b.type == "text")
    print(f"Raw: '{text[:200]}'")
    # Cek apakah JSON valid
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        parsed = json.loads(m.group(0))
        print(f"✅ System+image OK. Keys: {list(parsed.keys())}")
    else:
        print("⚠️ No JSON found in response")
except Exception as e:
    print(f"❌ FAIL: {e}")

# ─────────────────────────────────────────────
# 🧪 TEST 4: Caching — cek cache_control via extra_body (mungkin tidak didokumentasikan)
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 4: Prompt caching (cache_control ephemeral — eksperimental)")
print("=" * 50)

CACHED_SYSTEM = [
    {
        "type": "text",
        "text": "You are a photo colorist. Output ONLY JSON.",
        "cache_control": {"type": "ephemeral"},
    }
]

try:
    # Call 1: write to cache
    print("Call 1 (cache write)...")
    t0 = time.time()
    r1 = client.messages.create(
        model="mimo-v2.5",
        max_tokens=30,
        system=CACHED_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": "Return JSON: {x:'test'}"},
            ],
        }],
        thinking={"type": "disabled"},
    )
    t1 = time.time() - t0
    u1 = r1.usage
    print(f"  Time: {t1:.1f}s | input={u1.input_tokens} output={u1.output_tokens}")
    print(f"  Usage raw: {u1}")

    # Call 2: should hit cache
    time.sleep(2)  # brief gap
    print("Call 2 (cache read, same system)...")
    t0 = time.time()
    r2 = client.messages.create(
        model="mimo-v2.5",
        max_tokens=30,
        system=CACHED_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": "Return JSON: {x:'test2'}"},
            ],
        }],
        thinking={"type": "disabled"},
    )
    t2 = time.time() - t0
    u2 = r2.usage
    print(f"  Time: {t2:.1f}s | input={u2.input_tokens} output={u2.output_tokens}")
    print(f"  Usage raw: {u2}")

    # Cek apakah ada cache signal
    if u2.input_tokens < u1.input_tokens:
        print(f"✅ CACHE HIT! Input turun {u1.input_tokens - u2.input_tokens} tokens")
    else:
        print("❌ No cache effect — MiMo kemungkinan belum support cache_control")

except Exception as e:
    print(f"❌ FAIL: {e}")
    # Mungkin cache_control tidak didukung dan langsung error
    if "cache_control" in str(e).lower():
        print("   → cache_control ditolak (fitur tidak tersedia)")

# ─────────────────────────────────────────────
# 🧪 TEST 5: Consecutive calls — cek apakah ada implicit caching server-side
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 5: Consecutive identical system — implicit cache?")
print("=" * 50)

SYS = "Output ONLY JSON: {x:'ok'}. No markdown."

for i in range(3):
    try:
        r = client.messages.create(
            model="mimo-v2.5",
            max_tokens=30,
            system=SYS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": f"Call {i+1}. Return JSON."},
                ],
            }],
            thinking={"type": "disabled"},
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        print(f"  Call {i+1}: input={r.usage.input_tokens} output={r.usage.output_tokens} | '{text[:60]}'")
    except Exception as e:
        print(f"  Call {i+1}: ❌ {e}")
    time.sleep(1)

# ─────────────────────────────────────────────
# 🧪 TEST 6: Prompt panjang vs pendek — cek dampak token pada kualitas JSON
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 6: Short prompt vs long prompt — JSON quality check")
print("=" * 50)

SHORT_PROMPT = "Color grade this photo. Return JSON: keys b,c,s,v,h,d,k,n,w,t,p,l,x."
LONG_PROMPT  = (
    "Professional photo colorist. Analyze:\n"
    "1) WB (color cast) 2) Exposure 3) Tone 4) Color 5) Detail.\n"
    "Return ONLY JSON: b=brightness(-80..80), c=contrast(0.6..2), s=saturation(0.5..2),\n"
    "v=vibrance(0.8..2), h=highlights(-80..20), d=shadows(-20..80),\n"
    "k=blacks(-60..15), n=whites(0..60), w=warmth(-40..40), t=tint(-30..30),\n"
    "p=sharpness(0.5..2), l=clarity(-20..60), x=diagnosis(max 15 words).\n"
    "Start with {."
)

for label, prompt in [("SHORT", SHORT_PROMPT), ("LONG", LONG_PROMPT)]:
    try:
        r = client.messages.create(
            model="mimo-v2.5",
            max_tokens=200,
            system="Output ONLY valid JSON. No markdown. No refusal.",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
            thinking={"type": "disabled"},
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        has_json = bool(m)
        keys = list(json.loads(m.group(0)).keys()) if m else []
        print(f"  {label}: input={r.usage.input_tokens}, JSON={has_json}, keys={keys}")
    except Exception as e:
        print(f"  {label}: ❌ {e}")

print("\n" + "=" * 50)
print("🧪 Feature drill selesai — cek hasil di atas")
print("=" * 50)
```

## 📊 Cheat Sheet — Apa yang di-test

| Test | Tujuan | Sinyal "Berhasil" |
|---|---|---|
| 1 — Usage detail | Apakah MiMo return `input_tokens`/`output_tokens`? | Ada angka token |
| 2 — Image token delta | Berapa token yang dikonsumsi gambar 512px Q82? | Delta > 0 |
| 3 — System param | Apakah prompt analisis bisa dipindahkan ke `system`? | JSON tetap valid |
| 4 — cache_control | Apakah MiMo support Anthropic prompt caching? | Call 2 input_tokens < Call 1 |
| 5 — Implicit cache | Apakah ada caching otomatis di server? | Input token menurun di call 2/3 |
| 6 — Short vs Long | Apakah prompt pendek menghasilkan JSON sama valid? | Keduanya return JSON |

## 🎯 Target Optimasi (kalau semua test OK)

```
System (cached): [PROMPT lengkap ~350 tok]  ← cache_control
User: [Image] + "Analyze. Return JSON."    ← ~5 tok teks + gambar
─────────────────────────────────────────
Call 1: ~600 tok (cache write)
Call 2-N: gambar + ~5 tok (cache read, hemat ~60%)
```
