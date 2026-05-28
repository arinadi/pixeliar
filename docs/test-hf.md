```python
#@title 🧪 HF Router Test Cell
# ─────────────────────────────────────────────
# ⚙️  KONFIGURASI
# ─────────────────────────────────────────────
HF_TOKEN = ""  #@param {type:"string"}
HF_MODEL = "Qwen/Qwen3.6-35B-A3B"  #@param ["Qwen/Qwen3.6-35B-A3B", "google/gemma-3-12b-it", "google/gemma-3-4b-it"]

# ─────────────────────────────────────────────
# 📦  INSTALL
# ─────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai", "Pillow"], check=True)

import os, io, base64, json
from openai import OpenAI
from PIL import Image

if not HF_TOKEN:
    raise ValueError("❌ Isi HF_TOKEN dulu!")

# ─────────────────────────────────────────────
# 🔌  INIT
# ─────────────────────────────────────────────
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# ─────────────────────────────────────────────
# 🖼️  Buat test image (gradient sederhana)
# ─────────────────────────────────────────────
img = Image.new("RGB", (256, 256), color=(100, 80, 120))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=75)
data_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ─────────────────────────────────────────────
# 🧪  TEST 1: Text only
# ─────────────────────────────────────────────
print("=" * 50)
print("TEST 1: Text-only chat")
print("=" * 50)
try:
    r = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
    )
    print(f"✅ SUCCESS: {r.choices[0].message.content}")
    print(f"Model: {r.model}")
    print(f"Usage: {r.usage}")
except Exception as e:
    print(f"❌ FAIL: {e}")

# ─────────────────────────────────────────────
# 🧪  TEST 2: Image + text (base64)
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 2: Image (base64) + text")
print("=" * 50)
try:
    r = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
        max_tokens=50,
    )
    print(f"✅ SUCCESS: {r.choices[0].message.content}")
    print(f"Model: {r.model}")
except Exception as e:
    print(f"❌ FAIL: {e}")

# ─────────────────────────────────────────────
# 🧪  TEST 3: Image via URL (bukan base64)
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 3: Image via HTTP URL")
print("=" * 50)
try:
    r = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": "https://placehold.co/256x256/604070/white"}},
            ],
        }],
        max_tokens=50,
    )
    print(f"✅ SUCCESS: {r.choices[0].message.content}")
except Exception as e:
    print(f"❌ FAIL: {e}")

# ─────────────────────────────────────────────
# 🧪  TEST 4: Gemma 3 sebagai perbandingan
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 4: Gemma 3 (perbandingan)")
print("=" * 50)
try:
    r = client.chat.completions.create(
        model="google/gemma-3-4b-it",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
        max_tokens=50,
    )
    print(f"✅ SUCCESS: {r.choices[0].message.content}")
except Exception as e:
    print(f"❌ FAIL: {e}")

# ─────────────────────────────────────────────
# 🧪  TEST 5: Image + high max_tokens + check reasoning
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 5: Image base64 + max_tokens=500 + reasoning check")
print("=" * 50)
try:
    r = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
        max_tokens=500,
    )
    msg = r.choices[0].message
    print(f"content:           '{msg.content}'")
    print(f"reasoning_content: '{getattr(msg, 'reasoning_content', 'N/A')}'")
    print(f"Full message:      {msg}")
except Exception as e:
    print(f"❌ FAIL: {e}")

print("\n" + "=" * 50)
print("🧪 Test selesai — cek hasil di atas")
print("=" * 50)
```
