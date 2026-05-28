"""
pixeliar — PhotonLogger (Module 4)
Structured print log with timestamps, icons, and VRAM tracking.
All output via print(flush=True) for real-time Colab display.
"""

from datetime import datetime


class PhotonLogger:
    ICONS = {
        "START":    "🚀", "FOLDER": "📁", "DOWNLOAD": "📥",
        "MODEL":    "🤖", "VRAM":   "📊", "SEP":      "══",
        "IMAGE":    "🖼️", "TRIAGE": "🔍", "ISSUE":    "⚠️",
        "SKIP":     "⏩", "STEP":   "🔧", "UPSAMPLE": "📐",
        "SAVE":     "💾", "LOG":    "📝", "DONE":     "✅",
        "FAIL":     "❌", "COPY":   "📋", "SUMMARY":  "📊",
        "WARN":     "⚠️", "RESUME": "🔄", "DRIVE":    "☁️",
        "IDLE":     "⏰",
    }

    def __init__(self):
        self.session_start = datetime.now()

    def _ts(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _vram(self):
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return f"{used:.1f}GB/{total:.0f}GB"
        except Exception:
            pass
        return "CPU"

    def p(self, icon_key, msg, indent=0, **kv):
        prefix = "   " * indent
        ts = self._ts()
        icon = self.ICONS.get(icon_key, "  ")
        print(f"[{ts}] {prefix}{icon} {msg}", flush=True)
        for k, v in kv.items():
            print(f"[{ts}]    {prefix}{k}: {v}", flush=True)

    def sep(self):
        print(f"[{self._ts()}] {'═' * 50}", flush=True)

    def model_loaded(self, name, params, vram_mb):
        self.p("MODEL", f"{name:<16} loaded  "
               f"({params / 1e6:.2f}M params, {vram_mb:.1f} MB)", indent=1)

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
            b = before[k]
            a = after.get(k, "?")
            self.p("STEP", f"{k}: {b} → {a}", indent=2)
        self.p("STEP", f"⏱️  {ms}ms", indent=2)

    def skip(self, model, reason):
        self.p("SKIP", f"{model:<16} {reason}", indent=1)

    def resume(self, filename, fingerprint):
        self.p("RESUME",
               f"SKIP    {filename} (fingerprint={fingerprint[:8]}...)", indent=1)

    def drive_op(self, op, detail):
        self.p("DRIVE", f"{op}  {detail}", indent=1)

    def idle(self, msg):
        self.p("IDLE", msg, indent=1)
