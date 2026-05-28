"""
pixeliar — SessionManager (Module 3)
Fingerprint-based resume. Session persists across cell re-runs.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime


DEFAULT_SESSION_FILE = "/content/pixeliar_session.json"


class SessionManager:
    def __init__(self, session_file=None):
        self.path = session_file or DEFAULT_SESSION_FILE
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return self._new()

    def _new(self):
        return {
            "session_id": str(uuid.uuid4()),
            "started_at": datetime.now().isoformat(),
            "processed": {},
            "stats": {"total": 0, "ok": 0, "skip": 0, "fail": 0},
        }

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def fingerprint(self, file_id, md5=None):
        return md5 if md5 else hashlib.md5(file_id.encode()).hexdigest()

    def should_process(self, fp):
        return fp not in self.data["processed"]

    def is_failed(self, fp):
        entry = self.data["processed"].get(fp)
        return entry is not None and entry.get("status") == "fail"

    def mark(self, fp, status, log=None):
        self.data["processed"][fp] = {"status": status, **(log or {})}
        self.data["stats"]["total"] += 1
        self.data["stats"][status] = self.data["stats"].get(status, 0) + 1
        self.save()

    def stats(self):
        return self.data["stats"]

    def summary(self):
        s = self.data["stats"]
        return {
            "total": s.get("total", 0),
            "ok": s.get("ok", 0),
            "skip": s.get("skip", 0),
            "fail": s.get("fail", 0),
            "session_id": self.data["session_id"],
        }
