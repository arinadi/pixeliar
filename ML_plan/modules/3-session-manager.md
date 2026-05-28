# Module 3: SessionManager — Fingerprint Resume

| | |
|---|---|
| **Estimated Complexity** | M |
| **Estimated Files** | 1 |
| **Key Risks** | Session file corruption; cross-runtime session sync |

## Requirements
- Load existing session or create new one
- Generate fingerprint per image (MD5 or file_id)
- Check if image already processed (skip if yes)
- Mark image as processed with status + log
- Persist session to /content/photon_session.json
- Upload session to Drive for cross-runtime resume

## Data & API
```python
session = {
    "session_id": "uuid",
    "started_at": "ISO timestamp",
    "processed": {
        "fingerprint_abc123": {"status": "ok", "log": {...}},
        "fingerprint_def456": {"status": "fail", "error": "..."}
    },
    "stats": {"total": 48, "ok": 44, "skip": 3, "fail": 1}
}
```

## Technical Implementation
```python
import hashlib, json, os, uuid

SESSION_FILE = "/content/photon_session.json"

def load_session() -> dict:
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE) as f:
            return json.load(f)
    return {"session_id": str(uuid.uuid4()), "processed": {}, "stats": {...}}

def file_fingerprint(file_id: str, md5: str = None) -> str:
    return md5 if md5 else hashlib.md5(file_id.encode()).hexdigest()

def should_process(session: dict, fp: str) -> bool:
    return fp not in session["processed"]

def mark_processed(session: dict, fp: str, status: str, log: dict):
    session["processed"][fp] = {"status": status, **log}
    session["stats"][status] = session["stats"].get(status, 0) + 1
    save_session(session)
```

## Testing
- [ ] First run creates new session file
- [ ] Second run loads existing session
- [ ] should_process returns False for already-processed fingerprint
- [ ] mark_processed updates stats correctly
- [ ] Session file survives cell re-run
- [ ] Cross-runtime: session uploaded to Drive and re-downloaded
