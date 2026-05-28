# Module 0: runner.py — Colab Bootstrap

| | |
|---|---|
| **Estimated Complexity** | S |
| **Estimated Files** | 1 |
| **Key Risks** | Git clone failure; pip install timeout |

## Requirements
- Fetch & execute pipeline from GitHub in a single Colab cell
- Clone or update repo (fast: `--depth 1`)
- Install lightweight bootstrap deps
- Load Colab Secrets (MIMO_API_KEY, GITHUB_TOKEN)
- Launch start.py

## Data & API
```python
# Input: Colab cell command
# !curl -sL https://raw.githubusercontent.com/USER/REPO/main/runner.py | python3

# Output: subprocess launching start.py
```

## Technical Implementation
```python
import subprocess, sys, os, shutil

REPO_URL = "https://github.com/USER/REPO.git"
REPO_DIR = "/content/PHOTON"

# 1. Clone or update
if os.path.exists(REPO_DIR):
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "--depth", "1"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", "origin/main"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)

# 2. Install deps
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "-r", f"{REPO_DIR}/requirements.txt"], check=True)

# 3. Load secrets
try:
    from google.colab import userdata
    os.environ["MIMO_API_KEY"] = userdata.get("MIMO_API_KEY", "")
    os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN", "")
except ImportError:
    pass  # running outside Colab

# 4. Launch
subprocess.run([sys.executable, f"{REPO_DIR}/start.py"])
```

## Testing
- [ ] `python runner.py` clones repo to /content/PHOTON
- [ ]第二次 run updates (fetch + reset) without full re-clone
- [ ] Secrets loaded into os.environ
- [ ] start.py launched as subprocess
