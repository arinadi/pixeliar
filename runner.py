#!/usr/bin/env python3
"""
pixeliar — Colab Bootstrap (Module 0)
Clone repo, install deps, load secrets, launch start.py.

Usage:
    !curl -sL https://raw.githubusercontent.com/arinadi/pixeliar/main/runner.py | python3
"""

import subprocess
import sys
import os
import time

REPO_URL = "https://github.com/arinadi/pixeliar.git"
REPO_DIR = "/content/pixeliar"
BRANCH = "main"


def run(cmd, check=True, **kwargs):
    """Run a command, print it, return result."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"  > {cmd_str}", flush=True)
    return subprocess.run(cmd, check=check, **kwargs)


def step(msg):
    print(f"\n{'='*50}\n{msg}\n{'='*50}", flush=True)


def main():
    t0 = time.time()

    # ── Step 1: Clone or update ──────────────────────────
    step("[1/3] Clone or update repo")

    if os.path.exists(os.path.join(REPO_DIR, ".git")):
        print(f"  Repo exists at {REPO_DIR}, updating...", flush=True)
        run(["git", "-C", REPO_DIR, "fetch", "--depth", "1", "origin", BRANCH])
        run(["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"])
        # Clean untracked (remove old files from previous version)
        run(["git", "-C", REPO_DIR, "clean", "-fd", "--exclude=ML_plan", "--exclude=docs"])
    else:
        print(f"  Cloning {REPO_URL}...", flush=True)
        run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO_DIR])

    print(f"  ✅ Repo ready at {REPO_DIR}", flush=True)

    # ── Step 2: Install deps ────────────────────────────
    step("[2/3] Install dependencies")

    req_file = os.path.join(REPO_DIR, "requirements.txt")
    if os.path.exists(req_file):
        run([
            sys.executable, "-m", "pip", "install", "-q",
            "-r", req_file
        ])
        print("  ✅ Dependencies installed", flush=True)
    else:
        print("  ⚠️  No requirements.txt found, skipping pip install", flush=True)

    # ── Step 3: Launch start.py ─────────────────────────
    step("[3/3] Launch pipeline")

    start_py = os.path.join(REPO_DIR, "start.py")
    if not os.path.exists(start_py):
        print(f"  ❌ {start_py} not found!", flush=True)
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"  🚀 Bootstrap complete ({elapsed:.1f}s)", flush=True)
    print(f"  Starting start.py...\n", flush=True)

    # Replace this process with start.py (exec, not subprocess)
    # This avoids nested subprocess and keeps output flowing
    os.chdir(REPO_DIR)
    os.execv(sys.executable, [sys.executable, start_py])


if __name__ == "__main__":
    main()
