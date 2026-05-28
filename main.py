#!/usr/bin/env python3
"""
pixeliar — Main Pipeline (Module 16: BatchRunner)
Entry point: loads config, initializes, runs batch.
"""

import os
import sys
import time
import asyncio
import numpy as np
import cv2

from pixeliar.logger import PhotonLogger
from pixeliar.triage import TriageEngine
from pixeliar.session import SessionManager
from pixeliar.model_loader import load_all_models
from pixeliar.orchestrator import orchestrate
from pixeliar.hardware import detect_hardware
from pixeliar.idle_monitor import IdleMonitor


# ── CONFIG (edit before run) ────────────────────────────
# Paste your Google Drive folder URL here
SOURCE_FOLDER_URL = "https://drive.google.com/drive/folders/MASUKKAN_ID_FOLDER"  #@param {type:"string"}
# Name of the result folder created in your Drive
RESULT_FOLDER_NAME = "Pixeliar Enhanced"  #@param {type:"string"}
# Output JPEG quality (85-98 recommended)
JPEG_QUALITY = 95  #@param {type:"integer"}

CONFIG = {
    # INPUT
    "source_gdrive_url": SOURCE_FOLDER_URL,
    "recursive_folder_scan": False,
    # OUTPUT
    "target_mydrive_path": "MyDrive/pixeliar/enhanced",
    "result_folder_name": RESULT_FOLDER_NAME,
    # TRIAGE THRESHOLDS
    "thresholds": {
        "dark_lum": 60,
        "severe_dark_lum": 40,
        "bright_lum": 190,
        "flat_contrast_std": 45,
        "wb_cast_threshold": 8.0,
        "noise_threshold": 15.0,
        "blur_threshold": 150.0,
    },
    # RESOLUTION
    "inference_resolution": 512,
    "tiling_threshold": 1500,
    "tile_overlap": 64,
    "ml_blend": 0.4,
    # OUTPUT
    "jpeg_output_quality": JPEG_QUALITY,
    "csrnet_strength": 0.5,
    "save_thumbnails": False,
    # RUNTIME
    "load_restormer": False,
    "nafnet_model_dir": "/content/pixeliar_nafnet_weights",
    "weights_dir": "/content/pixeliar_weights",
    # SESSION
    "resume_enabled": True,
    "session_upload_to_drive": True,
    # COLAB
    "idle_shutdown_minutes": 5,
    "rate_limit_sleep": 0.5,
    "max_retries": 3,
    "fallback_on_cpu": True,
}


def download_image(drive_svc, file, dest_dir):
    """Download a Drive image to local path."""
    from pixeliar.gdrive import GDriveIO
    gdrive = GDriveIO(drive_svc, rate_limit=CONFIG["rate_limit_sleep"])
    local_path = os.path.join(dest_dir, file["name"])
    gdrive.download(file["id"], local_path)
    return local_path


def load_image(path):
    """Load image as float32 numpy RGB [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def save_image(img_np, path, quality=95):
    """Save float32 RGB [0,1] to JPEG."""
    img_u8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


def run_photon(config=None):
    """Main entry point."""
    cfg = config or CONFIG
    t0 = time.time()

    logger = PhotonLogger()
    logger.p("START", "pixeliar v4.0 starting")
    logger.sep()
    logger.p("FOLDER", f"Source: {cfg['source_gdrive_url']}")
    logger.p("FOLDER", f"Target: {cfg['target_mydrive_path']}")

    # ── Google Drive auth ──
    # Auth MUST happen in the Colab cell (before runner.py),
    # because os.execv / subprocess loses Colab kernel context.
    # Here we just build the service from existing credentials.
    drive_svc = None
    try:
        from google.auth import default
        from googleapiclient.discovery import build
        creds, _ = default()
        drive_svc = build("drive", "v3", credentials=creds)
        logger.drive_op("auth", "Google Drive connected")
    except Exception as e:
        logger.p("WARN", f"Drive auth failed: {e}", indent=1)
        logger.p("WARN", "Running in local-only mode", indent=1)

    # ── Hardware detect ──────────────────────────────────
    hw = detect_hardware()
    os.environ["PHOTON_MODE"] = hw["mode"]
    logger.p("MODEL", f"Mode: {hw['mode']}"
             + (f" ({hw['name']}, {hw['vram_gb']}GB)" if hw["name"] else ""))

    # ── Session ──────────────────────────────────────────
    session = SessionManager()
    logger.p("SESSION", f"Session {session.data['session_id'][:8]}... "
             f"({session.stats().get('ok', 0)} already processed)")

    # ── Load models ──────────────────────────────────────
    os.makedirs(cfg["weights_dir"], exist_ok=True)
    models = load_all_models(cfg, logger)

    # ── List images from Drive ───────────────────────────
    files = []
    if drive_svc:
        from pixeliar.gdrive import GDriveIO
        gdrive = GDriveIO(drive_svc, rate_limit=cfg["rate_limit_sleep"])
        folder_id = GDriveIO.extract_folder_id(cfg["source_gdrive_url"])
        files = gdrive.list_images(folder_id, cfg["recursive_folder_scan"])
        logger.drive_op("list", f"{len(files)} images found")
    else:
        logger.p("WARN", "No Drive — provide local images in /content/input/", indent=1)
        input_dir = "/content/input"
        if os.path.isdir(input_dir):
            for f in sorted(os.listdir(input_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.tiff')):
                    files.append({"id": f, "name": f, "md5Checksum": None})

    if not files:
        logger.p("FAIL", "No images found!", indent=1)
        return

    # ── Create result folder ─────────────────────────────
    result_folder_id = None
    result_link = ""
    if drive_svc:
        from pixeliar.gdrive import GDriveIO
        gdrive = GDriveIO(drive_svc, rate_limit=cfg["rate_limit_sleep"])
        result_folder_id, result_link = gdrive.create_folder(cfg["result_folder_name"])
        logger.drive_op("create", f"Result folder: {result_link}")

    # ── Batch processing ─────────────────────────────────
    triage_engine = TriageEngine(cfg)
    local_dir = "/content/pixeliar_temp"
    os.makedirs(local_dir, exist_ok=True)

    total = len(files)
    stats = {"ok": 0, "skip": 0, "fail": 0}
    model_usage = {}

    for idx, file in enumerate(files, 1):
        fname = file["name"]
        fp = session.fingerprint(file["id"], file.get("md5Checksum"))

        logger.sep()
        logger.p("IMAGE", f"[{idx:02d}/{total:02d}] {fname}", indent=0)

        # Resume check
        if cfg["resume_enabled"] and not session.should_process(fp):
            if not session.is_failed(fp):
                logger.resume(fname, fp)
                stats["skip"] += 1
                continue

        try:
            # Download
            if drive_svc and file.get("id"):
                local_path = download_image(drive_svc, file, local_dir)
            else:
                local_path = os.path.join(local_dir, fname)
                if not os.path.exists(local_path):
                    raise IOError(f"File not found: {local_path}")

            img = load_image(local_path)
            h, w = img.shape[:2]
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            logger.p("IMAGE", f"{w}x{h}, {size_mb:.1f}MB", indent=1)

            # Triage
            triage = triage_engine.analyze(img)
            logger.triage(triage)

            if triage_engine.get_no_correction_needed(triage):
                logger.skip("ALL", "no_correction_needed (all triage normal)")
                # Copy original to output
                output_name = fname.rsplit(".", 1)[0] + "_pixeliar.jpg"
                save_image(img, os.path.join(local_dir, output_name), cfg["jpeg_output_quality"])
                if drive_svc and result_folder_id:
                    gdrive.upload(os.path.join(local_dir, output_name), result_folder_id)
                session.mark(fp, "ok", {"steps": [], "reason": "no_correction_needed"})
                stats["ok"] += 1
                continue

            # Run pipeline
            result, steps = orchestrate(img, triage, cfg, models, logger)

            # Track model usage
            for s in steps:
                model_usage[s] = model_usage.get(s, 0) + 1

            # Save output
            output_name = fname.rsplit(".", 1)[0] + "_pixeliar.jpg"
            output_path = os.path.join(local_dir, output_name)
            save_image(result, output_path, cfg["jpeg_output_quality"])
            logger.p("SAVE", output_name, indent=1)

            # Upload to Drive
            if drive_svc and result_folder_id:
                gdrive.upload(output_path, result_folder_id)
                logger.drive_op("upload", output_name)

            session.mark(fp, "ok", {"steps": steps})
            stats["ok"] += 1

        except Exception as e:
            logger.p("FAIL", f"{fname}: {e}", indent=1)
            session.mark(fp, "fail", {"error": str(e)})
            stats["fail"] += 1

    # ── Session summary ──────────────────────────────────
    elapsed = time.time() - t0
    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
    avg = elapsed / total if total else 0

    logger.sep()
    logger.p("SUMMARY", "SESSION COMPLETE")
    logger.p("SUMMARY", f"Total images    : {total}", indent=1)
    logger.p("SUMMARY", f"Processed       : {stats['ok']}", indent=1)
    logger.p("SUMMARY", f"No correction   : {stats['skip']}", indent=1)
    logger.p("SUMMARY", f"Failed          : {stats['fail']}", indent=1)
    logger.p("SUMMARY", f"Total time      : {elapsed_str}", indent=1)
    logger.p("SUMMARY", f"Avg per image   : {avg:.1f}s", indent=1)

    if model_usage:
        logger.p("SUMMARY", "MODEL USAGE:", indent=1)
        for m, count in sorted(model_usage.items()):
            logger.p("SUMMARY", f"{m:<18}: {count} images", indent=2)

    # Upload session log
    if drive_svc and result_folder_id:
        session_log = {
            "config": {k: v for k, v in cfg.items() if k != "thresholds"},
            "stats": session.summary(),
            "model_usage": model_usage,
            "elapsed_seconds": round(elapsed, 1),
        }
        try:
            gdrive.upload_json(session_log, "pixeliar_session.json", result_folder_id)
            logger.drive_op("upload", "session log")
        except Exception as e:
            logger.p("WARN", f"Could not upload session log: {e}", indent=1)

    logger.p("DONE", f"All done — {result_link or local_dir}")
    logger.sep()

    # ── Idle monitor ─────────────────────────────────────
    if cfg.get("idle_shutdown_minutes", 0) > 0:
        monitor = IdleMonitor(cfg["idle_shutdown_minutes"], logger)
        asyncio.run(monitor.watch())


if __name__ == "__main__":
    # Allow config override via env (injected from Colab form)
    if os.environ.get("PHOTON_SOURCE_URL"):
        CONFIG["source_gdrive_url"] = os.environ["PHOTON_SOURCE_URL"]
    if os.environ.get("PHOTON_RESULT_NAME"):
        CONFIG["result_folder_name"] = os.environ["PHOTON_RESULT_NAME"]
    if os.environ.get("PHOTON_JPEG_QUALITY"):
        CONFIG["jpeg_output_quality"] = int(os.environ["PHOTON_JPEG_QUALITY"])

    run_photon()
