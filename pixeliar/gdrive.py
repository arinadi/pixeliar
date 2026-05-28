"""
pixeliar — GDriveI/O (Module 2)
Google Drive API v3 — list, download, upload, create folder.
Pattern adapted from simple.md.
"""

import io
import os
import re
import time
import mimetypes
from pathlib import Path

from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}
IMG_MIME = "image/jpeg,image/png,image/webp,image/tiff"


class GDriveIO:
    def __init__(self, svc, rate_limit=0.5, max_retries=3):
        self.svc = svc
        self.rate_limit = rate_limit
        self.max_retries = max_retries

    def _throttle(self):
        time.sleep(self.rate_limit)

    def _retry(self, fn, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  ⚠️  Retry {attempt + 1}/{self.max_retries} "
                      f"({e.__class__.__name__}), waiting {wait}s", flush=True)
                time.sleep(wait)

    @staticmethod
    def extract_folder_id(url):
        m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
        if not m:
            raise ValueError(f"URL tidak valid: {url}")
        return m.group(1)

    def list_images(self, folder_id, recursive=False):
        q = (f"'{folder_id}' in parents and trashed=false and "
             f"mimeType contains 'image/'")
        results = self._retry(
            self.svc.files().list,
            q=q,
            fields="files(id, name, mimeType, size, md5Checksum)",
            pageSize=500,
        ).execute()
        files = results.get("files", [])

        if recursive:
            fq = (f"'{folder_id}' in parents and trashed=false and "
                  f"mimeType='application/vnd.google-apps.folder'")
            subfolders = self._retry(
                self.svc.files().list,
                q=fq,
                fields="files(id, name)",
                pageSize=100,
            ).execute().get("files", [])
            for sf in subfolders:
                self._throttle()
                files.extend(self.list_images(sf["id"], recursive=True))

        return files

    def download(self, file_id, dest_path):
        request = self.svc.files().get_media(fileId=file_id)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return dest_path

    def upload(self, local_path, folder_id):
        name = os.path.basename(local_path)
        ext = Path(local_path).suffix.lower()
        mime = MIME_MAP.get(ext, "image/jpeg")
        meta = {"name": name, "parents": [folder_id]}
        media = MediaFileUpload(local_path, mimetype=mime)
        self._retry(
            self.svc.files().create,
            body=meta,
            media_body=media,
            fields="id",
        ).execute()

    def create_folder(self, name, parent_id=None):
        meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent_id:
            meta["parents"] = [parent_id]
        folder = self._retry(
            self.svc.files().create,
            body=meta,
            fields="id, webViewLink",
        ).execute()
        return folder["id"], folder.get("webViewLink", "")

    def upload_json(self, data, filename, folder_id):
        import json
        raw = json.dumps(data, indent=2, ensure_ascii=False).encode()
        content = io.BytesIO(raw)
        media = MediaFileUpload(content, mimetype="application/json", resumable=False)
        meta = {"name": filename, "parents": [folder_id]}
        self._retry(
            self.svc.files().create,
            body=meta,
            media_body=media,
            fields="id",
        ).execute()


def authenticate():
    """Colab OAuth → Drive API v3 service."""
    from google.colab import auth
    from google.auth import default
    from googleapiclient.discovery import build

    auth.authenticate_user()
    creds, _ = default()
    return build("drive", "v3", credentials=creds)
