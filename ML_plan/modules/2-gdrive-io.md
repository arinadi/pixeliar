# Module 2: GDriveI/O — Google Drive API v3

| | |
|---|---|
| **Estimated Complexity** | M |
| **Estimated Files** | 1 |
| **Key Risks** | OAuth expiry; API quota (20K/100s); large file download timeout |

## Requirements
- Authenticate via Colab OAuth → Drive API v3
- List images in folder (with optional recursive subfolder scan)
- Chunk-based download (MediaIoBaseDownload)
- Upload files with MIME type detection
- Create result folder
- Extract folder ID from Google Drive URL
- Rate limiting between API calls
- Retry with exponential backoff
- Fallback to gdown if API fails

## Data & API
```python
# Auth
drive_svc = build("drive", "v3", credentials=creds)

# List
files = drive_svc.files().list(q=query, fields="files(id, name, mimeType, size, md5Checksum)", pageSize=500)

# Download
downloader = MediaIoBaseDownload(fh, request)
_, done = downloader.next_chunk()

# Upload
drive_svc.files().create(body=meta, media_body=MediaFileUpload(path, mimetype=mime))

# Create folder
folder = drive_svc.files().create(body=meta, fields="id, webViewLink")
```

## Technical Implementation
- Full OAuth flow via `google.colab.auth.authenticate_user()`
- MIME detection from extension mapping (.jpg, .jpeg, .png, .webp, .tiff)
- Rate limit: `time.sleep(0.5)` between API calls
- Retry: up to 3 attempts with `2^attempt` second backoff
- Recursive folder: traverse subfolders via mimeType query
- Fallback: `gdown.download_folder()` on API failure

## Testing
- [ ] OAuth completes without error
- [ ] List returns correct image count
- [ ] Download handles 10MB+ files without OOM
- [ ] Upload creates file in correct Drive folder
- [ ] Rate limiting prevents quota errors
- [ ] Retry works on transient failures
- [ ] gdown fallback triggers on API failure
