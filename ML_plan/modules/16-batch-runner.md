# Module 16: BatchRunner — Main Loop

| | |
|---|---|
| **Estimated Complexity** | S |
| **Estimated Files** | 1 |
| **Key Risks** | Session timeout mid-batch; progress lost |

## Requirements
- Main loop: iterate all images with progress bar
- For each image: triage → orchestrate → save → upload
- Session resume: skip processed images (fingerprint)
- Session summary: total, processed, skipped, failed
- Upload session + log to Drive

## Data & API
```python
def run_photon(config):
    logger = PhotonLogger()
    session = load_session()
    models = load_all_models(config)
    files = list_images_in_folder(config["source_gdrive_url"])

    result_id, result_link = create_drive_folder(config["result_folder_name"])

    for file in tqdm(files, desc="PHOTON"):
        fp = file_fingerprint(file["id"], file.get("md5Checksum"))
        if not should_process(session, fp):
            logger.resume(file["name"], fp)
            continue

        try:
            local_path = download_file(file)
            triage = triage_engine.analyze(load_image(local_path))
            result, steps = orchestrate(load_image(local_path), triage, config, models, logger)
            save_image(result, output_path)
            upload_to_drive(output_path, result_id)
            mark_processed(session, fp, "ok", {"steps": steps})
        except Exception as e:
            mark_processed(session, fp, "fail", {"error": str(e)})

    print_session_summary(session)
    upload_session_log(session, result_id)
```

## Testing
- [ ] First run: all images processed
- [ ] Second run: all images skipped (resume)
- [ ] Failed images: logged, not skipped in next run
- [ ] Session summary shows correct stats
- [ ] Session uploaded to Drive
