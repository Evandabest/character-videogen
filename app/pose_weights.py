"""Download and verify the pinned YOLOv10/ViTPose ONNX checkpoint package."""

from __future__ import annotations

import argparse
import json
import os
import shutil

from app.weights import MODEL_ROOT, sha256


REPO = "Wan-AI/Wan2.2-Animate-14B"
REVISION = "cb93a225fbaf1ca100f54e79da8f994995b689b3"
PREFIXES = ("process_checkpoint/det/", "process_checkpoint/pose2d/")
DIRECTORY = MODEL_ROOT / "original"
MARKER = DIRECTORY / "process_checkpoint" / "verified.json"


def _files():
    from huggingface_hub import HfApi

    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    files = [item for item in info.siblings if item.rfilename.startswith(PREFIXES)]
    if len(files) < 390 or any(item.lfs is None or not item.size for item in files):
        raise RuntimeError("Unexpected pose checkpoint listing at the pinned revision")
    return files


def download() -> None:
    from huggingface_hub import snapshot_download

    files = _files()
    total = sum(item.size for item in files)
    free = shutil.disk_usage(DIRECTORY).free
    if free < total + 12 * 1024**3:
        raise RuntimeError(f"Insufficient disk space: need {(total + 12 * 1024**3) / 1024**3:.1f} GiB including reserve")
    snapshot_download(
        REPO,
        revision=REVISION,
        allow_patterns=[prefix + "**" for prefix in PREFIXES],
        local_dir=DIRECTORY,
        max_workers=4,
    )
    verified = 0
    for item in files:
        path = DIRECTORY / item.rfilename
        if not path.is_file() or path.stat().st_size != item.size or sha256(path) != item.lfs.sha256:
            raise RuntimeError(f"Pose checkpoint file failed verification: {item.rfilename}")
        verified += 1
    result = {"files": verified, "bytes": total, "revision": REVISION}
    temporary = MARKER.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, MARKER)
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download",))
    parser.parse_args()
    download()


if __name__ == "__main__":
    main()
