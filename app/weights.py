"""Pinned checkpoint downloads with disk preflight and SHA256 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from app.backend import ROOT


MODEL_ROOT = ROOT / "vendor" / "spielberg" / "models"


@dataclass(frozen=True)
class Weight:
    name: str
    repo: str
    revision: str
    filename: str
    sha256: str
    size: int
    directory: str

    @property
    def path(self) -> Path:
        return MODEL_ROOT / self.directory / self.filename


WEIGHTS = {
    "dit_q4": Weight(
        "dit_q4",
        "QuantStack/Wan2.2-Animate-14B-GGUF",
        "33c51bb84d4e70ffc0d088aeb6068d40d9446fa3",
        "Wan2.2-Animate-14B-Q4_K_M.gguf",
        "43d720f243c3cdf5346ca05b525ec662f89bc5ebeb8e998dc347b840406cdfa6",
        11496331072,
        "gguf",
    ),
    "vae": Weight(
        "vae",
        "Wan-AI/Wan2.2-Animate-14B",
        "cb93a225fbaf1ca100f54e79da8f994995b689b3",
        "Wan2.1_VAE.pth",
        "acf6c5aa49ad281d4b561e10656e2397c446a8ba4b8d8f19d3dd125c2628bc6a",
        507609928,
        "original",
    ),
    "t5": Weight(
        "t5",
        "Wan-AI/Wan2.2-Animate-14B",
        "cb93a225fbaf1ca100f54e79da8f994995b689b3",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
        11361920418,
        "original",
    ),
    "clip": Weight(
        "clip",
        "Wan-AI/Wan2.2-Animate-14B",
        "cb93a225fbaf1ca100f54e79da8f994995b689b3",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "628c9998b613391f193eb67ff68da9667d75f492911e4eb3decf23460a158c38",
        4772359047,
        "original",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def status(weight: Weight, *, verify: bool) -> dict:
    path = weight.path
    exists = path.is_file()
    actual_size = path.stat().st_size if exists else 0
    result = {**asdict(weight), "path": str(path), "exists": exists, "actual_size": actual_size}
    if verify and exists:
        result["sha256_ok"] = sha256(path) == weight.sha256
    return result


def download(weight: Weight) -> None:
    from huggingface_hub import hf_hub_download

    if weight.path.is_file() and sha256(weight.path) == weight.sha256:
        print(f"Verified existing {weight.path}")
        return
    free = shutil.disk_usage(MODEL_ROOT).free
    reserve = 12 * 1024**3
    if free < weight.size + reserve:
        raise RuntimeError(
            f"Need {(weight.size + reserve) / 1024**3:.1f} GiB free including reserve; "
            f"have {free / 1024**3:.1f} GiB"
        )
    weight.path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(weight.repo, weight.filename, revision=weight.revision, local_dir=weight.path.parent)
    )
    actual = sha256(downloaded)
    if actual != weight.sha256:
        raise RuntimeError(f"SHA256 mismatch for {weight.name}: {actual}")
    print(json.dumps({"name": weight.name, "path": str(downloaded), "bytes": downloaded.stat().st_size, "sha256": actual}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "download"))
    parser.add_argument("name", choices=sorted(WEIGHTS))
    args = parser.parse_args()
    weight = WEIGHTS[args.name]
    if args.action == "download":
        download(weight)
    else:
        print(json.dumps(status(weight, verify=True), indent=2))


if __name__ == "__main__":
    main()
