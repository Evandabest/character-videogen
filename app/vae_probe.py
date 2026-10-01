"""Download, convert, and test the official Wan2.1 VAE used by Wan-Animate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

from app.backend import ROOT, prepare_backend


REPO = "Wan-AI/Wan2.2-Animate-14B"
REVISION = "cb93a225fbaf1ca100f54e79da8f994995b689b3"
FILENAME = "Wan2.1_VAE.pth"
SHA256 = "acf6c5aa49ad281d4b561e10656e2397c446a8ba4b8d8f19d3dd125c2628bc6a"
MODEL_ROOT = ROOT / "vendor" / "spielberg" / "models"
ORIGINAL = MODEL_ROOT / "original" / FILENAME
CONVERTED = MODEL_ROOT / "mlx" / "vae.safetensors"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download() -> None:
    from huggingface_hub import hf_hub_download

    if ORIGINAL.is_file() and _sha256(ORIGINAL) == SHA256:
        print(f"Verified existing {ORIGINAL}")
        return
    ORIGINAL.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(REPO, FILENAME, revision=REVISION, local_dir=ORIGINAL.parent)
    actual = _sha256(ORIGINAL)
    if actual != SHA256:
        raise RuntimeError(f"VAE SHA256 mismatch: {actual}")
    print(f"Downloaded and verified {ORIGINAL}")


def convert() -> None:
    if not ORIGINAL.is_file() or _sha256(ORIGINAL) != SHA256:
        raise RuntimeError("Official VAE is missing or unverified; run download first")
    prepare_backend()
    import mlx.core as mx
    from mlx_video.models.wan_2.convert import load_torch_weights, sanitize_wan_vae_weights

    start = time.monotonic()
    weights = sanitize_wan_vae_weights(load_torch_weights(str(ORIGINAL)))
    weights = {key: value.astype(mx.float32) for key, value in weights.items()}
    CONVERTED.parent.mkdir(parents=True, exist_ok=True)
    # MLX appends .safetensors when the filename lacks that suffix.
    temporary = CONVERTED.with_name("vae.tmp.safetensors")
    mx.save_safetensors(str(temporary), weights)
    os.replace(temporary, CONVERTED)
    print(json.dumps({"output": str(CONVERTED), "tensors": len(weights),
                      "seconds": round(time.monotonic() - start, 2),
                      "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3, 3)}))


def probe() -> None:
    if not CONVERTED.is_file():
        raise RuntimeError("Converted VAE is missing; run convert first")
    prepare_backend()
    import mlx.core as mx
    import mlx.utils
    from engine.animate.config import AnimateConfig
    from mlx_video.models.wan_2.vae import WanVAE
    from mlx_video.models.wan_2.utils import load_vae_decoder, load_vae_encoder

    start = time.monotonic()
    cfg = AnimateConfig.animate_14b()
    checkpoint_keys = set(mx.load(str(CONVERTED)))
    model_keys = set(dict(mlx.utils.tree_flatten(WanVAE(z_dim=16, encoder=True).parameters())))
    missing = model_keys - checkpoint_keys - {"mean", "std", "inv_std"}
    if missing:
        raise RuntimeError(f"Converted VAE is missing {len(missing)} learned tensors")
    encoder = load_vae_encoder(CONVERTED, cfg)
    decoder = load_vae_decoder(CONVERTED, cfg)
    # One small legal temporal window; independent of DiT and text weights.
    rgb = mx.zeros((1, 3, 5, 64, 64), dtype=mx.float32)
    latent = encoder.encode(rgb)
    mx.eval(latent)
    from app.vae_decode import decode_streaming

    output = decode_streaming(decoder, latent)
    mx.eval(output)
    if output.shape != rgb.shape:
        raise RuntimeError(f"Unexpected decoded shape {output.shape}")
    print(json.dumps({"input_shape": list(rgb.shape), "latent_shape": list(latent.shape),
                      "decoded_shape": list(output.shape),
                      "extra_decoded_frames": output.shape[2] - rgb.shape[2],
                      "learned_tensors_matched": len(model_keys) - 3,
                      "seconds": round(time.monotonic() - start, 2),
                      "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3, 3)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "convert", "probe"))
    action = parser.parse_args().action
    {"download": download, "convert": convert, "probe": probe}[action]()


if __name__ == "__main__":
    main()
