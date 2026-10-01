"""Run one tiny synthetic Animate denoising forward pass under a memory cap."""

from __future__ import annotations

import argparse
import json
import resource
import time

from app.dit_loader import load_sharded_model


def probe(*, latent_size: int = 8, face_frames: int = 5) -> dict:
    import mlx.core as mx

    if latent_size < 4 or latent_size % 2:
        raise ValueError("latent_size must be even and at least 4")
    if face_frames != 5:
        raise ValueError("the first probe supports five face frames and three latent frames")
    begin = time.monotonic()
    model = load_sharded_model(memory_fraction=0.78)
    loaded = time.monotonic()
    active_after_load = mx.get_active_memory()
    latent_frames = 3  # one reference latent plus two temporal latents for 5 source frames
    x = mx.zeros((16, latent_frames, latent_size, latent_size))
    y = mx.zeros((20, latent_frames, latent_size, latent_size))
    pose = mx.zeros((16, latent_frames - 1, latent_size, latent_size))
    face = mx.zeros((face_frames, 512, 512, 3))
    clip = mx.zeros((257, 1280))
    text = mx.zeros((6, 4096))
    seq_len = latent_frames * latent_size * latent_size
    try:
        prediction = model(x, mx.array([500]), clip, text, seq_len, y, pose, face)
        mx.eval(prediction)
        result = {
            "output_shape": prediction.shape,
            "finite": bool(mx.all(mx.isfinite(prediction)).item()),
            "load_seconds": round(loaded - begin, 2),
            "forward_seconds": round(time.monotonic() - loaded, 2),
            "active_after_load_gib": round(active_after_load / 1024**3, 3),
            "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 3),
            "rss_peak_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3, 3),
        }
        return result
    finally:
        print(json.dumps({
            "mlx_active_gib": round(mx.get_active_memory() / 1024**3, 3),
            "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 3),
            "rss_peak_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3, 3),
        }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(probe(latent_size=args.latent_size)), flush=True)


if __name__ == "__main__":
    main()
