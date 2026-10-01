"""Extract only the visual CLIP tower from the pinned official checkpoint."""

from __future__ import annotations

import json
import os
import resource
import time

from app.backend import ROOT, prepare_backend
from app.weights import WEIGHTS, sha256


SOURCE = WEIGHTS["clip"].path
OUTPUT = ROOT / "vendor" / "spielberg" / "models" / "mlx" / "clip_visual.safetensors"


def convert() -> None:
    expected = WEIGHTS["clip"]
    if not SOURCE.is_file() or SOURCE.stat().st_size != expected.size or sha256(SOURCE) != expected.sha256:
        raise RuntimeError("Pinned CLIP checkpoint is missing or failed checksum verification")
    prepare_backend()
    import mlx.core as mx
    import mlx.utils
    import torch
    from engine.animate.clip_image import ClipVisual, remap_clip_key

    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    mx.set_memory_limit(int(physical * 0.7))
    if OUTPUT.is_file():
        saved = mx.load(str(OUTPUT))
        expected_keys = dict(mlx.utils.tree_flatten(ClipVisual().parameters()))
        if saved.keys() == expected_keys.keys() and all(saved[k].shape == v.shape for k, v in expected_keys.items()):
            print(f"Verified existing {OUTPUT}")
            return
        raise RuntimeError("Existing CLIP visual checkpoint does not match the model")
    begin = time.monotonic()
    state = torch.load(SOURCE, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise RuntimeError("Unexpected CLIP checkpoint format")
    weights = {}
    for key, tensor in state.items():
        name = remap_clip_key(key)
        if name is None:
            continue
        value = mx.array(tensor.numpy())
        if name == "patch_embedding.weight":
            value = mx.transpose(value, (0, 2, 3, 1))
        weights[name] = value
    model = ClipVisual()
    expected_shapes = {key: value.shape for key, value in mlx.utils.tree_flatten(model.parameters())}
    if weights.keys() != expected_shapes.keys():
        raise RuntimeError(f"CLIP visual key mismatch: missing={sorted(expected_shapes.keys() - weights.keys())[:8]}, extra={sorted(weights.keys() - expected_shapes.keys())[:8]}")
    for key, value in weights.items():
        if value.shape != expected_shapes[key]:
            raise RuntimeError(f"CLIP visual shape mismatch for {key}: {value.shape} != {expected_shapes[key]}")
    mx.eval(*weights.values())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_name(OUTPUT.stem + ".tmp.safetensors")
    mx.save_safetensors(str(temp), weights)
    os.replace(temp, OUTPUT)
    print(json.dumps({
        "path": str(OUTPUT),
        "tensors": len(weights),
        "bytes": OUTPUT.stat().st_size,
        "sha256": sha256(OUTPUT),
        "seconds": round(time.monotonic() - begin, 2),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3, 3),
        "peak_mlx_gib": round(mx.get_peak_memory() / 1024**3, 3),
    }), flush=True)


if __name__ == "__main__":
    convert()
