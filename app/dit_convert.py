"""Convert the pinned Animate GGUF into resumable, bounded MLX shards."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import time
from collections import defaultdict
from pathlib import Path

import gguf

from app.backend import ROOT, prepare_backend
from app.weights import WEIGHTS, sha256


SOURCE = WEIGHTS["dit_q4"].path
OUTPUT = ROOT / "vendor" / "spielberg" / "models" / "mlx" / "animate_dit_shards"
MANIFEST = OUTPUT / "manifest.json"
CONFIG = OUTPUT.parent / "config.json"


def _group(key: str) -> str:
    parts = key.split(".")
    if parts[0] == "blocks":
        return ".".join(parts[:2])
    if parts[0] == "motion_encoder":
        return ".".join(parts[: min(5, len(parts) - 1)])
    return parts[0]


def _write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def _converted_tensor(tensor) -> tuple[str, dict]:
    import mlx.core as mx
    from engine.weights.convert import _transpose_value, remap_key

    name = remap_key(tensor.name)
    source = gguf.dequantize(tensor.data, tensor.tensor_type)
    value = _transpose_value(name, mx.array(source))
    del source
    parent, _, leaf = name.rpartition(".")
    quantized = (
        leaf == "weight"
        and value.ndim == 2
        and any(parent.endswith(suffix) for suffix in (
            ".self_attn.q", ".self_attn.k", ".self_attn.v", ".self_attn.o",
            ".cross_attn.q", ".cross_attn.k", ".cross_attn.v", ".cross_attn.o",
            ".cross_attn.k_img", ".cross_attn.v_img", ".ffn.0", ".ffn.2",
        ))
    )
    if quantized:
        packed, scales, biases = mx.quantize(value.astype(mx.float16), group_size=64, bits=4)
        return name, {name: packed, parent + ".scales": scales, parent + ".biases": biases}
    return name, {name: value.astype(mx.float16) if value.ndim > 1 else value}


def _write_config() -> None:
    from engine.animate.config import AnimateConfig

    cfg = AnimateConfig.animate_14b()
    keys = (
        "dim", "num_heads", "num_layers", "ffn_dim", "in_dim", "out_dim",
        "text_dim", "clip_in_dim", "img_context_len", "motion_encoder_dim",
        "motion_dim", "face_adapter_heads", "patch_size", "vae_stride",
    )
    data = {key: getattr(cfg, key) for key in keys}
    data["patch_size"] = list(cfg.patch_size)
    data["vae_stride"] = list(cfg.vae_stride)
    data["quantization"] = {"bits": 4, "group_size": 64}
    _write_json(CONFIG, data)


def convert(*, max_shards: int | None = None) -> None:
    expected = WEIGHTS["dit_q4"]
    if not SOURCE.is_file() or SOURCE.stat().st_size != expected.size or sha256(SOURCE) != expected.sha256:
        raise RuntimeError("Pinned Q4 GGUF is missing or failed checksum verification")
    prepare_backend()
    import mlx.core as mx

    reader = gguf.GGUFReader(SOURCE)
    grouped: dict[str, list] = defaultdict(list)
    for tensor in reader.tensors:
        from engine.weights.convert import remap_key

        grouped[_group(remap_key(tensor.name))].append(tensor)
    groups = sorted(grouped)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        if manifest.get("source_sha256") != expected.sha256:
            raise RuntimeError("Existing shard manifest belongs to a different source")
    else:
        manifest = {
            "source_sha256": expected.sha256,
            "source_revision": expected.revision,
            "converter": 1,
            "bits": 4,
            "group_size": 64,
            "groups_total": len(groups),
            "shards": {},
        }
    processed = 0
    for group in groups:
        if group in manifest["shards"]:
            shard_info = manifest["shards"][group]
            path = OUTPUT / shard_info["file"]
            if path.is_file() and sha256(path) == shard_info["sha256"]:
                continue
            raise RuntimeError(f"Existing shard is missing or corrupt: {group}")
        begin = time.monotonic()
        weights = {}
        for tensor in grouped[group]:
            name, values = _converted_tensor(tensor)
            duplicate = weights.keys() & values.keys()
            if duplicate:
                raise RuntimeError(f"Duplicate converted keys in {group}: {sorted(duplicate)}")
            weights.update(values)
            del values
        mx.eval(*weights.values())
        filename = group.replace(".", "_") + ".safetensors"
        path = OUTPUT / filename
        temp = OUTPUT / (group.replace(".", "_") + ".tmp.safetensors")
        mx.save_safetensors(str(temp), weights)
        os.replace(temp, path)
        manifest["shards"][group] = {
            "file": filename,
            "sha256": sha256(path),
            "keys": len(weights),
            "source_tensors": len(grouped[group]),
            "seconds": round(time.monotonic() - begin, 2),
            "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3, 3),
        }
        _write_json(MANIFEST, manifest)
        print(json.dumps({"group": group, **manifest["shards"][group]}), flush=True)
        del weights
        gc.collect()
        mx.clear_cache()
        processed += 1
        if max_shards is not None and processed >= max_shards:
            break
    if len(manifest["shards"]) == len(groups):
        _write_config()
        print(f"Conversion complete: {len(groups)} verified shards", flush=True)
    else:
        print(f"Conversion paused: {len(manifest['shards'])}/{len(groups)} shards", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()
    if args.max_shards is not None and args.max_shards < 1:
        parser.error("--max-shards must be positive")
    convert(max_shards=args.max_shards)


if __name__ == "__main__":
    main()
