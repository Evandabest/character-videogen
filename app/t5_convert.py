"""Convert the official umT5 checkpoint into resumable 8-bit MLX shards."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
from collections import defaultdict

from app.backend import ROOT
from app.dit_convert import _write_json
from app.weights import WEIGHTS, sha256


SOURCE = WEIGHTS["t5"].path
OUTPUT = ROOT / "vendor" / "spielberg" / "models" / "mlx" / "t5_shards"
MANIFEST = OUTPUT / "manifest.json"
BITS = 8
GROUP_SIZE = 64


def remap_key(key: str) -> str:
    return key.replace(".ffn.gate.0.", ".ffn.gate_proj.")


def _group(key: str) -> str:
    parts = key.split(".")
    return ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]


def _quantized_key(key: str) -> bool:
    return key == "token_embedding.weight" or (
        key.startswith("blocks.")
        and key.endswith(".weight")
        and (".attn." in key or ".ffn." in key)
    )


def _convert_tensor(key, tensor):
    import mlx.core as mx
    import torch

    key = remap_key(key)
    value = mx.array(tensor.to(dtype=torch.float16).numpy())
    if _quantized_key(key):
        packed, scales, biases = mx.quantize(value, group_size=GROUP_SIZE, bits=BITS)
        prefix = key.removesuffix(".weight")
        return {key: packed, prefix + ".scales": scales, prefix + ".biases": biases}
    return {key: value}


def convert(*, max_shards: int | None = None) -> None:
    import mlx.core as mx
    import torch

    expected = WEIGHTS["t5"]
    if not SOURCE.is_file() or SOURCE.stat().st_size != expected.size or sha256(SOURCE) != expected.sha256:
        raise RuntimeError("Pinned umT5 checkpoint is missing or failed checksum verification")
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    mx.set_memory_limit(int(physical * 0.7))
    state = torch.load(SOURCE, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise RuntimeError("Unexpected umT5 checkpoint format")
    grouped = defaultdict(list)
    for key in state:
        grouped[_group(remap_key(key))].append(key)
    groups = sorted(grouped)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        if manifest.get("source_sha256") != expected.sha256 or manifest.get("bits") != BITS:
            raise RuntimeError("Existing umT5 shards have a different source or precision")
    else:
        manifest = {
            "source_sha256": expected.sha256,
            "source_revision": expected.revision,
            "converter": 1,
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "groups_total": len(groups),
            "shards": {},
        }
    processed = 0
    for group in groups:
        if group in manifest["shards"]:
            info = manifest["shards"][group]
            path = OUTPUT / info["file"]
            if path.is_file() and sha256(path) == info["sha256"]:
                continue
            raise RuntimeError(f"Existing umT5 shard is missing or corrupt: {group}")
        begin = time.monotonic()
        weights = {}
        for key in grouped[group]:
            converted = _convert_tensor(key, state[key])
            duplicate = weights.keys() & converted.keys()
            if duplicate:
                raise RuntimeError(f"Duplicate umT5 keys: {sorted(duplicate)}")
            weights.update(converted)
            mx.eval(*converted.values())
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
            "peak_mlx_gib": round(mx.get_peak_memory() / 1024**3, 3),
        }
        _write_json(MANIFEST, manifest)
        print(json.dumps({"group": group, **manifest["shards"][group]}), flush=True)
        del weights
        gc.collect()
        mx.clear_cache()
        processed += 1
        if max_shards is not None and processed >= max_shards:
            break
    print(f"umT5 conversion: {len(manifest['shards'])}/{len(groups)} shards", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    if args.max_shards is not None and args.max_shards < 1:
        parser.error("--max-shards must be positive")
    convert(max_shards=args.max_shards)


if __name__ == "__main__":
    main()
