"""Load the bounded, quantized Animate shards after full validation."""

from __future__ import annotations

import json
from pathlib import Path

from app.backend import prepare_backend
from app.dit_convert import CONFIG, MANIFEST, OUTPUT
from app.weights import WEIGHTS, sha256


def validate_shards() -> tuple[dict, list[Path]]:
    if not MANIFEST.is_file() or not CONFIG.is_file():
        raise RuntimeError("Animate shards or config are incomplete")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("source_sha256") != WEIGHTS["dit_q4"].sha256:
        raise RuntimeError("Shard manifest source hash does not match the pinned GGUF")
    if len(manifest.get("shards", {})) != manifest.get("groups_total"):
        raise RuntimeError("Animate shard conversion is incomplete")
    paths = []
    for info in manifest["shards"].values():
        path = OUTPUT / info["file"]
        if not path.is_file() or sha256(path) != info["sha256"]:
            raise RuntimeError(f"Missing or corrupt Animate shard: {path}")
        paths.append(path)
    return manifest, paths


def load_sharded_model(*, memory_fraction: float = 0.78):
    """Construct the quantized model and load verified shards sequentially."""
    prepare_backend()
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils
    from app.animate_model import WanAnimateModel
    from engine.weights.convert import _quantize_predicate
    from engine.weights.loader import load_config, set_memory_limit

    manifest, paths = validate_shards()
    allocation_limit = set_memory_limit(memory_fraction)
    # MLX defaults to zero wired memory. Keep weights resident within Apple's
    # recommendation; this is process-local and never changes system sysctls.
    resident_limit = min(allocation_limit, int(mx.device_info()["max_recommended_working_set_size"] * 0.9))
    try:
        mx.set_wired_limit(resident_limit)
    except (ValueError, RuntimeError) as error:
        print(json.dumps({"stage": "memory", "resident_budget_unavailable": str(error)}), flush=True)
        resident_limit = 0
    config, meta = load_config(CONFIG.parent)
    model = WanAnimateModel(config)
    model._resident_limit = resident_limit
    q = meta["quantization"]
    nn.quantize(model, group_size=q["group_size"], bits=q["bits"], class_predicate=_quantize_predicate)
    expected = {key: value.shape for key, value in mlx.utils.tree_flatten(model.parameters())}
    loaded: set[str] = set()
    for path in paths:
        weights = mx.load(str(path))
        extra = weights.keys() - expected.keys()
        if extra:
            raise RuntimeError(f"Unknown parameters in {path.name}: {sorted(extra)[:8]}")
        duplicate = loaded & weights.keys()
        if duplicate:
            raise RuntimeError(f"Duplicate parameters in {path.name}: {sorted(duplicate)[:8]}")
        for key, value in weights.items():
            if value.shape != expected[key]:
                raise RuntimeError(f"Shape mismatch for {key}: {value.shape} != {expected[key]}")
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(*weights.values())
        loaded.update(weights)
        del weights
    missing = expected.keys() - loaded - {"freqs"}
    if missing:
        raise RuntimeError(f"Animate shards are missing {len(missing)} model parameters")
    mx.eval(model.parameters())
    return model
