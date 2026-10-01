"""Load and validate the bounded 8-bit umT5 encoder shards."""

from __future__ import annotations

import json
from pathlib import Path

from app.backend import prepare_backend
from app.t5_convert import BITS, GROUP_SIZE, MANIFEST, OUTPUT
from app.weights import WEIGHTS, sha256


def validate_shards() -> list[Path]:
    if not MANIFEST.is_file():
        raise RuntimeError("umT5 shard manifest is missing")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("source_sha256") != WEIGHTS["t5"].sha256:
        raise RuntimeError("umT5 shard source hash does not match the pinned checkpoint")
    if manifest.get("bits") != BITS or manifest.get("group_size") != GROUP_SIZE:
        raise RuntimeError("umT5 shard quantization does not match the loader")
    if len(manifest.get("shards", {})) != manifest.get("groups_total"):
        raise RuntimeError("umT5 conversion is incomplete")
    paths = []
    for info in manifest["shards"].values():
        path = OUTPUT / info["file"]
        if not path.is_file() or sha256(path) != info["sha256"]:
            raise RuntimeError(f"Missing or corrupt umT5 shard: {path}")
        paths.append(path)
    return paths


def load_sharded_t5(*, memory_fraction: float = 0.7):
    prepare_backend()
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.utils
    from engine.animate.config import AnimateConfig
    from engine.weights.loader import set_memory_limit
    from mlx_video.models.wan_2.text_encoder import T5Encoder

    paths = validate_shards()
    set_memory_limit(memory_fraction)
    cfg = AnimateConfig.animate_14b()
    encoder = T5Encoder(
        vocab_size=cfg.t5_vocab_size,
        dim=cfg.t5_dim,
        dim_attn=cfg.t5_dim_attn,
        dim_ffn=cfg.t5_dim_ffn,
        num_heads=cfg.t5_num_heads,
        num_layers=cfg.t5_num_layers,
        num_buckets=cfg.t5_num_buckets,
        shared_pos=False,
    )
    nn.quantize(
        encoder,
        group_size=GROUP_SIZE,
        bits=BITS,
        class_predicate=lambda path, module: isinstance(module, nn.Linear) or path == "token_embedding",
    )
    expected = {key: value.shape for key, value in mlx.utils.tree_flatten(encoder.parameters())}
    loaded: set[str] = set()
    for path in paths:
        weights = mx.load(str(path))
        extra = weights.keys() - expected.keys()
        if extra:
            raise RuntimeError(f"Unknown umT5 parameters in {path.name}: {sorted(extra)[:8]}")
        duplicate = loaded & weights.keys()
        if duplicate:
            raise RuntimeError(f"Duplicate umT5 parameters in {path.name}: {sorted(duplicate)[:8]}")
        for key, value in weights.items():
            if value.shape != expected[key]:
                raise RuntimeError(f"Shape mismatch for {key}: {value.shape} != {expected[key]}")
        encoder.load_weights(list(weights.items()), strict=False)
        mx.eval(*weights.values())
        loaded.update(weights)
        del weights
    missing = expected.keys() - loaded
    if missing:
        raise RuntimeError(f"umT5 shards are missing {len(missing)} model parameters")
    mx.eval(encoder.parameters())
    return encoder
