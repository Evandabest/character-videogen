"""Wan2.1 temporal decoding with the official first-frame/cache behavior."""

from app.backend import prepare_backend

prepare_backend()

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.wan_2.vae import ResidualBlock, Resample


def _cached_conv(layer, x, cache, index):
    slot = index[0]
    previous = cache[slot]
    tail = x[:, :, -2:]
    if tail.shape[2] < 2 and previous is not None:
        tail = mx.concatenate([previous[:, :, -1:], tail], axis=2)
    result = layer(x, cache_x=previous)
    cache[slot] = tail
    index[0] += 1
    return result


def _upsample(layer, x, cache, index):
    if layer.mode == "upsample3d":
        slot = index[0]
        previous = cache[slot]
        if previous is None:
            # The first latent is one frame, not a pair of temporal frames.
            cache[slot] = "first"
        else:
            tail = x[:, :, -2:]
            first = isinstance(previous, str)
            if tail.shape[2] < 2:
                preceding = mx.zeros_like(tail) if first else previous[:, :, -1:]
                tail = mx.concatenate([preceding, tail], axis=2)
            b, c, t, h, w = x.shape
            expanded = layer.time_conv(x, cache_x=None if first else previous)
            expanded = expanded.reshape(b, 2, c, t, h, w)
            x = mx.stack([expanded[:, 0], expanded[:, 1]], axis=3).reshape(b, c, t * 2, h, w)
            cache[slot] = tail
        index[0] += 1
    b, c, t, h, w = x.shape
    pixels = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)
    pixels = mx.repeat(mx.repeat(pixels, 2, axis=1), 2, axis=2)
    pixels = layer.resample[1](pixels)
    return pixels.reshape(b, t, h * 2, w * 2, pixels.shape[-1]).transpose(0, 4, 1, 2, 3)


def _chunk(decoder, x, cache):
    index = [0]
    x = _cached_conv(decoder.conv1, x, cache, index)
    for layer in decoder.middle:
        x = layer(x, feat_cache=cache, feat_idx=index) if isinstance(layer, ResidualBlock) else layer(x)
    for layer in decoder.upsamples:
        if isinstance(layer, ResidualBlock):
            x = layer(x, feat_cache=cache, feat_idx=index)
        elif isinstance(layer, Resample):
            x = _upsample(layer, x, cache, index)
        else:
            x = layer(x)
    x = nn.silu(decoder.head[0](x))
    return _cached_conv(decoder.head[2], x, cache, index)


def decode_streaming(vae, latents):
    """Decode F latents to exactly 1+4*(F-1) frames with bounded activations."""
    mean = vae.mean.reshape(1, -1, 1, 1, 1)
    inv_std = vae.inv_std.reshape(1, -1, 1, 1, 1)
    z = vae.conv2(latents / inv_std + mean)
    mx.eval(z)
    # Some counted Conv3d shortcuts don't consume a cache slot; extra slots are harmless.
    slots = 2 + sum(2 for layer in [*vae.decoder.middle, *vae.decoder.upsamples]
                    if isinstance(layer, ResidualBlock))
    slots += sum(1 for layer in vae.decoder.upsamples
                 if isinstance(layer, Resample) and layer.mode == "upsample3d")
    cache = [None] * slots
    frames = []
    for i in range(z.shape[2]):
        frame = _chunk(vae.decoder, z[:, :, i:i + 1], cache)
        mx.eval(frame, *[item for item in cache if isinstance(item, mx.array)])
        frames.append(frame)
        mx.clear_cache()
    result = mx.concatenate(frames, axis=2)
    expected = 1 + 4 * (latents.shape[2] - 1)
    if result.shape[2] != expected:
        raise RuntimeError(f"Decoded {result.shape[2]} frames, expected {expected}")
    return mx.clip(result, -1, 1)
