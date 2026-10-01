from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from app.backend import prepare_backend

prepare_backend()

import mlx.core as mx

from app.vae_decode import _upsample, decode_streaming
from mlx_video.models.wan_2.vae import CausalConv3d, Decoder3d, Resample


def test_first_temporal_upsample_skips_time_conv_like_official_vae():
    layer = Resample(4, "upsample3d")
    x = mx.array(np.random.default_rng(2).normal(size=(1, 4, 1, 3, 3)).astype(np.float32))
    cache = [None]
    first = _upsample(layer, x, cache, [0])
    source = torch.from_numpy(np.asarray(x[:, :, 0]).copy())
    weight = torch.from_numpy(np.asarray(layer.resample[1].weight).copy()).permute(0, 3, 1, 2)
    bias = torch.from_numpy(np.asarray(layer.resample[1].bias).copy())
    expected = F.conv2d(F.interpolate(source, scale_factor=2, mode="nearest-exact"), weight, bias, padding=1)
    np.testing.assert_allclose(np.asarray(first[:, :, 0]), expected.numpy(), atol=2e-6)
    assert first.shape[2] == 1
    assert _upsample(layer, x, cache, [0]).shape[2] == 2
    assert _upsample(layer, x, cache, [0]).shape[2] == 2


def test_decoder_returns_exact_frame_count_and_resets_cache():
    decoder = Decoder3d(dim=4, z_dim=2, num_res_blocks=1)
    vae = SimpleNamespace(mean=mx.zeros(2), inv_std=mx.ones(2),
                          conv2=CausalConv3d(2, 2, 1), decoder=decoder)
    for latents, frames in [(1, 1), (2, 5), (3, 9), (1, 1)]:
        decoded = decode_streaming(vae, mx.zeros((1, 2, latents, 2, 2)))
        mx.eval(decoded)
        assert decoded.shape == (1, 3, frames, 16, 16)
        assert bool(mx.all(mx.isfinite(decoded)).item())
