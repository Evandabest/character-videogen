"""Numerical regression checks for differences from the official PyTorch model."""

import numpy as np
import pytest

from app.backend import prepare_backend

prepare_backend()

import mlx.core as mx
import mlx.nn as nn
import torch
import torch.nn.functional as F

from app.animate_model import AnimateClipVisual, WanAnimateModel, preprocess_clip
from engine.animate.model import WanAnimateModel as UpstreamAnimateModel


class TinyText(WanAnimateModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.text_len = 512
        self.text_embedding = [nn.Linear(4, 8), nn.GELU(approx="tanh"), nn.Linear(8, 8)]


def test_text_projection_matches_official_padding_before_projection():
    model = TinyText()
    context = mx.array(np.arange(12).reshape(3, 4).astype(np.float32) / 10)
    result = model._text(context)
    x = F.pad(torch.from_numpy(np.asarray(context).copy()), (0, 0, 0, 509))
    for index in (0, 2):
        layer = model.text_embedding[index]
        x = F.linear(x, torch.from_numpy(np.asarray(layer.weight).copy()),
                     torch.from_numpy(np.asarray(layer.bias).copy()))
        if index == 0:
            x = F.gelu(x, approximate="tanh")
    assert result.shape == (512, 8)
    np.testing.assert_allclose(np.asarray(result), x.numpy(), atol=1e-6)
    # Projected zero padding is learned content because the MLP has biases.
    assert np.any(np.asarray(result)[3:] != 0)
    with pytest.raises(ValueError):
        model._text(mx.zeros((513, 4)))


def test_clip_uses_exact_gelu_against_torch():
    model = AnimateClipVisual(dim=8, heads=2, layers=2, patch=2, image_size=4)
    pixels = mx.array(np.linspace(-2, 2, 48).reshape(1, 4, 4, 3).astype(np.float32))
    block = model.blocks[0]
    embedded = model.patch_embedding(pixels).reshape(1, 4, 8)
    x = model.pre_norm(mx.concatenate([model.cls_embedding, embedded], axis=1) + model.pos_embedding)
    x = x + block.attn(block.norm1(x))
    hidden = block.mlp[0](block.norm2(x))
    expected_activation = F.gelu(torch.from_numpy(np.asarray(hidden).copy())).numpy()
    expected = x + block.mlp[2](mx.array(expected_activation))
    np.testing.assert_allclose(np.asarray(model(pixels)), np.asarray(expected), atol=2e-6)


def test_single_device_tokens_keep_face_frames_aligned(monkeypatch):
    model = TinyText()
    model.patch_size = (1, 2, 2)
    observed = []
    monkeypatch.setattr(UpstreamAnimateModel, "__call__",
                        lambda self, x, t, clip, context, length, *args: observed.append(length))
    # Three latent frames, each containing 9*12 spatial tokens: 324 total.
    x = mx.zeros((16, 3, 18, 24))
    model(x, None, None, None, 1296, None, None, None)
    assert observed == [324]
    with pytest.raises(ValueError):
        model(mx.zeros((16, 3, 17, 24)), None, None, None, 0, None, None, None)


def test_clip_resize_matches_official_bicubic_interpolation():
    image = np.random.default_rng(17).uniform(-1, 1, (3, 144, 192)).astype(np.float32)
    resized = F.interpolate(torch.from_numpy(image)[None], size=(224, 224),
                            mode="bicubic", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
    expected = ((resized * 0.5 + 0.5 - mean) / std).permute(0, 2, 3, 1).numpy()
    # CPU/GPU coordinate rounding differs by <1e-4 after normalization.
    np.testing.assert_allclose(np.asarray(preprocess_clip(mx.array(image))), expected, atol=1e-4)


def test_cached_conditioning_matches_original_projection():
    model = TinyText()
    model.img_emb = nn.Linear(4, 8)
    model.motion_encoder = nn.Linear(4, 8)
    # Tiny deterministic stand-ins for the two face encoders.
    model.motion_encoder.get_motion = model.motion_encoder.__call__
    model.face_encoder = nn.Linear(8, 8)
    clip, context, faces = mx.ones((2, 4)), mx.ones((3, 4)), mx.ones((5, 4))
    text = model._text(context)
    image = model.img_emb(clip)
    motion = model.motion_encoder.get_motion(faces)
    tokens = model.face_encoder(motion[None])
    mx.eval(text, image, motion, tokens)
    model.prepare_conditioning(clip, context, faces)
    np.testing.assert_array_equal(np.asarray(model._text(context)), np.asarray(text))
    np.testing.assert_array_equal(np.asarray(model.img_emb(clip)), np.asarray(image))
    np.testing.assert_array_equal(np.asarray(model.face_encoder(motion[None])), np.asarray(tokens))
    assert model.text_embedding == []


def test_patch_embedding_matches_official_conv3d_layout():
    model = TinyText()
    model.dim = 8
    model.patch_size = (1, 2, 2)
    projection = nn.Linear(36 * 4, 8)
    pixels = mx.array(np.random.default_rng(19).normal(size=(36, 3, 18, 24)).astype(np.float32))
    actual, grid = model._patch_grid(pixels, projection)
    weights = torch.from_numpy(np.asarray(projection.weight).copy()).reshape(8, 36, 1, 2, 2)
    bias = torch.from_numpy(np.asarray(projection.bias).copy())
    expected = F.conv3d(torch.from_numpy(np.asarray(pixels).copy())[None], weights, bias, stride=(1, 2, 2))
    assert grid == (3, 9, 12)
    np.testing.assert_allclose(np.asarray(actual), expected[0].permute(1, 2, 3, 0).numpy(), atol=2e-6)
