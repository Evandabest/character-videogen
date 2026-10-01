"""Numerical regression checks for differences from the official PyTorch model."""

import numpy as np
import pytest

from app.backend import prepare_backend

prepare_backend()

import mlx.core as mx
import mlx.nn as nn
import torch
import torch.nn.functional as F

from app.animate_model import AnimateClipVisual, WanAnimateModel


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
