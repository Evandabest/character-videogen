"""Corrections to the pinned MLX port, checked against official Wan-Animate.

Keep these adapters in the project so the upstream submodules stay pinned.
"""

from app.backend import prepare_backend

prepare_backend()

import mlx.core as mx
import mlx.nn as nn

from engine.animate.model import WanAnimateModel as UpstreamAnimateModel
from engine.animate.clip_image import ClipVisual, preprocess_ref


class WanAnimateModel(UpstreamAnimateModel):
    def _text(self, context):
        # Official model_animate.py pads *before* the biased text projection.
        # These projected padding tokens participate in attention (no mask).
        length = context.shape[-2]
        if length > self.text_len:
            raise ValueError(f"Text context exceeds {self.text_len} tokens")
        padding = [(0, 0)] * context.ndim
        padding[-2] = (0, self.text_len - length)
        return super()._text(mx.pad(context, padding))


class AnimateClipVisual(ClipVisual):
    def __call__(self, pixels):
        batch = pixels.shape[0]
        x = self.patch_embedding(pixels)
        x = x.reshape(batch, -1, x.shape[-1])
        cls = mx.broadcast_to(self.cls_embedding, (batch, 1, x.shape[-1]))
        x = self.pre_norm(mx.concatenate([cls, x], axis=1) + self.pos_embedding)
        for block in self.blocks[:-1]:
            x = x + block.attn(block.norm1(x))
            # Official XLM-RoBERTa ViT-H uses exact GELU, not QuickGELU.
            x = x + block.mlp[2](nn.gelu(block.mlp[0](block.norm2(x))))
        return x


class ClipEncoder:
    def __init__(self, path):
        self.model = AnimateClipVisual()
        self.model.load_weights(list(mx.load(str(path)).items()), strict=True)
        mx.eval(self.model.parameters())

    def __call__(self, reference):
        return self.model(preprocess_ref(reference))[0]
