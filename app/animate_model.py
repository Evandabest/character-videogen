"""Corrections to the pinned MLX port, checked against official Wan-Animate.

Keep these adapters in the project so the upstream submodules stay pinned.
"""

from app.backend import prepare_backend

prepare_backend()

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from engine.animate.model import WanAnimateModel as UpstreamAnimateModel
from engine.animate.clip_image import ClipVisual, CLIP_MEAN, CLIP_STD, IMAGE_SIZE


class _CachedOutput(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __call__(self, unused):
        return self.value

    def get_motion(self, unused):
        return self.value


class WanAnimateModel(UpstreamAnimateModel):
    def prepare_conditioning(self, clip, context, faces):
        """Project fixed conditioning once and release its large encoders."""
        text = self._text(context)
        image = self.img_emb(clip)
        motions = []
        for start in range(0, len(faces), 4):
            encoded = self.motion_encoder.get_motion(faces[start:start + 4])
            mx.eval(encoded)
            motions.append(encoded)
            mx.clear_cache()
        motion = mx.concatenate(motions, axis=0)
        face_tokens = self.face_encoder(motion[None])
        mx.eval(text, image, motion, face_tokens)
        self._prepared_text = text
        self.img_emb = _CachedOutput(image)
        self.motion_encoder = _CachedOutput(motion)
        self.face_encoder = _CachedOutput(face_tokens)
        self.text_embedding = []
        mx.clear_cache()

    def __call__(self, x, t, clip_fea, context, seq_len, y, pose_latents, face_pixels):
        # This port runs on one device: no sequence-parallel padding is needed.
        # FaceAdapter groups the tokens by frame, so added tokens change alignment.
        actual = 1
        for size, patch in zip(x.shape[1:], self.patch_size):
            if size % patch:
                raise ValueError("Latent dimensions must divide evenly into patches")
            actual *= size // patch
        return super().__call__(x, t, clip_fea, context, actual, y, pose_latents, face_pixels)

    def _text(self, context):
        if hasattr(self, "_prepared_text"):
            return self._prepared_text
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
        return self.model(preprocess_clip(reference))[0]


def preprocess_clip(reference):
    """Match torch bicubic interpolation (align_corners=False, no antialias)."""
    import cv2

    image = np.asarray(reference).transpose(1, 2, 0)
    image = cv2.resize(image.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    return ((mx.array(image) * 0.5 + 0.5 - CLIP_MEAN) / CLIP_STD)[None]
