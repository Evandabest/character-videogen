import numpy as np
import pytest
import torch

from app.t5_convert import BITS, GROUP_SIZE, _convert_tensor, remap_key


def test_gate_key_remap():
    assert remap_key("blocks.0.ffn.gate.0.weight") == "blocks.0.ffn.gate_proj.weight"


def test_quantized_t5_linear_loads():
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    import mlx.nn as nn

    converted = _convert_tensor("blocks.0.attn.q.weight", torch.ones((8, 64), dtype=torch.bfloat16))
    prefix = "blocks.0.attn.q."
    layer = nn.QuantizedLinear(64, 8, bias=False, group_size=GROUP_SIZE, bits=BITS)
    layer.load_weights([(key.removeprefix(prefix), value) for key, value in converted.items()], strict=True)
    output = layer(mx.ones((1, 64), dtype=mx.float16))
    mx.eval(output)
    np.testing.assert_allclose(np.asarray(output), 64, atol=0.2)
