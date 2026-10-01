from types import SimpleNamespace

import gguf
import numpy as np
import pytest

from app.backend import prepare_backend
from app.dit_convert import _converted_tensor


def test_quantized_tensor_loads_into_mlx_linear():
    prepare_backend()
    mx = pytest.importorskip("mlx.core", exc_type=ImportError)
    import mlx.nn as nn

    tensor = SimpleNamespace(
        name="blocks.0.self_attn.q.weight",
        data=np.ones((8, 64), dtype=np.float16),
        tensor_type=gguf.GGMLQuantizationType.F16,
    )
    _, converted = _converted_tensor(tensor)
    prefix = "blocks.0.self_attn.q."
    layer = nn.QuantizedLinear(64, 8, bias=False, group_size=64, bits=4)
    layer.load_weights([(key.removeprefix(prefix), value) for key, value in converted.items()], strict=True)
    output = layer(mx.ones((1, 64), dtype=mx.float16))
    mx.eval(output)
    np.testing.assert_allclose(np.asarray(output), 64, atol=0.1)
