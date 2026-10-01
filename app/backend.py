"""Load the pinned Wan modules omitted from Spielberg's vendor snapshot.

Spielberg's ``engine`` package prepends an incomplete ``mlx_video`` copy to
``sys.path``. The pinned upstream checkout provides the missing Wan modules,
but its package initializers import unrelated LTX components. Register only
the package search paths required by the Wan engine before importing Spielberg.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
MLX_VIDEO = ROOT / "vendor" / "mlx-video" / "mlx_video"


def _package(name: str, path: Path) -> None:
    module = ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def prepare_backend() -> None:
    """Expose the pinned Wan source without importing unrelated model families."""
    if "engine" in sys.modules and "mlx_video" not in sys.modules:
        raise RuntimeError("prepare_backend must run before importing Spielberg engine")
    if "mlx_video" in sys.modules:
        return
    required = [
        MLX_VIDEO / "models" / "wan_2" / "config.py",
        MLX_VIDEO / "models" / "wan_2" / "attention.py",
        MLX_VIDEO / "models" / "wan_2" / "utils.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Pinned mlx-video source is missing; run git submodule update --init --recursive. "
            + ", ".join(missing)
        )
    _package("mlx_video", MLX_VIDEO)
    _package("mlx_video.models", MLX_VIDEO / "models")
    _package("mlx_video.models.ltx_2", MLX_VIDEO / "models" / "ltx_2")
    _package("mlx_video.models.wan_2", MLX_VIDEO / "models" / "wan_2")
