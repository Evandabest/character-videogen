"""Read-only setup checks; missing checkpoints remain explicit."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from app.backend import ROOT, prepare_backend


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _git_revision(path: Path) -> str | None:
    if not path.is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def checks() -> list[Check]:
    results: list[Check] = []
    arch = platform.machine()
    results.append(Check("architecture", "ok" if arch == "arm64" else "blocked", arch))
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        results.append(Check(tool, "ok" if path else "blocked", path or "missing"))
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    results.append(Check("free_disk", "ok" if free_gib >= 60 else "warning", f"{free_gib:.1f} GiB"))
    for name in ("spielberg", "mlx-video"):
        path = ROOT / "vendor" / name
        revision = _git_revision(path)
        results.append(Check(name, "ok" if revision else "blocked", revision or "not initialized"))
    try:
        prepare_backend()
        import mlx.core as mx
        from engine.animate.config import AnimateConfig

        value = mx.array([1.0, 2.0]).sum().item()
        config = AnimateConfig.animate_14b()
        results.append(
            Check("mlx_backend_import", "ok", f"MLX sum={value:g}; Animate layers={config.num_layers}")
        )
    except Exception as exc:
        results.append(Check("mlx_backend_import", "blocked", f"{type(exc).__name__}: {exc}"))
    model_root = ROOT / "vendor" / "spielberg" / "models" / "mlx"
    for filename in ("config.json", "animate_dit.safetensors", "t5_encoder.safetensors", "vae.safetensors", "clip_visual.safetensors"):
        path = model_root / filename
        detail = f"{path.stat().st_size / 1024**3:.2f} GiB" if path.is_file() else "missing"
        results.append(Check(filename, "ok" if path.is_file() else "pending", detail))
    return results


def run_doctor(*, as_json: bool = False) -> int:
    results = checks()
    if as_json:
        print(json.dumps([asdict(check) for check in results], indent=2))
    else:
        for check in results:
            print(f"{check.status:7} {check.name:26} {check.detail}")
    return 1 if any(check.status == "blocked" for check in results) else 0
