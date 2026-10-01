import subprocess
import sys

import numpy as np

from app.short_render import input_fingerprint, run_stage


def test_changed_reference_changes_input_fingerprint(tmp_path):
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"first reference")
    before = input_fingerprint(reference)
    reference.write_bytes(b"replacement reference")
    after = input_fingerprint(reference)
    assert before["path"] == after["path"]
    assert before["sha256"] != after["sha256"]


def test_failed_stage_keeps_diagnostic_log(tmp_path):
    import pytest

    with pytest.raises(subprocess.CalledProcessError):
        run_stage([sys.executable, "-c", "print('diagnostic failure'); raise SystemExit(3)"], tmp_path)
    assert "diagnostic failure" in (tmp_path / "stages.log").read_text()


def test_unipc_reaches_known_clean_latent_with_oracle_velocity():
    from app.backend import prepare_backend

    prepare_backend()
    import mlx.core as mx
    from mlx_video.models.wan_2.scheduler import FlowUniPCScheduler

    clean = mx.array(np.linspace(-1, 1, 16).reshape(1, 4, 2, 2).astype(np.float32))
    noise = mx.array(np.random.default_rng(8).normal(size=clean.shape).astype(np.float32))
    scheduler = FlowUniPCScheduler()
    scheduler.set_timesteps(20, shift=5)
    sigma = scheduler._sigmas_float[0]
    sample = (1 - sigma) * clean + sigma * noise
    for i, timestep in enumerate(scheduler.timesteps):
        velocity = (sample - clean) / scheduler._sigmas_float[i]
        sample = scheduler.step(velocity, timestep, sample)
        mx.eval(sample)
    np.testing.assert_allclose(np.asarray(sample), np.asarray(clean), atol=2e-6)
