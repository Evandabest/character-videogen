# Character Video Generation

This repository is implementing the feasibility gates in [the project spec](character_video_project_spec_v2.md). The first milestone is a measured short Wan-Animate render on the M3 Pro with 18 GB unified memory. Inference quality and memory fit have not yet been established on this machine.

## Setup

```bash
git submodule update --init --recursive
uv sync --frozen
uv run --no-sync python -m app doctor
```

The Spielberg checkout is pinned in `vendor/spielberg`. Its committed `mlx_video` vendor directory omits the Wan model modules it imports. The pinned `vendor/mlx-video` checkout supplies those modules, and `app.backend.prepare_backend()` loads only their package paths. Use that bootstrap before importing `engine` in project code.

The doctor reports available tools, pinned backend revisions, imports, disk space, and model checkpoints. A missing checkpoint is `pending`, not a successful inference test. The target disk currently has limited headroom for original and converted weights; inspect space before downloading. Model files belong under `vendor/spielberg/models/` and are ignored by Git.

Run `uv run --no-sync pytest` for local checks after setup. See the spec for the staged baseline, streaming, and replacement milestones.
