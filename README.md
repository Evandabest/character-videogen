# Character Video Generation

This repository is implementing the feasibility gates in [the project spec](character_video_project_spec_v2.md). The first milestone is a measured short Wan-Animate render on the M3 Pro with 18 GB unified memory. Inference quality and memory fit have not yet been established on this machine. Current measurements and limitations are in [the feasibility log](FEASIBILITY.md).

## Setup

```bash
git submodule update --init --recursive
uv sync --frozen
.venv/bin/python -m app doctor
```

The Spielberg checkout is pinned in `vendor/spielberg`. Its committed `mlx_video` vendor directory omits the Wan model modules it imports. The pinned `vendor/mlx-video` checkout supplies those modules, and `app.backend.prepare_backend()` loads only their package paths. Use that bootstrap before importing `engine` in project code.

The doctor reports available tools, pinned backend revisions, imports, disk space, and model checkpoints. A missing checkpoint is `pending`, not a successful inference test. The target disk has room for the current checkpoint experiments, but inspect space before downloading or rendering. Model files belong under `vendor/spielberg/models/` and are ignored by Git.

Run `.venv/bin/python -m pytest` for local checks after setup. See the spec for the staged baseline, streaming, and replacement milestones.

## VAE feasibility probe

This downloads only the official Wan2.1 VAE used by Animate, checks its SHA256, converts it to MLX, and runs a small encode/decode window. It does not download the full DiT or certify a complete render.

```bash
uv sync --frozen --group convert
.venv/bin/python -m app.vae_probe download
.venv/bin/python -m app.vae_probe convert
.venv/bin/python -m app.vae_probe probe
```

MLX computations may require an unsandboxed local terminal because sandboxed macOS sessions can hide the Metal device. Full-model conversion and inference remain unverified on 18 GB unified memory.

## Pinned Animate weights

The `app.weights` command downloads individual checkpoints from pinned repository revisions and verifies size and SHA256. It checks free space before each download. By default it uses resumable HTTP via `curl`; `--transport hub` is available as an alternative.

```bash
.venv/bin/python -m app.weights status dit_q4
.venv/bin/python -m app.weights download dit_q4
.venv/bin/python -m app.weights download t5
.venv/bin/python -m app.weights download clip
```

These files are the original checkpoints. The current Spielberg converter expands the entire GGUF model in memory before quantizing it, so downloading the weights does not make full Animate inference ready on this 18 GB machine. The bounded conversion path is the next implementation gate.

The project converter writes resumable MLX shards, one model group at a time. Start with a single group, inspect its memory and tensor results, then resume the remainder. A small quantized-layer compatibility test covers the layout, and the full checkpoint conversion was measured on the target Mac.

```bash
.venv/bin/python -m app.dit_convert --max-shards 1
.venv/bin/python -m app.dit_convert
```

The full DiT conversion and load have now passed on the target Mac; denoising remains unverified. The official umT5 checkpoint is converted to 8-bit shards and loaded through `app.t5_loader`, avoiding the upstream FP32 upcast. CLIP conversion extracts only the visual tower. Use these commands after downloading their pinned source checkpoints:

```bash
.venv/bin/python -m app.t5_convert --max-shards 1
.venv/bin/python -m app.t5_convert
.venv/bin/python -m app.clip_convert
.venv/bin/python -m app.tokenizer download
.venv/bin/python -m app.tokenizer probe
```

The tokenizer download fetches only five small files from a pinned `google/umt5-xxl` revision. Subsequent tokenizer loading is offline. The converted model checkpoints remain ignored by Git, while conversion code and checksum pins are committed.

For driving-video preprocessing, download and SHA-verify the official YOLOv10/ViTPose ONNX package (395 files, about 2.43 GiB):

```bash
.venv/bin/python -m app.pose_weights download
.venv/bin/python -m app doctor
```

The doctor checks that a complete pose package was verified and that all converted shard groups are present. Loading the shards performs full checksum and tensor-shape validation. The pose ONNX sessions have been initialized successfully on this Mac, but real human footage is still needed to assess extraction quality.

## Tiny denoising probe

With the converted DiT in place, run one synthetic five-frame forward pass under a 78%-of-RAM MLX allocation cap:

```bash
.venv/bin/python -m app.dit_probe
```

This verifies execution at a 64×64 working size only. It does not produce a useful video or establish feasibility at the intended delivery resolution. A meaningful baseline needs a reference character image and a short driving clip.

## Real-media preprocessing preview

Place input media under `app/input/` (ignored by Git), then inspect pose and face conditioning before starting a render:

```bash
.venv/bin/python -m app.preprocess_probe \
  --reference app/input/ref_img.png \
  --video app/input/ref_vid.mp4 \
  --frames 5
```

The preview and summary go to `outputs/preprocess_preview/` (also ignored by Git). Warnings about small poses or low-contrast face crops are heuristics, not a guarantee of failure, but they should be reviewed before an expensive render.
