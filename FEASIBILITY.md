# M3 Pro 18 GB Feasibility Log

Observed October 1, 2026 on the target MacBook Pro. These results establish setup and VAE operation only. They do not establish that the full Wan2.2-Animate DiT fits or that a useful video can be rendered.

## Environment

| Item | Observation |
| --- | --- |
| Machine | MacBook Pro, Apple M3 Pro, 18 GB unified memory |
| System | macOS 26.3.1, arm64 |
| Python | 3.12.10 through `uv` virtual environment |
| MLX | 0.32.3; Metal computation succeeds outside the sandbox |
| Media tools | ffmpeg and ffprobe 8.1.1 |
| Free disk after VAE setup | Approximately 45 GiB |
| Spielberg commit | `f692f93d73af996169244e9d8fa0178d6383f0d6` |
| mlx-video commit | `87db56a51758fefb748a359b90a5283bb8ba4837` |

Spielberg's committed vendor package omits the `mlx_video.models.wan_2` modules it imports. The project pins upstream mlx-video and loads the needed module paths through `app.backend.prepare_backend()`. The project import test and doctor check pass with these revisions.

## Completed VAE gate

The official Wan2.1 VAE was downloaded from revision `cb93a225fbaf1ca100f54e79da8f994995b689b3` of [`Wan-AI/Wan2.2-Animate-14B`](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B). Its SHA256 matches `acf6c5aa49ad281d4b561e10656e2397c446a8ba4b8d8f19d3dd125c2628bc6a`.

| Test | Observation |
| --- | --- |
| Conversion | 194 tensors; 1.5 seconds; 1.625 GiB peak process RSS |
| Checkpoint coverage | All 194 learned VAE tensors matched the MLX encoder model; three generated normalization buffers are not checkpoint tensors |
| Encode/decode | Input `[1, 3, 5, 64, 64]` became latent `[1, 16, 2, 8, 8]` and decoded `[1, 3, 8, 64, 64]` |
| Probe | 0.43 seconds; 0.793 GiB peak process RSS |

The timing is for a tiny zero-valued synthetic window and cannot be extrapolated to a full-resolution clip. The decoder emitted three extra frames for five input frames. Final output must be trimmed by the timeline/chunk assembler.

## Remaining feasibility risks

The full original checkpoint path is not ready on this disk. The pinned upstream sources report or expose these raw component sizes: Q4_K_M DiT GGUF 11.50 GB; umT5 checkpoint 11.36 GB; CLIP checkpoint 4.77 GB; VAE 0.51 GB. These already total approximately 28.1 GB before converted copies, temporary files, model caches, pose checkpoints, or output. Spielberg's converted DiT is approximately 11 GB and the converted BF16 T5 approximately 11 GB. Keeping both original and converted sets would exceed current free space. A carefully staged conversion with deletion might lower peak disk use, but the pinned DiT converter first expands GGUF tensors and constructs a full model before quantization; it has not been shown to fit 18 GB unified memory. Do not start that download/conversion path blindly.

No compatible preconverted Spielberg DiT checkpoint has been verified. The remaining gates are text/image encoding, DiT loading, denoising, short baseline quality, multi-chunk continuity, and replacement. A full render remains unavailable until a memory-safe weight preparation path and adequate disk space are established.

## Next engineering work

Find or produce a verified preconverted MLX DiT with the exact Spielberg key layout, or implement and measure a bounded-memory conversion that never instantiates the full floating-point model. Keep model downloads and conversion artifacts under a dedicated storage budget. After that, qualify text/image conditioning and one short baseline render with real reference/source media. Record load time, peak memory pressure, swap, output quality, and total job time before adding long-video processing.
