# M3 Pro 18 GB Feasibility Log

Observed October 1, 2026 on the target MacBook Pro. These results establish checkpoint preparation, VAE operation, and quantized DiT loading. They do not establish that denoising fits or that a useful video can be rendered.

## Environment

| Item | Observation |
| --- | --- |
| Machine | MacBook Pro, Apple M3 Pro, 18 GB unified memory |
| System | macOS 26.3.1, arm64 |
| Python | 3.12.10 through `uv` virtual environment |
| MLX | 0.32.3; Metal computation succeeds outside the sandbox |
| Media tools | ffmpeg and ffprobe 8.1.1 |
| Free disk after user cleanup | Approximately 115 GiB before the large checkpoint downloads; 88 GiB after DiT source and conversion |
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

## Completed DiT preparation gate

The pinned QuantStack Q4_K_M GGUF downloaded with the expected 11,496,331,072 bytes and SHA256 `43d720f243c3cdf5346ca05b525ec662f89bc5ebeb8e998dc347b840406cdfa6`. The upstream Spielberg converter expands the whole model before quantization, so the project instead converts one tensor group at a time to MLX 4-bit shards.

| Test | Observation |
| --- | --- |
| Conversion | All 1,441 GGUF tensors converted into 64 verified MLX shards; 80.56 seconds of per-shard work |
| Conversion memory | 4.119 GiB peak process RSS; the first shard peaked at 1.517 GiB |
| Converted size | Approximately 11 GiB on disk |
| Model load | Complete key and shape coverage; 8.59 seconds in the measured run |
| Load memory | 10.576 GiB MLX active / 10.577 GiB MLX peak; 3.415 GiB peak process RSS in the measured run |

MLX active memory and process RSS are different counters and should not be added as if they were independent physical allocations. The 78%-of-physical-RAM MLX allocation cap allows approximately 14 GiB, leaving narrow headroom above model weights. Model load success does not prove a denoising step will fit.

## Completed conditioning-encoder probes

The official umT5 and CLIP checkpoints downloaded from the pinned Wan revision and passed their recorded SHA256 checks. The local tokenizer was pinned to `google/umt5-xxl@66cb9e7e85526fe440a945569e42c72fb6cbc0ad`.

| Test | Observation |
| --- | --- |
| umT5 conversion | 242 BF16 tensors became 26 resumable 8-bit MLX shards; 6.064 GiB peak process RSS and 2.995 GiB peak MLX allocation |
| umT5 load | Complete key and shape coverage; 5.622 GiB MLX active memory |
| Text forward | The backend's 512-token padded encoding of “A person waves hello” returned finite `[6, 4096]` embeddings in 9.96 seconds including load; 6.444 GiB peak MLX memory |
| CLIP visual extraction | 389 visual-only tensors saved; 4.571 GiB peak process RSS and 2.353 GiB peak MLX memory |
| CLIP forward | A synthetic gray image returned finite `[257, 1280]` features in 15.0 seconds including load; 2.899 GiB peak MLX memory |

The encoders were run in separate processes. These probes establish execution and memory only, not conditioning quality. Real reference and driving media have not yet been tested.

## Remaining feasibility risks

The pinned raw component sizes are: Q4_K_M DiT GGUF 11.50 GB; umT5 checkpoint 11.36 GB; CLIP checkpoint 4.77 GB; VAE 0.51 GB. Converted and temporary copies add substantial disk use. Current free disk is sufficient for these setup experiments. The upstream umT5 loader upcasts the approximately 11 GB BF16 encoder to FP32; this project uses 8-bit shards and a separate loader to avoid that memory expansion. Accuracy relative to the BF16 encoder has not yet been compared.

The remaining gates are a DiT denoising step, end-to-end short baseline quality, multi-chunk continuity, and replacement. A full render remains unavailable until a real short inference test passes.

## Next engineering work

Run a short denoising and decode probe, then test with real reference/source media. Record peak memory pressure, swap, output quality, and total job time before adding long-video processing.
