# M3 Pro 18 GB Feasibility Log

Observed October 1, 2026 on the target MacBook Pro. These results establish checkpoint preparation, conditioning-encoder operation, VAE operation, and one tiny synthetic DiT forward pass. They do not establish that a useful video can be rendered.

## Environment

| Item | Observation |
| --- | --- |
| Machine | MacBook Pro, Apple M3 Pro, 18 GB unified memory |
| System | macOS 26.3.1, arm64 |
| Python | 3.12.10 through `uv` virtual environment |
| MLX | 0.32.3; Metal computation succeeds outside the sandbox |
| Media tools | ffmpeg and ffprobe 8.1.1 |
| Free disk after user cleanup | Approximately 115 GiB before the large checkpoint downloads; 63 GiB after current model setup |
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

The official pose-only package was downloaded from the pinned Wan revision: 395 files totaling 2,611,022,238 bytes, each checked against its LFS SHA256. YOLOv10 and ViTPose ONNX sessions initialized on CPU in 2.13 seconds, with 2.699 GiB peak process RSS. Pose extraction and face-crop quality still require real driving footage.

## Completed tiny denoising probe

One DiT forward pass used a five-frame synthetic face window, three latent frames, and an 8×8 latent grid (equivalent to a 64×64 working image). Text, image, pose, and temporal-conditioning tensors were zeros. The result was finite `[16, 3, 8, 8]` latents.

| Measurement | Observation |
| --- | --- |
| Model load | 11.23 seconds |
| One forward pass | 43.79 seconds |
| MLX active after load | 10.576 GiB |
| MLX peak through forward | 12.080 GiB |
| Memory cap | 78% of physical memory, approximately 14 GiB |

This is an execution smoke test only. It does not predict runtime or memory at 480p-class resolution, longer windows, multiple sampling steps, or real conditioning. A full render has not yet been completed.

## Supplied-media preprocessing preview

The supplied 15.53-second, 960×720, 30 fps H.264 driving clip and initial 957×2031 reference image were left untouched. A bounded preview of the first five frames at a 480×224 working shape completed in 5.63 seconds after renderer initialization. The first attempt exposed a missing `matplotlib` dependency; it is now in the lockfile.

The rendered pose occupied only 12.3–18.5% of the frame height across those frames. Face crops had grayscale spatial standard deviation 1.83–2.24 on a 0–255 scale. Preview inspection showed only a partial, small skeleton and nearly featureless face crops, consistent with the hazy/distant performer. Later sampled frames remain hazy or show a profile/occluded face. These are input-conditioning quality warnings; using this clip for a technical smoke test is possible, but it is not a good basis for judging expression transfer or a production-quality baseline.

Raw first-frame pose metadata placed the face crop at approximately 22×28 source pixels before upscaling it to 512×512, explaining why the face PNG looks like a flat brown square. The missing lower limbs in the fifth skeleton are a pose-estimation result, not an image file cropped at the bottom.

## Staged five-frame MP4 smoke test

The project ran preprocessing, umT5, CLIP, VAE encoding, DiT, and VAE decoding in separate processes using the supplied media. At `--area 28672`, the portrait working canvas was 112×240. The one-step run peaked at 12.511 GiB MLX memory in denoising and wrote a valid five-frame, 112×240 H.264 MP4. A four-step run peaked at 12.974 GiB, and the final decode/export filled the driving video's 960×720 frame by cropping the portrait output. `ffprobe` confirmed five frames, 30 fps, and 0.167-second duration.

Both outputs are visibly noisy and not recognizable character animation. They establish stage interoperability and a short memory envelope only. The poor pose/face conditioning, very low working resolution, aggressive portrait-to-landscape crop, and one/four-step sampling all prevent these clips from serving as quality evidence. Do not extrapolate their timing or quality to a 20-step, longer-window render.

## Updated-reference 20-step test

The reference was replaced with a 348×574 full-body cutout. The project made a video-aspect canvas containing the complete figure, selected five driving frames beginning at 3 seconds, and preprocessed a 192×144 working frame. All conditioning stages, 20 DiT steps, VAE decode, and H.264 export completed. The final MP4 is five frames at 30 fps and 960×720, matching the driving video's size. Peak reported MLX memory during denoising was 12.975 GiB.

The decoded frames remain abstract color noise rather than recognizable animation. This test removes one/four-step sampling and extreme portrait-to-landscape cropping as sufficient explanations for the earlier failure. It does not isolate the root cause: the very small 192×144, five-frame working window may itself be outside the model's useful regime, while checkpoint conversion/load mapping, scheduler behavior, latent scaling, and conditioning alignment still need verification against the upstream pipeline. Do not scale to the full clip until a short segment produces coherent frames. The updated reference and source media remain ignored local inputs, not committed assets.

## Remaining feasibility risks

The pinned raw component sizes are: Q4_K_M DiT GGUF 11.50 GB; umT5 checkpoint 11.36 GB; CLIP checkpoint 4.77 GB; VAE 0.51 GB. Converted and temporary copies add substantial disk use. Current free disk is sufficient for these setup experiments. The upstream umT5 loader upcasts the approximately 11 GB BF16 encoder to FP32; this project uses 8-bit shards and a separate loader to avoid that memory expansion. Accuracy relative to the BF16 encoder has not yet been compared.

The remaining gates are end-to-end short baseline quality, multi-chunk continuity, and replacement. A full render remains unavailable until the short inference quality gate passes.

## Next engineering work

Diagnose the incoherent 20-step result by comparing one denoising step and VAE reconstruction against the upstream pipeline, including model tensor mapping, scheduler inputs/outputs, latent scale, and conditioning arrays. Record system swap and total job time in the next instrumented run. Do not add long-video processing before a coherent short-window result.
