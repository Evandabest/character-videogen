# Local Character Animation and Video Replacement

Engineering specification v2 
Target machine: MacBook Pro with M3 Pro and 18 GB unified memory  
Initial model candidate: Wan2.2-Animate-14B with quantized MLX inference

## 1. Goal and feasibility decision

Build a fully local macOS pipeline that accepts a reference character image and a driving video, then generates a video in which the character follows the source performer's motion and expressions. Support Animate mode first and scene-preserving Replace mode after the baseline is stable.

Wan-Animate matches the required conditioning, but useful operation on an 18 GB M3 Pro is an unproven engineering assumption. Spielberg is the initial backend candidate, not a guarantee of hardware compatibility. Its upstream README reports validation on an M4 Pro with 48 GB unified memory.

Before building the full application, establish whether installation, conditioning, generation, and decoding can complete with useful quality, acceptable runtime, and sufficient memory headroom on the actual target machine. Record the result in a feasibility report. If the gate fails, document the limiting stage and viable alternatives: smaller working settings, a narrower animation task, or a local machine with more memory. Do not introduce cloud inference or change the product's motion and replacement requirements implicitly.

Long source videos must use bounded-memory processing rather than a fixed input-duration limit. This requirement does not promise unlimited temporal consistency or a particular render speed. Report measured quality limits, estimated completion time, and required disk space.

## 2. Architecture and upstream audit

Use Wan2.2-Animate-14B with 4-bit weights through a replaceable MLX backend. Begin by auditing and pinning Spielberg to an exact commit, along with dependency versions, checkpoint sources, conversion settings, and weight checksums.

Treat its model implementation as reusable engine code. Its current end-to-end orchestration requires changes for this product: preprocessing loads the source into arrays, generation retains the DiT and VAE components together, and generated chunks are accumulated before final concatenation. Native temporal conditioning is useful, but the existing pipeline is not a bounded-memory application.

The backend must expose separate operations for:

- Preparing and persisting text, image, pose, face, background, and mask conditioning.
- Encoding generated continuity frames when preparing the next chunk.
- Loading and unloading model components.
- Denoising one bounded window and persisting its latents.
- Decoding a bounded window and returning frames to a streaming sink.
- Reporting capabilities, effective settings, memory observations, and errors.

FastVideo is an optimization reference. Its distilled models and published speedups do not establish Wan-Animate performance. Keep temporal reduction, spatial reduction, conditioning reuse, and compilation independently measurable. VSA transplantation and Animate-specific distillation remain later research.

## 3. Feasibility gates

Instrumentation begins before the first full render. Test each stage separately so a setup or encoder failure cannot be mistaken for a diffusion-window problem.

| Gate | Required evidence | Decision |
| --- | --- | --- |
| Setup and weights | Reproducible download or import, verified checksums, conversion peak memory, disk usage, successful model loading | Select a setup path that fits the target machine |
| Conditioning | Pose/face preview, text and image embeddings, VAE encoding; timing and memory for each component | Resolve encoder bottlenecks before diffusion |
| Short baseline | A playable clip with recognizable identity, motion, and expression; full stage timing and memory observations | Establish useful quality and runtime before optimization |
| Two adjacent chunks | Correct frame count and timing, valid continuity conditioning, inspected boundary | Prove the streaming and continuity contract |
| Sustained render | A source spanning many windows and multiple shots; bounded memory, disk estimate, cancellation and resume | Qualify long-input support |
| Replacement | Stable subject masks, background preservation assessment, relighting and foreground occlusion checks | Qualify Replace separately from Animate |

Use approximately the official 20-step sampling configuration as the first quality reference, with a small frame window and resolution that fit. A four-step smoke test establishes execution only. Steps, resolution, and window size must be recorded as measured settings rather than asserted presets.

Before declaring a default preset, record the observed rendering time per output second, total job time, peak memory, sustained memory pressure and swap behavior, and quality results. Set product runtime and quality acceptance targets explicitly after the initial measurements; until then, report feasibility as provisional.

## 4. Product requirements and supported inputs

- Inference runs locally on Apple Silicon without a CUDA or cloud dependency.
- Reference formats: PNG, JPEG, and WebP. Driving formats: MP4 and MOV, subject to available decoder support.
- Output: playable MP4 with optional preservation of source audio.
- Support portrait, landscape, and square timelines with explicit resize, crop, and padding behavior.
- Accept sources longer than a model window; stream all stages and persist progress.
- Jobs support cancellation and resume after interruption or recoverable memory failure.
- Source media, conditioning, caches, and outputs remain local. Initial model/dependency downloads require network access; inference must work offline after setup.
- 480p-class inference is an initial experiment, not a proven minimum or output-resolution ceiling. Higher delivery resolutions normally use post-decode RGB processing.

The initial supported content is one clearly visible human performer, with a detectable face when expression transfer is required, and a reference character with compatible body proportions. The reference should show enough of the intended body and appearance to support the requested shot.

Multiple performers, prolonged disappearance, substantial occlusion, extreme camera changes, and unusual character anatomy are evaluation cases rather than guaranteed v1 inputs. Report low-confidence pose/face extraction before generation. Multiple people require subject selection and identity tracking before they can become supported inputs.

Pose retargeting is a separate capability. The official guide recommends it for mismatched proportions, and its basic method has reference-pose requirements. Spielberg's README states that retargeting is not wired. Expose its availability honestly; do not imply that resizing alone solves proportion differences.

## 5. Pipeline

```text
reference image + driving video
    -> media probe, timeline normalization, shot detection, input checks
    -> bounded pose / face / mask extraction
    -> preprocessing preview and persisted conditioning
    -> prepare one window including native generated continuity context
    -> load quantized DiT, denoise, persist latents
    -> VAE decode, persist frames and continuity tail
    -> commit chunk manifest
    -> repeat within the shot; reset generated continuity at shot cuts
    -> resolve boundaries and final timeline
    -> optional shot-aware RIFE interpolation
    -> optional RGB upscaling
    -> optional replacement compositing
    -> encode and preserve / transcode / trim source audio
    -> validate final MP4
```

Temporal and spatial reduction are decisions made before diffusion. RIFE and RGB upscaling run after VAE decode. Every downstream operation must also stream bounded ranges; final assembly must not load the complete output into memory.

## 6. Setup and weight preparation

Memory planning covers conversion and model construction as well as inference. Spielberg's converter loads GGUF weights into floating-point tensors before quantizing selected layers to MLX format. Consequently, the quantized output size does not bound conversion peak memory. Its auxiliary conversion code also documents an approximately 11 GB BF16 text encoder and a runtime FP32 upcast; verify the pinned loader's actual behavior and peak memory.

Prefer a verified preconverted MLX checkpoint when available. Otherwise investigate conversion one tensor or layer at a time, including relighting LoRA merging, with bounded temporary residency. Do not assume the upstream whole-model conversion fits on 18 GB.

Evaluate quantized or reduced-precision text encoding where supported and validated. Cache embeddings for fixed prompts, keyed by tokenizer, encoder weights, prompt, and precision. Cached embeddings must have a documented local creation/import path; setup must not silently depend on an external service.

Record original and converted weights, both Animate and relighting variants if needed, auxiliary models, temporary conversion files, and cache/output space. Check free disk space before setup and each job. Never silently substitute missing CLIP or relighting weights with reduced conditioning.

## 7. Media preprocessing and preview

Use ffprobe to inspect dimensions, frame rate, duration, rotation, codec, color metadata, and audio streams. Prefer presentation timestamps over unreliable container frame-count estimates.

Normalize variable-frame-rate media onto a stable working timeline and retain the mapping to source timestamps. Define source FPS, diffusion FPS, and delivery FPS separately. Resampling must preserve elapsed time rather than accidentally speeding up or slowing down motion.

Resize, crop, or pad consistently with model and VAE divisibility. Persist the spatial transform and apply it to pose, face crops, masks, backgrounds, reference images, and compositing coordinates.

Extract bounded source ranges and cache pose/face conditioning by content hash, extraction configuration, model versions, and spatial/timeline transforms. Persist confidence and missing-detection metadata. Provide an inexpensive preview of pose, face crops, subject masks, and shot boundaries before an expensive render.

Reset subject and generated continuity state at scene cuts. When a performer disappears or tracking confidence drops, use an explicit supported policy or flag the segment for review. Never silently switch to another person.

## 8. Memory scheduling on 18 GB

CPU and GPU share unified memory. Moving tensors to the CPU does not create an independent RAM budget. Optimize simultaneous residency and retain headroom for macOS and other applications.

For each stage, observe peak process memory, MLX active/cache memory where available, system memory pressure, and swap. Materialize outputs before releasing lazy computation graphs. Deleting a Python object or clearing an allocator cache is not proof that all referenced memory has been reclaimed.

Use this dependency-aware schedule:

1. Extract a bounded range and persist conditioning; release preprocessors when complete.
2. Prepare text/image embeddings and reference/pose/background latents; persist reusable outputs.
3. For later chunks, VAE-encode the previous generated tail into continuity conditioning.
4. Release components and arrays not needed during denoising.
5. Load/use the quantized DiT and persist completed chunk latents.
6. Release the DiT if required, then load/use the VAE decoder with supported tiling or temporal chunking.
7. Persist decoded output and the exact continuity tail; commit the chunk.
8. Repeat conditioning preparation for the next chunk, then run postprocessing in bounded ranges.

Benchmark two strategies: retain the VAE alongside the DiT only if safe, or unload/reload around denoising. Include reload time in job estimates. Previous generated context depends on newly decoded frames, so all continuity conditioning cannot be prepared in advance.

Use isolated workers for stages where process exit is the most reliable way to reclaim memory. Persist state outside workers. Limit heavyweight job concurrency to one on the target Mac.

Bound source buffers, conditioning, latents, decoded frames, interpolation queues, and encoder queues. Avoid full-video lists and final array concatenation. A memory retry must restart from committed state and record any changed settings; do not repeatedly retry a configuration that produces sustained severe memory pressure.

## 9. Long videos and continuity

Divide the source into shots, then split each shot into overlapping model windows. Preserve the original character reference throughout a shot and use model-native previous-segment conditioning. Reset generated temporal context across cuts and prevent interpolation across those cuts.

Window length, overlap, and temporal context must satisfy the backend's frame-count constraints. The current Spielberg parameters describe windows of the form `4n + 1` and temporal guidance of one or five frames. Query and validate capabilities rather than allowing arbitrary CLI values.

For a five-second window and one-second overlap, the long-run generated-duration overhead is approximately `5 / (5 - 1) = 1.25`, excluding initialization, padding, reloads, and postprocessing. Smaller windows may fit memory while increasing repeated work and the number of quality-sensitive boundaries.

Bounded memory does not guarantee bounded identity drift. Assess identity, clothing, color, and expression over many chunks, not only the first boundary. Native context can propagate errors from earlier generated frames. Retain reference conditioning, track quality trends, and allow shot-level regeneration.

Write completed chunks incrementally. Resolve overlaps with explicit ownership of final frame indices. Cross-fading is a documented fallback, not the default continuity method. Verify no missing or duplicated frames and exact intended duration after padding and trimming.

## 10. Persistence and resume

Use a versioned job manifest and atomic chunk commits. A chunk is complete only after its outputs, continuity context, and metadata have been persisted and validated.

Persist:

- Source/reference hashes, selected source range, normalized timeline, shot boundaries, and spatial transforms.
- Requested and effective settings for every chunk, including any fallback.
- Backend commit, dependency versions, model checksums, quantization, scheduler, and precision.
- Chunk indices, generated and retained frame ranges, overlap ownership, and padding/trimming rules.
- Completed latents or decoded output as required by the resume strategy, plus the exact generated continuity tail.
- Per-chunk seed and RNG strategy, or RNG state where needed for equivalent continuation.
- Stage status, timing, failures, file checksums, and output paths.

Validate compatibility before resume. Configuration changes invalidate affected chunks and downstream chunks that depend on their generated context. A committed video chunk alone is insufficient if the continuity state has been lost or changed through lossy encoding.

Define deterministic per-chunk randomness where feasible and document backend limits. Distinguish exact continuation from a retry that intentionally changes resolution, window size, or sampling settings. Never hide a fallback inside an otherwise unchanged job manifest.

## 11. Scene preservation in Replace mode

Replace mode requires background/mask conditioning and the appropriate relighting model variant. Pose-derived masks are an initial diagnostic fallback; do not equate them with precise silhouettes. SAM2 or another segmentation method is modular and must have its own memory and tracking evaluation.

Define two preservation policies:

- `model`: the model reconstructs a scene conditioned on source backgrounds and masks. Assess perceptual background consistency; unchanged source pixels are not guaranteed.
- `composite`: composite the generated region onto original source frames using temporally stable masks. Preserve original pixels outside the editable region in the compositing intermediate; final lossy video encoding can alter pixel values.

Mask size is a quality tradeoff. Tight masks can clip a larger character or new clothing, while wider masks permit shape changes and expose more background to regeneration. Provide mask dilation/feather controls and a preview of the editable region.

Test foreground objects, hair, hands, contact with props, shadows, reflections, and camera motion. Precise person segmentation alone does not solve foreground occlusion or recover background hidden by the original performer. Strict compositing requires an explicit occlusion policy and, where necessary, reconstruction of newly exposed regions.

Keep the first replacement scope to compatible body proportions and simple interactions. Report unsupported cases before rendering rather than promising arbitrary character insertion into any scene.

## 12. Conditioning reuse and MLX optimization

Profile before altering output quality. Spielberg's current model forward computes pose projections, face motion features, and text/image projections on each denoising call. Evaluate caching quantities that depend only on fixed conditioning and weights within a chunk.

Separate fixed conditioning from timestep-dependent and latent-dependent operations. Cache only numerically equivalent results. Invalidate caches when inputs, weights, transforms, precision, or configuration change. Measure both memory retained by caches and compute saved; large projected tensors can erase the benefit on 18 GB.

Consider fixed cross-attention context projections only after simpler reuse is validated. Do not cache activations dependent on the evolving noisy latents.

Use `mlx.compile` selectively on stable hot paths. Separate compilation warm-up from steady-state timing. Profile attention, MLP, face processing, VAE, allocation, synchronization, and model loading. Avoid custom Metal kernels until a measured bottleneck justifies them.

## 13. Temporal fast path

Optionally generate fewer diffusion frames at timestamps sampled from the same source duration, then reconstruct intermediate decoded RGB frames using an Apple-Silicon-compatible RIFE stage. Keep this independently switchable as `--temporal-fast`.

Align pose, face, mask, and background timestamps with generated frames. Resolve exact interval counts and VAE alignment before sampling; for example, doubling the intervals between 41 frames produces 81 frames, not 82. Preserve source time through padding and trimming.

RIFE cannot reliably recover events absent from generated frames. Test speech and expression timing, brief hand gestures, fast motion, occlusion, and nonlinear movement. Disable interpolation across shot cuts and support selective full-rate regeneration of failing segments.

FastVideo's cited measurements use a 1.3B INT8 QAD model, not Wan-Animate. Treat speed and quality on the target pipeline as unknown until measured. Do not use reconstruction similarity alone as evidence of correct motion or expression transfer.

## 14. Spatial fast path and delivery resolution

Optionally diffuse at a reduced working resolution, decode with the VAE, then upscale RGB frames. Expose `--spatial-fast` independently of temporal reduction. Clarify whether a scale reduces each spatial axis or total pixel area.

Transform all reference, pose, face, mask, and background coordinates consistently. Do not make latent interpolation/upscaling the default. Keep a basic RGB resampler available so generation has no mandatory learned-upscaler dependency.

Higher output dimensions do not establish recovered detail. Evaluate identity, hands, facial expressions, clothing, edges, background consistency, and temporal shimmer. Learned upscalers may invent detail and change appearance across frames; measure their temporal stability and memory independently.

Allow temporal and spatial fast paths together only after each is characterized separately. Report working resolution, generated FPS, output resolution, output FPS, and processing methods in job metadata.

## 15. Presets and automatic tuning

| Preset | Initial sampling intent | Fast paths | Qualification |
| --- | --- | --- | --- |
| Smoke | Small legal window and low resolution; about 4 steps | Off | Installation and execution only |
| Baseline reference | Small stable window; approximately official 20 steps | Off | Establish identity, motion, and timing quality |
| Fast preview | Measured reduced settings | Optional | Explicitly labeled preview quality |
| Balanced | Measured settings from qualified benchmarks | Independently selectable | Default only after feasibility gates pass |
| Quality | Highest useful stable settings | Mild or off | Measured quality improvement over Balanced |

Auto-tuning optimizes useful quality and completion time subject to memory headroom. Do not simply select the largest window that avoids an immediate allocation error. Include sustained pressure, swap growth, repeated-stage overhead, and boundary quality.

Persist a machine profile with hardware, OS, backend/dependency/weight identifiers, benchmark evidence, selected settings, and safety margin. Invalidate or requalify it after material changes. Profiles are configuration-specific; the filename alone does not certify support.

## 16. Benchmark and quality evaluation

Use a fixed short reference/source pair for reproducible timing, plus a small content suite covering speech, slow and fast full-body motion, occlusion, mismatched proportions, replacement edges, and multiple shots. Include portrait 9:16, landscape 16:9, and square media.

Measure baseline, conditioning reuse, temporal-fast only, spatial-fast only, and both fast paths. Sweep resolution, legal window lengths, temporal context, overlap, and steps within safe limits. Record seeds and repeat selected runs to distinguish variance from improvement.

Record setup/conversion, preprocessing, text/image encoding, conditioning VAE encoding, model load/reload, compile/warm-up, denoise, decode, RIFE, upscale, composite, encode, audio processing, and total time. Report useful output seconds per wall-clock hour as well as per-chunk timings.

Record peak memory by stage, memory pressure, swap changes, free disk space, cache/output bytes, failures, retries, and effective settings. Sustained runs must demonstrate that memory does not grow with total source duration.

Review identity, pose fidelity, face stability, expression timing, hands, clothing, temporal consistency, shot resets, chunk boundaries, and background preservation. A playable MP4 is necessary but insufficient. Retain comparison outputs and structured quality notes; apply explicit acceptance criteria before making a preset the default.

## 17. CLI contract

```bash
python -m app.generate \
  --mode animate \
  --reference ./inputs/character.png \
  --video ./inputs/driver.mp4 \
  --preset balanced \
  --output ./outputs/result.mp4
```

Core options:

```text
--mode animate|replace
--reference PATH; --video PATH; --output PATH
--start SECONDS; --duration SECONDS
--width PIXELS; --height PIXELS; --resolution PRESET
--fps DELIVERY_FPS; --working-fps DIFFUSION_FPS
--frames-per-chunk COUNT; --overlap COUNT
--steps COUNT; --seed INTEGER
--guidance VALUE                     # only when actually supported
--temporal-fast; --spatial-fast
--preserve-audio; --no-audio
--retarget none|basic                # reject if unavailable
--sam2-mask
--preservation model|composite       # Replace mode
--preview-preprocessing
--resume; --keep-intermediates
--upscale none|2x|4x|NAMED_PRESET
```

Provide separate setup/doctor and benchmark commands. Validate legal frame counts, overlap, resolution, backend features, weights, disk space, and cached-profile compatibility before generation. An option must fail clearly if unsupported; accepting a flag while ignoring it is not allowed.

Print requested and effective settings, estimated runtime and disk use, shot/chunk progress, and fallbacks. Estimates must include overlap and measured load/reload overhead. Preserve audio by remuxing when compatible, otherwise explicitly transcode; maintain timeline alignment and trim/pad according to the selected source range.

## 18. Repository layout

```text
character-video/
├── app/
│   ├── generate.py
│   ├── config.py
│   ├── media.py
│   ├── shots.py
│   ├── preprocess.py
│   ├── preview.py
│   ├── engines/
│   │   ├── base.py
│   │   └── wan_animate_mlx.py
│   ├── workers.py
│   ├── chunking.py
│   ├── continuity.py
│   ├── manifests.py
│   ├── memory.py
│   ├── postprocess/
│   │   ├── rife_mlx.py
│   │   ├── upscale.py
│   │   └── composite.py
│   ├── benchmark.py
│   └── telemetry.py
├── machine_profiles/
│   └── m3pro_18gb.json
├── benchmarks/                     # configs and evaluation metadata
├── models/                         # gitignored
├── cache/                          # gitignored
├── outputs/                        # gitignored
├── tests/
├── scripts/
├── pyproject.toml
└── README.md
```

Create the machine profile from actual results, not as a prefilled claim of hardware support. Keep engine-specific preprocessing and temporal constraints behind the capability interface.

## 19. Verification

- Unit checks: chunk/frame ownership, legal window math, timestamp conversion, spatial transforms, cache invalidation, atomic manifests, resume dependencies, audio alignment, and shot resets.
- Integration: a real short Animate render with expected duration and a playable MP4; Replace qualified separately with model and compositing policies where implemented.
- Continuity: two adjacent windows followed by a many-window shot; inspect drift and boundaries, not just frame count.
- Sustained operation: long input, bounded buffers, disk estimates, memory pressure, thermal/runtime changes, cancellation, and interruption/resume.
- Fast paths: compare against a qualified baseline using the content suite; verify timing and expression fidelity as well as completion.
- Failure handling: corrupt inputs, unsupported codec, missing or mismatched weights, low-confidence detections, insufficient disk, worker failure, memory pressure, and invalidated resume state.
- Reproducibility: record backend limits and compare resumed versus uninterrupted output under the documented randomness strategy.
- Offline operation: after setup, render without network access and verify no implicit downloads or external calls.

## 20. Implementation order

1. Audit and pin the backend, dependencies, model sources, conversion behavior, temporal constraints, and supported features.
2. Add minimal stage timing and memory observations; prove weight preparation/import, text/image encoding, VAE encoding, and DiT loading separately on 18 GB.
3. Produce a useful short baseline near official sampling settings. Publish the feasibility result and measured limits before expanding the application.
4. Implement streaming preprocessing, explicit component lifetimes, isolated workers where needed, versioned manifests, atomic commits, and resume.
5. Prove two-chunk native continuity and exact timing; then implement shot handling and qualify a sustained multi-shot render with audio.
6. Profile and validate fixed-conditioning reuse, safe precision choices, VAE tiling/chunking, and selective compilation.
7. Add temporal reduction plus shot-aware RIFE and evaluate against the baseline.
8. Add lower-resolution diffusion plus post-VAE RGB upscaling; evaluate independently, then together with temporal reduction.
9. Qualify machine-specific presets using sustained memory, runtime, and quality evidence.
10. Harden Replace mode, mask previews, relighting requirements, compositing, and occlusion policies. Qualify it separately.
11. Investigate VSA, Animate-specific distillation, or newer animation models only after the measured v1 pipeline is stable.

If an early feasibility stage fails, concentrate on that limiting stage and document alternatives. Do not build the remaining application on an unverified assumption that later fast paths will make the backend viable.

## 21. Non goals and hardware alternatives

V1 does not require native 1080p diffusion, real-time generation, training from scratch, a polished GUI, arbitrary multi-person tracking, or guaranteed animation of every character anatomy. Cloud inference remains outside the requested architecture.

Generic smaller text/image-to-video models are not equivalent substitutes unless they demonstrate the required driving-video motion, expression, identity, and replacement conditioning. A narrower portrait-animation product is an explicit scope alternative if full-body diffusion proves impractical.

Maintain a modular engine interface for a future local workstation. More unified memory would reduce residency constraints on Apple Silicon; a CUDA workstation offers access to upstream GPU implementations. Hardware purchases and new backend commitments require workload-specific benchmarks rather than capacity alone.

## 22. Upstream references

The review observations below reflect upstream material inspected during the project review. Recheck them against the exact commit selected for implementation; they are not measurements of this project on the target Mac.

- [Spielberg](https://github.com/dtellz/spielberg): initial MLX engine candidate, reported 48 GB validation, setup, features, and checkpoint requirements.
- [Spielberg conversion](https://github.com/dtellz/spielberg/blob/master/engine/weights/convert.py): floating-point conversion, selective quantization, auxiliary encoder precision, and relighting merge.
- [Spielberg pipeline](https://github.com/dtellz/spielberg/blob/master/engine/animate/pipeline.py): component residency, temporal-reference conditioning, chunk generation, and output accumulation.
- [Spielberg preprocessing](https://github.com/dtellz/spielberg/blob/master/engine/preprocess/extract.py): full-source loading and pose/face extraction.
- [Spielberg model](https://github.com/dtellz/spielberg/blob/master/engine/animate/model.py): candidate fixed-conditioning computations inside model forward.
- [Spielberg replacement preprocessing](https://github.com/dtellz/spielberg/blob/master/engine/preprocess/replace.py): pose-derived masks and optional SAM2 integration.
- [Official Wan2.2](https://github.com/Wan-Video/Wan2.2): Animate model and reference implementation.
- [Official Animate configuration](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_animate_14B.py): sampling defaults and model/VAE settings.
- [Official preprocessing guide](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md): retargeting, input assumptions, single-person limitations, and mask tradeoffs.
- [Wan-Animate project](https://humanaigc.github.io/wan-animate/): motion, expression, and relighting architecture.
- [FastVideo](https://github.com/hao-ai-lab/FastVideo): optimization and distillation research reference.
- [FastVideo Apple Silicon fast mode](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/design/apple_silicon_fast_mode.md): temporal reduction, RIFE, RGB upsampling, and model-specific measurements.
- [FastVideo offloading](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/inference/offloading.md): memory/offloading reference.
- [FastVideo VSA](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/attention/vsa/index.md) and [DMD](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/distillation/dmd.md): later research only.

## 23. Codex implementation handoff

Implement incrementally for the M3 Pro with 18 GB unified memory. Start by auditing and pinning Spielberg as a replaceable Wan2.2-Animate MLX backend. Instrument setup and every inference stage before attempting a full render. Prove that weight preparation or import, text/image encoding, VAE encoding, DiT loading, denoising, and decoding fit separately. Do not assume a 4-bit model file implies a feasible installation or runtime.

Establish a useful short baseline near official sampling settings and record quality, time, memory pressure, swap, and disk requirements. Treat four-step output as an execution smoke test. If feasibility fails, explain the limiting stage and supported alternatives without silently changing the local-only goal or animation task.

Next implement bounded preprocessing/output streaming, dependency-aware model lifetimes, isolated workers where useful, atomic chunk manifests, native generated-context conditioning, shot resets, exact timeline/audio handling, cancellation, and validated resume. Persist the exact continuity tail and randomness strategy. Test two adjacent chunks and a sustained multi-shot render before claiming long-video support.

Profile fixed-conditioning reuse before adding quality-reducing optimizations. Then evaluate temporal reduction with shot-aware RIFE and reduced-resolution diffusion with post-VAE RGB upscaling, independently and together. Build presets from measured quality, sustained memory, and total runtime. Qualify Replace mode separately with explicit preservation policies, mask previews, relighting, and occlusion limits. Keep VSA and Animate-specific distillation as later research.
