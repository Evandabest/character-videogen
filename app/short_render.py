"""Staged, single-window Animate render with bounded memory."""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def video_dimensions(video: Path) -> tuple[int, int]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "json", str(video)], capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def make_video_canvas(reference: Path, video: Path, output: Path) -> tuple[int, int]:
    """Keep the full reference person inside a canvas matching video aspect ratio."""
    from PIL import Image, ImageFilter, ImageOps

    video_width, video_height = video_dimensions(video)
    limit = min(1.0, 1024 / max(video_width, video_height))
    width = max(1, round(video_width * limit))
    height = max(1, round(video_height * limit))
    with Image.open(reference) as original:
        rgb = original.convert("RGB")
        background = ImageOps.fit(rgb, (width, height)).filter(
            ImageFilter.GaussianBlur(radius=max(1, round(min(width, height) * 0.04)))
        )
        foreground = ImageOps.contain(rgb, (width, height))
        background.paste(foreground, ((width - foreground.width) // 2, (height - foreground.height) // 2))
        background.save(output)
    return width, height


def fit_frame(frame, width: int, height: int, mode: str):
    """Resize an RGB frame to the delivery canvas with an explicit fit policy."""
    import cv2
    import numpy as np

    if mode not in ("pad", "crop") or width < 1 or height < 1:
        raise ValueError("Invalid delivery dimensions or fit mode")
    source_height, source_width = frame.shape[:2]
    scale = (min if mode == "pad" else max)(width / source_width, height / source_height)
    scaled_width = max(1, round(source_width * scale))
    scaled_height = max(1, round(source_height * scale))
    resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
    if mode == "pad":
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        left, top = (width - scaled_width) // 2, (height - scaled_height) // 2
        canvas[top:top + scaled_height, left:left + scaled_width] = resized
        return canvas
    left, top = (scaled_width - width) // 2, (scaled_height - height) // 2
    return resized[top:top + height, left:left + width]


def _save(job: Path, name: str, arrays: dict) -> None:
    import mlx.core as mx

    mx.eval(*arrays.values())
    mx.save_safetensors(str(job / f"{name}.safetensors"), arrays)


def _prepare(args) -> None:
    import numpy as np
    from app.backend import prepare_backend

    prepare_backend()
    from engine.preprocess.extract import preprocess

    reference = args.reference
    if args.canvas == "video":
        reference = args.job / "reference_video_canvas.png"
        make_video_canvas(args.reference, args.video, reference)
    source = args.video
    if args.start > 0:
        source = args.job / "source_segment.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(args.start),
             "-i", str(args.video), "-frames:v", str(args.frames), "-an", "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", str(source)], check=True,
        )
    data = preprocess(str(source), str(reference), resolution_area=args.area, max_frames=args.frames)
    if len(data["pose_frames"]) != args.frames:
        raise RuntimeError(f"The render needs at least {args.frames} driving frames after the start time")
    np.savez_compressed(args.job / "prepared.npz", pose=data["pose_frames"], face=data["face_frames"],
                        reference=data["ref_image"], fps=np.array(data["fps"]))
    print(json.dumps({"stage": "prepare", "shape": list(data["ref_image"].shape), "fps": data["fps"],
                      "start": args.start, "canvas": args.canvas}), flush=True)


def _text(args) -> None:
    import mlx.core as mx
    from app.backend import prepare_backend
    from app.t5_loader import load_sharded_t5
    from app.tokenizer import load

    prepare_backend()
    from mlx_video.models.wan_2.utils import encode_text

    encoder = load_sharded_t5()
    context = encode_text(encoder, load(), args.prompt, 512)
    _save(args.job, "text", {"context": context})
    print(json.dumps({"stage": "text", "shape": list(context.shape),
                      "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 3)}), flush=True)


def _image(args) -> None:
    import mlx.core as mx
    import numpy as np
    from app.backend import prepare_backend
    from app.clip_convert import OUTPUT

    prepare_backend()
    from app.animate_model import ClipEncoder

    with np.load(args.job / "prepared.npz") as data:
        reference = data["reference"]
    chw = mx.array(reference.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
    features = ClipEncoder(OUTPUT)(chw)
    _save(args.job, "image", {"features": features})
    print(json.dumps({"stage": "image", "shape": list(features.shape),
                      "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 3)}), flush=True)


def _vae_encode(args) -> None:
    import mlx.core as mx
    import numpy as np
    from app.backend import prepare_backend
    from app.vae_probe import CONVERTED

    prepare_backend()
    from engine.animate.config import AnimateConfig
    from engine.animate.pipeline import _to_bcthw, get_i2v_mask
    from mlx_video.models.wan_2.utils import load_vae_encoder

    with np.load(args.job / "prepared.npz") as data:
        reference = data["reference"]
        pose = data["pose"]
    encoder = load_vae_encoder(CONVERTED, AnimateConfig.animate_14b())
    h, w = reference.shape[:2]
    count = len(pose)
    ref_lat = encoder.encode(_to_bcthw(reference[None]))[0]
    pose_lat = encoder.encode(_to_bcthw(pose))[0]
    temporal_lat = encoder.encode(mx.zeros((1, 3, count, h, w)))[0]
    mx.eval(ref_lat, pose_lat, temporal_lat)
    lat_h, lat_w = ref_lat.shape[-2:]
    ref = mx.concatenate([get_i2v_mask(1, lat_h, lat_w, 1), ref_lat], axis=0)
    temporal = mx.concatenate([get_i2v_mask(count // 4 + 1, lat_h, lat_w, 0), temporal_lat], axis=0)
    conditioning = mx.concatenate([ref, temporal], axis=1)
    _save(args.job, "vae_conditioning", {"y": conditioning, "pose": pose_lat})
    print(json.dumps({"stage": "vae_encode", "y": list(conditioning.shape),
                      "pose": list(pose_lat.shape), "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 3)}), flush=True)


def _denoise(args) -> None:
    import mlx.core as mx
    import numpy as np
    from app.backend import prepare_backend
    from app.dit_loader import load_sharded_model

    prepare_backend()
    from mlx_video.models.wan_2.scheduler import FlowUniPCScheduler

    with np.load(args.job / "prepared.npz") as data:
        face = data["face"]
    y_and_pose = mx.load(str(args.job / "vae_conditioning.safetensors"))
    context = mx.load(str(args.job / "text.safetensors"))["context"]
    features = mx.load(str(args.job / "image.safetensors"))["features"]
    y, pose = y_and_pose["y"], y_and_pose["pose"]
    face_pixels = mx.array(face.astype(np.float32) / 127.5 - 1.0)
    model = load_sharded_model(memory_fraction=0.78)
    model.prepare_conditioning(features, context, face_pixels)
    mx.set_cache_limit(128 * 1024**2)
    print(json.dumps({"stage": "denoise_prepare", "conditioning_cached": True,
                      "resident_budget_gib": round(model._resident_limit / 1024**3, 3),
                      "mlx_active_gib": round(mx.get_active_memory() / 1024**3, 3),
                      "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 3)}), flush=True)
    mx.random.seed(args.seed)
    latents = mx.random.normal((16, *y.shape[1:]))
    seq_len = math.prod(size // patch for size, patch in zip(y.shape[1:], model.patch_size))
    scheduler = FlowUniPCScheduler()
    scheduler.set_timesteps(args.steps, shift=5.0)
    for step, timestep in enumerate(scheduler.timesteps, 1):
        start = time.monotonic()
        prediction = model(latents, mx.array([timestep]), features, context, seq_len, y, pose, face_pixels)
        latents = scheduler.step(prediction[None], timestep, latents[None]).squeeze(0)
        mx.eval(latents)
        if not bool(mx.all(mx.isfinite(latents)).item()):
            raise RuntimeError(f"Non-finite latents at denoising step {step}")
        print(json.dumps({"stage": "denoise", "step": step, "seconds": round(time.monotonic() - start, 2),
                          "latent_std": round(float(mx.std(latents).item()), 4),
                          "prediction_std": round(float(mx.std(prediction).item()), 4),
                          "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 3)}), flush=True)
    _save(args.job, "latents", {"latents": latents})


def _decode(args) -> None:
    import mlx.core as mx
    import numpy as np
    from app.backend import prepare_backend
    from app.vae_probe import CONVERTED

    prepare_backend()
    from engine.animate.config import AnimateConfig
    from mlx_video.models.wan_2.utils import load_vae_decoder
    from app.vae_decode import decode_streaming

    latents = mx.load(str(args.job / "latents.safetensors"))["latents"]
    decoder = load_vae_decoder(CONVERTED, AnimateConfig.animate_14b())
    decoded = decode_streaming(decoder, latents[None][:, :, 1:])[0]
    mx.eval(decoded)
    frames = np.transpose(np.asarray(decoded[:, :args.frames]), (1, 2, 3, 0))
    frames = ((np.clip(frames, -1, 1) + 1) * 127.5).astype(np.uint8)
    with np.load(args.job / "prepared.npz") as data:
        fps = float(data["fps"])
    width, height = video_dimensions(args.video)
    if width % 2 or height % 2:
        raise RuntimeError("The current H.264 smoke encoder requires even source-video dimensions")
    fitted = [fit_frame(frame, width, height, args.fit) for frame in frames]
    ffmpeg = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
              "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
              "-pix_fmt", "yuv420p", "-y", str(args.output)]
    process = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in fitted:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the smoke clip")
    print(json.dumps({"stage": "decode", "frames": len(fitted), "size": [width, height], "fit": args.fit,
                      "output": str(args.output),
                      "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 3)}), flush=True)


STAGES = {"prepare": _prepare, "text": _text, "image": _image, "vae_encode": _vae_encode,
          "denoise": _denoise, "decode": _decode}


def input_fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def run_stage(command: list[str], job: Path) -> list[dict]:
    """Keep stage logs and metrics even if the child process fails."""
    records = []
    with (job / "stages.log").open("a") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--area", type=int, default=128 * 224)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="视频中的人在做动作")
    parser.add_argument("--fit", choices=("pad", "crop"), default="crop")
    parser.add_argument("--canvas", choices=("video", "reference"), default="video")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.area < 64 * 64 or args.start < 0:
        parser.error("steps must be positive, area at least 4096 pixels, and start nonnegative")
    if args.frames < 5 or args.frames % 4 != 1:
        parser.error("frames must be at least five and have the form 4n+1 (5, 9, 17, ...)")
    if args.stage:
        if args.job is None:
            parser.error("--job is required for internal stages")
        STAGES[args.stage](args)
        return
    if not args.reference.is_file() or not args.video.is_file():
        parser.error("reference and video must exist")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jobs = Path("outputs/jobs")
    jobs.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix="short-", dir=jobs))
    code_root = Path(__file__).resolve().parent
    metadata = {"reference": input_fingerprint(args.reference), "video": input_fingerprint(args.video),
                "parameters": {key: getattr(args, key) for key in
                               ("area", "steps", "frames", "seed", "prompt", "fit", "canvas", "start")},
                "status": "running", "quality_review": "pending", "stages": {}}
    begin = time.monotonic()
    manifest = job / "run.json"
    try:
        for stage in STAGES:
            metadata["current_stage"] = stage
            code_hashes = {path.name: input_fingerprint(path)["sha256"]
                           for path in sorted(code_root.glob("*.py"))}
            metadata["current_code_sha256"] = code_hashes
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
            command = [sys.executable, "-m", "app.short_render", "--stage", stage, "--job", str(job),
                       "--reference", str(args.reference.resolve()), "--video", str(args.video.resolve()),
                       "--output", str(args.output.resolve()), "--area", str(args.area), "--steps", str(args.steps),
                       "--frames", str(args.frames), "--seed", str(args.seed), "--prompt", args.prompt,
                       "--fit", args.fit, "--canvas", args.canvas, "--start", str(args.start)]
            started = time.monotonic()
            records = run_stage(command, job)
            metadata["stages"][stage] = {"seconds": round(time.monotonic() - started, 2),
                                         "records": records, "code_sha256": code_hashes}
        metadata["status"] = "completed"
    except BaseException:
        metadata["status"] = "failed"
        raise
    finally:
        metadata["seconds"] = round(time.monotonic() - begin, 2)
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "job": str(job)}), flush=True)


if __name__ == "__main__":
    main()
