"""Preview a bounded Animate pose/face extraction before an expensive render."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from app.backend import prepare_backend


def probe(video: Path, reference: Path, output: Path, *, frames: int = 5, area: int = 256 * 448) -> dict:
    if frames < 1 or frames > 25:
        raise ValueError("frames must be between 1 and 25 for this bounded preview")
    if not video.is_file() or not reference.is_file():
        raise FileNotFoundError("Reference or driving video is missing")
    prepare_backend()
    from engine.preprocess.extract import preprocess

    begin = time.monotonic()
    data = preprocess(str(video), str(reference), resolution_area=area, max_frames=frames)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data["ref_image"]).save(output / "reference_resized.png")
    pose_pixels = []
    face_pixels = []
    pose_height_fraction = []
    face_luma_std = []
    for index, (pose, face) in enumerate(zip(data["pose_frames"], data["face_frames"])):
        Image.fromarray(pose).save(output / f"pose_{index:03d}.png")
        Image.fromarray(face).save(output / f"face_{index:03d}.png")
        pose_pixels.append(int(np.count_nonzero(pose)))
        face_pixels.append(int(np.count_nonzero(face)))
        occupied_rows = np.where(np.any(pose > 0, axis=(1, 2)))[0]
        pose_height_fraction.append(round((int(occupied_rows[-1] - occupied_rows[0] + 1) / pose.shape[0]) if len(occupied_rows) else 0.0, 3))
        face_luma_std.append(round(float(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY).std()), 2))
    warnings = []
    if any(height < 0.4 for height in pose_height_fraction):
        warnings.append("Rendered pose is under 40% of frame height in at least one frame; inspect subject detection and framing")
    if any(contrast < 8 for contrast in face_luma_std):
        warnings.append("Face crop has very low grayscale contrast in at least one frame; expression transfer may be unreliable")
    result = {
        "frames": len(data["pose_frames"]),
        "fps": data["fps"],
        "working_shape": list(data["ref_image"].shape[:2]),
        "nonzero_pose_values": pose_pixels,
        "nonzero_face_values": face_pixels,
        "pose_height_fraction": pose_height_fraction,
        "face_luma_std": face_luma_std,
        "warnings": warnings,
        "seconds": round(time.monotonic() - begin, 2),
        "output": str(output),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/preprocess_preview"))
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--area", type=int, default=256 * 448)
    args = parser.parse_args()
    print(json.dumps(probe(args.video, args.reference, args.output, frames=args.frames, area=args.area)), flush=True)


if __name__ == "__main__":
    main()
