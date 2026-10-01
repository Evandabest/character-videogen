import numpy as np
from PIL import Image

from app.short_render import fit_frame, make_video_canvas


def test_pad_matches_video_canvas_without_cropping():
    portrait = np.full((24, 12, 3), 180, dtype=np.uint8)
    output = fit_frame(portrait, 40, 30, "pad")
    assert output.shape == (30, 40, 3)
    assert np.all(output[:, 0] == 0)
    assert np.all(output[:, 20] == 180)


def test_crop_fills_video_canvas():
    portrait = np.full((24, 12, 3), 180, dtype=np.uint8)
    output = fit_frame(portrait, 40, 30, "crop")
    assert output.shape == (30, 40, 3)
    assert np.all(output == 180)


def test_video_canvas_matches_driving_aspect_and_preserves_reference(tmp_path, monkeypatch):
    monkeypatch.setattr("app.short_render.video_dimensions", lambda _: (40, 30))
    reference = tmp_path / "reference.png"
    output = tmp_path / "canvas.png"
    Image.fromarray(np.full((20, 10, 3), (220, 20, 20), dtype=np.uint8)).save(reference)
    assert make_video_canvas(reference, tmp_path / "video.mp4", output) == (40, 30)
    canvas = np.asarray(Image.open(output))
    assert canvas.shape == (30, 40, 3)
    assert np.array_equal(canvas[15, 20], (220, 20, 20))
