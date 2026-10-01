import numpy as np

from app.short_render import fit_frame


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
