import json
import os
import subprocess
import sys
import tempfile

from PIL import Image


def _make_fixture(tmp):
    images = os.path.join(tmp, "images")
    masks = os.path.join(tmp, "masks")
    os.makedirs(images)
    os.makedirs(masks)
    # One frame named "0000001"; frame_idx 0 maps to it.
    Image.new("RGB", (40, 30), (120, 120, 120)).save(os.path.join(images, "0000001.jpg"))
    # Palettized mask: a block of object id 1.
    m = Image.new("P", (40, 30), 0)
    for yy in range(5, 15):
        for xx in range(5, 20):
            m.putpixel((xx, yy), 1)
    m.putpalette([0, 0, 0] + [255, 0, 0] + [0] * (256 * 3 - 6))
    m.save(os.path.join(masks, "0000001.png"))
    with open(os.path.join(tmp, "clicks.json"), "w") as f:
        json.dump({"1": {"0": {"pos": [[10, 10]], "neg": [[25, 25]], "dims": [30, 40]}}}, f)


def test_render_writes_valid_png():
    with tempfile.TemporaryDirectory() as tmp:
        _make_fixture(tmp)
        out = os.path.join(tmp, "fig.png")
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "render_click_figure.py")
        result = subprocess.run(
            [sys.executable, script, "--workspace", tmp, "--obj", "1",
             "--frame", "0", "--out", out],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert os.path.exists(out)
        with Image.open(out) as im:
            im.verify()  # raises if the PNG is corrupt
