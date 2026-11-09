import json
import os
import subprocess
import sys
import tempfile

from PIL import Image

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "derive_workspace.py")
PALETTE = [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255] + [7] * (256 * 3 - 12)


def _write(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _load(ws, name):
    with open(os.path.join(ws, name)) as f:
        return json.load(f)


def _mask(path):
    m = Image.new("P", (6, 2), 0)
    for x, obj in enumerate([0, 1, 2, 3, 3, 1]):
        m.putpixel((x, 0), obj)
    m.putpalette(PALETTE)
    m.save(path)


def _make_src(tmp, optional=True):
    src = os.path.join(tmp, "src")
    for d in ("images", "masks", "visualization"):
        os.makedirs(os.path.join(src, d))
    Image.new("RGB", (6, 2)).save(os.path.join(src, "images", "0000001.jpg"))
    _mask(os.path.join(src, "masks", "0000001.png"))
    _mask(os.path.join(src, "masks", "0000002.png"))
    for obj in (1, 2, 3):
        d = os.path.join(src, "soft_masks", str(obj))
        os.makedirs(d)
        Image.new("L", (6, 2)).save(os.path.join(d, "0000001.png"))
    Image.new("RGB", (6, 2)).save(os.path.join(src, "visualization", "x.png"))
    _write(os.path.join(src, "object_labels.json"), {"1": "A", "2": "B", "3": "C"})
    _write(os.path.join(src, "models.json"), {"1": "M", "2": "M", "3": "X"})
    _write(os.path.join(src, "sizes.json"), {"1": "DN300", "2": "DN300"})
    _write(os.path.join(src, "references.json"), {"1": True, "3": True})
    _write(os.path.join(src, "workspace_config.json"), {"scale_factor": 1640.0})
    _write(os.path.join(src, "connections.json"), {"1": [2]})
    if optional:
        _write(os.path.join(src, "clicks.json"),
               {"3": {"0": {"pos": [], "neg": [], "dims": [1, 1]}}})
        _write(os.path.join(src, "inverted.json"), {"2": True})
    return src


def _run(src, dst, *extra):
    return subprocess.run(
        [sys.executable, SCRIPT, "--src", src, "--dst", dst, *extra],
        capture_output=True, text=True,
    )


FULL_ARGS = ("--model", "2=M-inverted", "--remove-ids", "3", "--disable-inverted-flag")


def test_derive_full():
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_src(tmp)
        dst = os.path.join(tmp, "dst")
        r = _run(src, dst, *FULL_ARGS)
        assert r.returncode == 0, r.stderr
        assert _load(dst, "models.json") == {"1": "M", "2": "M-inverted"}
        assert _load(dst, "object_labels.json")["3"] == "None"
        assert "3" not in _load(dst, "sizes.json")
        assert "3" not in _load(dst, "references.json")
        assert "3" not in _load(dst, "clicks.json")
        assert not os.path.exists(os.path.join(dst, "inverted.json"))
        assert _load(dst, "workspace_config.json") == {"scale_factor": 1640.0, "use_inverted_flag": False}
        assert not os.path.exists(os.path.join(dst, "soft_masks", "3"))
        assert os.path.exists(os.path.join(dst, "soft_masks", "2"))
        assert _load(dst, "connections.json") == {"1": [2]}
        assert os.path.exists(os.path.join(dst, "visualization", "x.png"))


def test_masks_keep_palette_and_other_ids():
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_src(tmp)
        dst = os.path.join(tmp, "dst")
        assert _run(src, dst, *FULL_ARGS).returncode == 0
        for name in ("0000001.png", "0000002.png"):
            with Image.open(os.path.join(dst, "masks", name)) as m:
                assert m.mode == "P"
                assert m.getpalette() == PALETTE
                assert [m.getpixel((x, 0)) for x in range(6)] == [0, 1, 2, 0, 0, 1]


def test_source_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_src(tmp)
        assert _run(src, os.path.join(tmp, "dst"), *FULL_ARGS).returncode == 0
        with Image.open(os.path.join(src, "masks", "0000001.png")) as m:
            assert m.getpixel((3, 0)) == 3
        assert os.path.exists(os.path.join(src, "inverted.json"))
        assert os.path.exists(os.path.join(src, "soft_masks", "3"))
        assert _load(src, "models.json") == {"1": "M", "2": "M", "3": "X"}
        assert _load(src, "workspace_config.json") == {"scale_factor": 1640.0}


def test_missing_optional_files_ok():
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_src(tmp, optional=False)
        dst = os.path.join(tmp, "dst")
        r = _run(src, dst, *FULL_ARGS)
        assert r.returncode == 0, r.stderr
        assert not os.path.exists(os.path.join(dst, "clicks.json"))


def test_refuses_existing_destination():
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_src(tmp)
        dst = os.path.join(tmp, "dst")
        os.makedirs(dst)
        _write(os.path.join(dst, "keep.json"), {"k": 1})
        r = _run(src, dst, *FULL_ARGS)
        assert r.returncode != 0
        assert os.listdir(dst) == ["keep.json"]


def test_refuses_dst_inside_src_or_equal():
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_src(tmp)
        before = sorted(os.listdir(src))
        assert _run(src, src, *FULL_ARGS).returncode != 0
        assert _run(src, os.path.join(src, "sub"), *FULL_ARGS).returncode != 0
        assert _run(src, os.path.dirname(src), *FULL_ARGS).returncode != 0
        assert sorted(os.listdir(src)) == before
        with Image.open(os.path.join(src, "masks", "0000001.png")) as m:
            assert m.getpixel((3, 0)) == 3
