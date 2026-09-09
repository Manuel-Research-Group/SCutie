import json
import os
import tempfile


class _FakeController:
    """Minimal stand-in — only the fields needed for click persistence logic."""
    def __init__(self, workspace):
        self.cfg = {'workspace': workspace}
        self.object_clicks = {}
        self.clicks_file_path = os.path.join(workspace, 'clicks.json')

    def load_clicks(self):
        from gui.main_controller import MainController
        MainController.load_clicks(self)

    def save_clicks(self):
        from gui.main_controller import MainController
        MainController.save_clicks(self)

    def record_click(self, obj_id, frame_idx, x, y, is_neg, dims=None):
        from gui.main_controller import MainController
        MainController.record_click(self, obj_id, frame_idx, x, y, is_neg, dims)


def test_load_clicks_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.load_clicks()
        assert ctrl.object_clicks == {}
        # load must NOT create the file (matches load_polylines behavior)
        assert not os.path.exists(ctrl.clicks_file_path)


def test_save_and_load_clicks_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.object_clicks = {
            1: {5: {"pos": [[10, 20]], "neg": [[30, 40]], "dims": [100, 200]}}
        }
        ctrl.save_clicks()

        ctrl2 = _FakeController(tmp)
        ctrl2.load_clicks()
        assert ctrl2.object_clicks == {
            1: {5: {"pos": [[10, 20]], "neg": [[30, 40]], "dims": [100, 200]}}
        }


def test_save_clicks_writes_string_keys():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.object_clicks = {1: {5: {"pos": [[10, 20]], "neg": [], "dims": [100, 200]}}}
        ctrl.save_clicks()
        with open(ctrl.clicks_file_path) as f:
            raw = json.load(f)
        assert raw == {"1": {"5": {"pos": [[10, 20]], "neg": [], "dims": [100, 200]}}}


def test_record_click_positive_creates_entry_and_persists():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.record_click(1, 5, 10, 20, is_neg=False, dims=[100, 200])
        assert ctrl.object_clicks[1][5]["pos"] == [[10, 20]]
        assert ctrl.object_clicks[1][5]["neg"] == []
        with open(ctrl.clicks_file_path) as f:
            raw = json.load(f)
        assert raw == {"1": {"5": {"pos": [[10, 20]], "neg": [], "dims": [100, 200]}}}


def test_record_click_accumulates_across_calls():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.record_click(1, 5, 10, 20, is_neg=False)
        ctrl.record_click(1, 5, 11, 21, is_neg=False)
        ctrl.record_click(1, 5, 30, 40, is_neg=True)
        assert ctrl.object_clicks[1][5]["pos"] == [[10, 20], [11, 21]]
        assert ctrl.object_clicks[1][5]["neg"] == [[30, 40]]


def test_record_click_separates_objects_and_frames():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.record_click(1, 5, 10, 20, is_neg=False)
        ctrl.record_click(2, 5, 15, 25, is_neg=False)
        ctrl.record_click(1, 6, 12, 22, is_neg=False)
        assert ctrl.object_clicks[1][5]["pos"] == [[10, 20]]
        assert ctrl.object_clicks[2][5]["pos"] == [[15, 25]]
        assert ctrl.object_clicks[1][6]["pos"] == [[12, 22]]
