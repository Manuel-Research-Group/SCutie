import json
import os
import tempfile
from unittest.mock import MagicMock


class _FakeController:
    """Minimal stand-in exposing only the state used by the inverted-flag logic."""
    def __init__(self, workspace):
        self.cfg = {'workspace': workspace}
        self.workspace_config_path = os.path.join(workspace, 'workspace_config.json')
        self.inverted_file_path = os.path.join(workspace, 'inverted.json')
        self.scale_factor = None
        self.use_inverted_flag = True
        self.object_inverted = {}
        self.object_labels = {}
        self.object_models = {}
        self.object_sizes = {}
        self.object_references = {}
        self.gui = MagicMock()

    def _call(self, name, *args):
        from gui.main_controller import MainController
        return getattr(MainController, name)(self, *args)

    def load_workspace_config(self):
        return self._call('load_workspace_config')

    def save_workspace_config(self, value):
        return self._call('save_workspace_config', value)

    def load_inverted(self):
        return self._call('load_inverted')

    def save_inverted(self):
        return self._call('save_inverted')

    def _find_reference_for(self, obj_id):
        return self._call('_find_reference_for', obj_id)


def _write(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def _same_valves(ctrl):
    ctrl.object_labels = {1: "WaferButterflyValve", 2: "WaferButterflyValve"}
    ctrl.object_models = {1: "VPI-4448-02", 2: "VPI-4448-02"}
    ctrl.object_sizes = {1: "DN300", 2: "DN300"}
    ctrl.object_references = {1: True}
    ctrl.object_inverted = {1: True}


def test_flag_defaults_true_when_key_missing():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.use_inverted_flag = None
        _write(ctrl.workspace_config_path, {"scale_factor": 1640.0})
        ctrl.load_workspace_config()
        assert ctrl.use_inverted_flag is True
        assert ctrl.scale_factor == 1640.0


def test_flag_defaults_true_when_file_missing():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.load_workspace_config()
        assert ctrl.use_inverted_flag is True


def test_flag_false_read_from_config():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        _write(ctrl.workspace_config_path, {"scale_factor": 1640.0, "use_inverted_flag": False})
        ctrl.load_workspace_config()
        assert ctrl.use_inverted_flag is False


def test_flag_string_false_is_false():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        _write(ctrl.workspace_config_path, {"use_inverted_flag": "false"})
        ctrl.load_workspace_config()
        assert ctrl.use_inverted_flag is False


def test_save_workspace_config_preserves_other_keys():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        _write(ctrl.workspace_config_path, {"scale_factor": 1.0, "use_inverted_flag": False})
        ctrl.save_workspace_config(1640.0)
        with open(ctrl.workspace_config_path) as f:
            assert json.load(f) == {"scale_factor": 1640.0, "use_inverted_flag": False}
        assert ctrl.scale_factor == 1640.0


def test_flag_on_loads_inverted_json():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        _write(ctrl.inverted_file_path, {"8": True})
        ctrl.load_inverted()
        assert ctrl.object_inverted == {8: True}


def test_flag_off_ignores_existing_inverted_json():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.use_inverted_flag = False
        _write(ctrl.inverted_file_path, {"8": True})
        ctrl.load_inverted()
        assert ctrl.object_inverted == {}


def test_flag_off_save_inverted_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.use_inverted_flag = False
        ctrl.object_inverted = {8: True}
        ctrl.save_inverted()
        assert not os.path.exists(ctrl.inverted_file_path)


def test_find_reference_ignores_inversion_when_flag_off():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        _same_valves(ctrl)
        ctrl.use_inverted_flag = False
        assert ctrl._find_reference_for(2) == 1


def test_find_reference_respects_inversion_when_flag_on():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        _same_valves(ctrl)
        assert ctrl._find_reference_for(2) is None
