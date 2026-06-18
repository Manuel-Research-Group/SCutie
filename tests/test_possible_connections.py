import json
import os
import tempfile
import pytest


class _FakeController:
    """Minimal stand-in — only the fields needed for possible_connections logic."""
    def __init__(self, workspace):
        self.cfg = {'workspace': workspace}
        self.object_possible_connections = {}
        self.possible_connections_file_path = os.path.join(workspace, 'possible_connections.json')
        self.object_labels = {}
        self.object_models = {}
        self.object_sizes = {}
        self.object_inverted = {}
        self.object_references = {}

    def load_possible_connections(self):
        from gui.main_controller import MainController
        MainController.load_possible_connections(self)

    def save_possible_connections(self):
        from gui.main_controller import MainController
        MainController.save_possible_connections(self)


def test_load_possible_connections_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.load_possible_connections()
        assert ctrl.object_possible_connections == {}
        assert os.path.exists(ctrl.possible_connections_file_path)


def test_load_possible_connections_reads_json():
    with tempfile.TemporaryDirectory() as tmp:
        data = {"3": 4, "5": 2}
        with open(os.path.join(tmp, 'possible_connections.json'), 'w') as f:
            json.dump(data, f)
        ctrl = _FakeController(tmp)
        ctrl.load_possible_connections()
        assert ctrl.object_possible_connections == {3: 4, 5: 2}


def test_save_possible_connections_writes_json():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.object_possible_connections = {3: 4, 5: 2}
        ctrl.save_possible_connections()
        with open(os.path.join(tmp, 'possible_connections.json')) as f:
            data = json.load(f)
        assert data == {"3": 4, "5": 2}


def test_find_reference_for_returns_none_when_no_refs():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.object_labels   = {1: "valve", 2: "valve"}
        ctrl.object_models   = {1: "VPI-A", 2: "VPI-A"}
        ctrl.object_sizes    = {1: "DN50",  2: "DN50"}
        ctrl.object_inverted = {1: False,   2: False}
        ctrl.object_references = {1: False, 2: False}
        from gui.main_controller import MainController
        result = MainController._find_reference_for(ctrl, 2)
        assert result is None


def test_find_reference_for_returns_matching_ref():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.object_labels   = {1: "valve", 2: "valve", 3: "pump"}
        ctrl.object_models   = {1: "VPI-A", 2: "VPI-A", 3: "VPI-A"}
        ctrl.object_sizes    = {1: "DN50",  2: "DN50",  3: "DN50"}
        ctrl.object_inverted = {1: False,   2: False,   3: False}
        ctrl.object_references = {1: True, 2: False, 3: False}
        from gui.main_controller import MainController
        result = MainController._find_reference_for(ctrl, 2)
        assert result == 1


def test_find_reference_for_ignores_different_label():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.object_labels   = {1: "pump",  2: "valve"}
        ctrl.object_models   = {1: "VPI-A", 2: "VPI-A"}
        ctrl.object_sizes    = {1: "DN50",  2: "DN50"}
        ctrl.object_inverted = {1: False,   2: False}
        ctrl.object_references = {1: True, 2: False}
        from gui.main_controller import MainController
        result = MainController._find_reference_for(ctrl, 2)
        assert result is None


def test_find_reference_for_ignores_self():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.object_labels   = {1: "valve"}
        ctrl.object_models   = {1: "VPI-A"}
        ctrl.object_sizes    = {1: "DN50"}
        ctrl.object_inverted = {1: False}
        ctrl.object_references = {1: True}
        from gui.main_controller import MainController
        result = MainController._find_reference_for(ctrl, 1)
        assert result is None


def test_on_possible_connections_changed_stores_and_saves():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.curr_object = 3
        ctrl.object_references = {3: True}
        from gui.main_controller import MainController
        MainController.on_possible_connections_changed(ctrl, 4)
        assert ctrl.object_possible_connections[3] == 4
        with open(ctrl.possible_connections_file_path) as f:
            data = json.load(f)
        assert data == {"3": 4}


def test_on_possible_connections_changed_non_ref_does_not_write():
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = _FakeController(tmp)
        ctrl.curr_object = 2
        ctrl.object_references = {2: False}
        from gui.main_controller import MainController
        MainController.on_possible_connections_changed(ctrl, 3)
        assert 2 not in ctrl.object_possible_connections
        assert not os.path.exists(ctrl.possible_connections_file_path)
