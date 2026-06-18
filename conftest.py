import sys
from unittest.mock import MagicMock

# Stub heavy dependencies before any project code is imported
_MOCKED = [
    'cv2',
    'torch',
    'torchvision',
    'torchvision.transforms',
    'torchvision.transforms.functional',
    'numpy',
    'omegaconf',
    'cutie',
    'cutie.model',
    'cutie.model.cutie',
    'cutie.inference',
    'cutie.inference.inference_core',
    'cutie.utils',
    'cutie.utils.download_models',
    'cotracker',
    'cotracker.predictor',
    'PySide6',
    'PySide6.QtWidgets',
    'gui.interaction',
    'gui.interactive_utils',
    'gui.resource_manager',
    'gui.gui',
    'gui.click_controller',
    'gui.reader',
    'gui.exporter',
    'gui.remote_sam_controller',
]

for _mod in _MOCKED:
    sys.modules.setdefault(_mod, MagicMock())

# Make os.environ.pop a no-op for the QT env var removal at import time
import os as _os
_orig_pop = _os.environ.pop

def _safe_pop(key, *args):
    return _orig_pop(key, *args) if key in _os.environ else (args[0] if args else None)

_os.environ.pop = _safe_pop

import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
