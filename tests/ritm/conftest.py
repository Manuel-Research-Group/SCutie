import sys
from unittest.mock import MagicMock

import pytest

_removed_mocks = {}


def _unmock_real_modules():
    """Undo the global mocks from the root conftest.py for modules this
    package's tests need to import for real (numpy/torch + gui.ritm.*)."""
    to_remove = []
    for name, mod in sys.modules.items():
        if not isinstance(mod, MagicMock):
            continue
        if name in ('numpy', 'torch', 'torchvision',
                     'torchvision.transforms',
                     'torchvision.transforms.functional'):
            to_remove.append(name)

    for name in to_remove:
        _removed_mocks[name] = sys.modules.pop(name)


_unmock_real_modules()


@pytest.fixture(scope='module', autouse=True)
def _restore_mocked_modules():
    """After the tests in this module have run, put the original mocks
    back so other test modules see the mocked numpy/torch as expected by
    the root conftest.py."""
    yield
    for name, mock in _removed_mocks.items():
        sys.modules[name] = mock
