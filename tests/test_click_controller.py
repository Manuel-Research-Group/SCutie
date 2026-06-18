import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def click_controller_module():
    """Import the real gui.click_controller module, bypassing the
    MagicMock() placeholder registered by the root conftest.py.

    gui.click_controller imports gui.ritm.controller and
    gui.ritm.inference.utils, both of which transitively import heavy real
    dependencies (torch internals, scipy, ...) that aren't safely mocked by
    the root conftest. Stub those two modules directly so click_controller
    can be imported for real without pulling in that chain.
    """
    utils_mock = MagicMock()
    inference_mock = MagicMock()
    inference_mock.utils = utils_mock
    sys.modules['gui.ritm.controller'] = MagicMock()
    sys.modules['gui.ritm.inference'] = inference_mock
    sys.modules['gui.ritm.inference.utils'] = utils_mock

    sys.modules.pop('gui.click_controller', None)
    import importlib
    import gui.click_controller as ccm
    importlib.reload(ccm)
    return ccm


def test_default_predictor_params_use_fbrs_and_flip(click_controller_module):
    ClickController = click_controller_module.ClickController

    with patch.object(click_controller_module.utils, 'load_is_model', return_value=MagicMock()), \
         patch.object(click_controller_module, 'InteractiveController') as MockController:
        ctrl = ClickController('dummy_checkpoint.pth', device='cpu')

    params = ctrl._build_predictor_params()
    assert params['brs_mode'] == 'f-BRS-B'
    assert params['with_flip'] is True


def test_set_fast_mode_true_switches_to_nobrs_no_flip(click_controller_module):
    ClickController = click_controller_module.ClickController

    with patch.object(click_controller_module.utils, 'load_is_model', return_value=MagicMock()), \
         patch.object(click_controller_module, 'InteractiveController') as MockController:
        mock_instance = MockController.return_value
        ctrl = ClickController('dummy_checkpoint.pth', device='cpu')

        ctrl.set_fast_mode(True)

    params = ctrl._build_predictor_params()
    assert params['brs_mode'] == 'NoBRS'
    assert params['with_flip'] is False

    mock_instance.reset_predictor.assert_called_once()
    called_params = mock_instance.reset_predictor.call_args[0][0]
    assert called_params['brs_mode'] == 'NoBRS'
    assert called_params['with_flip'] is False


def test_set_fast_mode_same_value_does_not_call_reset_predictor_again(click_controller_module):
    ClickController = click_controller_module.ClickController

    with patch.object(click_controller_module.utils, 'load_is_model', return_value=MagicMock()), \
         patch.object(click_controller_module, 'InteractiveController') as MockController:
        mock_instance = MockController.return_value
        ctrl = ClickController('dummy_checkpoint.pth', device='cpu')

        ctrl.set_fast_mode(True)
        assert mock_instance.reset_predictor.call_count == 1

        ctrl.set_fast_mode(True)
        assert mock_instance.reset_predictor.call_count == 1
