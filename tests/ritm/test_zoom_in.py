import logging

import numpy as np
import torch

from gui.ritm.inference.transforms.zoom_in import ZoomIn
from gui.ritm.inference.clicker import Click


def _make_zoom_in():
    return ZoomIn(
        target_size=400,
        skip_clicks=1,
        expansion_ratio=1.4,
        min_crop_size=200,
        recompute_thresh_iou=0.5,
        prob_thresh=0.5,
    )


def _image_nd(height, width):
    return torch.zeros((1, 3, height, width), dtype=torch.float32)


def _clicks_list(n):
    # Enough clicks to exceed skip_clicks=1 so transform() doesn't early-return.
    return [Click(is_positive=True, coords=(10.0, 10.0)) for _ in range(n)]


def test_exploded_roi_is_ignored_and_warns(caplog):
    height, width = 400, 400

    zoom_in = _make_zoom_in()

    # First call: prev_probs has a small "real" blob only (no ghost pixel).
    # This establishes a sane self._object_roi.
    probs1 = np.zeros((1, 1, height, width), dtype=np.float32)
    probs1[0, 0, 10:30, 10:30] = 1.0  # small blob near (10, 10)
    zoom_in._prev_probs = probs1

    image_nd = _image_nd(height, width)
    clicks_lists = [_clicks_list(2)]

    zoom_in.transform(image_nd, clicks_lists)

    assert zoom_in._object_roi is not None
    first_roi = zoom_in._object_roi
    first_area = (first_roi[1] - first_roi[0] + 1) * (first_roi[3] - first_roi[2] + 1)
    assert first_area < 0.70 * (height * width)

    # Second call: prev_probs now also has an isolated "ghost" pixel far away,
    # which would blow up get_bbox_from_mask's bbox to cover most of the image.
    probs2 = probs1.copy()
    probs2[0, 0, height - 1, width - 1] = 1.0  # ghost pixel, opposite corner
    zoom_in._prev_probs = probs2

    with caplog.at_level(logging.WARNING):
        zoom_in.transform(image_nd, clicks_lists)

    # The ROI must not have been updated to the exploded bbox.
    assert zoom_in._object_roi == first_roi

    new_area = (zoom_in._object_roi[1] - zoom_in._object_roi[0] + 1) * \
               (zoom_in._object_roi[3] - zoom_in._object_roi[2] + 1)
    assert new_area <= 0.70 * (height * width)

    assert any('ZoomIn' in record.message and 'explos' in record.message.lower()
                for record in caplog.records)
