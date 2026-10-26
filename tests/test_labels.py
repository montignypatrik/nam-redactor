"""Property tests for stamping a patient ID inside a blackout box.

The governing rule is that the stamp lives *inside* the box: the box is never
resized to accommodate it.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from nam_redactor.labels import (
    PADDING_PX,
    TARGET_INK_HEIGHT_RATIO,
    _ink_box,
    draw_labels,
    fit_label,
)

PAGE = (1700, 2200)
LABEL = "Patient_0001"
#: A NAM-shaped box, the size seen on a real scan.
NAM_BOX = (600.0, 300.0, 770.0, 336.0)
#: Wide enough that the height target, not the width, is what binds.
ROOMY = (600.0, 300.0, 900.0, 336.0)
SHORT = (600.0, 300.0, 900.0, 314.0)


def _ink_size(font):
    left, top, right, bottom = _ink_box(font, LABEL)
    return right - left, bottom - top


def _bright(image: Image.Image):
    return np.nonzero(np.asarray(image).min(axis=2) > 60)


# -- the stamp stays inside the box ---------------------------------------


def test_stamp_ink_never_escapes_the_box():
    """No widening, no overflow -- this is the whole instruction."""
    for box in [NAM_BOX, ROOMY, SHORT, (600.0, 300.0, 620.0, 336.0)]:
        image = Image.new("RGB", PAGE, (0, 0, 0))
        if draw_labels(image, [(box, LABEL)]) == 0:
            continue
        ys, xs = _bright(image)
        assert xs.min() >= box[0], "ink escaped to the left"
        assert xs.max() <= box[2], "ink escaped to the right"
        assert ys.min() >= box[1], "ink escaped upwards"
        assert ys.max() <= box[3], "ink escaped downwards"


def test_fit_label_only_picks_a_font_and_touches_no_geometry():
    font = fit_label(NAM_BOX, LABEL)
    assert font is not None
    ink_w, ink_h = _ink_size(font)
    assert ink_w + 2 * PADDING_PX <= NAM_BOX[2] - NAM_BOX[0]
    assert ink_h <= NAM_BOX[3] - NAM_BOX[1]


def test_stamp_ink_is_centred_on_both_axes():
    """Centred on the ink, not on the font's line box."""
    image = Image.new("RGB", PAGE, (0, 0, 0))
    assert draw_labels(image, [(ROOMY, LABEL)]) == 1

    ys, xs = _bright(image)
    ink_cy = (ys.min() + ys.max()) / 2
    ink_cx = (xs.min() + xs.max()) / 2
    assert abs(ink_cy - (ROOMY[1] + ROOMY[3]) / 2) <= 1.5, "not vertically centred"
    assert abs(ink_cx - (ROOMY[0] + ROOMY[2]) / 2) <= 1.5, "not horizontally centred"


# -- how full the stamp gets ----------------------------------------------


def test_a_wide_box_lets_the_stamp_reach_the_height_target():
    font = fit_label(ROOMY, LABEL)
    _, ink_h = _ink_size(font)
    box_h = ROOMY[3] - ROOMY[1]
    assert ink_h >= 0.90 * box_h, f"only {ink_h / box_h:.0%} of the height"


def test_the_height_target_is_a_ceiling():
    font = fit_label(ROOMY, LABEL)
    _, ink_h = _ink_size(font)
    assert ink_h <= (ROOMY[3] - ROOMY[1]) * TARGET_INK_HEIGHT_RATIO + 1


def test_a_nam_shaped_box_binds_on_width_not_height():
    """A 12-character stamp is wider than a NAM, so the height target is out
    of reach without widening the box -- which we do not do."""
    font = fit_label(NAM_BOX, LABEL)
    _, ink_h = _ink_size(font)
    box_h = NAM_BOX[3] - NAM_BOX[1]

    assert ink_h < TARGET_INK_HEIGHT_RATIO * box_h, "should be width-bound here"
    assert ink_h > 0.3 * box_h, "but still a usable size"


def test_a_box_too_small_for_a_legible_stamp_is_left_blank():
    """Better a solid black box than an unreadable or overflowing stamp."""
    assert fit_label((0.0, 0.0, 40.0, 8.0), LABEL) is None
    image = Image.new("RGB", PAGE, (0, 0, 0))
    assert draw_labels(image, [((0.0, 0.0, 40.0, 8.0), LABEL)]) == 0


# -- drawing ---------------------------------------------------------------


def test_draw_labels_puts_bright_ink_inside_the_box():
    image = Image.new("RGB", PAGE, (255, 255, 255))
    ImageDraw.Draw(image).rectangle(list(ROOMY), fill=(0, 0, 0))

    assert draw_labels(image, [(ROOMY, LABEL)]) == 1
    assert int((np.asarray(image).min(axis=2) > 200).sum()) > 0


def test_draw_labels_skips_a_box_that_would_overlap_a_stamped_one():
    """Overlapping stamps would destroy one of the two IDs.

    Leaving the second box solid black is the safe failure: the number stays
    hidden either way.
    """
    image = Image.new("RGB", PAGE, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    first = (600.0, 300.0, 900.0, 336.0)
    second = (700.0, 305.0, 1000.0, 341.0)
    draw.rectangle(list(first), fill=(0, 0, 0))
    draw.rectangle(list(second), fill=(0, 0, 0))

    assert draw_labels(image, [(first, LABEL), (second, "Patient_0002")]) == 1


def test_draw_labels_with_nothing_to_do_is_a_no_op():
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    assert draw_labels(image, []) == 0
    assert image.getpixel((50, 50)) == (255, 255, 255)


def test_drawing_is_deterministic():
    """Same input, same pixels -- the stamp is not randomly placed."""
    first = Image.new("RGB", PAGE, (0, 0, 0))
    second = Image.new("RGB", PAGE, (0, 0, 0))
    draw_labels(first, [(NAM_BOX, LABEL)])
    draw_labels(second, [(NAM_BOX, LABEL)])
    assert (np.asarray(first) == np.asarray(second)).all()


@pytest.mark.parametrize("label", ["Patient_0001", "Patient_9999", "Patient_10000"])
def test_longer_labels_never_overflow(label):
    font = fit_label(NAM_BOX, label)
    assert font is not None
    left, top, right, bottom = _ink_box(font, label)
    assert (right - left) + 2 * PADDING_PX <= NAM_BOX[2] - NAM_BOX[0]
