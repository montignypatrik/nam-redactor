"""Draw patient-ID labels in white inside blackout boxes.

The label must be legible in the output PDF, so the font is sized from the
box rather than passed in: that makes the whole module resolution-independent,
which is what lets the review screen and the final render agree.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

#: How much of the box height the drawn ink should fill.
#:
#: This is a target for the *ink*, not the font size: a font's em box is
#: noticeably taller than the glyphs it draws (ascender/descender space plus
#: the underscore's own depth), so sizing the font at this fraction would come
#: out much smaller than intended.
TARGET_INK_HEIGHT_RATIO = 0.95
#: The bundled font is thin; below this it reads poorly as white-on-black.
MIN_FONT_PX = 12
#: Upper bound, so a pathologically tall box cannot pick an absurd size.
MAX_FONT_PX = 400
#: Breathing room between the glyphs and the box edge, in pixels.
PADDING_PX = 3


def _load_font(size: float) -> ImageFont.FreeTypeFont:
    """A scalable bundled font.

    ``load_default`` must always be given a size: called bare it returns the
    unscalable legacy bitmap font. Pillow bundles this font, so there is no
    system font file to depend on -- CI runs on Linux.
    """
    clamped = int(max(MIN_FONT_PX, min(MAX_FONT_PX, size)))
    return ImageFont.load_default(size=clamped)


@lru_cache(maxsize=512)
def _raster_ink(label: str, size: int) -> tuple[int, int, int, int]:
    """Ink extents of the *rendered* glyphs, relative to the draw origin.

    ``font.getbbox`` reports unhinted metrics, but the rasteriser hints glyphs,
    so the ink it actually paints comes out a few pixels narrower and would
    sit off-centre if the metrics were trusted for placement. Measuring a real
    render costs one small draw, cached per (label, size).
    """
    font = _load_font(size)
    _left, top, right, bottom = font.getbbox(label)
    margin = 4
    # getbbox reports top/bottom as offsets FROM THE DRAW ORIGIN, so the ink
    # occupies rows [margin + top, margin + bottom]. Sizing the canvas from
    # (bottom - top) alone would clip the descender and under-report the
    # height -- which then mis-sizes and mis-centres the stamp.
    canvas = Image.new(
        "L",
        (max(1, int(right) + 2 * margin), max(1, int(bottom) + 2 * margin)),
        0,
    )
    ImageDraw.Draw(canvas).text((margin, margin), label, font=font, fill=255)
    ink = canvas.getbbox()
    if ink is None:
        return (0, 0, 0, 0)
    # Back into text-origin space.
    return (ink[0] - margin, ink[1] - margin, ink[2] - margin, ink[3] - margin)


def _ink_box(
    font: ImageFont.FreeTypeFont, label: str
) -> tuple[float, float, float, float]:
    """Ink extents of the glyphs themselves, as actually painted.

    ``getlength`` would be advance width, which includes trailing sidebearing
    and would over-reserve space.
    """
    left, top, right, bottom = _raster_ink(label, int(font.size))
    return float(left), float(top), float(right), float(bottom)


def _fit_font(
    label: str, avail_w: float, avail_h: float
) -> ImageFont.FreeTypeFont | None:
    """Largest font whose ink fits both limits, or ``None`` if none reaches
    the minimum legible size.

    Ink is monotone in font size, so a binary search lands exactly on the
    biggest size that overshoots neither constraint. Sizing by arithmetic
    would drift with the font's own metrics.
    """
    best: int | None = None
    low, high = MIN_FONT_PX, MAX_FONT_PX
    while low <= high:
        middle = (low + high) // 2
        left, top, right, bottom = _ink_box(_load_font(middle), label)
        if (right - left) + 2 * PADDING_PX <= avail_w and (bottom - top) <= avail_h:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return _load_font(best) if best is not None else None


def fit_label(
    box: tuple[float, float, float, float], label: str
) -> ImageFont.FreeTypeFont | None:
    """Largest stamp that fits *inside* ``box``, or ``None`` if none legibly does.

    The stamp targets ``TARGET_INK_HEIGHT_RATIO`` of the box height, but is
    bounded by the box in both dimensions and **the box is never resized**.
    Enlarging a blackout would over-redact the content around the number, so
    the box stays exactly where detection put it -- and where the user, having
    seen it, expects it to stay.

    A 12-character ID is wide relative to a NAM-shaped box, so on those the
    width is what binds and the stamp ends up filling less than the height
    target. That is the correct outcome: the alternative is a wider blackout.
    """
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    if box_w <= 0 or box_h <= 0:
        return None

    font = _fit_font(label, box_w, box_h * TARGET_INK_HEIGHT_RATIO)
    if font is None:
        return None

    left, top, right, bottom = _ink_box(font, label)
    if (right - left) + 2 * PADDING_PX > box_w or (bottom - top) > box_h:
        return None
    return font


def _rects_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def draw_labels(
    image: Image.Image,
    labelled: Sequence[tuple[tuple[float, float, float, float], str]],
) -> int:
    """Draw each ID in white, centred inside its box, which must already be black.

    Call this *after* every black rectangle is drawn: otherwise a later box's
    fill would paint over an earlier box's label.

    The box is never resized. A box too small for a legibly sized stamp, or one
    whose rect overlaps a box already stamped, is skipped and left solid black
    -- the number stays hidden either way, which is the safe way to fail.

    Returns how many labels were actually drawn.
    """
    if not labelled:
        return 0

    draw = ImageDraw.Draw(image)
    placed: list[tuple[float, float, float, float]] = []
    drawn = 0

    for box, label in labelled:
        font = fit_label(box, label)
        if font is None:
            continue
        if any(_rects_overlap(box, other) for other in placed):
            continue
        placed.append(box)

        left, top, right, bottom = _ink_box(font, label)
        # Put the ink's centre on the box centre. ``anchor="mm"`` would centre
        # the font's line box instead, which sits the glyphs a little high
        # because that box includes descender space the ink never touches.
        # Rounded because the rasteriser floors the origin, and _ink_box
        # measured integer offsets from it.
        draw.text(
            (
                round((box[0] + box[2]) / 2.0 - (right - left) / 2.0 - left),
                round((box[1] + box[3]) / 2.0 - (bottom - top) / 2.0 - top),
            ),
            label,
            font=font,
            fill=(255, 255, 255),
        )
        drawn += 1

    return drawn
