"""Map boxes on working pages back to the original uploaded PDF.

Detection, review and appointment rows all happen on *working pages*: either
the uploaded pages as-is, or pages rebuilt one per zone (see
:mod:`nam_redactor.zones`). Working-page numbers change whenever the zones do,
so they are never a reference to the source. Each working page instead
carries a :class:`PageProvenance` naming its source page and the exact pixel
crop taken from it.

Coordinate conventions, used throughout the app:

* **normalized** -- ``x1, y1, x2, y2`` in 0..1, origin top-left, y down,
  relative to the page *as displayed* (``/Rotate`` applied; what
  ``page.rect`` and the rendered image show).
* **pixels** -- the same, on a raster rendered at ``render_dpi``.
* **points** -- PDF points (1/72 in), displayed orientation, origin top-left.
* **pdf_unrotated** -- PDF user space of the source page with rotation
  undone, as stored in the file (``page.derotation_matrix`` applied).

A working page is rendered from its crop pixel for pixel: its raster at
``render_dpi`` is exactly ``crop_px`` in size, so a working-normalized
coordinate maps to a source pixel with no resampling error.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import pymupdf

COORDINATE_CONVENTION = {
    "normalized": (
        "x1,y1,x2,y2 in 0..1; origin top-left, y down; relative to the page as "
        "displayed (PDF /Rotate applied)"
    ),
    "pixels": "same orientation, on a raster rendered at render_dpi",
    "points": "PDF points (1/72 in), displayed orientation, origin top-left",
    "pdf_unrotated": (
        "PDF user space of the source page with its rotation undone "
        "(page.derotation_matrix applied)"
    ),
}


@dataclass(frozen=True, slots=True)
class PageProvenance:
    """Where one working page came from."""

    working_page: int
    source_page: int  # 1-based, in the original upload
    render_dpi: int
    #: Source page raster at ``render_dpi``, displayed orientation.
    source_raster_px: tuple[int, int]
    #: The crop on that raster, ``(x1, y1, x2, y2)`` pixels, end-exclusive.
    crop_px: tuple[int, int, int, int]
    #: Source page size in points, displayed orientation (``page.rect``).
    source_page_pt: tuple[float, float]
    source_rotation: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("source_raster_px", "crop_px", "source_page_pt"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PageProvenance":
        return cls(
            working_page=int(data["working_page"]),
            source_page=int(data["source_page"]),
            render_dpi=int(data["render_dpi"]),
            source_raster_px=tuple(int(v) for v in data["source_raster_px"]),  # type: ignore[arg-type]
            crop_px=tuple(int(v) for v in data["crop_px"]),  # type: ignore[arg-type]
            source_page_pt=tuple(float(v) for v in data["source_page_pt"]),  # type: ignore[arg-type]
            source_rotation=int(data["source_rotation"]),
        )

    @property
    def crop_size_px(self) -> tuple[int, int]:
        return (self.crop_px[2] - self.crop_px[0], self.crop_px[3] - self.crop_px[1])


def source_page_info(page: pymupdf.Page) -> tuple[tuple[float, float], int]:
    """``(displayed size in points, rotation)`` for a source page."""
    return (float(page.rect.width), float(page.rect.height)), int(page.rotation)


def source_crop(prov: PageProvenance) -> dict[str, float]:
    """The working page's footprint on its source page, normalized."""
    return working_to_source({"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}, prov)


def working_to_source(
    bounds: Mapping[str, float], prov: PageProvenance
) -> dict[str, float]:
    """A working-page normalized box as a source-page normalized box."""
    cx1, cy1, cx2, cy2 = prov.crop_px
    width, height = prov.source_raster_px
    return {
        "x1": round((cx1 + bounds["x1"] * (cx2 - cx1)) / width, 6),
        "y1": round((cy1 + bounds["y1"] * (cy2 - cy1)) / height, 6),
        "x2": round((cx1 + bounds["x2"] * (cx2 - cx1)) / width, 6),
        "y2": round((cy1 + bounds["y2"] * (cy2 - cy1)) / height, 6),
    }


def source_to_points(
    source_bounds: Mapping[str, float], prov: PageProvenance
) -> dict[str, float]:
    """A source-normalized box in displayed PDF points."""
    width_pt, height_pt = prov.source_page_pt
    return {
        "x1": round(source_bounds["x1"] * width_pt, 3),
        "y1": round(source_bounds["y1"] * height_pt, 3),
        "x2": round(source_bounds["x2"] * width_pt, 3),
        "y2": round(source_bounds["y2"] * height_pt, 3),
    }


def points_to_unrotated(
    points: Mapping[str, float], page: pymupdf.Page
) -> dict[str, float]:
    """Displayed points in the source page's unrotated user space."""
    rect = pymupdf.Rect(points["x1"], points["y1"], points["x2"], points["y2"])
    rect = (rect * page.derotation_matrix).normalize()
    return {
        "x1": round(rect.x0, 3), "y1": round(rect.y0, 3),
        "x2": round(rect.x1, 3), "y2": round(rect.y1, 3),
    }


def trace(
    bounds: Mapping[str, float], prov: PageProvenance
) -> dict[str, Any]:
    """Everything needed to find a working-page box on the original page."""
    source = working_to_source(bounds, prov)
    return {
        "source_page": prov.source_page,
        "bounds": source,
        "bounds_pt": source_to_points(source, prov),
    }
