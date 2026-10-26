"""Split pages into zones, each of which becomes a page of its own.

A dense page -- a schedule, a list of visits -- carries many patients at
once. Cropping it into zones first gives detection and review small pages
with little on them, and the output PDF is made of those zones.

Crops are rasterized rather than clipped as vectors. Rendering the page
exactly as it is displayed makes rotation a non-issue (``page.rect`` and the
browser preview agree by construction), and each new page is sized so that
rasterizing it again at the same DPI reproduces the crop pixel for pixel.
Nothing is lost: the output was always going to be an image-only PDF.

Each output page is described by a
:class:`~nam_redactor.provenance.PageProvenance`, so anything placed on it
later can be traced back to the original page and pixels.
"""
from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pymupdf

from .provenance import PageProvenance, source_page_info
from .redactor import _reading_order, rasterize_page

#: A zone smaller than this on either side, in rendered pixels, is a stray
#: click rather than a region anyone meant to keep.
MIN_ZONE_PX = 8


def _crop_px(
    zone: Mapping[str, float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(width, int(round(zone["x1"] * width))))
    y1 = max(0, min(height, int(round(zone["y1"] * height))))
    x2 = max(0, min(width, int(round(zone["x2"] * width))))
    y2 = max(0, min(height, int(round(zone["y2"] * height))))
    if x2 - x1 < MIN_ZONE_PX or y2 - y1 < MIN_ZONE_PX:
        return None
    return x1, y1, x2, y2


def crop_to_zones(
    input_path: Path,
    out_path: Path,
    zones_by_page: Mapping[int, Sequence[Mapping[str, Any]]],
    dpi: int = 200,
) -> list[PageProvenance]:
    """Write a PDF with one page per zone.

    Returns the provenance of each output page, in output order: its source
    page and the exact pixel crop taken from it.

    ``zones_by_page`` maps a 1-based page number to normalized (0..1) boxes.
    Output follows the document: page by page, and within a page in reading
    order (top to bottom, then left to right), however the zones were drawn.
    Pages with no zone are left out -- choosing zones is choosing what to keep.
    """
    src = pymupdf.open(str(input_path))
    out = pymupdf.open()
    sources: list[PageProvenance] = []
    try:
        for page_num in sorted(zones_by_page):
            if not 1 <= page_num <= len(src):
                continue
            zones = _reading_order([dict(zone) for zone in zones_by_page[page_num]])
            if not zones:
                continue
            page_pt, rotation = source_page_info(src[page_num - 1])
            rendered = rasterize_page(src[page_num - 1], dpi=dpi)
            try:
                width, height = rendered.size
                for zone in zones:
                    box = _crop_px(zone, width, height)
                    if box is None:
                        continue
                    crop = rendered.crop(box)
                    try:
                        new_page = out.new_page(
                            width=crop.width * 72.0 / dpi,
                            height=crop.height * 72.0 / dpi,
                        )
                        buf = io.BytesIO()
                        crop.save(buf, format="PNG")
                        new_page.insert_image(new_page.rect, stream=buf.getvalue())
                        sources.append(PageProvenance(
                            working_page=len(sources) + 1,
                            source_page=page_num,
                            render_dpi=dpi,
                            source_raster_px=(width, height),
                            crop_px=box,
                            source_page_pt=page_pt,
                            source_rotation=rotation,
                        ))
                    finally:
                        crop.close()
            finally:
                rendered.close()

        if not sources:
            raise ValueError("No usable zone was given.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(str(out_path), deflate=True)
        return sources
    finally:
        src.close()
        out.close()


def whole_page_provenance(input_path: Path, dpi: int = 200) -> list[PageProvenance]:
    """Provenance for keeping the document as uploaded: each working page is
    its source page, uncropped.

    The raster size is the one :func:`rasterize_page` produces at ``dpi``,
    computed without rendering.
    """
    scale = dpi / 72.0
    out: list[PageProvenance] = []
    with pymupdf.open(str(input_path)) as src:
        for index, page in enumerate(src):
            irect = (page.rect * pymupdf.Matrix(scale, scale)).irect
            page_pt, rotation = source_page_info(page)
            out.append(PageProvenance(
                working_page=index + 1,
                source_page=index + 1,
                render_dpi=dpi,
                source_raster_px=(irect.width, irect.height),
                crop_px=(0, 0, irect.width, irect.height),
                source_page_pt=page_pt,
                source_rotation=rotation,
            ))
    return out
