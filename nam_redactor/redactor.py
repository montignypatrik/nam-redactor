"""Core PDF rasterization and black-box redaction pipeline.

The pipeline is split in two so a caller can review (and edit) the detected
regions before committing them to an output PDF:

``detect_nam_pages``
    Slow. Rasterizes and OCRs every page, returning normalized NAM and
    patient-name boxes. Runs the OCR engine exactly once per page and never
    writes a PDF.
``render_redacted_pdf``
    Fast. Re-rasterizes the source and applies whatever boxes it is handed.
    Never runs OCR, so hand-edited boxes cost nothing extra.

``redact_pdf`` composes the two for callers that just want one call.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pymupdf
from PIL import Image, ImageDraw

from .labels import draw_labels
from .locator import _locate_pairs
from .names import NameModel, NamePage, locate_name_regions
from .ocr import OCRResult, PaddleOCREngine


@dataclass(frozen=True, slots=True)
class PageProgress:
    """Progress after one detect step.

    ``page_num`` of 0 is a sentinel meaning "starting": the first page stalls
    while PaddleOCR lazily loads its models, so callers get a signal before
    the long wait rather than after it.
    """

    page_num: int
    total_pages: int
    nams_on_page: int
    total_nams: int


@dataclass(frozen=True, slots=True)
class PageDetection:
    """Detected regions for one page, normalized 0..1 against the raster.

    NAM boxes carry ``patient_id`` alongside their geometry, so the pseudonym
    travels with the region. The canonical NAM behind it does not: it is used
    to assign the ID and then dropped, so no downstream payload or artifact
    can leak it.

    Name boxes carry ``"kind": "name"`` and no ID -- a name is blacked out,
    not pseudonymized. NAM boxes are listed first.
    """

    page_num: int
    width: int
    height: int
    boxes: tuple[dict[str, Any], ...]


def _reading_order(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort boxes top-to-bottom, left-to-right.

    ``_dedupe`` orders by descending confidence, which is arbitrary as a
    numbering order. Centre-y is bucketed by half a median box height so two
    NAMs on one visual line whose OCR tops differ by a few pixels still order
    left-to-right instead of being split across rows.
    """
    if not boxes:
        return []
    heights = sorted(b["y2"] - b["y1"] for b in boxes)
    tolerance = heights[len(heights) // 2] / 2.0 or 1.0

    def key(box: dict[str, Any]) -> tuple[int, float]:
        centre = (box["y1"] + box["y2"]) / 2.0
        return (round(centre / tolerance), box["x1"])

    return sorted(boxes, key=key)


def resolve_device(requested: str = "auto") -> str:
    """Resolve which device inference runs on.

    ``"auto"`` uses the GPU when this Paddle build can actually reach one and
    falls back to CPU otherwise, so the same code runs on a GPU workstation
    and in GPU-less CI. Anything else is passed through unchanged.
    """
    if requested != "auto":
        return requested
    try:
        import paddle
    except ImportError:
        return "cpu"
    try:
        if (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        ):
            return "gpu:0"
    except Exception:  # noqa: BLE001 - any probe failure means "no GPU"
        return "cpu"
    return "cpu"


def default_engine(device: str = "auto") -> PaddleOCREngine:
    """The engine configuration this project targets.

    Models are named explicitly, which is why no ``lang`` is passed: PaddleOCR
    ignores ``lang`` whenever model names or directories are given, so setting
    it would be silently inert. To use the Latin-script recognition model for
    accented French labels, name it here directly
    (``latin_PP-OCRv5_mobile_rec``) -- but note it cannot help NAM *digits*,
    which are what detection depends on.
    """
    return PaddleOCREngine(
        device=resolve_device(device),
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        enable_mkldnn=False,
    )


def rasterize_page(page: pymupdf.Page, dpi: int = 200) -> Image.Image:
    """Rasterize a single PDF page at specified DPI with page rotation applied."""
    scale = dpi / 72.0
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return image


def redact_image(
    image: Image.Image,
    regions: Sequence[dict[str, Any]],
) -> Image.Image:
    """Black out the given boxes, labelling any that carry a patient ID.

    Two passes -- all rectangles, then all labels. Drawing each rectangle
    immediately would let a later box's fill paint over an earlier box's white
    label wherever two boxes overlap.

    A label is drawn only when the region carries a ``patient_id``, so callers
    that pass plain geometry get an unlabelled blackout exactly as before.
    """
    width, height = image.size
    draw = ImageDraw.Draw(image)

    boxes: list[tuple[tuple[float, float, float, float], str | None]] = []
    for reg in regions:
        x1 = max(0, int(round(reg["x1"] * width)))
        y1 = max(0, int(round(reg["y1"] * height)))
        x2 = min(width, int(round(reg["x2"] * width)))
        y2 = min(height, int(round(reg["y2"] * height)))
        if x2 > x1 and y2 > y1:
            label = reg.get("patient_id")
            boxes.append(((x1, y1, x2, y2), str(label) if label else None))

    for box, _label in boxes:
        draw.rectangle(list(box), fill=(0, 0, 0))

    labelled = [(box, label) for box, label in boxes if label]
    if labelled:
        draw_labels(image, labelled)

    return image


def _ocr_image(engine: PaddleOCREngine, image: Image.Image, page_num: int) -> list[OCRResult]:
    """Run OCR on an in-memory image via the engine's file-path interface."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_image_path = tmp_file.name
        image.save(tmp_image_path, format="PNG")
    try:
        return engine.process_image(tmp_image_path, page_num=page_num)
    finally:
        if os.path.exists(tmp_image_path):
            try:
                os.unlink(tmp_image_path)
            except OSError:
                pass


def _check_paths(input_path: Path, out_path: Path | None = None) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if out_path is not None and input_path.resolve() == out_path.resolve():
        raise ValueError("Output file path must be different from input file path.")


def detect_nam_pages(
    input_path: Path,
    dpi: int = 200,
    engine: PaddleOCREngine | None = None,
    progress_callback: Callable[[PageProgress], None] | None = None,
    patient_ids: dict[str, str] | None = None,
    name_model: NameModel | None = None,
    detect_names: bool = True,
) -> tuple[list[PageDetection], int]:
    """Locate NAM and patient-name regions on every page. Slow phase; no PDF
    is written.

    The callback is invoked once with ``page_num=0`` before any work, then
    once per page. It is the callback's job not to raise -- exceptions are
    not swallowed here, because a failure to report progress should not be
    mistaken for a failure to redact.

    ``patient_ids`` maps a canonical NAM to its pseudonym. Pass a dict to have
    it populated in place, which lets a caller keep numbering consistent with
    identities assigned later (see the web layer's typed-NAM endpoint);
    otherwise a private one is used.

    Names are found once every page has been read, because a name is learned
    from a NAM that may sit on a later page (see :mod:`nam_redactor.names`).
    ``name_model`` adds NER on top of that; without one, only NAM-anchored and
    labelled names are found. Progress counts NAMs only.

    Returns ``(pages, total_pages)``.
    """
    _check_paths(input_path)
    if engine is None:
        engine = default_engine()
    if patient_ids is None:
        patient_ids = {}

    doc = pymupdf.open(str(input_path))
    try:
        total_pages = len(doc)
        if progress_callback is not None:
            progress_callback(PageProgress(0, total_pages, 0, 0))

        pages: list[PageDetection] = []
        name_pages: list[NamePage] = []
        canonicals: set[str] = set()
        running_total = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            rendered = rasterize_page(doc[page_idx], dpi=dpi)
            try:
                width, height = rendered.size
                blocks = _ocr_image(engine, rendered, page_num)
            finally:
                rendered.close()

            merged: list[dict[str, Any]] = []
            for geometry, canonical in _locate_pairs(blocks, width, height):
                item = dict(geometry)
                item["canonical"] = canonical
                merged.append(item)
                canonicals.add(canonical)

            boxes: list[dict[str, Any]] = []
            # Order first, then number: assigning IDs before sorting would
            # number by confidence, since that is the order _dedupe returns.
            for item in _reading_order(merged):
                canonical = item.pop("canonical")
                patient_id = patient_ids.get(canonical)
                if patient_id is None:
                    patient_id = f"Patient_{len(patient_ids) + 1:04d}"
                    patient_ids[canonical] = patient_id
                # Geometry stays exactly as detected. The stamp is sized to fit
                # the box; the box is never resized to fit the stamp.
                item["patient_id"] = patient_id
                boxes.append(item)

            running_total += len(boxes)
            if detect_names:
                name_pages.append(NamePage(blocks, width, height, tuple(boxes)))
            pages.append(
                PageDetection(
                    page_num=page_num,
                    width=width,
                    height=height,
                    boxes=tuple(boxes),
                )
            )
            if progress_callback is not None:
                progress_callback(
                    PageProgress(page_num, total_pages, len(boxes), running_total)
                )

        if detect_names:
            name_boxes = locate_name_regions(name_pages, canonicals, name_model)
            pages = [
                PageDetection(
                    page_num=page.page_num,
                    width=page.width,
                    height=page.height,
                    boxes=page.boxes + tuple(found),
                )
                for page, found in zip(pages, name_boxes)
            ]

        return pages, total_pages
    finally:
        doc.close()


def render_redacted_pdf(
    input_path: Path,
    out_path: Path,
    pages: Sequence[PageDetection],
    dpi: int = 200,
    debug: bool = False,
) -> tuple[int, list[int], dict[str, Any]]:
    """Apply the given (possibly hand-edited) boxes and write the output PDF.

    Pages absent from ``pages`` are rendered unredacted rather than skipped,
    so a partial payload cannot silently drop pages from the output.

    Returns ``(total_nams, per_page_counts, debug_info)``.
    """
    _check_paths(input_path, out_path)

    boxes_by_page = {page.page_num: page.boxes for page in pages}

    doc = pymupdf.open(str(input_path))
    out_doc = pymupdf.open()

    per_page_counts: list[int] = []
    debug_data: list[dict[str, Any]] = []

    debug_dir: Path | None = None
    if debug:
        debug_dir = out_path.parent / f"{out_path.stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            rendered = rasterize_page(doc[page_idx], dpi=dpi)
            try:
                width, height = rendered.size
                regions = list(boxes_by_page.get(page_num, ()))
                per_page_counts.append(len(regions))

                redacted = redact_image(rendered, regions)

                if debug and debug_dir is not None:
                    page_debug_info = {
                        "page": page_num,
                        "width": width,
                        "height": height,
                        "boxes": [
                            {
                                "x1": r["x1"],
                                "y1": r["y1"],
                                "x2": r["x2"],
                                "y2": r["y2"],
                                "confidence": r["confidence"],
                            }
                            for r in regions
                        ],
                    }
                    debug_data.append(page_debug_info)

                    page_json_path = debug_dir / f"page_{page_num}_boxes.json"
                    with open(page_json_path, "w", encoding="utf-8") as f:
                        json.dump(page_debug_info, f, indent=2)

                    preview_png_path = debug_dir / f"page_{page_num}_preview.png"
                    redacted.save(preview_png_path, format="PNG")

                page_w_pts = width * 72.0 / dpi
                page_h_pts = height * 72.0 / dpi
                new_page = out_doc.new_page(width=page_w_pts, height=page_h_pts)
                buf = io.BytesIO()
                redacted.save(buf, format="PNG")
                new_page.insert_image(new_page.rect, stream=buf.getvalue())
                buf.close()
            finally:
                rendered.close()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_doc.save(str(out_path), deflate=True)

        all_debug: dict[str, Any] = {
            "input": str(input_path),
            "output": str(out_path),
            "dpi": dpi,
            "total_pages": len(doc),
            "total_nams": sum(per_page_counts),
            "pages": debug_data,
        }

        if debug and debug_dir is not None:
            combined_json_path = debug_dir / "boxes.json"
            with open(combined_json_path, "w", encoding="utf-8") as f:
                json.dump(all_debug, f, indent=2)

        return sum(per_page_counts), per_page_counts, all_debug

    finally:
        doc.close()
        out_doc.close()


def redact_pdf(
    input_path: Path,
    out_path: Path,
    dpi: int = 200,
    debug: bool = False,
    engine: PaddleOCREngine | None = None,
    name_model: NameModel | None = None,
) -> tuple[int, list[int], dict[str, Any]]:
    """Detect and redact every NAM and patient name in a PDF, producing a
    rasterized output PDF.

    Convenience composition of :func:`detect_nam_pages` and
    :func:`render_redacted_pdf`. Callers that need to review or edit the
    detected regions before writing should use those two directly.

    Returns:
        (total_boxes, per_page_counts, debug_info)
    """
    pages, _ = detect_nam_pages(
        input_path, dpi=dpi, engine=engine, name_model=name_model
    )
    return render_redacted_pdf(input_path, out_path, pages, dpi=dpi, debug=debug)
