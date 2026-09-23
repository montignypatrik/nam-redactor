"""Core PDF rasterization and black-box redaction pipeline."""
from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import pymupdf
from PIL import Image, ImageDraw

from .locator import locate_nam_regions
from .ocr import PaddleOCREngine


def rasterize_page(page: pymupdf.Page, dpi: int = 200) -> Image.Image:
    """Rasterize a single PDF page at specified DPI with page rotation applied."""
    scale = dpi / 72.0
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return image


def redact_image(
    image: Image.Image,
    regions: Sequence[dict[str, float]],
) -> Image.Image:
    """Draw opaque black rectangles over normalized bounding boxes."""
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for reg in regions:
        x1 = max(0, int(round(reg["x1"] * width)))
        y1 = max(0, int(round(reg["y1"] * height)))
        x2 = min(width, int(round(reg["x2"] * width)))
        y2 = min(height, int(round(reg["y2"] * height)))
        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
    return image


def redact_pdf(
    input_path: Path,
    out_path: Path,
    dpi: int = 200,
    debug: bool = False,
    engine: PaddleOCREngine | None = None,
) -> tuple[int, list[int], dict[str, Any]]:
    """Redact all NAMs in a PDF file, producing a rasterized output PDF.

    Returns:
        (total_nams, per_page_counts, debug_info)
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.resolve() == out_path.resolve():
        raise ValueError("Output file path must be different from input file path.")

    if engine is None:
        engine = PaddleOCREngine(
            device="cpu",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
        )

    doc = pymupdf.open(str(input_path))
    out_doc = pymupdf.open()
    total_pages = len(doc)

    per_page_counts: list[int] = []
    debug_data: list[dict[str, Any]] = []

    debug_dir: Path | None = None
    if debug:
        debug_dir = out_path.parent / f"{out_path.stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]
            rendered_image = rasterize_page(page, dpi=dpi)
            width, height = rendered_image.size

            # Save temporary image for PaddleOCR inference
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_image_path = tmp_file.name
                rendered_image.save(tmp_image_path, format="PNG")

            try:
                blocks = engine.process_image(tmp_image_path, page_num=page_num)
            finally:
                if os.path.exists(tmp_image_path):
                    try:
                        os.unlink(tmp_image_path)
                    except OSError:
                        pass

            # Locate normalized NAM regions
            regions = locate_nam_regions(blocks, width, height)
            count = len(regions)
            per_page_counts.append(count)

            # Redact image
            redacted = redact_image(rendered_image, regions)

            # Debug output per page
            if debug and debug_dir:
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

                # Save per-page JSON
                page_json_path = debug_dir / f"page_{page_num}_boxes.json"
                with open(page_json_path, "w", encoding="utf-8") as f:
                    json.dump(page_debug_info, f, indent=2)

                # Save preview PNG
                preview_png_path = debug_dir / f"page_{page_num}_preview.png"
                redacted.save(preview_png_path, format="PNG")

            # Add to output PDF as raster page
            page_w_pts = width * 72.0 / dpi
            page_h_pts = height * 72.0 / dpi
            new_page = out_doc.new_page(width=page_w_pts, height=page_h_pts)
            buf = io.BytesIO()
            redacted.save(buf, format="PNG")
            new_page.insert_image(new_page.rect, stream=buf.getvalue())
            buf.close()
            rendered_image.close()

        # Save new redacted PDF
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_doc.save(str(out_path), deflate=True)

        all_debug: dict[str, Any] = {
            "input": str(input_path),
            "output": str(out_path),
            "dpi": dpi,
            "total_pages": total_pages,
            "total_nams": sum(per_page_counts),
            "pages": debug_data,
        }

        if debug and debug_dir:
            combined_json_path = debug_dir / "boxes.json"
            with open(combined_json_path, "w", encoding="utf-8") as f:
                json.dump(all_debug, f, indent=2)

        return sum(per_page_counts), per_page_counts, all_debug

    finally:
        doc.close()
        out_doc.close()
