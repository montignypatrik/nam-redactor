"""End-to-end PDF rasterization and redaction tests."""
import json

import numpy as np
import pymupdf
import pytest
from PIL import Image

from nam_redactor.redactor import (
    rasterize_page,
    redact_image,
    redact_pdf,
    resolve_device,
)


def _bright_inside(image: Image.Image, box) -> int:
    """Label glyphs are antialiased, so count bright pixels, not exact white."""
    x1, y1, x2, y2 = (int(v) for v in box)
    patch = np.asarray(image.crop((x1, y1, x2 + 1, y2 + 1)))
    return int((patch.min(axis=2) > 200).sum())


def test_resolve_device_passes_explicit_values_through():
    """An explicit choice is never second-guessed -- this is what CI relies on."""
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("gpu:0") == "gpu:0"


def test_resolve_device_auto_returns_a_usable_device():
    device = resolve_device("auto")
    assert device == "cpu" or device.startswith("gpu")


def test_redact_image_draws_black_rectangle():
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    regions = [{"x1": 0.2, "y1": 0.2, "x2": 0.5, "y2": 0.4, "confidence": 0.99}]
    redacted = redact_image(img, regions)

    # Pixel inside box must be black (0, 0, 0)
    assert redacted.getpixel((30, 30)) == (0, 0, 0)
    # Pixel outside box must be untouched white (255, 255, 255)
    assert redacted.getpixel((10, 10)) == (255, 255, 255)


def test_redact_image_leaves_a_box_plain_when_it_has_no_patient_id():
    """Regions without an identity must render exactly as before."""
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    redacted = redact_image(
        image, [{"x1": 0.2, "y1": 0.2, "x2": 0.5, "y2": 0.4, "confidence": 0.99}]
    )
    patch = np.asarray(redacted.crop((20, 20, 51, 41)))
    # Solid black: no bright ink anywhere inside.
    assert int((patch.min(axis=2) > 200).sum()) == 0


def test_redact_image_draws_every_rectangle_before_any_label():
    """A later box's fill must not paint over an earlier box's white ID.

    Drawing each rectangle immediately would erase the first label wherever
    two boxes overlap.
    """
    image = Image.new("RGB", (400, 200), (255, 255, 255))
    first = {
        "x1": 0.05, "y1": 0.25, "x2": 0.45, "y2": 0.55,
        "confidence": 1.0, "patient_id": "Patient_0001",
    }
    overlapping = {
        "x1": 0.35, "y1": 0.30, "x2": 0.85, "y2": 0.60,
        "confidence": 1.0, "patient_id": "Patient_0002",
    }

    redacted = redact_image(image, [first, overlapping])

    assert _bright_inside(
        redacted,
        (first["x1"] * 400, first["y1"] * 200, first["x2"] * 400, first["y2"] * 200),
    ) > 0, "the first box's label was painted over"


def test_redact_pdf_labels_the_blackouts(tmp_path):
    """The patient ID must be visible in white inside the blacked-out box."""
    pdf_path = tmp_path / "labelled.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((20, 80), "NAM: HOGP 3361 0615", fontsize=14)
    doc.save(str(pdf_path))
    doc.close()

    out_path = tmp_path / "labelled.redacted.pdf"
    total, per_page, _ = redact_pdf(pdf_path, out_path, dpi=200, debug=True)
    assert total == 1
    assert per_page == [1]

    with open(tmp_path / "labelled.redacted_debug" / "boxes.json", encoding="utf-8") as handle:
        debug = json.load(handle)
    box = debug["pages"][0]["boxes"][0]

    with pymupdf.open(str(out_path)) as out_doc:
        pixmap = out_doc[0].get_pixmap(dpi=200)
        rendered = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    width, height = rendered.size
    assert _bright_inside(rendered, (
        box["x1"] * width, box["y1"] * height, box["x2"] * width, box["y2"] * height,
    )) > 0, "no label ink found inside the blackout"


def test_redact_pdf_synthetic_e2e(tmp_path):
    pdf_path = tmp_path / "synthetic_medical.pdf"
    doc = pymupdf.open()

    # Page 1: contains NAM
    page1 = doc.new_page(width=300, height=200)
    page1.insert_text((20, 40), "Clinique Médicale", fontsize=14)
    page1.insert_text((20, 80), "NAM: HOGP 3361 0615", fontsize=14)
    page1.insert_text((20, 120), "Date: 2026-09-22", fontsize=12)

    # Page 2: no NAM
    page2 = doc.new_page(width=300, height=200)
    page2.insert_text((20, 40), "Page 2 - Sommaire sans numero", fontsize=12)
    page2.insert_text((20, 80), "Code d'acte: 08123", fontsize=12)

    doc.save(str(pdf_path))
    doc.close()

    out_path = tmp_path / "synthetic_medical.redacted.pdf"

    total, per_page, debug_info = redact_pdf(
        input_path=pdf_path,
        out_path=out_path,
        dpi=150,
        debug=True,
    )

    # Counts
    assert len(per_page) == 2
    assert per_page[0] == 1
    assert per_page[1] == 0
    assert total == 1

    # Output PDF check
    assert out_path.exists()
    out_doc = pymupdf.open(str(out_path))
    assert len(out_doc) == 2

    # Original text layer must be completely removed
    for p in out_doc:
        assert p.get_text().strip() == ""

    out_doc.close()

    # Debug artifacts
    debug_dir = tmp_path / "synthetic_medical.redacted_debug"
    assert debug_dir.exists()
    assert (debug_dir / "boxes.json").exists()
    assert (debug_dir / "page_1_boxes.json").exists()
    assert (debug_dir / "page_1_preview.png").exists()

    # Debug JSON must contain only boxes and confidence (no text, no PII)
    with open(debug_dir / "page_1_boxes.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "boxes" in data
    assert len(data["boxes"]) == 1
    assert set(data["boxes"][0].keys()) == {"x1", "y1", "x2", "y2", "confidence"}
    assert "HOGP" not in json.dumps(data)
