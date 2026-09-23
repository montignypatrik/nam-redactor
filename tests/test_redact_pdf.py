"""End-to-end PDF rasterization and redaction tests."""
import json
import pymupdf
import pytest
from PIL import Image

from nam_redactor.redactor import rasterize_page, redact_image, redact_pdf


def test_redact_image_draws_black_rectangle():
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    regions = [{"x1": 0.2, "y1": 0.2, "x2": 0.5, "y2": 0.4, "confidence": 0.99}]
    redacted = redact_image(img, regions)

    # Pixel inside box must be black (0, 0, 0)
    assert redacted.getpixel((30, 30)) == (0, 0, 0)
    # Pixel outside box must be untouched white (255, 255, 255)
    assert redacted.getpixel((10, 10)) == (255, 255, 255)


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
