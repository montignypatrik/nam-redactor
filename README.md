# nam-redactor

[![Tests](https://github.com/montignypatrik/nam-redactor/actions/workflows/test.yml/badge.svg)](https://github.com/montignypatrik/nam-redactor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Standalone, local CLI tool that detects and black-box redacts Quebec health insurance numbers (RAMQ / NAM) from PDF documents.

Runs **100% locally on CPU** — zero cloud services, no GPU, no Redis, no database, and no LLMs.

---

## Key Features

- **True Raster Redaction**: Pages are rendered at target DPI, redacted, and saved as a fresh image-only PDF. The original vector text layer and NAM pixels are completely eliminated (not just masked with overlay rectangles).
- **Sub-Line Box Tightness**: Refines PaddleOCR line-level boxes to the matched characters via character offset interpolation. A line such as `NAM: HOGP 3361 0615` blacks out only `HOGP 3361 0615`, leaving the `NAM:` label intact.
- **Split Token Merging**: Automatically joins NAM numbers split across adjacent OCR bounding blocks on the same row.
- **Deterministic Validation**: Tolerant of OCR misrecognitions (O/0, I/1, B/8, S/5, Z/2, etc.) and accounts for RAMQ female month offsets (+50). Rejects common clinical and administrative stopwords.
- **Lightweight CPU Profile**: Uses mobile detection & recognition models (`PP-OCRv5_mobile_det`, `PP-OCRv5_mobile_rec`) with `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` to minimize CPU footprint and memory (<500 MB RAM).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/montignypatrik/nam-redactor.git
cd nam-redactor

# Install dependencies (Python 3.11+ / 3.12 recommended)
pip install -e .
```

Or install directly via `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Usage

### Basic Redaction
By default, outputs to `<input>.redacted.pdf` at 200 DPI:

```bash
nam-redact document.pdf
# or
python redact_nam.py document.pdf
```

### Options

```
usage: nam-redact [-h] [--dpi DPI] [--out OUT] [--debug] input_pdf

positional arguments:
  input_pdf   Path to input PDF file

options:
  -h, --help  Show help message and exit
  --dpi DPI   Rasterization DPI (default: 200; use 150 for lower memory)
  --out OUT   Path to output redacted PDF (default: <input>.redacted.pdf)
  --debug     Export detected bounding boxes (JSON) and preview PNGs per page
```

### Examples

```bash
# Custom output path and 150 DPI for lower memory usage
nam-redact medical_records.pdf --dpi 150 --out /path/to/anonymized.pdf

# Run with debug artifacts (boxes.json and preview PNGs)
nam-redact claim.pdf --debug
```

In `--debug` mode, artifacts are saved to `<output_stem>_debug/`:
- `boxes.json` & `page_<N>_boxes.json`: Coordinates `(x1, y1, x2, y2)` normalized 0..1 and confidence scores only. **No PII text is stored**.
- `page_<N>_preview.png`: Rendered image with redacted black boxes for visual inspection.

---

## Testing

Run the test suite with `pytest`:

```bash
pytest
```

Includes unit tests for isolated NAMs, label exclusions, adjacent block splits, non-NAM word filtering, and end-to-end PDF vector text removal.

---

## License

[MIT](LICENSE) © Patrik Montigny
