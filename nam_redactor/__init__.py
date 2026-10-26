"""nam-redactor: Standalone local tool to redact RAMQ health insurance numbers (NAM) from PDFs."""
from __future__ import annotations

import os

# Paddle reads these at import time, so they must be set before anything can
# reach ``import paddle``. Set here rather than in an entry point so no caller
# can get the ordering wrong. ``setdefault`` means an explicit choice by the
# caller always wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from .locator import locate_nam_regions
from .ocr import OCRResult, PaddleOCREngine
from .redactor import (
    PageDetection,
    PageProgress,
    default_engine,
    detect_nam_pages,
    rasterize_page,
    redact_image,
    redact_pdf,
    render_redacted_pdf,
    resolve_device,
)
from .validation import clean_ramq_run, validate_ramq

__all__ = [
    "PageDetection",
    "PageProgress",
    "clean_ramq_run",
    "default_engine",
    "detect_nam_pages",
    "locate_nam_regions",
    "OCRResult",
    "PaddleOCREngine",
    "rasterize_page",
    "redact_image",
    "redact_pdf",
    "render_redacted_pdf",
    "resolve_device",
    "validate_ramq",
]
__version__ = "0.1.0"
