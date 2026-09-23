"""nam-redactor: Standalone local tool to redact RAMQ health insurance numbers (NAM) from PDFs."""
from .locator import locate_nam_regions
from .ocr import OCRResult, PaddleOCREngine
from .redactor import rasterize_page, redact_image, redact_pdf
from .validation import clean_ramq_run, validate_ramq

__all__ = [
    "clean_ramq_run",
    "locate_nam_regions",
    "OCRResult",
    "PaddleOCREngine",
    "rasterize_page",
    "redact_image",
    "redact_pdf",
    "validate_ramq",
]
__version__ = "0.1.0"
