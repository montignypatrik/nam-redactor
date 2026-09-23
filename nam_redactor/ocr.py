"""OCR connector for local PaddleOCR inference."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class OCRResult:
    """Standardized OCR text line result."""
    page: int
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float
    block_type: str = "text"


def _paddle_field(result: Any, key: str) -> list[Any]:
    """Read a field from PaddleOCR 3.x result object or dict payload."""
    try:
        return result[key]
    except (TypeError, KeyError, IndexError):
        payload = getattr(result, "json", None)
        if isinstance(payload, dict) and "res" in payload:
            return payload["res"].get(key, [])
        return []


class PaddleOCREngine:
    """Connecteur PaddleOCR 3.x configuré pour CPU avec modèles mobiles légers."""

    def __init__(
        self,
        lang: str = "fr",
        device: str = "cpu",
        text_detection_model_name: Optional[str] = "PP-OCRv5_mobile_det",
        text_recognition_model_name: Optional[str] = "PP-OCRv5_mobile_rec",
        use_textline_orientation: bool = False,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        enable_mkldnn: bool = False,
        **extra_ocr_kwargs,
    ):
        self.lang = lang
        self.device = device
        self.text_detection_model_name = text_detection_model_name
        self.text_recognition_model_name = text_recognition_model_name
        self.use_textline_orientation = use_textline_orientation
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.enable_mkldnn = enable_mkldnn
        self.extra_ocr_kwargs = extra_ocr_kwargs
        self._ocr = None

    def name(self) -> str:
        return f"PaddleOCR-{self.text_detection_model_name or 'default'}"

    def validate(self) -> None:
        self._get_ocr()

    def _get_ocr(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "paddleocr is not installed. Install via `pip install paddleocr paddlepaddle`"
                ) from exc

            ocr_kwargs = {
                "lang": self.lang,
                "device": self.device,
                "use_doc_orientation_classify": self.use_doc_orientation_classify,
                "use_doc_unwarping": self.use_doc_unwarping,
                "use_textline_orientation": self.use_textline_orientation,
                "enable_mkldnn": self.enable_mkldnn,
                **self.extra_ocr_kwargs,
            }
            if self.text_detection_model_name is not None:
                ocr_kwargs["text_detection_model_name"] = self.text_detection_model_name
            if self.text_recognition_model_name is not None:
                ocr_kwargs["text_recognition_model_name"] = self.text_recognition_model_name
            self._ocr = PaddleOCR(**ocr_kwargs)
        return self._ocr

    def process_image(self, image_path: str, page_num: int = 1) -> List[OCRResult]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        ocr = self._get_ocr()
        results: List[OCRResult] = []
        for res in ocr.predict(image_path):
            texts = _paddle_field(res, "rec_texts")
            scores = _paddle_field(res, "rec_scores")
            polys = _paddle_field(res, "rec_polys")
            for text, score, poly in zip(texts, scores, polys):
                clean_text = str(text).strip()
                if not clean_text:
                    continue
                xs = [float(pt[0]) for pt in poly]
                ys = [float(pt[1]) for pt in poly]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                results.append(
                    OCRResult(
                        page=page_num,
                        text=clean_text,
                        bbox=bbox,
                        confidence=round(float(score), 4),
                        block_type="number" if clean_text.isdigit() else "text",
                    )
                )
        return results
