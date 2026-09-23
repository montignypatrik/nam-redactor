"""Locate NAM (RAMQ health-insurance number) blackout regions on OCR pages.

PaddleOCR reports one box per text line. A NAM often shares a line with a
label, so each match is refined to the matched characters by interpolating
their offsets across the line's bounding box — the smallest box that still
covers the whole NAM. Detection reuses the deterministic ``validate_ramq``
rules (OCR-confusion tolerant, no LLM).

Regions are returned normalized (0..1) against the processed page image.
Only boxes are produced — never raw PII values.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .validation import clean_ramq_run, validate_ramq

#: A run of alphanumerics optionally split by spaces/dashes, matching both
#: "abcd12345678" and "abcd 1234 5678".
_NAM_RUN = re.compile(r"[A-Za-z0-9]+(?:[\s\-–—]+[A-Za-z0-9]+){0,5}")
_SEPARATORS = frozenset(" \t-–—")
MIN_CONFIDENCE = 0.6
#: Safety margin around a matched run, in source pixels.
BOX_PADDING_PX = 2


def _bbox_of(block: Any) -> tuple[int, int, int, int] | None:
    bbox = getattr(block, "bbox", None)
    if bbox is None or len(bbox) != 4:
        return None
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


def _clean_with_positions(raw: str, offset: int) -> tuple[str, list[int]]:
    """Cleaned uppercase run plus each cleaned char's index in the source."""
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(raw):
        if char.isspace() or char in _SEPARATORS:
            continue
        chars.append(char.upper())
        positions.append(offset + index)
    return "".join(chars), positions


def _refine_bbox(
    bbox: tuple[int, int, int, int], text_length: int, start: int, end: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    length = max(1, text_length)
    left = x1 + (x2 - x1) * (start / length)
    right = x1 + (x2 - x1) * (end / length)
    return (left, y1, right, y2)


def _same_row(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ca = (a[1] + a[3]) / 2
    cb = (b[1] + b[3]) / 2
    height = max(a[3] - a[1], b[3] - b[1], 1)
    return abs(ca - cb) <= 0.5 * height


def _overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def _single_block_regions(
    blocks: Sequence[Any], min_confidence: float
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for block in blocks:
        text = getattr(block, "text", "") or ""
        bbox = _bbox_of(block)
        if bbox is None or not text.strip():
            continue
        for match in _NAM_RUN.finditer(text):
            cleaned, positions = _clean_with_positions(match.group(0), match.start())
            if len(cleaned) < 12:
                continue
            ok, confidence, _problems = validate_ramq(cleaned)
            if not ok or confidence < min_confidence:
                continue
            found.append({
                "bbox": _refine_bbox(bbox, len(text), positions[0], positions[-1] + 1),
                "confidence": round(confidence, 3),
            })
    return found


def _split_block_regions(
    blocks: Sequence[Any], min_confidence: float
) -> list[dict[str, Any]]:
    """Join a NAM split across two adjacent blocks on the same row."""
    items = [
        (block, _bbox_of(block))
        for block in blocks
        if _bbox_of(block) is not None and (getattr(block, "text", "") or "").strip()
    ]
    found: list[dict[str, Any]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            left_block, left = items[i]
            right_block, right = items[j]
            assert left is not None and right is not None
            if not _same_row(left, right):
                continue
            if left[0] > right[0]:
                left_block, right_block = right_block, left_block
                left, right = right, left
            height = max(left[3] - left[1], right[3] - right[1], 1)
            if right[0] - left[2] > height:  # too far apart to be one token
                continue
            left_clean = clean_ramq_run(left_block.text)
            right_clean = clean_ramq_run(right_block.text)
            if validate_ramq(left_clean)[0] or validate_ramq(right_clean)[0]:
                continue
            ok, confidence, _problems = validate_ramq(left_clean + right_clean)
            if not ok or confidence < min_confidence:
                continue
            found.append({
                "bbox": (
                    float(min(left[0], right[0])), float(min(left[1], right[1])),
                    float(max(left[2], right[2])), float(max(left[3], right[3])),
                ),
                "confidence": round(max(0.0, confidence - 0.05), 3),
            })
    return found


def _normalize(
    regions: Sequence[dict[str, Any]], width: int, height: int, padding: int
) -> list[dict[str, float]]:
    normalized: list[dict[str, float]] = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        x1, y1 = x1 - padding, y1 - padding
        x2, y2 = x2 + padding, y2 + padding
        nx1 = max(0.0, min(1.0, x1 / width))
        ny1 = max(0.0, min(1.0, y1 / height))
        nx2 = max(0.0, min(1.0, x2 / width))
        ny2 = max(0.0, min(1.0, y2 / height))
        if nx2 - nx1 <= 0 or ny2 - ny1 <= 0:
            continue
        normalized.append({
            "x1": round(nx1, 6), "y1": round(ny1, 6),
            "x2": round(nx2, 6), "y2": round(ny2, 6),
            "confidence": float(region["confidence"]),
        })
    return normalized


def _dedupe(regions: Sequence[dict[str, float]]) -> list[dict[str, float]]:
    kept: list[dict[str, float]] = []
    for region in sorted(regions, key=lambda item: -item["confidence"]):
        box = (region["x1"], region["y1"], region["x2"], region["y2"])
        if any(
            _overlap_ratio(box, (k["x1"], k["y1"], k["x2"], k["y2"])) > 0.5
            for k in kept
        ):
            continue
        kept.append(region)
    return kept


def locate_nam_regions(
    blocks: Sequence[Any],
    width: int,
    height: int,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    padding: int = BOX_PADDING_PX,
) -> list[dict[str, float]]:
    """Return normalized NAM blackout regions for one rendered page."""
    if width <= 0 or height <= 0:
        return []
    regions = _single_block_regions(blocks, min_confidence)
    regions.extend(_split_block_regions(blocks, min_confidence))
    return _dedupe(_normalize(regions, width, height, padding))
