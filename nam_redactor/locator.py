"""Locate NAM (RAMQ health-insurance number) blackout regions on OCR pages.

PaddleOCR reports one box per text line. A NAM often shares a line with a
label, so each match is refined to the matched characters by interpolating
their offsets across the line's bounding box -- the smallest box that still
covers the whole NAM. Detection reuses the deterministic ``validate_ramq``
rules (OCR-confusion tolerant, no LLM).

A NAM is also frequently *glued* to its label: a line reading
``NAM HOGP 3361 0615`` is one OCR run of 15 cleaned characters, so validating
the run as a whole rejects it and discards a perfectly good 12-character NAM.
Each run is therefore searched for NAM-length spans -- see
:func:`_run_candidates`.

Regions are returned normalized (0..1) against the processed page image.
Only boxes are produced — never raw PII values.
"""
from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

from .validation import canonical_nam, clean_ramq_run, validate_ramq

#: A run of alphanumerics optionally split by spaces/dashes, matching both
#: "abcd12345678" and "abcd 1234 5678".
_NAM_RUN = re.compile(r"[A-Za-z0-9]+(?:[\s\-–—]+[A-Za-z0-9]+){0,5}")
#: A single alphanumeric group within a run.
_ALNUM_GROUP = re.compile(r"[A-Za-z0-9]+")
_SEPARATORS = frozenset(" \t-–—")
MIN_CONFIDENCE = 0.6
#: Safety margin around a matched run, in source pixels.
BOX_PADDING_PX = 2
#: A NAM spans at most four groups ("HOGP 33 61 06 15" is five, so allow five).
_MAX_RUN_GROUPS = 5
#: NAM length: 4 letters + 8 digits. The 13th character is an optional check
#: digit that is ignored here.
_NAM_LENGTHS = (12, 13)


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


def _run_candidates(run_text: str, offset: int) -> Iterator[tuple[str, list[int]]]:
    """Yield NAM-length candidates found *inside* one OCR run.

    Newlines aside, a run is what OCR read as one uninterrupted string, so a
    label sharing the line lands in the same run. Validating only the whole
    run means ``NAM HOGP 3361 0615`` (15 cleaned characters) is rejected
    outright and the NAM inside it is never examined.

    Two passes, in order:

    1. Contiguous **groups** of alphanumerics. This is how NAMs are written --
       letters then digit groups -- and it never splits a word, so it cannot
       invent a NAM out of the tail of one word plus the head of another.
    2. Only if step 1 found nothing for this run, slide a 12-character window.
       This rescues a label fused to the name letters (``NAMHOGP 3361 0615``),
       where there is no group boundary to cut at. Scoped per run rather than
       per page so it does not loosen the search everywhere.

    Yields ``(cleaned, positions)`` where ``positions`` are the source indices
    of each cleaned character, for bounding-box refinement.
    """
    groups = list(_ALNUM_GROUP.finditer(run_text))
    yielded = False
    for i in range(len(groups)):
        for j in range(i, min(i + _MAX_RUN_GROUPS, len(groups))):
            start, stop = groups[i].start(), groups[j].end()
            cleaned, positions = _clean_with_positions(
                run_text[start:stop], offset + start
            )
            if len(cleaned) in _NAM_LENGTHS:
                yielded = True
                yield cleaned, positions

    if yielded:
        return

    cleaned, positions = _clean_with_positions(run_text, offset)
    for start in range(max(0, len(cleaned) - 11)):
        window = cleaned[start:start + 12]
        if len(window) == 12:
            yield window, positions[start:start + 12]


class _Candidate(NamedTuple):
    """A detected NAM region plus the identity behind it.

    ``canonical`` has to ride on the candidate rather than sit in a parallel
    array: ``_normalize`` and ``_dedupe`` both *drop* entries, so any
    index-based pairing would silently misalign.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    canonical: str


def _candidate_dict(candidate: _Candidate) -> dict[str, float]:
    x1, y1, x2, y2 = candidate.bbox
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": candidate.confidence,
    }


def _single_block_regions(
    blocks: Sequence[Any], min_confidence: float
) -> list[_Candidate]:
    found: list[_Candidate] = []
    for block in blocks:
        text = getattr(block, "text", "") or ""
        bbox = _bbox_of(block)
        if bbox is None or not text.strip():
            continue
        for match in _NAM_RUN.finditer(text):
            for cleaned, positions in _run_candidates(match.group(0), match.start()):
                ok, confidence, _problems = validate_ramq(cleaned)
                if not ok or confidence < min_confidence:
                    continue
                canonical = canonical_nam(cleaned)
                if canonical is None:
                    continue
                found.append(_Candidate(
                    bbox=_refine_bbox(
                        bbox, len(text), positions[0], positions[-1] + 1
                    ),
                    confidence=round(confidence, 3),
                    canonical=canonical,
                ))
    return found


def _split_block_regions(
    blocks: Sequence[Any], min_confidence: float
) -> list[_Candidate]:
    """Join a NAM split across two adjacent blocks on the same row."""
    items = [
        (block, _bbox_of(block))
        for block in blocks
        if _bbox_of(block) is not None and (getattr(block, "text", "") or "").strip()
    ]
    found: list[_Candidate] = []
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
            joined = left_clean + right_clean
            ok, confidence, _problems = validate_ramq(joined)
            if not ok or confidence < min_confidence:
                continue
            canonical = canonical_nam(joined)
            if canonical is None:
                continue
            found.append(_Candidate(
                bbox=(
                    float(min(left[0], right[0])), float(min(left[1], right[1])),
                    float(max(left[2], right[2])), float(max(left[3], right[3])),
                ),
                confidence=round(max(0.0, confidence - 0.05), 3),
                canonical=canonical,
            ))
    return found


def _normalize(
    candidates: Sequence[_Candidate], width: int, height: int, padding: int
) -> list[_Candidate]:
    normalized: list[_Candidate] = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.bbox
        x1, y1 = x1 - padding, y1 - padding
        x2, y2 = x2 + padding, y2 + padding
        nx1 = max(0.0, min(1.0, x1 / width))
        ny1 = max(0.0, min(1.0, y1 / height))
        nx2 = max(0.0, min(1.0, x2 / width))
        ny2 = max(0.0, min(1.0, y2 / height))
        if nx2 - nx1 <= 0 or ny2 - ny1 <= 0:
            continue
        normalized.append(_Candidate(
            bbox=(round(nx1, 6), round(ny1, 6), round(nx2, 6), round(ny2, 6)),
            confidence=float(candidate.confidence),
            canonical=candidate.canonical,
        ))
    return normalized


def _dedupe(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    kept: list[_Candidate] = []
    for candidate in sorted(candidates, key=lambda item: -item.confidence):
        if any(_overlap_ratio(candidate.bbox, k.bbox) > 0.5 for k in kept):
            continue
        kept.append(candidate)
    return kept


def _locate_candidates(
    blocks: Sequence[Any],
    width: int,
    height: int,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    padding: int = BOX_PADDING_PX,
) -> list[_Candidate]:
    if width <= 0 or height <= 0:
        return []
    candidates = _single_block_regions(blocks, min_confidence)
    candidates.extend(_split_block_regions(blocks, min_confidence))
    return _dedupe(_normalize(candidates, width, height, padding))


def locate_nam_regions(
    blocks: Sequence[Any],
    width: int,
    height: int,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    padding: int = BOX_PADDING_PX,
) -> list[dict[str, float]]:
    """Return normalized NAM blackout regions for one rendered page.

    Geometry and confidence only -- never the recognised text. Callers that
    need to know *which* number a region covers use :func:`_locate_pairs`,
    which is deliberately separate so this function cannot leak PII.
    """
    return [
        _candidate_dict(candidate)
        for candidate in _locate_candidates(
            blocks, width, height, min_confidence=min_confidence, padding=padding
        )
    ]


def _locate_pairs(
    blocks: Sequence[Any],
    width: int,
    height: int,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    padding: int = BOX_PADDING_PX,
) -> list[tuple[dict[str, float], str]]:
    """Regions paired with the canonical NAM behind each one.

    Internal: the canonical value is PII-derived and must not be forwarded to
    a client or written to an artifact.
    """
    return [
        (_candidate_dict(candidate), candidate.canonical)
        for candidate in _locate_candidates(
            blocks, width, height, min_confidence=min_confidence, padding=padding
        )
    ]


def canonical_from_text(text: str) -> str | None:
    """First valid NAM found in arbitrary text, or ``None``.

    Reuses the same in-run search as detection, so a user who retypes what
    they see on the page -- including the label, as in
    ``NAM: HOGP 3361 0615`` -- is not rejected for the surrounding characters.
    """
    for match in _NAM_RUN.finditer(text):
        for cleaned, _positions in _run_candidates(match.group(0), match.start()):
            canonical = canonical_nam(cleaned)
            if canonical is not None:
                return canonical
    return None
