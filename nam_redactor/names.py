"""Locate patient-name blackout regions on OCR pages.

Names get no pseudonym -- they are simply blacked out. Three sources feed one
set of boxes per page:

**NAM-anchored.** A NAM encodes its holder's name: letters 1-3 are the first
three letters of the last name, letter 4 is the first-name initial. So once a
NAM like ``HOGP 3361 0615`` is found, a row reading ``HOGUE, Pierre`` or
``Pierre Hogue`` is recognisably that patient -- a word starting ``HOG`` next
to a word starting ``P``. Requiring the *pair* is what keeps this precise: a
lone word starting ``HOG`` could be anything.

**Labelled.** A value following ``Nom :``, ``Prénom :``, ``Patient :`` and the
like. A colon (or a label standing alone in its own OCR block, as in a table)
is required, so running prose such as "le patient se plaint" is not taken as
a label.

Every name word learned from those two sources is then redacted **wherever it
appears in the document**, on any page, which is what catches a name printed
in a header on page 1 when the NAM only appears on page 5.

**NER** (optional). A spaCy French model flags person names anywhere --
doctors and relatives included. Its hits are redacted where found but *not*
propagated: a model false positive should cost one box, not one per page.

Like :mod:`nam_redactor.locator`, only geometry leaves this module. The name
text is used to find boxes and then dropped.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from .validation import is_non_name_word

#: A word: letters (accents included), optionally joined by hyphens or
#: apostrophes, so ``Marie-Ève`` and ``O'Neil`` stay one name.
_WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*")
#: What may separate two words of one name: "HOGUE, Pierre", "Pierre  Hogue".
_NAME_GAP = re.compile(r"^[\s,]*$")

#: Confidence reported for each source. Display only; nothing thresholds on it.
KNOWN_NAME_CONFIDENCE = 0.9
NER_CONFIDENCE = 0.75
BOX_PADDING_PX = 2
#: Extra margin on each side of a name box, in average character widths.
#: Character positions inside an OCR line are estimated, not measured, and a
#: sliver of a first letter is enough to leak a name -- so err wide.
SIDE_MARGIN_CHARS = 0.6
#: How many words apart, within one row, a last name and first name may be.
#: Two allows a middle initial or a stray comma token between them.
MAX_PAIR_DISTANCE = 2
#: Longest labelled value taken, in words: first, middle, and a compound last.
MAX_LABEL_VALUE_WORDS = 4

#: Label phrases, normalized (see :func:`_norm`), longest first so
#: ``NOM DE FAMILLE`` wins over ``NOM``.
_LABELS: tuple[tuple[str, ...], ...] = tuple(sorted(
    (
        ("NOM",), ("PRENOM",), ("PRENOMS",), ("NOM", "DE", "FAMILLE"),
        ("NOM", "DU", "PATIENT"), ("NOM", "DE", "LA", "PATIENTE"),
        ("NOM", "COMPLET"), ("PATIENT",), ("PATIENTE",), ("USAGER",),
        ("USAGERE",), ("BENEFICIAIRE",), ("NAME",), ("FIRST", "NAME"),
        ("LAST", "NAME"), ("SURNAME",), ("GIVEN", "NAME"), ("GIVEN", "NAMES"),
        ("FAMILY", "NAME"), ("PATIENT", "NAME"),
    ),
    key=len,
    reverse=True,
))
_LABEL_WORDS = frozenset(word for label in _LABELS for word in label)

#: Courtesy and professional titles. A single-word NER hit is kept only
#: right after one of these ("Dr Gagnon"): alone, the model tags too many
#: place and clinic names ("Laval") as people.
_TITLES = frozenset({
    "DR", "DRE", "DOCTEUR", "DOCTEURE", "M", "MR", "MME", "MLLE", "MM",
    "MONSIEUR", "MADAME", "MADEMOISELLE", "PR", "PROF", "INF", "MRS", "MS",
    "MISS", "SIR",
})

#: Words that sit next to names but are not names: courtesy titles, and the
#: label words themselves. Trimmed off the edges of NER hits, and never
#: learned as a name.
_NOT_NAME_WORDS = frozenset({
    "DE", "DU", "LA", "LE", "LES", "ET", "PAR", "VU", "AVEC", "POUR", "CHEZ",
    "SIGNE", "SIGNEE",
}) | _TITLES | _LABEL_WORDS


def _norm(text: str) -> str:
    """Uppercase ASCII letters only: ``Marie-Ève`` -> ``MARIEEVE``.

    Accents and punctuation are dropped so OCR that misses an accent, or
    splits a hyphen, still compares equal.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if "A" <= ch.upper() <= "Z").upper()


def _is_name_word(norm: str, raw: str) -> bool:
    return (
        len(norm) >= 2
        and norm not in _NOT_NAME_WORDS
        and not is_non_name_word(raw)
    )


@dataclass(frozen=True, slots=True)
class _Word:
    block: int  # index into the page's block list
    start: int  # character span within that block's text
    end: int
    raw: str
    norm: str


@dataclass(frozen=True, slots=True)
class _Line:
    """One OCR block's text and pixel box."""

    text: str
    bbox: tuple[int, int, int, int]


class NameModel(Protocol):
    """Anything that can flag person names in lines of text.

    Returns, per input line, ``(start, end)`` character spans. The seam tests
    use to stand in for spaCy.
    """

    def find_names(self, lines: Sequence[str]) -> list[list[tuple[int, int]]]: ...


# -- page structure ---------------------------------------------------------


def _lines_of(blocks: Sequence[Any]) -> list[_Line]:
    lines: list[_Line] = []
    for block in blocks:
        text = getattr(block, "text", "") or ""
        bbox = getattr(block, "bbox", None)
        if not text.strip() or bbox is None or len(bbox) != 4:
            continue
        lines.append(_Line(text, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
    return lines


def _words_of(lines: Sequence[_Line], index: int) -> list[_Word]:
    return [
        _Word(index, m.start(), m.end(), m.group(0), _norm(m.group(0)))
        for m in _WORD.finditer(lines[index].text)
    ]


def _same_row(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    height = max(a[3] - a[1], b[3] - b[1], 1)
    return abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) <= 0.5 * height


def _rows(lines: Sequence[_Line]) -> list[list[int]]:
    """Group blocks into visual rows, each ordered left to right.

    PaddleOCR often splits a table row -- ``Nom :`` | ``HOGUE, Pierre`` --
    into separate blocks, and a label or a name pair can straddle them.
    """
    rows: list[list[int]] = []
    for index in sorted(range(len(lines)), key=lambda i: lines[i].bbox[1]):
        for row in rows:
            if _same_row(lines[row[0]].bbox, lines[index].bbox):
                row.append(index)
                break
        else:
            rows.append([index])
    for row in rows:
        row.sort(key=lambda i: lines[i].bbox[0])
    return rows


def _row_words(lines: Sequence[_Line], row: Sequence[int]) -> list[_Word]:
    return [word for index in row for word in _words_of(lines, index)]


def _gap(lines: Sequence[_Line], left: _Word, right: _Word) -> str | None:
    """Text between two words, or ``None`` when they are in different blocks."""
    if left.block != right.block:
        return None
    return lines[left.block].text[left.end:right.start]


# -- learning names from NAMs and labels ------------------------------------


def _last_name_matches(words: Sequence[_Word], i: int, prefix: str) -> int:
    """How many words from ``i`` on spell a last name starting ``prefix``.

    0 for no match, 1 for ``HOGUE``, 2 for a split particle name like
    ``De Grandpré`` (NAM prefix ``DEG``). RAMQ pads a last name shorter than
    three letters with ``X``, so ``LIX`` also matches a bare ``LI``.
    """
    first = words[i].norm
    if first.startswith(prefix):
        return 1
    unpadded = prefix.rstrip("X")
    if unpadded != prefix and first == unpadded:
        return 1
    if i + 1 < len(words) and len(first) < 3:
        if (first + words[i + 1].norm).startswith(prefix):
            return 2
    return 0


def _anchored_names(
    rows: Sequence[Sequence[_Word]], canonicals: Iterable[str]
) -> set[str]:
    """Name words confirmed by a NAM: a last name *and* first initial together."""
    learned: set[str] = set()
    for canonical in set(canonicals):
        prefix, initial, nam_letters = canonical[:3], canonical[3], canonical[:4]
        for words in rows:
            for i, word in enumerate(words):
                # The NAM's own letters (``HOGP``) start with the prefix too.
                if word.norm == nam_letters:
                    continue
                span = _last_name_matches(words, i, prefix)
                if not span:
                    continue
                last = words[i:i + span]
                if not all(_is_name_word(w.norm, w.raw) or len(w.norm) < 3 for w in last):
                    continue
                lo = max(0, i - MAX_PAIR_DISTANCE)
                hi = min(len(words), i + span + MAX_PAIR_DISTANCE)
                firsts = [
                    w for j, w in enumerate(words[lo:hi], start=lo)
                    if not (i <= j < i + span)
                    and w.norm.startswith(initial)
                    and w.norm != nam_letters
                    and _is_name_word(w.norm, w.raw)
                ]
                if not firsts:
                    continue
                # A particle like ``De`` helps match the prefix but is not
                # learned: it would then be redacted in every sentence.
                learned.update(w.norm for w in last if _is_name_word(w.norm, w.raw))
                learned.update(w.norm for w in firsts)
    return learned


def _match_label(words: Sequence[_Word], i: int) -> int:
    """Length in words of the label phrase starting at ``i``, or 0."""
    for label in _LABELS:
        if tuple(w.norm for w in words[i:i + len(label)]) == label:
            return len(label)
    return 0


def _labelled_names(lines: Sequence[_Line], rows: Sequence[Sequence[_Word]]) -> set[str]:
    """Name words that follow a ``Nom :``-style label on the same row."""
    learned: set[str] = set()
    for words in rows:
        i = 0
        while i < len(words):
            size = _match_label(words, i)
            if not size:
                i += 1
                continue
            label_end = words[i + size - 1]
            j = i + size
            if j >= len(words):
                break
            after = _gap(lines, label_end, words[j])
            if after is None:
                # Value in the next block: accept only a label that is its
                # block's whole text, the way a form or table cell reads.
                block_text = lines[label_end.block].text
                ok = _norm(block_text) == "".join(w.norm for w in words[i:i + size])
            else:
                ok = ":" in after
            if ok:
                taken = 0
                while j < len(words) and taken < MAX_LABEL_VALUE_WORDS:
                    word = words[j]
                    if _match_label(words, j) or not _is_name_word(word.norm, word.raw):
                        break
                    if not word.raw[0].isupper():
                        break  # names are capitalized; prose after a colon is not
                    if taken:
                        gap = _gap(lines, words[j - 1], word)
                        if gap is not None and not _NAME_GAP.match(gap):
                            break
                    learned.add(word.norm)
                    taken += 1
                    j += 1
            i = max(j, i + 1)
    return learned


# -- spans -> boxes ---------------------------------------------------------


def _known_spans(
    lines: Sequence[_Line], known: set[str]
) -> dict[int, list[tuple[int, int]]]:
    spans: dict[int, list[tuple[int, int]]] = {}
    if not known:
        return spans
    for index in range(len(lines)):
        for word in _words_of(lines, index):
            if word.norm in known:
                spans.setdefault(index, []).append((word.start, word.end))
    return spans


def _trim_ner_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink an NER hit to its name words: ``Dre Sophie Gagnon`` -> ``Sophie Gagnon``.

    Titles and label words ride along in NER output but are not the name, and
    a hit with digits in it is not a person at all. A lone word survives only
    after a title, in the hit or just before it: single-word hits are where
    the model's place-name false positives live.
    """
    if any(ch.isdigit() for ch in text[start:end]):
        return None
    words = [
        (start + m.start(), start + m.end(), m.group(0))
        for m in _WORD.finditer(text[start:end])
    ]
    kept = [(s, e) for s, e, raw in words if _is_name_word(_norm(raw), raw)]
    if not kept:
        return None
    if len(kept) == 1:
        before = [_norm(m.group(0)) for m in _WORD.finditer(text[:kept[0][0]])]
        if not before or before[-1] not in _TITLES:
            return None
    return kept[0][0], kept[-1][1]


def _merge_spans(spans: Iterable[tuple[int, int]], text: str) -> list[tuple[int, int]]:
    """Union overlapping spans, and join neighbours separated only by a gap
    like ``", "`` so ``HOGUE, Pierre`` becomes one box rather than two."""
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and (
            start <= merged[-1][1] or _NAME_GAP.match(text[merged[-1][1]:start])
        ):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


#: Approximate advance widths of a proportional sans-serif (Arial/Helvetica,
#: per 1000 em). Only the *relative* sizes matter: they turn a character
#: offset into a position along the OCR box far better than assuming every
#: character is equally wide, which misplaces a name after "Il" or "mm".
_NARROW = dict.fromkeys("iljI.,:;'|!", 250)
_NARROW.update(dict.fromkeys("ftr -()/[]", 320))
_WIDE = dict.fromkeys("mwMW", 870)
_UPPER = 690
_DEFAULT = 540


def _char_width(ch: str) -> int:
    if ch in _NARROW:
        return _NARROW[ch]
    if ch in _WIDE:
        return _WIDE[ch]
    return _UPPER if ch.isupper() else _DEFAULT


def _span_box(
    line: _Line, start: int, end: int, width: int, height: int, padding: int
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = line.bbox
    advances = [_char_width(ch) for ch in line.text]
    total = max(1, sum(advances))
    scale = (x2 - x1) / total
    margin = SIDE_MARGIN_CHARS * scale * total / max(1, len(advances))
    left = x1 + sum(advances[:start]) * scale - margin - padding
    right = x1 + sum(advances[:end]) * scale + margin + padding
    nx1 = max(0.0, min(1.0, left / width))
    nx2 = max(0.0, min(1.0, right / width))
    ny1 = max(0.0, min(1.0, (y1 - padding) / height))
    ny2 = max(0.0, min(1.0, (y2 + padding) / height))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return (round(nx1, 6), round(ny1, 6), round(nx2, 6), round(ny2, 6))


def _overlap_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


@dataclass(frozen=True, slots=True)
class NamePage:
    """One page's OCR output plus what NAM detection already covered."""

    blocks: Sequence[Any]
    width: int
    height: int
    #: Normalized NAM boxes on this page; a name box mostly inside one is
    #: redundant and dropped.
    nam_boxes: Sequence[dict[str, Any]] = ()


def locate_name_regions(
    pages: Sequence[NamePage],
    canonicals: Iterable[str] = (),
    model: NameModel | None = None,
    *,
    padding: int = BOX_PADDING_PX,
) -> list[list[dict[str, Any]]]:
    """Name blackout boxes for every page of one document.

    ``canonicals`` are the NAMs detected anywhere in the document. Names are
    learned document-wide and then searched for on every page, so this takes
    all pages at once rather than one at a time.

    Each box is ``{x1, y1, x2, y2, confidence, "kind": "name"}``, normalized
    against its page -- never the name itself.
    """
    page_lines = [_lines_of(page.blocks) for page in pages]
    page_rows = [
        [_row_words(lines, row) for row in _rows(lines)] for lines in page_lines
    ]
    all_rows = [words for rows in page_rows for words in rows]

    known = _anchored_names(all_rows, canonicals)
    for lines, rows in zip(page_lines, page_rows):
        known |= _labelled_names(lines, rows)

    results: list[list[dict[str, Any]]] = []
    for page, lines in zip(pages, page_lines):
        known_spans = _known_spans(lines, known)
        ner_spans: dict[int, list[tuple[int, int]]] = {}
        if model is not None and lines:
            for index, found in enumerate(model.find_names([ln.text for ln in lines])):
                for start, end in found:
                    trimmed = _trim_ner_span(lines[index].text, start, end)
                    if trimmed is not None:
                        ner_spans.setdefault(index, []).append(trimmed)

        boxes: list[dict[str, Any]] = []
        for index in sorted(set(known_spans) | set(ner_spans)):
            line = lines[index]
            known_here = known_spans.get(index, [])
            for start, end in _merge_spans(known_here + ner_spans.get(index, []), line.text):
                bbox = _span_box(line, start, end, page.width, page.height, padding)
                if bbox is None:
                    continue
                if any(
                    _overlap_ratio(bbox, (n["x1"], n["y1"], n["x2"], n["y2"])) > 0.5
                    for n in page.nam_boxes
                ):
                    continue
                confident = any(s < end and start < e for s, e in known_here)
                boxes.append({
                    "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                    "confidence": KNOWN_NAME_CONFIDENCE if confident else NER_CONFIDENCE,
                    "kind": "name",
                })
        results.append(boxes)
    return results


# -- spaCy ------------------------------------------------------------------

SPACY_MODEL = "fr_core_news_md"


class SpacyNameModel:
    """French NER via spaCy, reporting ``PER`` entities only.

    Each OCR line is its own document: joined, entities run across line
    breaks (``PIERRE\\nConsultation``). All-caps lines are title-cased first,
    because the model reads ``HOGUE`` as an acronym but ``Hogue`` as a name.
    """

    def __init__(self, nlp: Any) -> None:
        self._nlp = nlp

    def find_names(self, lines: Sequence[str]) -> list[list[tuple[int, int]]]:
        prepared = [_title_if_shouting(line) for line in lines]
        return [
            [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "PER"]
            for doc in self._nlp.pipe(prepared)
        ]


def _title_if_shouting(line: str) -> str:
    if not line.isupper():
        return line
    titled = line.title()
    # Offsets map back onto the original text, so the length must not move.
    return titled if len(titled) == len(line) else line


@lru_cache(maxsize=1)
def default_name_model() -> SpacyNameModel | None:
    """The spaCy model, loaded once per process, or ``None`` if unavailable.

    Missing spaCy is not fatal: NAM-anchored and labelled names are still
    found, only the NER layer is skipped.
    """
    try:
        import spacy

        nlp = spacy.load(
            SPACY_MODEL,
            # Only NER is used; the rest costs load time and memory.
            exclude=["parser", "lemmatizer", "tagger", "morphologizer", "attribute_ruler"],
        )
    except (ImportError, OSError):
        return None
    return SpacyNameModel(nlp)

