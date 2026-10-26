"""Tests for patient-name detection. No spaCy: NER is stubbed where used."""
from __future__ import annotations

from nam_redactor.names import NamePage, SpacyNameModel, locate_name_regions
from nam_redactor.ocr import OCRResult

W, H = 1000, 1000
NAM = "HOGP33610615"  # canonical of "HOGP 3361 0615": last name HOG..., first P...


def line(text: str, y: int = 100, x1: int = 100, x2: int | None = None) -> OCRResult:
    """One OCR block at x=100, 10 px per character unless told otherwise."""
    return OCRResult(
        page=1, text=text, bbox=(x1, y, x2 if x2 is not None else x1 + 10 * len(text), y + 20),
        confidence=0.99,
    )


def boxes_for(*blocks: OCRResult, canonicals=(NAM,), model=None) -> list[dict]:
    return locate_name_regions([NamePage(blocks, W, H)], canonicals, model)[0]


def spans_px(boxes: list[dict]) -> list[tuple[int, int]]:
    """Horizontal extents in pixels, relative to a block at x=100."""
    return sorted((round(b["x1"] * W) - 100, round(b["x2"] * W) - 100) for b in boxes)


#: How far past its characters a box may reach. Positions within a line are
#: estimated, and boxes err wide on purpose -- but not by whole words.
SLACK_PX = 25


def assert_covers(boxes: list[dict], expected: list[tuple[int, int]]) -> None:
    """Each box fully covers its expected pixel span, with bounded overshoot.

    ``expected`` is in the tests' nominal 10 px-per-character layout.
    """
    got = spans_px(boxes)
    assert len(got) == len(expected), got
    for (lo, hi), (want_lo, want_hi) in zip(got, sorted(expected)):
        assert lo <= want_lo and hi >= want_hi, (got, expected)
        assert want_lo - lo <= SLACK_PX and hi - want_hi <= SLACK_PX, (got, expected)


class StubModel:
    """Returns fixed spans per line text."""

    def __init__(self, hits: dict[str, list[tuple[int, int]]]):
        self.hits = hits

    def find_names(self, lines):
        return [self.hits.get(text, []) for text in lines]


# -- NAM-anchored ----------------------------------------------------------


def test_name_pair_matching_the_nam_is_redacted_as_one_box():
    boxes = boxes_for(line("HOGUE, Pierre"))
    # "HOGUE, Pierre" is 13 chars -> one box spanning all of it.
    assert_covers(boxes, [(0, 130)])
    assert boxes[0]["kind"] == "name"
    assert "patient_id" not in boxes[0]


def test_first_last_order_is_also_found():
    assert_covers(boxes_for(line("Pierre Hogue")), [(0, 120)])


def test_prefix_alone_without_the_first_initial_is_not_a_name():
    """A lone word starting HOG could be anything; the pair is the evidence."""
    assert boxes_for(line("Hogwarts Express")) == []


def test_no_nam_means_nothing_is_anchored():
    assert boxes_for(line("HOGUE, Pierre"), canonicals=()) == []


def test_nam_letters_themselves_are_not_mistaken_for_a_last_name():
    assert boxes_for(line("NAM HOGP 3361 0615 Pomme")) == []


def test_learned_name_is_redacted_on_every_page():
    """The name on page 2 is found because the pair on page 1 taught it."""
    pages = [
        NamePage([line("Patient HOGUE Pierre")], W, H),
        NamePage([line("Suivi de M. Hogue en clinique")], W, H),
    ]
    page1, page2 = locate_name_regions(pages, [NAM])
    assert_covers(page1, [(80, 200)])
    # "Hogue" is chars 12..17 of the page-2 line.
    assert_covers(page2, [(120, 170)])


def test_pair_split_across_two_blocks_on_one_row():
    boxes = boxes_for(line("HOGUE", x1=100), line("Pierre", x1=400))
    assert_covers(boxes, [(0, 50), (300, 360)])


def test_accents_and_case_do_not_matter():
    boxes = boxes_for(line("Éloïse Tremblay"), canonicals=("TREE85012312",))
    assert_covers(boxes, [(0, 150)])


def test_short_last_name_padded_with_x_in_the_nam():
    """RAMQ pads a two-letter last name: Li -> LIX."""
    boxes = boxes_for(line("LI Wei"), canonicals=("LIXW90051512",))
    assert_covers(boxes, [(0, 60)])


def test_name_box_inside_a_nam_box_is_dropped():
    nam_box = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "confidence": 0.99}
    pages = [NamePage([line("HOGUE, Pierre")], W, H, (nam_box,))]
    assert locate_name_regions(pages, [NAM]) == [[]]


# -- labels ----------------------------------------------------------------


def test_value_after_a_label_is_redacted():
    boxes = boxes_for(line("Nom: Tremblay Prénom: Marie-Ève"), canonicals=())
    # "Tremblay" 5..13 and "Marie-Ève" 22..31; the "Prénom:" label stays.
    assert_covers(boxes, [(50, 130), (220, 310)])


def test_label_in_its_own_cell_takes_the_next_block():
    boxes = boxes_for(line("Patient", x1=100), line("Gagnon Luc", x1=400), canonicals=())
    assert_covers(boxes, [(300, 400)])


def test_label_word_in_prose_is_not_a_label():
    assert boxes_for(line("Le patient Revient demain"), canonicals=()) == []


def test_lowercase_prose_after_a_label_is_not_a_name():
    assert boxes_for(line("Patient: stable ce matin"), canonicals=()) == []


# -- NER -------------------------------------------------------------------


def test_ner_hits_are_redacted_without_their_title():
    text = "Vu par Dre Sophie Gagnon ce matin"
    model = StubModel({text: [(7, 24)]})  # "Dre Sophie Gagnon"
    boxes = boxes_for(line(text), canonicals=(), model=model)
    assert_covers(boxes, [(110, 240)])
    assert boxes[0]["confidence"] < 0.9


def test_ner_hit_with_digits_is_ignored():
    text = "Chambre 12 B"
    model = StubModel({text: [(0, 12)]})
    assert boxes_for(line(text), canonicals=(), model=model) == []


def test_single_word_ner_hit_needs_a_title():
    """Alone, the model calls places people: "Clinique Medicale Laval"."""
    text = "Clinique Medicale Laval"
    model = StubModel({text: [(18, 23)]})
    assert boxes_for(line(text), canonicals=(), model=model) == []


def test_single_word_ner_hit_after_a_title_is_kept():
    text = "Signé: Dr Gagnon"
    model = StubModel({text: [(10, 16)]})  # model left the title out
    assert_covers(boxes_for(line(text), canonicals=(), model=model), [(100, 160)])


def test_ner_hits_are_not_propagated():
    """A model false positive costs one box, not one per occurrence."""
    first, second = "Signé par Dr Gagnon", "Gagnon"
    model = StubModel({first: [(10, 19)]})
    pages = [NamePage([line(first)], W, H), NamePage([line(second)], W, H)]
    page1, page2 = locate_name_regions(pages, (), model)
    assert len(page1) == 1
    assert page2 == []


def test_ner_and_known_overlap_merge_into_one_box():
    text = "Pierre Hogue"
    model = StubModel({text: [(0, 12)]})
    boxes = boxes_for(line(text), model=model)
    assert len(boxes) == 1
    assert boxes[0]["confidence"] == 0.9


def test_spacy_adapter_title_cases_shouting_lines():
    seen: list[str] = []

    class Nlp:
        def pipe(self, texts):
            seen.extend(texts)
            return [type("Doc", (), {"ents": []})() for _ in texts]

    SpacyNameModel(Nlp()).find_names(["HOGUE, PIERRE", "Déjà vu"])
    assert seen == ["Hogue, Pierre", "Déjà vu"]
