"""Unit tests for the NAM locator with synthetic OCR blocks."""
import pytest
from nam_redactor.locator import locate_nam_regions


class MockBlock:
    def __init__(self, text: str, bbox: tuple[int, int, int, int], page: int = 1):
        self.text = text
        self.bbox = bbox
        self.page = page


def test_locate_nam_alone():
    """A NAM alone."""
    block = MockBlock("HOGP 3361 0615", (0, 0, 120, 20))
    regions = locate_nam_regions([block], 120, 20)
    assert len(regions) == 1
    assert regions[0]["x1"] == pytest.approx(0.0)
    assert regions[0]["x2"] == pytest.approx(1.0)
    assert regions[0]["confidence"] >= 0.6


def test_locate_nam_excludes_label():
    """A line 'NAM: HOGP 3361 0615' whose box excludes the label."""
    block = MockBlock("NAM: HOGP 3361 0615", (0, 0, 190, 20))
    regions = locate_nam_regions([block], 190, 20)
    assert len(regions) == 1
    region = regions[0]
    # "NAM: " takes up the first ~26% of characters
    assert 0.25 < region["x1"] < 0.30
    assert region["x2"] == pytest.approx(1.0)


def test_locate_nam_joins_split_blocks():
    """A NAM split across two adjacent blocks on the same row."""
    left = MockBlock("HOGP 336", (0, 0, 100, 20))
    right = MockBlock("1 0615", (105, 0, 160, 20))
    regions = locate_nam_regions([left, right], 200, 20)
    assert len(regions) == 1
    assert regions[0]["confidence"] >= 0.55


def test_locate_nam_ignores_non_nam_text():
    """Non-NAM text that must be ignored."""
    blocks = [
        MockBlock("Clinique Médicale 12345", (0, 0, 150, 20)),
        MockBlock("Dr. Martin Tremblay", (0, 30, 150, 50)),
        MockBlock("Date: 2026-09-22", (0, 60, 150, 80)),
        MockBlock("Code d'acte: 08123 Montant: 48.75 $", (0, 90, 250, 110)),
    ]
    assert locate_nam_regions(blocks, 300, 300) == []


def test_locate_nam_glued_to_a_space_separated_label():
    """A NAM sharing its line with a label and no colon separator.

    'NAM HOGP 3361 0615' is a single 15-character run, so validating the run
    as a whole rejects it and the NAM inside is lost. This was the real-scan
    failure: OCR read the number correctly and the locator discarded it.
    """
    block = MockBlock("NAM HOGP 3361 0615", (0, 0, 210, 20))
    regions = locate_nam_regions([block], 210, 20)

    assert len(regions) == 1
    # The box must start after the label, not at the beginning of the line.
    assert regions[0]["x1"] > 0.15
    assert regions[0]["confidence"] >= 0.6


def test_locate_nam_fused_label_and_name_letters():
    """No group boundary to cut at: the window fallback must rescue it."""
    block = MockBlock("NAMHOGP 3361 0615", (0, 0, 190, 20))
    regions = locate_nam_regions([block], 190, 20)

    assert len(regions) == 1
    assert regions[0]["x1"] > 0.1


def test_locate_nam_in_a_crowded_line_yields_exactly_one_region():
    """Extra words around the NAM must not produce duplicate boxes."""
    block = MockBlock("Dossier 4471 NAM HOGP 3361 0615", (0, 0, 400, 20))
    regions = locate_nam_regions([block], 400, 20)

    assert len(regions) == 1
