"""Unit tests for deterministic RAMQ validation and OCR confusion tolerance."""
import pytest
from nam_redactor.validation import canonical_nam, clean_ramq_run, validate_ramq


def test_clean_ramq_run():
    assert clean_ramq_run("abcd 1234 5678") == "ABCD12345678"
    assert clean_ramq_run("HOGP-3361-0615") == "HOGP33610615"
    assert clean_ramq_run("  dup m   8254   1234  ") == "DUPM82541234"


def test_validate_valid_nam_male():
    # Male born 1982-04-12
    ok, conf, problems = validate_ramq("DUPM82041234")
    assert ok is True
    assert conf >= 0.95
    assert len(problems) == 0


def test_validate_valid_nam_female():
    # Female born 1975-11-28 (+50 to month -> 61)
    ok, conf, problems = validate_ramq("GAGR75612890")
    assert ok is True
    assert conf >= 0.95
    assert len(problems) == 0


def test_validate_ocr_confusion_substitutions():
    # 'O' instead of '0', 'I' instead of '1'
    # DUPM82O4I234
    ok, conf, problems = validate_ramq("DUPM82O4I234")
    assert ok is True
    assert conf < 0.99  # penalized for substitutions
    assert conf >= 0.85


def test_validate_invalid_length():
    ok, _, problems = validate_ramq("DUPM82041")
    assert ok is False
    assert "longueur" in problems


def test_validate_invalid_month():
    # Month 13 is invalid (allowed: 1-12 or 51-62)
    ok, _, problems = validate_ramq("DUPM82131234")
    assert ok is False
    assert "mois_invalide" in problems


def test_validate_rejects_impossible_day_for_month():
    """February 30 and April 31 are not real birth dates."""
    ok, _, problems = validate_ramq("DUPM82023034")  # Feb 30
    assert ok is False
    assert "jour_invalide" in problems

    ok, _, problems = validate_ramq("DUPM82043134")  # Apr 31
    assert ok is False
    assert "jour_invalide" in problems


def test_validate_accepts_last_real_day_of_month():
    assert validate_ramq("DUPM82043034")[0] is True  # Apr 30
    assert validate_ramq("DUPM82022934")[0] is True  # Feb 29 (century unknown)
    assert validate_ramq("DUPM82013134")[0] is True  # Jan 31


def test_validate_female_offset_uses_the_real_calendar_month():
    """Month 61 is November (+50), which has 30 days -- not 31."""
    ok, _, problems = validate_ramq("GAGR75613134")  # Nov 31
    assert ok is False
    assert "jour_invalide" in problems

    assert validate_ramq("GAGR75613034")[0] is True  # Nov 30


def test_validate_normalises_a_confused_zero_in_the_date():
    """OCR reads 0 as O often; the date must still parse after normalising."""
    ok, conf, problems = validate_ramq("DUPM8204O934")  # day "O9" -> 09
    assert ok is True
    assert conf < 0.99
    assert problems == []


# -- canonical_nam: the pseudonym key -------------------------------------


def test_canonical_nam_folds_ocr_confusions_together():
    """Two readings of one number must produce one key, or one patient
    would be split into two identities."""
    assert canonical_nam("DUPM82O4I234") == canonical_nam("DUPM82041234")
    assert canonical_nam("DUPM82041234") == "DUPM82041234"


def test_canonical_nam_ignores_the_optional_check_digit():
    """12- and 13-character readings are the same patient.

    ``validate_ramq`` accepts both lengths, so if OCR catches the trailing
    check digit on one occurrence but not another, keeping it in the key
    would split the record.
    """
    assert canonical_nam("DUPM820412349") == canonical_nam("DUPM82041234")


def test_canonical_nam_normalises_the_letter_zone_to_letters():
    # '1' in the letter zone is a misread 'I'.
    assert canonical_nam("DUP182041234") == "DUPI82041234"


def test_canonical_nam_is_none_for_anything_that_is_not_a_nam():
    assert canonical_nam("NOTANAM") is None
    assert canonical_nam("SANS82041234") is None  # stopword in the name zone
    assert canonical_nam("DUPM82043134") is None  # April 31 does not exist


def test_canonical_nam_distinguishes_different_patients():
    assert canonical_nam("DUPM82041234") != canonical_nam("GAGR75612890")


def test_validate_stopwords_rejected():
    # Medical stopwords like "SANS" or "DATE" in letter zone should be rejected
    ok, _, problems = validate_ramq("SANS82041234")
    assert ok is False
    assert "zone_nom" in problems
