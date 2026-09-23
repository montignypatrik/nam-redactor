"""Unit tests for deterministic RAMQ validation and OCR confusion tolerance."""
import pytest
from nam_redactor.validation import clean_ramq_run, validate_ramq


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
    ok, conf, problems = validate_ramq("DUPM82131234")
    # Substitutions = 0, so month_invalide penalizes confidence
    assert "mois_invalide" in problems


def test_validate_stopwords_rejected():
    # Medical stopwords like "SANS" or "DATE" in letter zone should be rejected
    ok, _, problems = validate_ramq("SANS82041234")
    assert ok is False
    assert "zone_nom" in problems
