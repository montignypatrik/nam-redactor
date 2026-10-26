"""Deterministic RAMQ (Quebec Health Insurance Number / NAM) validation.

A NAM is 4 letters followed by 8 digits, sometimes grouped ``abcd 1234 5678``:

- letters 1-3: first three letters of the patient's **last name**
- letter 4: first letter of the patient's **first name**
- digits 1-2: birth **year** (two digits; no usable range check, patients may
  be very young or very old)
- digits 3-4: birth **month**, 01-12, or **51-62** for female records (+50)
- digits 5-6: birth **day**, which must exist in that month
- digits 7-8: sequence, carries no meaning here

Rule-based validation tolerant of OCR character confusions (O/0, I/1, B/8, S/5, Z/2...).
Never uses an LLM.
"""
from __future__ import annotations

import re

# Frequent OCR confusions in digit zone:
_DIGIT_ZONE = {
    "O": "0", "o": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "l": "1", "L": "1", "|": "1", "i": "1",
    "B": "8", "S": "5", "Z": "2", "z": "2", "G": "6",
}

# Frequent OCR confusions in letter zone (first 4 positions):
_LETTER_ZONE = {
    "1": "I", "0": "O", "5": "S", "8": "B", "2": "Z", "6": "G",
    "l": "L", "|": "I",
}

_CLEAN_RE = re.compile(r"[\s\-–—]+")
_RUN_RE = re.compile(r"[A-Za-z0-9]+(?:[\s\-–—]+[A-Za-z0-9]+){0,5}")

#: Days per calendar month, indexed 1-12. February allows 29 because a
#: two-digit year cannot settle the century (1900 and 2000 differ).
_DAYS_IN_MONTH = (0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

NON_NAME_STOPWORDS = {
    "sans", "rendez", "vous", "sansrendez-vous", "sans-rendez-vous", "sansrendezvous",
    "regulier", "régulier", "urgence", "urgent", "urgente", "urgences",
    "suivi", "gmf", "gap", "clinique", "medicale", "médicale", "agenda",
    "visite", "consultation", "saisie", "manuelle", "inf", "vaginal", "infection",
    "abces", "abcès", "panari", "drainage", "fasceite", "plantaire", "faceite",
    "bouton", "gorge", "fievre", "fièvre", "pneumonie", "repartiteur", "dispatcher",
    "medecin", "résident", "resident", "contact", "numero", "numéro", "courriel",
    "email", "imprime", "imprimé", "genere", "généré", "omnimed", "confidentiel",
    "document", "rectomrage", "dre", "dr", "docteur", "docteure", "annierivest",
    "annie", "rivest", "samedi", "dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi",
    "date", "dates", "dob", "naissance", "heure", "heures", "quart", "garde",
    "plaie", "face", "forfait", "jour", "soir", "nuit", "luxation", "exploration",
    "onycect", "onycectomie", "bourse", "ans", "par", "transfert", "canada",
    "given", "family", "federal", "interim", "certificate", "eligibility",
    "travel", "resettlement", "protected", "muscle", "tendon", "peau", "bloc",
    "platre", "lavage", "assign", "rappel", "trousse", "kystes", "conjonctivite",
    "lombaire", "caps", "bague", "corhee", "hematome", "hématome", "debridement",
    "débridement", "evaluation", "évaluation", "evacuation", "évacuation", "prise",
    "charge", "evolution", "évolution", "final", "certificat", "prelevement",
    "prélèvement", "epistaxis", "épistaxis", "epitaxis", "epaule", "épaule",
    "orteil", "orteils", "doigt", "doigts", "membre", "membres", "fracture",
    "entorse", "plie", "point", "points", "suture", "sutures", "corps", "etranger",
}


def is_non_name_word(text: str) -> bool:
    clean = re.sub(r"[^a-zA-ZÀ-ÿ]", "", text).lower()
    if not clean or len(clean) < 2:
        return True
    if clean.startswith("sansrendez") or clean.startswith("sansrdv"):
        return True
    for sw in NON_NAME_STOPWORDS:
        if clean == re.sub(r"[^a-zA-ZÀ-ÿ]", "", sw).lower():
            return True
    return False


def clean_ramq_run(text: str) -> str:
    """Strip spaces, tabs, and dashes from an OCR run and convert to uppercase."""
    return _CLEAN_RE.sub("", text).upper()


def validate_ramq(cleaned: str) -> tuple[bool, float, list[str]]:
    """Validate a cleaned run as a probable RAMQ / NAM number.

    Returns:
        (is_valid, confidence, issues)

    Confidence reflects OCR character confusions only. Date rules are strict:
    an impossible month or a day that does not exist in that month is rejected
    outright, because no real patient has one. OCR confusion is absorbed
    beforehand by normalising the digit zone, so ``O`` reads as a zero.
    """
    if len(cleaned) not in (12, 13):  # 13th is optional check digit, ignored here
        return False, 0.0, ["longueur"]

    name_part, digit_part = cleaned[:4], cleaned[4:12]

    # Letter prefix must have at least 3 genuine letters and not be a stopword
    genuine_letters = sum(1 for ch in name_part if ch.isalpha() and ch.isupper())
    if genuine_letters < 3 or is_non_name_word(name_part):
        return False, 0.0, ["zone_nom"]

    # Digit section must contain at least 5 genuine digits
    genuine_digits = sum(1 for ch in digit_part if ch.isdigit())
    if genuine_digits < 5:
        return False, 0.0, ["zone_chiffres"]

    substitutions = 0
    for ch in name_part:
        if ch.isalpha() and ch.isupper():
            continue
        if ch in _LETTER_ZONE:
            substitutions += 1
            continue
        return False, 0.0, ["zone_nom"]

    for ch in digit_part:
        if ch.isdigit():
            continue
        if ch in _DIGIT_ZONE:
            substitutions += 1
            continue
        return False, 0.0, ["zone_chiffres"]

    norm_digits = "".join(
        ch if ch.isdigit() else _DIGIT_ZONE[ch] for ch in digit_part
    )
    mois = int(norm_digits[2:4])
    jour = int(norm_digits[4:6])

    # Month must be 01-12, or 51-62 for female records (+50 offset).
    if not (1 <= mois <= 12 or 51 <= mois <= 62):
        return False, 0.0, ["mois_invalide"]

    # Day must exist in that month. Strip the female +50 offset first so the
    # month length is the real calendar one -- February 30 and April 31 are
    # impossible dates, and no patient has one.
    real_month = mois - 50 if mois >= 51 else mois
    if not (1 <= jour <= _DAYS_IN_MONTH[real_month]):
        return False, 0.0, ["jour_invalide"]

    conf = 0.99 - 0.04 * substitutions
    return True, max(0.0, min(1.0, conf)), []


def canonical_nam(cleaned: str) -> str | None:
    """Stable pseudonym key for a valid NAM, or ``None`` if it is not one.

    Two readings of the same number must produce the same key, so OCR
    confusions are normalised away: the digit zone becomes digits and the
    letter zone becomes letters, using the same maps ``validate_ramq`` uses.

    The optional 13th check digit is dropped. ``validate_ramq`` accepts 12 or
    13 characters, so if OCR picks the check digit up on one occurrence of a
    number but not another, keeping it would split one patient into two
    identities -- which is exactly what this key exists to prevent.
    """
    ok, _, _ = validate_ramq(cleaned)
    if not ok:
        return None

    body = cleaned[:12]
    letters = "".join(
        ch.upper() if ch.isalpha() else _LETTER_ZONE[ch] for ch in body[:4]
    )
    digits = "".join(
        ch if ch.isdigit() else _DIGIT_ZONE[ch] for ch in body[4:]
    )
    return letters + digits
