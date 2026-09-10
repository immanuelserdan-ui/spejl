"""Lexicon snap — the stage that earns more than a better OCR model.

Fast, no OCR: these are the exact substitutions observed on the golden
fixture, locked in so a lexicon edit can't silently regress them.
"""

from __future__ import annotations

import pytest

from spejl.lexicon.snap import snap


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Kokken", "Køkken"),      # ø dropped by a Latin-only recogniser
        ("Vaer.", "Vær."),         # æ dropped
        ("Vaer. 1", "Vær. 1"),     # ... with the room number preserved
        ("Kokken ", "Køkken"),     # trailing whitespace
        ("Vaerelse", "Værelse"),
        ("Vaer.-1", "Vær. 1"),     # regression: real-plan OCR misread the
                                   # space before the room number as a
                                   # hyphen; must still split and correct
        ("Vaer.‑1", "Vær. 1"),  # non-breaking hyphen variant too
    ],
)
def test_diacritics_are_restored(raw: str, expected: str):
    result = snap(raw)
    assert result.text == expected
    assert result.changed
    assert result.kind == "room"


@pytest.mark.parametrize("text", ["Stue", "Bad", "Toilet", "Entre", "Entré", "Køkken"])
def test_correct_strings_are_left_alone(text: str):
    """Regression guard: fuzzy ranking once swapped 'Entre' for 'Entré'.
    Snapping a valid string to a *different* valid string is a defect."""
    result = snap(text)
    assert result.text == text
    assert not result.changed


def test_dimensions_are_not_run_past_the_word_list():
    result = snap("2900")
    assert result.kind == "dimension"
    assert result.text == "2900"
    assert result.warning is None


def test_implausible_dimension_is_flagged_despite_high_ocr_confidence():
    """'2105' read as '105' arrives at 0.99 confidence, so confidence
    cannot catch it — the millimetre range check has to."""
    result = snap("105")
    assert result.kind == "dimension"
    assert result.warning is not None
    assert "outside the plausible range" in result.warning


def test_plausible_dimension_range_boundaries():
    assert snap("200").warning is None
    assert snap("20000").warning is None
    assert snap("199").warning is not None
    assert snap("20001").warning is not None


def test_annotation_glyph_is_recognised():
    result = snap("H*")
    assert result.kind == "annotation"
    assert result.text == "H*"


def test_area_figure_with_comma_decimal():
    result = snap("12,4 m²")
    assert result.kind == "area"
    assert result.text == "12,4 m²"


def test_unknown_string_is_reported_not_guessed():
    result = snap("Qzxwv")
    assert result.kind == "unknown"
    assert result.warning is not None
    assert result.text == "Qzxwv"  # returned unchanged, never invented


@pytest.mark.parametrize("raw", ["entre", "ENTRE", "eNtRe"])
def test_case_only_difference_resolves_to_the_matching_entry_not_its_accented_twin(raw: str):
    """Regression: 'Entre' and 'Entré' both fold to the same key, so a
    lowercase 'entre' used to skip the case-sensitive Tier 1 guard,
    land in Tier 2's correctly-detected ambiguity, and fall through to
    Tier 3 fuzzy matching — which resolved the tie by arbitrarily
    picking 'Entré', the exact regression Tier 1 exists to prevent."""
    result = snap(raw)
    assert result.text == "Entre"
    assert result.kind == "room"


def test_accented_case_variant_still_resolves_to_itself():
    result = snap("entré")
    assert result.text == "Entré"


def test_abbreviation_with_a_diacritic_is_reachable():
    """Regression: da_dk.json's abbreviations dict was keyed with plain
    .casefold(), which leaves æ/ø/å untouched — but the lookup folds
    with _fold(), which always rewrites them. A key built the first way
    can never match a lookup built the second way, so 'Vær' -> 'Værelse'
    was dead code. (Currently masked for this exact input by Tier 3
    fuzzy matching resolving it first — this test locks in the
    abbreviation-table fix itself, not just the visible symptom, so a
    future entry that Tier 3 doesn't confidently catch stays reachable.)
    """
    from spejl.lexicon.snap import _fold, _vocabulary

    _rooms, abbrev, _annotations = _vocabulary()
    assert "vær" not in abbrev  # the broken plain-.casefold() key (æ kept)
    assert _fold("Vær") in abbrev  # the correct _fold()-consistent key (æ->ae)
