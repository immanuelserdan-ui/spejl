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
