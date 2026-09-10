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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("---Entre", "Entre"),   # regression: a dashed reference line on a
        ("-Kokken", "Køkken"),   # real drawing crosses near/through a label;
        ("--Bad", "Bad"),        # RapidOCR reads a fragment of it as leading
        ("Toilet--", "Toilet"),  # dash characters prepended (or appended) to
    ],                           # the room name it detected
)
def test_dash_noise_from_a_crossing_reference_line_is_stripped(raw: str, expected: str):
    """Regression: real project drawing 722-0553-0006-1001 uses dashed
    reference lines running through several rooms. OCR read '---Entre'
    for the room labelled 'Entre' — enough noise to dodge Tier 1's
    exact-match guard (the point of which is to stop fuzzy matching
    picking the WRONG lexicon twin) without being different enough to
    hit Tier 2's unambiguous fold match either, so it fell through to
    Tier 3 fuzzy matching and resolved to 'Entré' instead of the plain
    'Entre' the drawing actually says. Stripped before any tier runs.
    """
    result = snap(raw)
    assert result.text == expected


def test_kaelderrum_is_not_truncated_to_kaelder():
    """Regression: OCR on a real project drawing read 'Kaelderrum'
    (basement STORAGE ROOM) perfectly, needing only diacritic
    restoration — but 'Kælderrum' was missing from the lexicon while
    the shorter, related 'Kælder' (basement) was present, so Tier 3
    fuzzy matching confidently substituted the wrong, shorter word for
    a correctly-read longer one. A closed vocabulary needs both real
    words present, not just one standing in for its relative."""
    assert snap("Kaelderrum").text == "Kælderrum"
    assert snap("Kaelder").text == "Kælder"  # the distinct, shorter word is unaffected


def test_meaningful_internal_hyphen_survives_the_edge_only_strip():
    """The dash-noise strip only touches the string's edges — a
    hyphen used as a real separator (the 'Vaer.-1' room-number-suffix
    case, also a real OCR misread) sits between two other characters
    and must still resolve correctly."""
    assert snap("Vaer.-1").text == "Vær. 1"


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


def test_room_number_glued_directly_to_the_abbreviation_still_splits():
    """Regression: real-plan OCR on 'Vær. 3' dropped the 'æ' outright
    (not just its diacritic) AND left no separator at all between the
    abbreviation's own period and the room number — '--Vr.3' after the
    dash-noise strip. _split_trailing_number's original pattern required
    at least one space/hyphen separator, so this fell through unsplit as
    one token, and 'Vr.3' as a whole matched nothing in any tier —
    genuinely different from the 'Vaer.-1' regression above (that one
    DOES have a separator, just a misread one).

    A room label never legitimately ends in a bare digit on its own
    account, so treating any trailing digit run as this suffix —
    separator or not — is safe.
    """
    result = snap("--Vr.3")
    assert result.text == "Vær. 3"
    assert result.kind == "room"
    assert result.warning is None


def test_vr_period_abbreviation_is_a_curated_lexicon_entry():
    """'Vr.' is missing 'æ' entirely, not just its diacritic (WRatio
    fuzzy score against 'Vær.' is 75, below the 82 min_score — correctly
    too uncertain for Tier 3 to guess), so it needs an explicit
    abbreviation entry, the same way 'Vær' and 'Vaer' are both already
    curated variants of the identical word for the identical reason:
    OCR corrupts the same abbreviation different ways across different
    real plans."""
    result = snap("Vr.")
    assert result.text == "Vær."
    assert result.kind == "room"
