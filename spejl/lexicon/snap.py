"""Snap OCR output to the plan vocabulary, and sanity-check dimensions.

Stage S3 of the build plan. Two jobs, both cheap, both worth more than a
better recognition model:

1. **Diacritic restoration.** OCR reliably drops Danish Ø/Æ/Å —
   ``Køkken`` comes back ``Kokken``, ``Vær. 1`` comes back ``Vaer. 1``.
   Because plan vocabulary is a closed set, fuzzy-matching against it
   restores the right glyph deterministically.
2. **Dimension plausibility.** A millimetre wall run is 200-20000 mm. A
   rotated ``2105`` recognised as ``105`` arrives with 0.99 confidence,
   so confidence cannot catch it — but a range check can.

Every snap keeps the raw string beside the corrected one so the review
pane can show exactly what the tool changed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz, process

_LEXICON_PATH = Path(__file__).with_name("da_dk.json")

# A dimension on these plans is a bare millimetre integer; area figures
# carry a comma decimal and a unit ("12,4 m²") and are matched separately.
_DIM_RE = re.compile(r"^\d{2,5}$")
_AREA_RE = re.compile(r"^\d{1,3}(?:[,.]\d{1,2})?\s*m[²2]?$")

DIM_MIN_MM = 200
DIM_MAX_MM = 20000


@dataclass(frozen=True)
class SnapResult:
    """What the lexicon decided about one string."""

    text: str                 # corrected — what should be drawn
    raw: str                  # exactly what OCR returned
    kind: str                 # "room" | "dimension" | "area" | "annotation" | "unknown"
    changed: bool
    confidence: float         # match score 0-1 for a snap; 1.0 when untouched
    warning: str | None = None


@lru_cache(maxsize=1)
def _vocabulary() -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...]]:
    data = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
    rooms = tuple(data["rooms"])
    abbrev = {k.casefold(): v for k, v in data["abbreviations"].items()}
    annotations = tuple(data["annotations"])
    return rooms, abbrev, annotations


def _fold(s: str) -> str:
    """Diacritic-insensitive key, with the Danish digraphs OCR substitutes.

    NFKD alone maps ``é`` to ``e`` but leaves ``ø`` and ``æ`` untouched —
    they are distinct letters in Danish, not accented vowels — so those
    are mapped explicitly to what a Latin-only model tends to emit.
    """
    lowered = s.casefold()
    for src, dst in (("ø", "o"), ("æ", "ae"), ("å", "aa"), ("ö", "o"), ("ä", "ae")):
        lowered = lowered.replace(src, dst)
    stripped = unicodedata.normalize("NFKD", lowered)
    return "".join(c for c in stripped if not unicodedata.combining(c))


def snap(raw: str, *, min_score: float = 82.0) -> SnapResult:
    """Correct one OCR string against the lexicon, or classify it."""
    text = raw.strip()
    if not text:
        return SnapResult(text="", raw=raw, kind="unknown", changed=False, confidence=0.0)

    rooms, abbrev, annotations = _vocabulary()

    # Dimensions and areas are numeric — never run them past a word list.
    if _DIM_RE.match(text):
        value = int(text)
        warning = None
        if not (DIM_MIN_MM <= value <= DIM_MAX_MM):
            warning = (
                f"{value} mm is outside the plausible range "
                f"{DIM_MIN_MM}-{DIM_MAX_MM} mm — likely a truncated or misread run."
            )
        return SnapResult(text, raw, "dimension", False, 1.0, warning)

    if _AREA_RE.match(text):
        return SnapResult(text, raw, "area", False, 1.0)

    if text in annotations:
        return SnapResult(text, raw, "annotation", False, 1.0)

    # Room labels can carry a trailing number ("Vær. 1"); match the word
    # part and reattach the rest, so the numbering survives the snap.
    head, sep, tail = _split_trailing_number(text)
    candidates = {r: _fold(r) for r in rooms}
    folded_head = _fold(head)

    # Tier 1 — the string is already a lexicon entry. Never "correct" it.
    # Danish plans use both "Entre" and "Entré"; fuzzy ranking alone will
    # happily swap one for the other, which is a regression, not a fix.
    if head in rooms:
        return SnapResult(text, raw, "room", False, 1.0)

    # Tier 2 — an unambiguous diacritic-only difference ("Kokken" for
    # "Køkken", "Vaer." for "Vær."): exactly one entry folds to the same
    # key, so the correction is deterministic rather than a guess.
    fold_matches = [r for r, folded in candidates.items() if folded == folded_head]
    if len(fold_matches) == 1:
        corrected = fold_matches[0] + sep + tail
        return SnapResult(corrected, raw, "room", corrected != text, 1.0)

    # Tier 3 — genuine fuzzy match, for OCR damage beyond diacritics.
    match = process.extractOne(
        folded_head, candidates, scorer=fuzz.WRatio, processor=None
    )
    if match is not None:
        _folded_value, score, matched_room = match
        if score >= min_score:
            corrected = matched_room + sep + tail
            return SnapResult(
                text=corrected,
                raw=raw,
                kind="room",
                changed=corrected != text,
                confidence=score / 100.0,
            )

    lowered = _fold(head)
    if lowered in abbrev:
        corrected = abbrev[lowered] + sep + tail
        return SnapResult(corrected, raw, "room", corrected != text, 0.9)

    return SnapResult(text, raw, "unknown", False, 0.0,
                      "No lexicon match — confirm this string manually.")


def _split_trailing_number(text: str) -> tuple[str, str, str]:
    """``"Vaer. 1"`` -> ``("Vaer.", " ", "1")``; ``"Stue"`` -> ``("Stue", "", "")``."""
    m = re.match(r"^(.*?)(\s+)(\d+)$", text)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return text, "", ""
