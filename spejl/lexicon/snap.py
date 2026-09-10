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

# Below this, a fuzzy match is guessing, not correcting. A bare single
# letter shares SOME similarity with almost any short word by chance
# alone (WRatio scores 'T' against 'Stue' at 90 -- comfortably over
# min_score -- because 't' is literally a substring of 'stue'), so
# without a floor, any stray single-character detection ANYWHERE on the
# sheet confidently relabels itself as a full room name. Confirmed on a
# real plan: an unrelated 'T' annotation roughly 800px away from the
# room labelled 'Stue' snapped to 'Stue' anyway, rendering a spurious
# duplicate label at the wrong (tiny) size. Tier 1 (exact) and Tier 2
# (unambiguous fold) are naturally immune -- a single character can
# never equal a whole word -- so only Tier 3 needs this guard.
_MIN_FUZZY_HEAD_LEN = 3


@dataclass(frozen=True)
class SnapResult:
    """What the lexicon decided about one string."""

    text: str                 # corrected — what should be drawn
    raw: str                  # exactly what OCR returned
    kind: str                 # "room" | "dimension" | "area" | "annotation" | "unknown"
    changed: bool
    confidence: float         # match score 0-1 for a snap; 1.0 when untouched
    warning: str | None = None


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


@lru_cache(maxsize=1)
def _vocabulary() -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...]]:
    data = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
    rooms = tuple(data["rooms"])
    # Keyed with _fold(), matching the lookup site below — not
    # .casefold(): _fold() unconditionally rewrites æ/ø/å, so a key built
    # with plain .casefold() (which keeps those letters) could never
    # match a folded lookup string and the entry would be silently dead.
    # ("Vær" -> .casefold() "vær", but _fold(head) can never produce "vær"
    # since _fold always rewrites æ to "ae".)
    abbrev = {_fold(k): v for k, v in data["abbreviations"].items()}
    annotations = tuple(data["annotations"])
    return rooms, abbrev, annotations


def snap(raw: str, *, min_score: float = 82.0) -> SnapResult:
    """Correct one OCR string against the lexicon, or classify it."""
    text = raw.strip()
    if not text:
        return SnapResult(text="", raw=raw, kind="unknown", changed=False, confidence=0.0)

    # Strip leading/trailing dash-family noise before anything else. Real
    # plans routinely run a dashed reference line (a ceiling-height or
    # ownership-boundary line) straight through a room, and confirmed on
    # a real project's own drawing: RapidOCR reads a fragment of that
    # dashed line sitting right next to a label as literal leading
    # characters prepended to the room name — '---Entre', '-Kokken'.
    # Left in, that noise is enough to dodge Tier 1's exact-match guard
    # below (the whole point of which is to stop fuzzy matching from
    # resolving a room name to the WRONG lexicon twin) without being
    # different enough to trip Tier 2's unambiguous fold match either,
    # so it falls through to Tier 3 fuzzy matching and can pick the
    # wrong one anyway — confirmed: '---Entre' resolved to 'Entré'
    # instead of plain 'Entre'. No real Danish room label or dimension
    # starts or ends with a bare dash, so stripping this is safe; a
    # MEANINGFUL hyphen (the "Vaer.-1" room-number-suffix case) sits
    # between two other characters, never at either edge, so it is
    # untouched by an edge-only strip.
    text = text.strip("-‐‑‒–—―")

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

    # Tier 1 — the string is already a lexicon entry, exactly or up to
    # case. Never "correct" it. Danish plans use both "Entre" and
    # "Entré"; fuzzy ranking alone will happily swap one for the other,
    # which is a regression, not a fix — and that includes case variants:
    # an all-lowercase "entre" must resolve here too, or it falls through
    # past this guard into the exact ambiguity (Tier 2 sees both "Entre"
    # and "Entré" fold to the same key) it exists to prevent.
    # Deliberately `.casefold()`, not `_fold()`: casefold ignores case but
    # keeps diacritics, so "entre" matches only "Entre", not "Entré" —
    # `_fold()` would strip the accent and reintroduce the ambiguity.
    if head in rooms:
        return SnapResult(text, raw, "room", False, 1.0)
    case_matches = [r for r in rooms if r.casefold() == head.casefold()]
    if len(case_matches) == 1:
        corrected = case_matches[0] + sep + tail
        return SnapResult(corrected, raw, "room", corrected != text, 1.0)

    # Tier 2 — an unambiguous diacritic-only difference ("Kokken" for
    # "Køkken", "Vaer." for "Vær."): exactly one entry folds to the same
    # key, so the correction is deterministic rather than a guess.
    fold_matches = [r for r, folded in candidates.items() if folded == folded_head]
    if len(fold_matches) == 1:
        corrected = fold_matches[0] + sep + tail
        return SnapResult(corrected, raw, "room", corrected != text, 1.0)

    # Tier 3 — genuine fuzzy match, for OCR damage beyond diacritics.
    # See _MIN_FUZZY_HEAD_LEN's own comment: a head this short is too
    # little evidence for a similarity score to mean anything, no matter
    # how high it comes back.
    match = (
        process.extractOne(folded_head, candidates, scorer=fuzz.WRatio, processor=None)
        if len(folded_head) >= _MIN_FUZZY_HEAD_LEN
        else None
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
    """``"Vaer. 1"`` -> ``("Vaer.", " ", "1")``; ``"Stue"`` -> ``("Stue", "", "")``.

    The separator before the number is normalised to a single space on
    the way out, not preserved verbatim: OCR on a real (non-synthetic)
    plan misread "Vær. 1"'s space as a hyphen — "Vaer.-1" — and the
    original whitespace-only pattern didn't match that at all, so the
    whole string fell through as one unsplit token with no lexicon
    match. A room-number suffix is a drafting convention, not a
    hyphenated compound word, so any run of space/hyphen/en-dash
    characters here is read as that separator and rewritten as a plain
    space, regardless of which one OCR happened to produce.

    The separator is optional (``*``, not ``+``), not just multi-form:
    on the same real plan, "Vær. 3" came back "Vr.3" — OCR dropped "æ"
    outright rather than just its diacritic, and with nothing between
    the abbreviation's own period and the room number, there was no
    separator character at all to require. A room label never ends in
    a bare digit on its own account, so a trailing digit run is always
    this suffix, separator or not.
    """
    m = re.match(r"^(.*?)[\s\-‐-―]*(\d+)$", text)
    if m:
        return m.group(1), " ", m.group(2)
    return text, "", ""
