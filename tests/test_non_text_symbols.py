"""Regression: a graphical symbol (an arrow, a bullet, a stray dash)
mis-detected by OCR as a short "text" run must not be erased and
redrawn as garbled text. Found on a real, non-synthetic plan: a '→'
direction arrow, detected at 0.50 confidence with no lexicon match,
was erased and re-rendered as a nonsense string overlapping the real
'Entré' room label after mirroring.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spejl.detect.ocr import Detection
from spejl.models import Axis
from spejl.raster.pipeline import _looks_like_a_graphical_symbol, mirror_raster


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("→", True),
        ("—", True),
        ("•", True),
        ("--", True),
        ("H*", False),   # a real annotation glyph — has an alnum char
        ("2105", False),
        ("Stue", False),
        ("Kok", False),
        ("", True),      # empty: no alnum char present either
    ],
)
def test_looks_like_a_graphical_symbol(text: str, expected: bool):
    assert _looks_like_a_graphical_symbol(text) is expected


class _FixedBackend:
    """Reports fixed detections scaled to whatever image size it's
    given — see tests/test_protected_regions.py for the full rationale
    (a naive fake that ignores rotated-pass input duplicates runs)."""

    def __init__(self, detections: list[Detection], base_size: tuple[int, int]) -> None:
        self._detections = detections
        self._base_h, self._base_w = base_size

    def detect_and_recognise(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        base_h, base_w = self._base_h, self._base_w
        if h * base_w != w * base_h:  # a 90°-rotated pass
            return []
        scale = w / base_w
        return [
            Detection(
                text=d.text,
                quad=tuple((x * scale, y * scale) for x, y in d.quad),
                conf=d.conf,
            )
            for d in self._detections
        ]


def test_an_arrow_symbol_is_not_rendered_as_garbled_text(tmp_path):
    img = np.full((300, 400, 3), 255, np.uint8)
    cv2.putText(img, "Entre", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    src = tmp_path / "plan.png"
    cv2.imwrite(str(src), img)

    entre = Detection(text="Entre", quad=((148, 192), (185, 192), (185, 205), (148, 205)), conf=0.97)
    arrow = Detection(text="→", quad=((220, 195), (235, 195), (235, 205), (220, 205)), conf=0.50)
    backend = _FixedBackend([entre, arrow], base_size=(300, 400))

    result = mirror_raster(src, tmp_path / "out.png", axis=Axis.VERTICAL, backend=backend)

    texts = [r.text for r in result.runs]
    assert "Entre" in texts
    assert not any(_looks_like_a_graphical_symbol(t) for t in texts), texts
    assert "→" not in texts  # never turned into a rendered run at all

    flag_codes = {f.code for p in result.document.pages for f in p.flags}
    assert "non-text-symbol" in flag_codes


def test_symbol_pixels_pass_through_as_geometry_not_erased(tmp_path):
    """The arrow's original pixels should survive the mirror (as
    ordinary flipped geometry), not be erased and left blank."""
    img = np.full((200, 200, 3), 255, np.uint8)
    cv2.line(img, (100, 50), (100, 150), (0, 0, 0), 2)  # a stand-in "arrow" mark
    src = tmp_path / "plan.png"
    cv2.imwrite(str(src), img)

    symbol = Detection(text="—", quad=((90, 88), (110, 88), (110, 112), (90, 112)), conf=0.4)
    backend = _FixedBackend([symbol], base_size=(200, 200))

    mirror_raster(src, tmp_path / "out.png", axis=Axis.VERTICAL, backend=backend)
    out = cv2.imread(str(tmp_path / "out.png"))
    # The vertical line, now mirrored to x' = 200-100 = 100, should still be dark.
    column = cv2.cvtColor(out[50:150, 95:105], cv2.COLOR_BGR2GRAY)
    assert column.min() < 100, "the symbol's own geometry was erased instead of passed through"
