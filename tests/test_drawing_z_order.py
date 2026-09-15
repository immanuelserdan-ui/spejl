"""The drawing layer sits in front of the re-rendered type.

A mirrored sheet has two kinds of content on it and they are not equally
recoverable. A text run is reconstructed — Spejl knows the string, the
measured style and the mirrored anchor, and can redraw it at any time.
Linework is not reconstructed at all; it is the source plate's own
pixels, flipped. So when the two land on the same pixel the drawing has
to win, and these tests pin that down end to end, through the real
pipeline rather than the renderer alone.

Every fixture here draws its type in GREY on BLACK geometry, which is
the only way these tests can see what they are testing. Black type over
black linework is indistinguishable from the linework, so a sheet whose
labels overwrite every line they cross looks pixel-perfect under any
check that only asks "is this pixel dark?" — the damage shows up solely
where type and geometry differ in tone. Real sheets differ in tone
constantly: grey annotation text, and the antialiased flank of every
glyph and every line on a CAD export.
"""

from __future__ import annotations

import cv2
import numpy as np

from spejl.detect.ocr import Detection
from spejl.models import Axis
from spejl.raster.pipeline import mirror_raster


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


# The label, and where it lands once the 420px-wide sheet is mirrored.
_LABEL_QUAD = ((168, 112), (215, 112), (215, 132), (168, 132))
_MIRRORED_COLS = slice(420 - 215, 420 - 168)


def _mirror(tmp_path, image: np.ndarray, text: str):
    src = tmp_path / "plan.png"
    cv2.imwrite(str(src), image)
    label = Detection(text=text, quad=_LABEL_QUAD, conf=0.97)
    result = mirror_raster(
        src, tmp_path / "out.png", axis=Axis.VERTICAL,
        backend=_FixedBackend([label], base_size=(image.shape[0], image.shape[1])),
    )
    return result, cv2.imread(str(tmp_path / "out.png"))


def test_a_dimension_line_keeps_its_full_weight_under_the_label_sitting_on_it(tmp_path):
    """The regression this layer exists for: mirroring a sheet must not
    thin or nick a line just because a label crossed it.

    The line is drawn the way a CAD export actually draws one — a solid
    core with an antialiased flank either side — because that flank is
    where the damage was. The erase takes everything below *local*
    paper, so it removes the flank; the repair used to hand back only
    what Otsu called linework, which is the core alone. The line came
    out a pixel thin down both sides for exactly the width of the label,
    and the label's own ink was then painted over what was left.

    Asserted column by column: a nick is a *contiguous* run of pale
    columns, and a mean or a total across 420 columns would average it
    away and pass while the sheet visibly had a gap in it.
    """
    img = np.full((240, 420, 3), 255, np.uint8)
    cv2.rectangle(img, (0, 0), (419, 14), (0, 0, 0), -1)   # poché wall along the top
    img[120, :] = 0                                        # dimension line, solid core
    img[119, :] = 150                                      # ...and its antialiased flank
    img[121, :] = 150
    cv2.putText(img, "2650", (170, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)

    _result, out = _mirror(tmp_path, img, "2650")
    grey = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    assert grey[120, :].max() < 128, "the line's core is broken somewhere along its length"
    for row in (119, 121):
        under_label = grey[row, _MIRRORED_COLS]
        assert under_label.max() <= 150, (
            f"row {row}: the line's flank was dropped under the label — "
            f"lightest pixel {under_label.max()}, expected no paler than the 150 it was drawn at"
        )


def test_a_grey_label_crossing_a_wall_does_not_lighten_it(tmp_path):
    """The z-order claim itself, stated where it is unambiguous. Type
    composited OVER geometry blends its own ink into whatever it lands
    on, so a grey label crossing a black wall lifts those pixels toward
    grey. Composited UNDER, the wall keeps precisely the bytes it had —
    which is the guarantee: a re-rendered run cannot alter linework at
    all, whatever its colour.

    The label crosses the wall rather than sitting inside it, which is
    both the realistic arrangement and the one this test can read: a
    label wholly inside poché is light-on-dark, and the erase's local
    paper estimate (the patch's own light end) then picks out the
    *label* as paper and the wall as the thing to remove — a separate
    matter entirely from which layer ends up on top.
    """
    img = np.full((240, 420, 3), 255, np.uint8)
    cv2.rectangle(img, (185, 0), (205, 239), (0, 0, 0), -1)  # a wall the label crosses
    cv2.putText(img, "2650", (170, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)

    _result, out = _mirror(tmp_path, img, "2650")

    # The wall, at its mirrored position (x' = 420 - x), across the band
    # of rows the label occupies. Every pixel of it must still be ink.
    wall = cv2.cvtColor(out[110:135, 420 - 205 + 2:420 - 185 - 1], cv2.COLOR_BGR2GRAY)
    assert wall.max() == 0, f"the wall was lightened to {wall.max()} by the label drawn across it"


def test_a_label_that_lands_in_poche_is_flagged_rather_than_punched_through_it(tmp_path):
    """The one thing drawing-over-type costs, and the reason it is
    reported. A run whose mirrored anchor lands on solid geometry is now
    *under* it — the plan stays intact and the string is still recorded
    on the run, but the label cannot be read off the output sheet until
    a human moves it, so it must not pass silently.
    """
    img = np.full((240, 420, 3), 255, np.uint8)
    cv2.rectangle(img, (0, 90), (419, 150), (0, 0, 0), -1)
    cv2.putText(img, "2650", (170, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)

    result, _out = _mirror(tmp_path, img, "2650")

    codes = {f.code for r in result.runs for f in r.flags}
    assert "hidden-behind-linework" in codes
    # ...and reported once, not twice: being buried implies touching, so
    # the plain collision warning must stand down rather than doubling up.
    assert "collision" not in codes
