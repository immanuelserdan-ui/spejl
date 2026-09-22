"""qa/verify.py — the post-hoc, file-based spatial-integrity check a
human can trigger on demand against any already-saved mirror, distinct
from qa/self_correct.py's own in-memory Rule.SPATIAL_INTEGRITY.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spejl.detect.ocr import Detection
from spejl.models import Axis
from spejl.qa.verify import render_diff_overlay, verify_mirror
from spejl.transform import mirror as M


class _FixedBackend:
    """An OcrBackend stand-in that reports one fixed detection for the
    original (non-rotated, non-rescaled) frame only — matches the same
    pattern tests/test_protected_regions.py's own _FixedBackend uses,
    so detect_all_orientations' extra 90°-rotated passes see nothing
    rather than three ghost copies of the one real detection."""

    def __init__(self, detections: list[Detection], base_size: tuple[int, int]) -> None:
        self._detections = detections
        self._base_h, self._base_w = base_size
        self._base_calls = 0

    def detect_and_recognise(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        if (h, w) != (self._base_h, self._base_w):
            return []
        self._base_calls += 1
        if self._base_calls > 1:
            return [Detection(d.text, tuple((w-x,y) for x,y in d.quad),d.conf)
                    for d in self._detections]
        return self._detections


def _make_source_and_mirror(tmp_path):
    """A wall well clear of a text run, mirrored correctly (flip, then
    the text region alone replaced with a differently-shaped mark — the
    same "erase and redraw" a real run always does)."""
    source = np.full((200, 300, 3), 255, np.uint8)
    source[20:180, 140:160] = 0  # a vertical wall, far from the text below
    cv2.putText(source, "42", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
    text_box = (20.0, 65.0, 95.0, 115.0)  # comfortably covers "42"

    source_path = tmp_path / "source.png"
    cv2.imwrite(str(source_path), source)

    flipped = cv2.flip(source, 1)
    h, w = flipped.shape[:2]
    mx0, my0, mx1, my1 = (int(v) for v in M.mirror_bbox(text_box, w, h, Axis.VERTICAL))
    flipped[my0:my1, mx0:mx1] = 255  # erase the (now-mirrored) old text
    cv2.putText(flipped, "42", (mx0, my1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

    backend = _FixedBackend(
        [Detection(text="42", quad=((20.0, 65.0), (95.0, 65.0), (95.0, 115.0), (20.0, 115.0)), conf=0.99)],
        base_size=source.shape[:2],
    )
    return source_path, flipped, backend


def test_verify_mirror_passes_on_a_clean_mirror(tmp_path):
    source_path, flipped, backend = _make_source_and_mirror(tmp_path)
    output_path = tmp_path / "output.png"
    cv2.imwrite(str(output_path), flipped)

    report = verify_mirror(source_path, output_path, Axis.VERTICAL, backend=backend)
    assert report.passed
    assert report.outside_text_regions == ()
    assert report.text_regions_checked == 1
    # The redrawn text itself DOES differ from a plain flip — that's
    # expected and exactly what this test confirms gets excluded, not
    # that nothing differs at all.
    assert report.differing_px > 0


def test_verify_mirror_catches_missing_geometry_outside_text(tmp_path):
    source_path, flipped, backend = _make_source_and_mirror(tmp_path)
    corrupted = flipped.copy()
    # The wall (source x:140-160 of a 300-wide canvas) sits exactly on
    # the mirror axis's own centre line, so it lands back at the SAME
    # x:140-160 after flipping — punching the hole there, well above
    # the text (y:65-115), corrupts real geometry nowhere near it.
    corrupted[20:60, 140:160] = 255
    output_path = tmp_path / "corrupted.png"
    cv2.imwrite(str(output_path), corrupted)

    report = verify_mirror(source_path, output_path, Axis.VERTICAL, backend=backend)
    assert not report.passed
    assert len(report.outside_text_regions) >= 1
    assert report.outside_text_px >= 700  # the 20x40 hole, give or take antialiasing


def test_verify_mirror_raises_on_a_size_mismatch(tmp_path):
    source_path, flipped, backend = _make_source_and_mirror(tmp_path)
    wrong_size = cv2.resize(flipped, (flipped.shape[1] // 2, flipped.shape[0] // 2))
    output_path = tmp_path / "wrong_size.png"
    cv2.imwrite(str(output_path), wrong_size)

    with pytest.raises(ValueError, match="size mismatch"):
        verify_mirror(source_path, output_path, Axis.VERTICAL, backend=backend)


def test_verify_mirror_raises_on_an_unreadable_file(tmp_path):
    source_path, flipped, backend = _make_source_and_mirror(tmp_path)
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(ValueError, match="Could not read"):
        verify_mirror(source_path, missing, Axis.VERTICAL, backend=backend)


def test_render_diff_overlay_writes_a_visible_image(tmp_path):
    source_path, flipped, _backend = _make_source_and_mirror(tmp_path)
    output_path = tmp_path / "output.png"
    cv2.imwrite(str(output_path), flipped)

    dest = tmp_path / "diff.png"
    render_diff_overlay(source_path, output_path, Axis.VERTICAL, dest)

    assert dest.exists()
    overlay = cv2.imread(str(dest))
    assert overlay is not None
    assert overlay.shape[:2] == flipped.shape[:2]
    # Some pixels marked red (BGR: high B/R channel... actually pure
    # red is (0, 0, 255) in BGR) where the redrawn text differs.
    red_mask = (overlay[:, :, 2] == 255) & (overlay[:, :, 0] == 0) & (overlay[:, :, 1] == 0)
    assert red_mask.any(), "diff overlay marked nothing red despite a known text difference"
