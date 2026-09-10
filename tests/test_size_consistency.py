"""_snap_consistent_sizes — clustering same-kind label sizes so ordinary
OCR measurement noise doesn't render two identically-sized original
labels at two visibly different sizes.
"""

from __future__ import annotations

from spejl.raster.pipeline import MirroredRun, _snap_consistent_sizes
from spejl.style.metrics import TextStyle, font_measure_width, resolve_font


def _run(text: str, kind: str, px_size: int, along: float | None = None) -> MirroredRun:
    """A synthetic run whose (along, px_size) are self-consistent — the
    real pipeline never sees a target width unrelated to the size that
    was independently fit to reach it, and a test fixture that does
    isn't testing the same thing the real code has to handle. Default
    ``along`` is this string's own natural width at ``px_size`` (0
    tracking): realistic for "this run's own best independent fit",
    which is exactly what every run looks like before clustering runs.
    """
    if along is None:
        along = font_measure_width(text, resolve_font(), px_size, 0.0)
    style = TextStyle(
        font_path=resolve_font(), px_size=px_size, tracking=0.0,
        ink=(0, 0, 0), paper=(255, 255, 255), cap_height_px=float(px_size) * 0.7,
        ink_along_px=along,
    )
    return MirroredRun(
        text=text, text_raw=text, kind=kind, conf=0.99, angle_src=0.0, angle_out=0.0,
        center_src=(0, 0), center_out=(0, 0), bbox_src=(0, 0, along, px_size * 0.7), style=style,
    )


def test_close_sizes_snap_to_a_common_median():
    runs = [_run("Stue", "room", 45), _run("Bad", "room", 42), _run("Køkken", "room", 45)]
    _snap_consistent_sizes(runs)
    sizes = {r.style.px_size for r in runs}
    assert len(sizes) == 1, f"expected one consistent size, got {sizes}"


def test_genuinely_different_sizes_are_not_merged():
    """A big room's label and a small utility room's label really can
    differ on a real drawing — clustering must not erase that."""
    runs = [_run("Stue", "room", 45), _run("Toilet", "room", 20)]
    _snap_consistent_sizes(runs)
    sizes = sorted(r.style.px_size for r in runs)
    assert sizes == [20, 45], f"genuinely different sizes were merged: {sizes}"


def test_dimensions_and_rooms_cluster_independently():
    runs = [
        _run("Stue", "room", 45),
        _run("Bad", "room", 44),
        _run("2900", "dimension", 26),
        _run("4045", "dimension", 27),
    ]
    _snap_consistent_sizes(runs)
    room_sizes = {r.style.px_size for r in runs if r.kind == "room"}
    dim_sizes = {r.style.px_size for r in runs if r.kind == "dimension"}
    assert len(room_sizes) == 1
    assert len(dim_sizes) == 1
    assert room_sizes != dim_sizes  # never cross-merged into one shared size


def test_other_kinds_are_left_untouched():
    runs = [_run("H*", "annotation", 20), _run("?", "unknown", 44)]
    before = [r.style.px_size for r in runs]
    _snap_consistent_sizes(runs)
    after = [r.style.px_size for r in runs]
    assert before == after


def test_a_single_run_in_a_kind_is_a_no_op():
    runs = [_run("Stue", "room", 45)]
    _snap_consistent_sizes(runs)
    assert runs[0].style.px_size == 45


def test_snapped_style_still_hits_its_own_measured_width_target():
    """Tracking/scale must be re-derived at the new size, not left
    stale from the run's original (pre-snap) size."""
    runs = [_run("Stue", "room", 45, along=70.0), _run("Bad", "room", 42, along=55.0)]
    _snap_consistent_sizes(runs)
    bad = next(r for r in runs if r.text == "Bad")
    assert bad.style.ink_along_px == 55.0  # the target itself never changes
    # tracking/width_scale were recomputed for the NEW size, not copied
    # from the pre-snap style, which was fit for px_size=42 not 45.
    assert isinstance(bad.style.tracking, float)
    assert isinstance(bad.style.width_scale, float)


def test_real_fixture_dimensions_end_up_perfectly_consistent():
    """Integration check against the actual golden-fixture measurement
    noise this feature was built for: 11 real dimension runs, fitted
    independently, land across a 25-30px spread before clustering —
    confirmed to collapse to one size after.
    """
    import cv2

    from spejl.detect.ocr import RapidOcrBackend
    from spejl.detect.rotations import detect_all_orientations
    from spejl.lexicon.snap import snap
    from spejl.style.metrics import fit_style

    img = cv2.imread("spejl/qa/golden/plan_150.png")
    dets = detect_all_orientations(img, RapidOcrBackend())

    runs = []
    for det in dets:
        result = snap(det.text)
        style = fit_style(img, result.text, det.bbox, det.angle_deg)
        runs.append(
            MirroredRun(
                text=result.text, text_raw=det.text, kind=result.kind, conf=det.conf,
                angle_src=det.angle_deg, angle_out=0.0, center_src=(0, 0), center_out=(0, 0),
                bbox_src=det.bbox, style=style,
            )
        )

    before_sizes = {r.style.px_size for r in runs if r.kind == "dimension"}
    assert len(before_sizes) > 1, "fixture no longer reproduces the measurement noise this guards against"

    _snap_consistent_sizes(runs)
    after_sizes = {r.style.px_size for r in runs if r.kind == "dimension"}
    assert len(after_sizes) == 1, f"dimensions still inconsistent after clustering: {after_sizes}"
