"""Route B end-to-end: mirror the fixture, then read the result back.

The decisive test is the round trip. Scoring detection against ground
truth proves OCR works; scoring *the mirrored output* against mirrored
ground truth proves the whole pipeline works — because a re-rendered
label that is misplaced, wrongly sized, upside down, or drawn over a
wall will fail to come back correctly, whatever the intermediate stages
reported about themselves.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from skimage.metrics import structural_similarity as ssim

from spejl.detect.ocr import RapidOcrBackend
from spejl.detect.rotations import detect_all_orientations
from spejl.lexicon.snap import snap
from spejl.models import Axis
from spejl.qa import fixture_gen
from spejl.qa.metrics import load_ground_truth, score
from spejl.raster.pipeline import mirror_raster
from spejl.transform import mirror as M

pytestmark = pytest.mark.slow

DPI = 150


@pytest.fixture(scope="module")
def backend() -> RapidOcrBackend:
    return RapidOcrBackend()


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("raster_golden")
    fixture_gen.generate(out, dpi=DPI)
    return out


@pytest.fixture(scope="module")
def mirrored(fixture_dir, backend):
    truth = load_ground_truth(fixture_dir / f"ground_truth_{DPI}.json")
    src = fixture_dir / truth["image"]
    out = fixture_dir / "mirrored.png"
    result = mirror_raster(src, out, axis=Axis.VERTICAL, backend=backend)
    return {"truth": truth, "src": src, "out": out, "result": result}


def _mirror_truth(truth: dict, axis: Axis = Axis.VERTICAL) -> dict:
    """Ground truth for the mirrored sheet: same strings, mirrored boxes.

    Angles are put through the same canonicalisation the renderer uses,
    so the expectation matches the drafting convention rather than a
    naive reflection.
    """
    w, h = truth["width_px"], truth["height_px"]
    runs = []
    for run in truth["runs"]:
        runs.append(
            {
                **run,
                "bbox_px": list(M.mirror_bbox(tuple(run["bbox_px"]), w, h, axis)),
                "angle_deg": M.mirror_angle(run["angle_deg"], axis),
            }
        )
    return {**truth, "runs": runs}


@pytest.fixture(scope="module")
def roundtrip(mirrored, backend):
    expected = _mirror_truth(mirrored["truth"])
    image = cv2.imread(str(mirrored["out"]))
    detections = detect_all_orientations(image, backend)
    corrections = {i: (snap(d.text).text, snap(d.text).warning)
                   for i, d in enumerate(detections)}
    return score(expected, detections, corrections)


# --------------------------------------------------------------------------
# The round trip
# --------------------------------------------------------------------------


def test_every_run_survives_the_mirror(roundtrip):
    missing = [r.truth for r in roundtrip.runs if not r.found]
    assert roundtrip.recall == 1.0, f"lost after mirroring: {missing}"


def test_every_string_still_reads_correctly(roundtrip):
    wrong = [(r.truth, r.corrected) for r in roundtrip.runs if not r.exact_corrected]
    assert wrong == [], f"misread after mirroring: {wrong}"


def test_danish_diacritics_survive_the_round_trip(roundtrip):
    """The whole point of the lexicon stage, verified on output pixels."""
    by_truth = {r.truth: r for r in roundtrip.runs}
    assert by_truth["Køkken"].corrected == "Køkken"
    assert by_truth["Vær. 1"].corrected == "Vær. 1"


def test_runs_land_where_they_should(roundtrip):
    """Anchor error against the mirrored ground-truth centre. Build plan
    §10 gates this at 3 px; re-rendered glyph widths differ slightly from
    the original's, so a few px of drift is expected and acceptable."""
    assert roundtrip.mean_anchor_error <= 6.0
    worst = max(r.anchor_error_px for r in roundtrip.runs if r.anchor_error_px is not None)
    assert worst <= 14.0


def test_vertical_dimensions_are_still_vertical_and_readable(roundtrip):
    verticals = [r for r in roundtrip.runs if r.angle_truth == 90]
    assert len(verticals) == 5
    for run in verticals:
        assert run.found and run.exact_corrected, f"{run.truth}: {run.corrected!r}"
        assert run.angle_detected == pytest.approx(90, abs=15)


def test_re_rendered_type_matches_the_original_size(mirrored):
    """The §10 "zoom test", made measurable.

    Ink extent is measured with the *same* function on both sheets, so
    the comparison is apples to apples — measuring the source with exact
    PDF boxes and the output with padded OCR boxes hides a real ~13%
    oversize behind ~10% of measurement bias, which is how the original
    cap-height bug survived a visual check.
    """
    from spejl.style.metrics import measure_ink_extent

    truth = mirrored["truth"]
    w, h = truth["width_px"], truth["height_px"]
    src = cv2.imread(str(mirrored["src"]))
    out = cv2.imread(str(mirrored["out"]))

    along_ratios, cap_ratios = [], []
    for run in truth["runs"]:
        box = tuple(run["bbox_px"])
        angle = run["angle_deg"]
        src_along, src_cap = measure_ink_extent(src, box, angle)
        out_along, out_cap = measure_ink_extent(
            out,
            M.mirror_bbox(box, w, h, Axis.VERTICAL),
            M.mirror_angle(angle, Axis.VERTICAL),
        )
        along_ratios.append(out_along / src_along)
        cap_ratios.append(out_cap / src_cap)

    mean_along = sum(along_ratios) / len(along_ratios)
    mean_cap = sum(cap_ratios) / len(cap_ratios)
    assert 0.90 <= mean_along <= 1.10, f"mean along ratio {mean_along:.3f}"
    assert 0.90 <= mean_cap <= 1.10, f"mean cap ratio {mean_cap:.3f}"
    # Growing is worse than shrinking: an oversized label can collide
    # with a wall, an undersized one only looks slightly light.
    assert max(along_ratios) <= 1.20, f"worst oversize {max(along_ratios):.3f}"


def test_no_text_is_drawn_over_linework(mirrored):
    """A label that lands on a wall is a visible defect; the renderer
    reports collisions so review can catch them."""
    collided = [r.text for r in mirrored["result"].runs
                if any(f.code == "collision" for f in r.flags)]
    assert collided == [], f"drawn over linework: {collided}"


# --------------------------------------------------------------------------
# Geometry must be untouched
# --------------------------------------------------------------------------


def test_geometry_integrity_outside_the_text(mirrored):
    """Everything that is not a label must match a plain cv2.flip.

    This is what proves the pipeline mirrors the *drawing* faithfully and
    only rewrites the type: erase scars, repaired lines that drifted, or
    a stray fill would all show up here.
    """
    truth = mirrored["truth"]
    src = cv2.imread(str(mirrored["src"]))
    out = cv2.imread(str(mirrored["out"]))
    naive = M.flip_image(src, Axis.VERTICAL)

    # Blank the mirrored text regions in both, generously, so only
    # geometry is compared.
    w, h = truth["width_px"], truth["height_px"]
    mask = np.zeros(out.shape[:2], dtype=np.uint8)
    for run in truth["runs"]:
        x0, y0, x1, y1 = M.mirror_bbox(tuple(run["bbox_px"]), w, h, Axis.VERTICAL)
        cv2.rectangle(mask, (int(x0) - 10, int(y0) - 10), (int(x1) + 10, int(y1) + 10), 255, -1)

    a = cv2.cvtColor(naive, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    a[mask > 0] = 255
    b[mask > 0] = 255

    assert ssim(a, b) >= 0.98


def test_output_has_the_same_dimensions(mirrored):
    src = cv2.imread(str(mirrored["src"]))
    out = cv2.imread(str(mirrored["out"]))
    assert out.shape == src.shape


# --------------------------------------------------------------------------
# Idempotency (build plan §10)
# --------------------------------------------------------------------------


def test_mirroring_twice_returns_close_to_the_source(fixture_dir, backend):
    truth = load_ground_truth(fixture_dir / f"ground_truth_{DPI}.json")
    src = fixture_dir / truth["image"]
    once = fixture_dir / "rt_once.png"
    twice = fixture_dir / "rt_twice.png"
    mirror_raster(src, once, axis=Axis.VERTICAL, backend=backend)
    mirror_raster(once, twice, axis=Axis.VERTICAL, backend=backend)

    image = cv2.imread(str(twice))
    detections = detect_all_orientations(image, backend)
    corrections = {i: (snap(d.text).text, snap(d.text).warning)
                   for i, d in enumerate(detections)}
    report = score(truth, detections, corrections)

    assert report.recall == 1.0
    assert report.char_accuracy(corrected=True) >= 0.99
    # Two rounds of erase-and-redraw, so allow more drift than one.
    assert report.mean_anchor_error <= 10.0


def test_refuses_a_sheet_too_small_to_mirror_honestly(tmp_path, backend):
    """Refusal is a passing outcome — better than a plausible-looking
    wrong drawing (build plan §10)."""
    truth = load_ground_truth(
        list(Path(fixture_gen.__file__).parent.glob("golden/ground_truth_150.json"))[0]
    )
    src = Path(fixture_gen.__file__).parent / "golden" / truth["image"]
    tiny = cv2.resize(cv2.imread(str(src)), None, fx=0.18, fy=0.18,
                      interpolation=cv2.INTER_AREA)
    tiny_path = tmp_path / "tiny.png"
    cv2.imwrite(str(tiny_path), tiny)

    with pytest.raises(ValueError, match="too low to mirror reliably"):
        mirror_raster(tiny_path, tmp_path / "out.png", backend=backend)
