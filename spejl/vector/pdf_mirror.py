"""Mirror PDFs with a vector route and a complete-page raster fallback.

Pages containing images (including scans and mixed content) are rendered at
300 dpi and processed by the text-aware raster pipeline. They retain visible
content but lose vector editability, which is reported in the result flags.
Vector-only pages reconstruct paths and text. Text baselines are reflected as
directions, then reversed when necessary to preserve normal reading order. An
arbitrary-angle PDF text matrix retains slanted dimensions instead of rounding
them to quarter turns.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pymupdf
import pikepdf

from spejl.models import Axis, Document, Flag, PageResult, Route
from spejl.transform import mirror as M
from spejl.vector.text_page import TextPage, compose
from spejl.vector.pdf_save import save_compact_pdf

if TYPE_CHECKING:
    import numpy as np

# Point/direction mirroring and the readability convention live in
# transform/mirror.py, shared with Route B — two copies of that rule
# would be free to drift, and it is the rule the whole tool turns on.
_mirror_point = M.mirror_point

# Font-family fallback: CAD-exported PDFs overwhelmingly set a
# Helvetica/Arial-alike, so that is the default; a document that
# clearly asks for a serif or monospace face gets one. Exact family
# matching (the build plan's style/family.py, NCC template matching)
# is Route B's job — Route A only ever loses a glyph's *exact* outline
# for a font PyMuPDF doesn't ship, never its position or orientation.
_HELV = {False: {False: "helv", True: "heit"}, True: {False: "hebo", True: "hebi"}}
_TIMES = {False: {False: "tiro", True: "tiit"}, True: {False: "tibo", True: "tibi"}}
_COURIER = {False: {False: "cour", True: "coit"}, True: {False: "cobo", True: "cobi"}}


def mirror_pdf(
    input_path: Path,
    output_path: Path,
    axis: Axis = Axis.VERTICAL,
    *,
    pictures_as_linework: bool = False,
) -> Document:
    """Mirror every page of ``input_path`` and write ``output_path``.

    ``pictures_as_linework`` keeps image-bearing pages on the vector route:
    their pictures are reflected with the page like any other drawing.
    Only pass it once every piece of lettering inside those pictures has
    been taken out (see :mod:`spejl.vector.mixed_pdf`), or that lettering
    comes out backwards.

    Returns the :class:`Document` record — the same object that
    ``to_sidecar()`` turns into the ``*.spejl.json`` written beside the
    output.
    """
    result = Document(source=input_path, output=output_path, axis=axis, route=Route.VECTOR)

    src = pymupdf.open(str(input_path))
    pike_src = pikepdf.Pdf.open(str(input_path))
    try:
        out = pymupdf.open()
    except Exception:
        src.close()  # opening `out` failing must not leak the already-open `src`
        raise

    try:
        for page_index in range(src.page_count):
            page = src[page_index]
            W, H = page.rect.width, page.rect.height
            # Raster and mixed pages cannot be reconstructed from paths and
            # spans: that drops images (including inline images). Render the
            # complete page and use the text-aware raster route instead.
            if page.get_image_info() and not pictures_as_linework:
                from spejl.raster.pipeline import mirror_raster
                with TemporaryDirectory(prefix="spejl-pdf-") as temporary:
                    source_png = Path(temporary) / "source.png"
                    output_png = Path(temporary) / "mirror.png"
                    page.get_pixmap(dpi=300, alpha=False).save(str(source_png))
                    raster = mirror_raster(source_png, output_png, axis=axis)
                    new_page = out.new_page(width=W, height=H)
                    new_page.insert_image(new_page.rect, filename=str(output_png))
                result.pages.append(PageResult(
                    index=page_index, width=W, height=H,
                    text_runs_mirrored=len(raster.runs),
                    flags=[Flag("pdf-rasterized", "Page contains images; the complete page "
                                "was mirrored through the raster pipeline at 300 dpi. "
                                "Vector editability is not retained on this page.", "warn")]
                          + raster.document.pages[0].flags,
                ))
                continue
            with pikepdf.Pdf.new() as working:
                layout = TextPage(working, pike_src.pages[page_index])
                mirrored, expected = _mirror_text_page(layout, page, W, H, axis)
                with mirrored:
                    out.insert_pdf(mirrored)

            result.pages.append(
                PageResult(
                    index=page_index,
                    width=W,
                    height=H,
                    text_runs_mirrored=len(expected),
                    flags=[],
                )
            )

        save_compact_pdf(out, output_path)
        # Validate the serialized file, not only the in-memory page used by
        # the matrix QA above.  Some PDF writers defer resource resolution
        # until save; reopening and rendering catches a corrupt stream before
        # the GUI reports a successful job with a blank preview.
        with pymupdf.open(str(output_path)) as written:
            if written.page_count != src.page_count:
                raise VectorTextTransformError(
                    "Saved mirrored PDF has a different page count."
                )
            blank_pages = []
            for page_index, written_page in enumerate(written):
                pix = written_page.get_pixmap(alpha=False)
                if not pix.samples:
                    raise VectorTextTransformError(
                        "Saved mirrored PDF page rendered no image data."
                    )
                if not _pixmap_has_ink(pix):
                    blank_pages.append(page_index)
        if blank_pages:
            # A few CAD exporters produce a page whose resource graph cannot
            # survive stream replacement even though text extraction works.
            # Do not return a blank deliverable: render the original page and
            # mirror that complete image as a safe, explicitly flagged PDF.
            fallback_pages = _write_raster_pdf_fallback(
                src, output_path, axis, page_indexes=set(blank_pages),
            )
            replacements = {page.index: page for page in fallback_pages}
            result.pages = [replacements.get(page.index, page) for page in result.pages]
    finally:
        out.close()
        src.close()
        pike_src.close()

    return result


def remove_pdf_text_runs(
    input_path: Path, output_path: Path, removals: dict[int, set[int]]
) -> None:
    """Remove selected vector text objects while preserving all other PDF data.

    ``removals`` maps zero-based page numbers to zero-based visible text-run
    indices in PyMuPDF reading order.  Only complete ``BT``/``ET`` objects
    containing a show operator are removed; fonts, paths, images and every
    other content operator remain untouched.
    """
    pdf = pikepdf.Pdf.open(str(input_path))
    try:
        for page_index, indices in removals.items():
            if not indices:
                continue
            page = pdf.pages[page_index]
            instructions = list(pikepdf.parse_content_stream(page))
            rewritten: list[pikepdf.ContentStreamInstruction] = []
            text_index = 0
            block: list[pikepdf.ContentStreamInstruction] = []
            in_text = False

            def flush() -> None:
                nonlocal text_index, block
                has_show = any(str(op) in {"Tj", "TJ", "'", '"'} for _, op in block)
                if has_show:
                    if text_index not in indices:
                        rewritten.extend(block)
                    text_index += 1
                else:
                    rewritten.extend(block)
                block = []

            for instruction in instructions:
                operands, operator = instruction
                name = str(operator)
                if name == "BT":
                    if in_text:
                        flush()
                    in_text = True
                    block = [instruction]
                elif in_text:
                    block.append(instruction)
                    if name == "ET":
                        flush()
                        in_text = False
                else:
                    rewritten.append(instruction)
            if block:
                flush()
            page.Contents = pikepdf.Stream(pdf, pikepdf.unparse_content_stream(rewritten))
        pdf.save(str(output_path))
    finally:
        pdf.close()


def translate_pdf_text_runs(
    input_path: Path,
    output_path: Path,
    translations: dict[tuple[int, int], tuple[float, float]],
    axis: Axis = Axis.VERTICAL,
) -> None:
    """Move selected visible PDF text runs by screen-space points.

    The translation is applied only to each selected ``Tm`` origin.  The
    mirrored PDF has an outer reflection CTM, so the requested screen-space
    nudge is converted through that CTM before changing ``Tm``. All font,
    kerning and text-show operators remain exactly as stored.
    """
    pdf = pikepdf.Pdf.open(str(input_path))
    try:
        for page_index, page in enumerate(pdf.pages):
            requested = {
                run_index: delta
                for (wanted_page, run_index), delta in translations.items()
                if wanted_page == page_index
            }
            if not requested:
                continue
            rewritten: list[pikepdf.ContentStreamInstruction] = []
            in_text = False
            visible_run = 0
            pending_index: int | None = None
            block_has_show = False
            for operands, operator in pikepdf.parse_content_stream(page):
                name = str(operator)
                if name == "BT":
                    in_text, pending_index, block_has_show = True, None, False
                    rewritten.append(pikepdf.ContentStreamInstruction(operands, operator))
                    continue
                if in_text and name == "Tm" and len(operands) == 6:
                    pending_index = visible_run
                    if pending_index in requested:
                        right, down = requested[pending_index]
                        matrix = [float(value) for value in operands]
                        matrix[4] += -right if axis in {Axis.VERTICAL, Axis.BOTH} else right
                        matrix[5] += down if axis in {Axis.HORIZONTAL, Axis.BOTH} else -down
                        operands = matrix
                if in_text and name in {"Tj", "TJ", "'", '"'} and pending_index is not None:
                    block_has_show = True
                if name == "ET":
                    if block_has_show:
                        visible_run += 1
                    in_text = False
                rewritten.append(pikepdf.ContentStreamInstruction(operands, operator))
            page.Contents = pikepdf.Stream(pdf, pikepdf.unparse_content_stream(rewritten))
        pdf.save(str(output_path))
    finally:
        pdf.close()


def selected_text_overlaps_drawing(
    pdf_path: Path, selected_runs: dict[int, set[int]]
) -> bool:
    """Return whether selected text intersects a vector or image drawing.

    This uses the actual post-edit PDF geometry, rather than a rendered OCR
    approximation. Stroked path segments are expanded by their line width;
    filled rectangles and image blocks are treated as occupied areas.
    """
    document = pymupdf.open(str(pdf_path))
    try:
        for page_index, indexes in selected_runs.items():
            if not indexes or page_index >= document.page_count:
                continue
            page = document[page_index]
            selected_boxes: list[pymupdf.Rect] = []
            run_index = 0
            for trace in page.get_texttrace():
                if trace.get("type") != 0 or not trace.get("chars"):
                    continue
                if run_index in indexes:
                    selected_boxes.append(pymupdf.Rect(trace["bbox"]))
                run_index += 1
            if not selected_boxes:
                continue
            occupied = _drawing_element_rectangles(page)
            pictures = _picture_ink(page)
            if any(text_box.intersects(element) for text_box in selected_boxes for element in occupied):
                return True
            if any(_box_on_picture_ink(text_box, pictures) for text_box in selected_boxes):
                return True
        return False
    finally:
        document.close()


def text_drawing_overlap_findings(pdf_path: Path) -> list[tuple[int, str]]:
    """List visible PDF text runs whose glyph bounds intersect drawing geometry.

    Each result is ``(one_based_page_number, text)``. Findings are deduplicated
    per run even when a label crosses several wall or dimension segments.
    """
    return [
        (page_index + 1, text)
        for page_index, _run_index, text in text_drawing_overlap_runs(pdf_path)
    ]


def text_drawing_overlap_runs(pdf_path: Path) -> list[tuple[int, int, str]]:
    """Return zero-based page and text-run indexes for every overlap.

    The indexes let the GUI apply a correction to exactly the runs shown in
    its overlap panel, including repeated labels and findings on other pages.
    """
    document = pymupdf.open(str(pdf_path))
    findings: list[tuple[int, int, str]] = []
    try:
        for page_index, page in enumerate(document):
            occupied = _drawing_element_rectangles(page)
            pictures = _picture_ink(page)
            if not occupied and not pictures:
                continue
            for run_index, trace in enumerate(_trace_runs(page)):
                box = pymupdf.Rect(trace["bbox"])
                if not any(box.intersects(element) for element in occupied) \
                        and not _box_on_picture_ink(box, pictures):
                    continue
                label = "".join(chr(char[0]) for char in trace["chars"]).strip()
                if label:
                    findings.append((page_index, run_index, label))
        return findings
    finally:
        document.close()


def _picture_ink(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, pymupdf.Matrix, np.ndarray]]:
    """Each picture on the page with a dark-pixel mask of its drawing.

    A raster export's picture spans the whole sheet; treating its rectangle
    as occupied would flag every label on the page. Only where the picture
    actually has linework does text overlap the drawing. The placement
    matrix is kept because a mirrored page draws the same pixels flipped.
    """
    import numpy as np

    pictures: list[tuple[pymupdf.Rect, pymupdf.Matrix, np.ndarray]] = []
    for block in page.get_text("rawdict").get("blocks", []):
        if block.get("type") != 1 or "bbox" not in block:
            continue
        rect = pymupdf.Rect(block["bbox"])
        transform = pymupdf.Matrix(block.get("transform") or (rect.width, 0, 0, rect.height, rect.x0, rect.y0))
        try:
            pix = pymupdf.Pixmap(block["image"])
            if pix.alpha:
                pix = pymupdf.Pixmap(pix, 0)
            if pix.colorspace is None or pix.colorspace.n != 1:
                pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
            gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
            ink = gray < 128
        except Exception:  # noqa: BLE001 — an undecodable picture counts as fully occupied
            ink = np.ones((1, 1), dtype=bool)
        pictures.append((rect, transform, ink))
    return pictures


def _box_on_picture_ink(
    box: pymupdf.Rect, pictures: list[tuple[pymupdf.Rect, pymupdf.Matrix, np.ndarray]]
) -> bool:
    """Whether ``box`` covers any dark pixel of any picture under it."""
    for rect, transform, ink in pictures:
        hit = box & rect
        if hit.is_empty:
            continue
        inverse = ~transform
        if inverse == pymupdf.Matrix(0, 0, 0, 0, 0, 0):
            continue
        h, w = ink.shape
        # Page point -> unit square of the picture -> pixel; image row 0 is at v = 0.
        corners = [pymupdf.Point(x, y) * inverse for x in (hit.x0, hit.x1) for y in (hit.y0, hit.y1)]
        us, vs = [c.x for c in corners], [c.y for c in corners]
        x0 = max(int(min(us) * w), 0)
        y0 = max(int(min(vs) * h), 0)
        x1 = min(max(math.ceil(max(us) * w), x0 + 1), w)
        y1 = min(max(math.ceil(max(vs) * h), y0 + 1), h)
        if x0 < w and y0 < h and ink[y0:y1, x0:x1].any():
            return True
    return False


def _drawing_element_rectangles(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Return tight occupied rectangles for individual PDF path elements."""
    occupied: list[pymupdf.Rect] = []
    for drawing in page.get_drawings():
        pad = max(float(drawing.get("width") or 0.0) / 2.0, 0.5)

        def add_segment(first: pymupdf.Point, second: pymupdf.Point) -> None:
            rect = pymupdf.Rect(first, second)
            rect.normalize()
            rect.x0 -= pad
            rect.y0 -= pad
            rect.x1 += pad
            rect.y1 += pad
            occupied.append(rect)

        for item in drawing.get("items", []):
            operator = item[0]
            if operator == "l":
                add_segment(item[1], item[2])
            elif operator == "re":
                rect = pymupdf.Rect(item[1])
                if drawing.get("fill") is not None:
                    occupied.append(rect)
                else:
                    add_segment(rect.top_left, rect.top_right)
                    add_segment(rect.top_right, rect.bottom_right)
                    add_segment(rect.bottom_right, rect.bottom_left)
                    add_segment(rect.bottom_left, rect.top_left)
            elif operator == "c":
                # A Bézier can curve through this control-point envelope;
                # it is conservative but avoids missing a real collision.
                points = item[1:5]
                xs, ys = [point.x for point in points], [point.y for point in points]
                occupied.append(pymupdf.Rect(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad))
            elif operator == "qu":
                quad = item[1]
                points = (quad.ul, quad.ur, quad.ll, quad.lr)
                xs, ys = [point.x for point in points], [point.y for point in points]
                occupied.append(pymupdf.Rect(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad))
    return occupied


def _pixmap_has_ink(pixmap: pymupdf.Pixmap) -> bool:
    """Return whether a rendered page contains visible non-white content."""
    samples = pixmap.samples
    channels = pixmap.n
    # Sample every fourth pixel for speed; floor plans contain large black
    # walls, so this remains decisively different from an all-white page.
    step = max(channels * 4, channels)
    return any(min(samples[i : i + channels]) < 245 for i in range(0, len(samples), step))


def _write_raster_pdf_fallback(
    source: pymupdf.Document, output_path: Path, axis: Axis,
    *, page_indexes: set[int] | None = None,
) -> list[PageResult]:
    """Replace only failed pages, preserving successful vector pages verbatim."""
    from spejl.raster.pipeline import mirror_raster

    pages: list[PageResult] = []
    with TemporaryDirectory(prefix="spejl-pdf-fallback-") as temporary, pymupdf.open() as raster_pdf:
        # Read the serialized candidate before replacing it. Closing this
        # handle before save also releases the Windows file lock.
        if page_indexes is not None:
            with pymupdf.open(str(output_path)) as candidate:
                raster_pdf.insert_pdf(candidate)
        for index, page in enumerate(source):
            if page_indexes is not None and index not in page_indexes:
                continue
            source_png = Path(temporary) / f"source-{index}.png"
            output_png = Path(temporary) / f"mirror-{index}.png"
            page.get_pixmap(dpi=300, alpha=False).save(str(source_png))
            raster = mirror_raster(source_png, output_png, axis=axis)
            if page_indexes is not None:
                raster_pdf.delete_page(index)
                target = raster_pdf.new_page(pno=index, width=page.rect.width, height=page.rect.height)
            else:
                target = raster_pdf.new_page(width=page.rect.width, height=page.rect.height)
            target.insert_image(target.rect, filename=str(output_png))
            pages.append(PageResult(
                index=index,
                width=page.rect.width,
                height=page.rect.height,
                text_runs_mirrored=len(raster.runs),
                flags=[Flag(
                    "pdf-rasterized-fallback",
                    "Vector PDF serialization rendered blank; the complete page was mirrored as a raster image.",
                    "warn",
                )] + raster.document.pages[0].flags,
            ))
        save_compact_pdf(raster_pdf, output_path)
    return pages


# --------------------------------------------------------------------------
# Lossless vector content transformation
# --------------------------------------------------------------------------


_TEXT_TRANSFORM_TOLERANCE_PT = 0.75


@dataclass(frozen=True)
class _TextExpectation:
    text: str
    start: tuple[float, float]
    end: tuple[float, float]
    direction: tuple[float, float]


class VectorTextTransformError(ValueError):
    """Raised when post-write text-matrix QA finds a non-mirrored run."""


def _reflection_pdf_matrix(width: float, height: float, axis: Axis) -> tuple[float, ...]:
    """PDF-space (y-up) reflection matrix around the page centre line."""
    if axis is Axis.VERTICAL:
        return (-1.0, 0.0, 0.0, 1.0, width, 0.0)
    if axis is Axis.HORIZONTAL:
        return (1.0, 0.0, 0.0, -1.0, 0.0, height)
    return (-1.0, 0.0, 0.0, -1.0, width, height)


def _pdf_reflection_matrix(
    pike_page: pikepdf.Page, axis: Axis
) -> tuple[float, float, float, float, float, float]:
    """Reflect in the actual PDF page box, including centered/negative boxes."""
    box = [float(value) for value in pike_page.MediaBox]
    min_x, min_y, max_x, max_y = box[0], box[1], box[2], box[3]
    if axis is Axis.VERTICAL:
        return (-1.0, 0.0, 0.0, 1.0, min_x + max_x, 0.0)
    if axis is Axis.HORIZONTAL:
        return (1.0, 0.0, 0.0, -1.0, 0.0, min_y + max_y)
    return (-1.0, 0.0, 0.0, -1.0, min_x + max_x, min_y + max_y)


def _trace_runs(page: pymupdf.Page) -> list[dict]:
    """Return physical glyph runs in source-content order.

    ``get_texttrace`` exposes the PDF interpreter's exact glyph origins,
    including PDF kerning and character spacing.  We use it only to derive
    the local translation needed when a readable mirror reverses a baseline;
    the original ``Tj`` / ``TJ`` operators remain byte-for-byte untouched.
    """
    return [
        run for run in page.get_texttrace()
        if run.get("type") == 0 and run.get("chars")
    ]


def _trace_end(run: dict) -> tuple[float, float]:
    """Farthest baseline endpoint of one traced glyph run, screen space.

    A traced glyph bbox is the axis-aligned hull of an ``advance x size``
    box laid along ``dir``.  Projecting that hull's corners onto a slanted
    baseline overshoots the advance, so solve the hull for the advance
    instead (exact for cardinal runs too, where it reduces to the width).
    """
    start = run["chars"][0][2]
    dx, dy = run["dir"]
    last_origin = run["chars"][-1][2]
    last_bbox = run["chars"][-1][3]
    cos, sin = abs(dx), abs(dy)
    size = float(run["size"])
    if cos >= sin:
        glyph_advance = ((last_bbox[2] - last_bbox[0]) - size * sin) / cos
    else:
        glyph_advance = ((last_bbox[3] - last_bbox[1]) - size * cos) / sin
    extent = (
        (last_origin[0] - start[0]) * dx + (last_origin[1] - start[1]) * dy
        + max(glyph_advance, 0.0)
    )
    return (start[0] + dx * extent, start[1] + dy * extent)


def _run_text(run: dict) -> str:
    return "".join(chr(char[0]) for char in run["chars"])


def _glyph_box_flip_shift(run: dict) -> float:
    """Glyph-box span that a local y flip moves a run by, in trace size units.

    Glyphs occupy ``size * [descender, ascender]`` across the baseline.  A
    local y flip mirrors that span through the baseline, so the box moves by
    ``size * (ascender + descender)``; shifting the baseline back by the same
    amount puts the box on the reflection of its source footprint.  The
    trace ``size`` is measured along the baseline, so callers rescale by the
    text matrix for condensed (non-uniform) fonts.
    """
    return float(run["size"]) * (float(run["ascender"]) + float(run["descender"]))


def _mirror_text_page(layout, source_page, width, height, axis):
    with layout.render() as normalized:
        traces = _trace_runs(normalized[0])
    # Splitting text objects may split a trace, but must preserve every glyph
    # and origin before any reflection is applied.
    def glyphs(runs):
        return (char for run in runs for char in run["chars"])

    for before, after in zip_longest(glyphs(_trace_runs(source_page)), glyphs(traces)):
        if (before is None or after is None or before[:2] != after[:2]
                or math.dist(before[2], after[2]) > 0.01):
            raise VectorTextTransformError("Resolving PDF text positions changed a source glyph.")
    if len(traces) != len(layout.placements):
        raise VectorTextTransformError("Cannot match positioned PDF labels to visible glyph runs.")
    expected = []
    for slot, run in zip(layout.placements, traces, strict=True):
        start = tuple(float(v) for v in run["chars"][0][2])
        end = _trace_end(run)
        dx, dy = run["dir"]
        raw_dx, raw_dy = _mirrored_screen_direction(dx, dy, axis)
        out_dx, out_dy = M.mirror_direction(dx, dy, axis)
        reverse = raw_dx * out_dx + raw_dy * out_dy < 0

        # ``Tm x CTM`` maps text-space x/y to page space.  The traced
        # physical advance projected on the x column gives the local
        # translation required by the text-space reflection.
        matrix = list(slot.matrix)
        to_page = compose(slot.ctm, matrix)
        unit = math.hypot(to_page[0], to_page[1])
        if unit < 1e-9:
            raise VectorTextTransformError("Degenerate PDF text matrix.")
        advance = math.dist(start, end) / unit
        raw_start = _mirror_point(*start, width, height, axis)
        raw_end = _mirror_point(*end, width, height, axis)
        want_start, want_end = (raw_end, raw_start) if reverse else (raw_start, raw_end)
        a, b, c, d, e, f = matrix
        if reverse:
            e, f = e + a * advance, f + b * advance
            a, b = -a, -b
        # A single-axis reflection changes handedness. A two-axis
        # reflection preserves it, so reversing x also requires reversing y.
        flip_y = reverse if axis is Axis.BOTH else not reverse
        if flip_y:
            text_shift = _glyph_box_flip_shift(run) / unit
            e, f = e + c * text_shift, f + d * text_shift
            shift = text_shift * math.hypot(*to_page[2:4])
            up_x, up_y = _mirrored_screen_direction(dy, -dx, axis)
            want_start = (want_start[0] + up_x * shift, want_start[1] + up_y * shift)
            want_end = (want_end[0] + up_x * shift, want_end[1] + up_y * shift)
            c, d = -c, -d
        slot.set_matrix((a, b, c, d, e, f))
        expected.append(_TextExpectation(_run_text(run), want_start, want_end, (out_dx, out_dy)))

    reflection = _pdf_reflection_matrix(layout.page, axis)
    document = layout.render(reflection)
    try:
        actual = _trace_runs(document[0])
        if len(actual) != len(expected):
            raise VectorTextTransformError("Text-run count changed after vector mirror.")
        changed = False
        for slot, want, run in zip(layout.placements, expected, actual, strict=True):
            start = run["chars"][0][2]
            end = _trace_end(run)
            if max(math.dist(start, want.start), math.dist(end, want.end)) <= _TEXT_TRANSFORM_TOLERANCE_PT:
                continue
            # Convert the mean screen-space error through the reflected form CTM.
            dx = (want.start[0] - start[0] + want.end[0] - end[0]) / 2
            dy = -(want.start[1] - start[1] + want.end[1] - end[1]) / 2
            a, b, c, d, _, _ = compose(reflection, slot.ctm)
            det = a*d - b*c
            if abs(det) < 1e-12:
                raise VectorTextTransformError("Degenerate PDF placement transform.")
            matrix = list(slot.matrix)
            matrix[4] += (d*dx - c*dy) / det
            matrix[5] += (a*dy - b*dx) / det
            slot.set_matrix(matrix)
            changed = True
        if changed:
            document.close()
            document = layout.render(reflection)
        _verify_text_transforms(document[0], expected)
        return document, expected
    except Exception:
        document.close()
        raise


def _mirrored_screen_direction(dx: float, dy: float, axis: Axis) -> tuple[float, float]:
    if axis is Axis.VERTICAL:
        return (-dx, dy)
    if axis is Axis.HORIZONTAL:
        return (dx, -dy)
    return (-dx, -dy)


def _mirror_bbox(
    bbox: tuple[float, float, float, float], width: float, height: float, axis: Axis
) -> tuple[float, float, float, float]:
    points = [_mirror_point(x, y, width, height, axis) for x, y in (
        (bbox[0], bbox[1]), (bbox[0], bbox[3]),
        (bbox[2], bbox[1]), (bbox[2], bbox[3]),
    )]
    return (
        min(point[0] for point in points), min(point[1] for point in points),
        max(point[0] for point in points), max(point[1] for point in points),
    )


def _verify_text_transforms(output_page, expected):
    """Verify text and both reflected endpoints on the serialized page."""
    actual = _trace_runs(output_page)
    if len(actual) != len(expected):
        raise VectorTextTransformError(
            f"Text-run count changed after vector mirror ({len(expected)} -> {len(actual)})."
        )
    for index, (want, run) in enumerate(zip(expected, actual, strict=True)):
        got_text = _run_text(run)
        got_start = tuple(float(v) for v in run["chars"][0][2])
        got_end = _trace_end(run)
        if got_text != want.text or math.dist(run["dir"], want.direction) > 1e-4 or max(
            math.dist(got_start, want.start), math.dist(got_end, want.end)
        ) > _TEXT_TRANSFORM_TOLERANCE_PT:
            raise VectorTextTransformError(
                f"Text transform QA failed for run {index} {want.text!r}: "
                f"baseline differs from the reflected source by more than "
                f"{_TEXT_TRANSFORM_TOLERANCE_PT} pt."
            )


def _extract_text_spans(page: pymupdf.Page) -> list[dict]:
    """One dict per glyph run, carrying exactly what re-rendering needs."""
    spans: list[dict] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        for line in block["lines"]:
            direction = line["dir"]  # (dx, dy), screen coords, unit vector
            for span in line["spans"]:
                text = span["text"]
                if not text.strip():
                    continue
                spans.append(
                    {
                        "text": text,
                        "bbox": span["bbox"],  # (x0, y0, x1, y1)
                        # Exact PDF baseline start.  Using the centre of the
                        # axis-aligned bbox for rotated text shifts the run
                        # across its wall because that box contains ascender /
                        # descender offsets.  The origin is the stable anchor
                        # the source content stream actually used.
                        "origin": span["origin"],
                        "size": span["size"],
                        "color": _int_to_rgb(span["color"]),
                        "font": span["font"],
                        "bold": bool(span["flags"] & (1 << 4)),
                        "italic": bool(span["flags"] & (1 << 1)),
                        "dir": direction,
                    }
                )
    return spans


def _int_to_rgb(color_int: int) -> tuple[float, float, float]:
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return (r / 255, g / 255, b / 255)


def _rotate_param(dx: float, dy: float) -> tuple[int, float]:
    """Map a canonicalised direction to PyMuPDF's ``insert_text(rotate=)``
    quarter-turn steps. This remains useful for cardinal text, where
    PyMuPDF's native ``rotate`` produces the cleanest PDF content stream.

    ``rotate`` is the negative of the screen-space ``atan2(dy, dx)``
    angle — i.e. exactly Spejl's own angle convention, confirmed
    empirically during the build: ``rotate=90`` produces direction
    (0, -1). Quarter turns only, hence the rounding; the module docstring
    covers what that costs for diagonal text.

    Also returns the signed deviation (degrees) between the run's own
    true angle and the cardinal it was rounded to, so the caller can
    judge whether that rounding actually cost anything worth flagging —
    see ``_DIAGONAL_ROUNDING_TOLERANCE_DEG``.
    """
    angle = M.angle_from_direction(dx, dy)
    nearest_quarter_turn = round(angle / 90)
    return int(nearest_quarter_turn * 90) % 360, angle - nearest_quarter_turn * 90


def _font_alias(font_name: str, bold: bool, italic: bool) -> str:
    name = font_name.lower()
    if "courier" in name or "mono" in name:
        table = _COURIER
    elif "times" in name or "serif" in name or "georgia" in name or "garamond" in name:
        table = _TIMES
    else:
        table = _HELV
    return table[bold][italic]


def _reinsert_mirrored_span(
    page: pymupdf.Page, span: dict, W: float, H: float, axis: Axis
) -> None:
    """Re-insert ``span`` upright at its mirrored anchor.

    Geometry is reflected first, which changes the baseline direction.
    ``mirror_direction`` then reverses the traversal when reflection would
    make glyphs read backwards or upside down.  Crucially, a reversal does
    not change the wall-aligned baseline: it only chooses its readable
    direction, mirroring AutoCAD's MIRRTEXT behaviour.
    """
    dx, dy = span["dir"]
    dxf, dyf = M.mirror_direction(dx, dy, axis)
    target_angle = M.angle_from_direction(dxf, dyf)
    rotate, deviation_deg = _rotate_param(dxf, dyf)

    fontname = _font_alias(span["font"], span["bold"], span["italic"])
    fontsize = span["size"]
    width = pymupdf.get_text_length(span["text"], fontname=fontname, fontsize=fontsize)

    # Mirror the source's exact baseline segment, not its axis-aligned bbox.
    # If reflection reverses the baseline's readable traversal, start at the
    # mirrored END of the original run.  This is the matrix equivalent of
    # AutoCAD MIRRTEXT=0: position follows the reflection, while character
    # order remains readable.  It also preserves the perpendicular offset
    # from a horizontal, vertical or slanted wall exactly.
    ox, oy = span["origin"]
    raw_start = _mirror_point(ox, oy, W, H, axis)
    raw_end = _mirror_point(ox + dx * width, oy + dy * width, W, H, axis)
    raw_dx = raw_end[0] - raw_start[0]
    raw_dy = raw_end[1] - raw_start[1]
    point = raw_start if (raw_dx * dxf + raw_dy * dyf) >= 0 else raw_end

    if abs(deviation_deg) < 1e-6 and rotate == 0:
        morph = None
    elif abs(deviation_deg) < 1e-6 and rotate == 90:
        morph = None
    elif abs(deviation_deg) < 1e-6 and rotate == 180:
        morph = None
    elif abs(deviation_deg) < 1e-6:  # 270
        morph = None
    else:
        # Passing the exact mirrored baseline origin as the morph pivot makes
        # arbitrary rotation change glyph orientation without moving it.
        morph = (
            pymupdf.Point(*point),
            pymupdf.Matrix(1, 0, 0, 1, 0, 0).prerotate(target_angle),
        )

    page.insert_text(
        point,
        span["text"],
        fontsize=fontsize,
        fontname=fontname,
        color=span["color"],
        rotate=rotate if morph is None else 0,
        morph=morph,
    )


# --------------------------------------------------------------------------
# Geometry: reconstruct every drawn path, mirrored, from get_drawings()
# --------------------------------------------------------------------------


def _draw_mirrored_path(shape, dwg: dict, W: float, H: float, axis: Axis) -> None:
    for item in dwg["items"]:
        op = item[0]
        if op == "re":
            rect: pymupdf.Rect = item[1]
            p0 = _mirror_point(rect.x0, rect.y0, W, H, axis)
            p1 = _mirror_point(rect.x1, rect.y1, W, H, axis)
            shape.draw_rect(
                pymupdf.Rect(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
            )
        elif op == "l":
            p0 = _mirror_point(item[1].x, item[1].y, W, H, axis)
            p1 = _mirror_point(item[2].x, item[2].y, W, H, axis)
            shape.draw_line(pymupdf.Point(*p0), pymupdf.Point(*p1))
        elif op == "c":
            pts = [pymupdf.Point(*_mirror_point(p.x, p.y, W, H, axis)) for p in item[1:5]]
            shape.draw_bezier(*pts)
        elif op == "qu":
            quad: pymupdf.Quad = item[1]
            pts = [
                pymupdf.Point(*_mirror_point(p.x, p.y, W, H, axis))
                for p in (quad.ul, quad.ur, quad.ll, quad.lr)
            ]
            shape.draw_quad(pymupdf.Quad(*pts))
        # Unrecognised ops are rare (this covers line/rect/curve/quad, which
        # is everything CAD/vector-PDF exporters emit for plan geometry)
        # and are skipped rather than raising, so one odd path never aborts
        # a whole-document mirror.

    line_cap = dwg.get("lineCap")
    # PDF/CAD convention: an explicit stroke width of 0 means "hairline —
    # thinnest the device can render". `dwg.get("width") or 1.0` treats
    # that 0 as falsy and silently thickens every hairline wall/gridline
    # (what DWG->PDF exporters commonly use) to a full 1pt stroke.
    #
    # The fix is not simply "pass 0 through", though: PyMuPDF's own
    # Shape.finish() gives `width=0` a THIRD, different meaning again —
    # its source sets `color = None` whenever `width == 0` ("border
    # color makes no sense then"), which suppresses the stroke operator
    # entirely, so the line would vanish rather than render thin. A
    # small positive width is the only value that survives PyMuPDF's own
    # writer as a visibly hairline-thin *stroke*, which is what "0 w" in
    # the source actually meant. Only a genuinely missing key (None)
    # falls back to the ordinary default.
    _HAIRLINE_PT = 0.1
    dwg_width = dwg.get("width")
    if dwg_width is None:
        resolved_width = 1.0
    elif dwg_width == 0:
        resolved_width = _HAIRLINE_PT
    else:
        resolved_width = dwg_width
    shape.finish(
        width=resolved_width,
        color=dwg.get("color"),
        fill=dwg.get("fill"),
        lineCap=line_cap[0] if isinstance(line_cap, tuple) else (line_cap or 0),
        lineJoin=int(dwg.get("lineJoin") or 0),
        dashes=dwg.get("dashes"),
        closePath=bool(dwg.get("closePath", False)),
        even_odd=bool(dwg.get("even_odd", False)),
        fill_opacity=dwg.get("fill_opacity") if dwg.get("fill_opacity") is not None else 1,
        stroke_opacity=dwg.get("stroke_opacity") if dwg.get("stroke_opacity") is not None else 1,
    )
