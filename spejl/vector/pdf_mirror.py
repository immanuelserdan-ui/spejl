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
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf
import pikepdf

from spejl.models import Axis, Document, Flag, PageResult, Route
from spejl.transform import mirror as M

# Point/direction mirroring and the readability convention live in
# transform/mirror.py, shared with Route B — two copies of that rule
# would be free to drift, and it is the rule the whole tool turns on.
_mirror_point = M.mirror_point

_MIN_PAGE_CONTENT_MARGIN_PT = 5.0

# Font-family fallback: CAD-exported PDFs overwhelmingly set a
# Helvetica/Arial-alike, so that is the default; a document that
# clearly asks for a serif or monospace face gets one. Exact family
# matching (the build plan's style/family.py, NCC template matching)
# is Route B's job — Route A only ever loses a glyph's *exact* outline
# for a font PyMuPDF doesn't ship, never its position or orientation.
_HELV = {False: {False: "helv", True: "heit"}, True: {False: "hebo", True: "hebi"}}
_TIMES = {False: {False: "tiro", True: "tiit"}, True: {False: "tibo", True: "tibi"}}
_COURIER = {False: {False: "cour", True: "coit"}, True: {False: "cobo", True: "cobi"}}


def mirror_pdf(input_path: Path, output_path: Path, axis: Axis = Axis.VERTICAL) -> Document:
    """Mirror every page of ``input_path`` and write ``output_path``.

    Returns the :class:`Document` record — the same object that
    ``to_sidecar()`` turns into the ``*.spejl.json`` written beside the
    output.
    """
    result = Document(source=input_path, output=output_path, axis=axis, route=Route.VECTOR)
    fitted_pages: set[int] = set()

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
            if page.get_image_info():
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
            # Copy the complete original page first.  This retains embedded
            # fonts, XObjects, colour spaces and every CAD drawing operator.
            # We then replace only its content stream with an equivalent
            # mirrored stream whose text matrices have been corrected.
            out.insert_pdf(src, from_page=page_index, to_page=page_index)
            new_page = out[-1]
            transformed, expected = _transform_vector_content(
                pike_src.pages[page_index], page, W, H, axis
            )
            xref = out.get_new_xref()
            out.update_object(xref, "<<>>")
            out.update_stream(xref, transformed)
            new_page.set_contents(xref)
            transformed, expected = _calibrate_text_translations(
                transformed, new_page, expected, width=W, height=H, axis=axis
            )
            if transformed != new_page.read_contents():
                out.update_stream(xref, transformed)
                new_page.set_contents(xref)
            _verify_text_transforms(
                page, new_page, expected, W, H, axis, transformed,
                _pdf_reflection_matrix(pike_src.pages[page_index], axis),
            )

            fitted_stream, did_fit = _fit_page_content_stream(
                new_page, transformed, width=W, height=H,
            )
            if did_fit:
                out.update_stream(xref, fitted_stream)
                new_page.set_contents(xref)
                fitted_pages.add(page_index)

            result.pages.append(
                PageResult(
                    index=page_index,
                    width=W,
                    height=H,
                    text_runs_mirrored=len(expected),
                    flags=[],
                )
            )

        out.save(str(output_path))
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
                if page_index in fitted_pages:
                    bounds = _page_visible_content_bounds(written_page)
                    margin = _MIN_PAGE_CONTENT_MARGIN_PT
                    tolerance = 0.1
                    if bounds is not None and (
                        bounds.x0 < margin - tolerance
                        or bounds.y0 < margin - tolerance
                        or bounds.x1 > written_page.rect.width - margin + tolerance
                        or bounds.y1 > written_page.rect.height - margin + tolerance
                    ):
                        raise VectorTextTransformError(
                            f"Fitted page {page_index + 1} still has content inside the "
                            f"{margin:g}-point safety margin."
                        )
        if blank_pages:
            # A few CAD exporters produce a page whose resource graph cannot
            # survive stream replacement even though text extraction works.
            # Do not return a blank deliverable: render the original page and
            # mirror that complete image as a safe, explicitly flagged PDF.
            fallback_pages = _write_raster_pdf_fallback(src, output_path, axis)
            result.pages = fallback_pages
    finally:
        out.close()
        src.close()
        pike_src.close()

    return result


def _page_visible_content_bounds(page: pymupdf.Page) -> pymupdf.Rect | None:
    """Union visible PDF object bounds, including strokes and text glyphs."""
    bounds: pymupdf.Rect | None = None
    for kind, bbox in page.get_bboxlog():
        if kind.startswith("clip-") or kind == "group":
            continue
        rect = pymupdf.Rect(bbox)
        if rect.is_empty or not rect.is_valid:
            continue
        if bounds is None:
            bounds = rect
        else:
            bounds.include_rect(rect)
    return bounds


def _fit_page_content_stream(
    page: pymupdf.Page,
    content: bytes,
    *,
    width: float,
    height: float,
    margin: float = _MIN_PAGE_CONTENT_MARGIN_PT,
) -> tuple[bytes, bool]:
    """Translate/scale page graphics just enough to keep them 5pt in-bounds.

    A uniform fit keeps walls, door swings, dimensions, and text aligned. The
    existing page size is retained; content is only changed when its rendered
    bounds cross the safety inset. Rotated PDF pages use a different default
    user-space basis, so they are left untouched rather than risk a bad CTM.
    """
    if page.rotation or width <= 2 * margin or height <= 2 * margin:
        return content, False
    bounds = _page_visible_content_bounds(page)
    if bounds is None:
        return content, False
    safe = pymupdf.Rect(margin, margin, width - margin, height - margin)
    tolerance = 0.05
    if (
        bounds.x0 >= safe.x0 - tolerance
        and bounds.y0 >= safe.y0 - tolerance
        and bounds.x1 <= safe.x1 + tolerance
        and bounds.y1 <= safe.y1 + tolerance
    ):
        return content, False

    scale = min(
        1.0,
        safe.width / max(bounds.width, 1e-6),
        safe.height / max(bounds.height, 1e-6),
    )
    dx = safe.x0 + (safe.width - bounds.width * scale) / 2 - bounds.x0 * scale
    dy = safe.y0 + (safe.height - bounds.height * scale) / 2 - bounds.y0 * scale
    # PDF's default user space has its origin at the bottom-left, while the
    # bounds above use PyMuPDF's top-left page coordinates.
    pdf_dy = height * (1 - scale) - dy
    matrix = f"q\n{scale:.10f} 0 0 {scale:.10f} {dx:.10f} {pdf_dy:.10f} cm\n".encode("ascii")
    return matrix + content + b"\nQ\n", True


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
            for block in page.get_text("rawdict").get("blocks", []):
                if block.get("type") == 1 and "bbox" in block:
                    occupied.append(pymupdf.Rect(block["bbox"]))
            if any(text_box.intersects(element) for text_box in selected_boxes for element in occupied):
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
            for block in page.get_text("rawdict").get("blocks", []):
                if block.get("type") == 1 and "bbox" in block:
                    occupied.append(pymupdf.Rect(block["bbox"]))
            if not occupied:
                continue
            for run_index, trace in enumerate(_trace_runs(page)):
                box = pymupdf.Rect(trace["bbox"])
                if not any(box.intersects(element) for element in occupied):
                    continue
                label = "".join(chr(char[0]) for char in trace["chars"]).strip()
                if label:
                    findings.append((page_index, run_index, label))
        return findings
    finally:
        document.close()


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
    source: pymupdf.Document, output_path: Path, axis: Axis
) -> list[PageResult]:
    """Write a visible image-PDF when a vector page serializes blank."""
    from spejl.raster.pipeline import mirror_raster

    pages: list[PageResult] = []
    with TemporaryDirectory(prefix="spejl-pdf-fallback-") as temporary:
        raster_pdf = pymupdf.open()
        for index, page in enumerate(source):
            source_png = Path(temporary) / f"source-{index}.png"
            output_png = Path(temporary) / f"mirror-{index}.png"
            page.get_pixmap(dpi=300, alpha=False).save(str(source_png))
            raster = mirror_raster(source_png, output_png, axis=axis)
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
        raster_pdf.save(str(output_path))
        raster_pdf.close()
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
    source_matrix: tuple[float, float, float, float, float, float]
    output_matrix: tuple[float, float, float, float, float, float]


@dataclass
class _TextState:
    """Complete PDF text state preserved while walking one ``BT`` block."""

    font: object | None = None
    size: float | None = None
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    horizontal_scale: float = 100.0
    rise: float = 0.0
    matrix: tuple[float, float, float, float, float, float] | None = None


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
    """Farthest baseline endpoint of one traced glyph run, screen space."""
    start = run["chars"][0][2]
    dx, dy = run["dir"]
    last_bbox = run["chars"][-1][3]
    corners = (
        (last_bbox[0], last_bbox[1]), (last_bbox[0], last_bbox[3]),
        (last_bbox[2], last_bbox[1]), (last_bbox[2], last_bbox[3]),
    )
    extent = max((x - start[0]) * dx + (y - start[1]) * dy for x, y in corners)
    return (start[0] + dx * extent, start[1] + dy * extent)


def _run_text(run: dict) -> str:
    return "".join(chr(char[0]) for char in run["chars"])


def _mul_text_matrix_by_reflection(
    matrix: list[float], advance: float, reverse_baseline: bool
) -> list[float]:
    """Apply a readable local reflection to one PDF ``Tm``.

    The page is globally reflected.  A second, local reflection makes the
    combined glyph transform orientation-preserving.  Flipping the local y
    axis keeps an already-readable reflected baseline; flipping local x and
    translating by the exact traced advance reverses a backwards baseline
    while retaining the original string and its raw ``TJ`` spacing.
    """
    a, b, c, d, e, f = matrix
    if reverse_baseline:
        # M * [-1 0 0 1 advance 0]
        return [-a, -b, c, d, e + a * advance, f + b * advance]
    # M * [1 0 0 -1 0 0]
    return [a, b, -c, -d, e, f]


def _transform_vector_content(
    pike_page: pikepdf.Page,
    source_page: pymupdf.Page,
    width: float,
    height: float,
    axis: Axis,
) -> tuple[bytes, list[_TextExpectation]]:
    """Mirror a page while retaining its native PDF text operators.

    Geometry and resources are left in the original content stream and the
    whole page is wrapped in the reflection CTM.  Only ``Tm`` operators are
    changed; font selection, ``Tc``, ``Tw``, ``Tz``, ``Ts`` and every
    ``Tj``/``TJ`` operand are preserved exactly as supplied by the CAD PDF.
    """
    instructions = list(pikepdf.parse_content_stream(pike_page))
    traces = _trace_runs(source_page)
    trace_index = 0
    expected: list[_TextExpectation] = []
    rewritten: list[pikepdf.ContentStreamInstruction] = []
    in_text = False
    state: _TextState | None = None

    for instruction in instructions:
        operands, operator = instruction
        name = str(operator)
        if name == "BT":
            if in_text:
                raise VectorTextTransformError("Nested BT in PDF content stream.")
            in_text, state = True, _TextState()
            rewritten.append(instruction)
            continue
        if name == "ET":
            if not in_text:
                raise VectorTextTransformError("ET without a matching BT in PDF content stream.")
            in_text, state = False, None
            rewritten.append(instruction)
            continue
        if in_text and state is not None:
            _record_text_state(state, name, operands)
        if name != "Tm" or len(operands) != 6:
            rewritten.append(instruction)
            continue
        if not in_text or state is None:
            raise VectorTextTransformError("Tm outside BT/ET in PDF content stream.")
        if trace_index >= len(traces):
            raise VectorTextTransformError(
                "PDF text matrix has no corresponding glyph trace; refusing a lossy vector mirror."
            )

        run = traces[trace_index]
        trace_index += 1
        start = tuple(float(v) for v in run["chars"][0][2])
        end = _trace_end(run)
        dx, dy = run["dir"]
        raw_dx, raw_dy = _mirrored_screen_direction(dx, dy, axis)
        out_dx, out_dy = M.mirror_direction(dx, dy, axis)
        reverse = raw_dx * out_dx + raw_dy * out_dy < 0

        # ``Tm``'s first column maps text-space x to page space.  The traced
        # physical advance projected on that vector gives the local
        # translation required by the text-space reflection.
        a, b = float(operands[0]), float(operands[1])
        unit = math.hypot(a, b)
        if unit < 1e-9:
            raise VectorTextTransformError("Degenerate PDF text matrix.")
        advance = math.hypot(end[0] - start[0], end[1] - start[1]) / unit
        matrix = [float(value) for value in operands]
        replacement = _mul_text_matrix_by_reflection(matrix, advance, reverse)
        rewritten.append(pikepdf.ContentStreamInstruction(replacement, operator))

        raw_start = _mirror_point(*start, width, height, axis)
        raw_end = _mirror_point(*end, width, height, axis)
        expected.append(_TextExpectation(
            text=_run_text(run),
            start=raw_end if reverse else raw_start,
            end=raw_start if reverse else raw_end,
            source_matrix=tuple(matrix),
            output_matrix=tuple(replacement),
        ))

    if trace_index != len(traces):
        raise VectorTextTransformError(
            "PDF glyph trace has text with no Tm matrix; refusing a lossy vector mirror."
        )

    reflection = " ".join(f"{value:g}" for value in _pdf_reflection_matrix(pike_page, axis))
    return b"q\n" + reflection.encode("ascii") + b" cm\n" + pikepdf.unparse_content_stream(rewritten) + b"\nQ\n", expected


def _record_text_state(state: _TextState, name: str, operands: object) -> None:
    """Record text state without replacing any original content operands."""
    if name == "Tf" and len(operands) == 2:
        state.font, state.size = operands[0], float(operands[1])
    elif name == "Tc" and len(operands) == 1:
        state.char_spacing = float(operands[0])
    elif name == "Tw" and len(operands) == 1:
        state.word_spacing = float(operands[0])
    elif name == "Tz" and len(operands) == 1:
        state.horizontal_scale = float(operands[0])
    elif name == "Ts" and len(operands) == 1:
        state.rise = float(operands[0])
    elif name == "Tm" and len(operands) == 6:
        state.matrix = tuple(float(value) for value in operands)


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


def _calibrate_text_translations(
    stream: bytes,
    output_page: pymupdf.Page,
    expected: list[_TextExpectation],
    *,
    width: float,
    height: float,
    axis: Axis,
) -> tuple[bytes, list[_TextExpectation]]:
    """Remove PDF side-bearing drift while leaving all show operators intact.

    Some CAD exporters put a non-zero left/right side bearing on short
    annotations such as ``H*``.  The reflected ``Tm`` is mathematically
    correct, but PyMuPDF reports the final ink endpoint rather than the
    text-matrix advance.  If the discrepancy is a pure translation, adjust
    only ``Tm.e/f`` by that reflected delta.  A changed baseline *length* is
    rejected later by strict QA; this helper never changes font metrics or
    ``Tj``/``TJ`` spacing.
    """
    actual = _trace_runs(output_page)
    if len(actual) != len(expected):
        return stream, expected
    corrections: list[tuple[float, float]] = []
    changed = False
    for want, run in zip(expected, actual, strict=True):
        got_start = tuple(float(v) for v in run["chars"][0][2])
        got_end = _trace_end(run)
        start_delta = (want.start[0] - got_start[0], want.start[1] - got_start[1])
        end_delta = (want.end[0] - got_end[0], want.end[1] - got_end[1])
        if max(math.dist(got_start, want.start), math.dist(got_end, want.end)) <= _TEXT_TRANSFORM_TOLERANCE_PT:
            corrections.append((0.0, 0.0))
            continue
        # Use the midpoint of the endpoint deltas.  CAD exporters sometimes
        # report a slightly different side bearing for the final glyph (most
        # visible on ``H*``); correcting the midpoint removes the translation
        # component while strict QA below still rejects any real length error.
        correction = (
            (start_delta[0] + end_delta[0]) / 2.0,
            (start_delta[1] + end_delta[1]) / 2.0,
        )
        # Convert screen-space movement back through the page reflection.
        if axis is Axis.VERTICAL:
            raw_delta = (-correction[0], correction[1])
        elif axis is Axis.HORIZONTAL:
            raw_delta = (correction[0], -correction[1])
        else:
            raw_delta = (-correction[0], -correction[1])
        corrections.append(raw_delta)
        changed = True
    if not changed:
        return stream, expected

    holder = pikepdf.Pdf.new()
    rewritten: list[pikepdf.ContentStreamInstruction] = []
    matrix_index = 0
    for operands, operator in pikepdf.parse_content_stream(pikepdf.Stream(holder, stream)):
        if str(operator) == "Tm" and len(operands) == 6:
            matrix = [float(value) for value in operands]
            if matrix_index < len(corrections):
                de, df = corrections[matrix_index]
                matrix[4] += de
                matrix[5] += df
                matrix_index += 1
            rewritten.append(pikepdf.ContentStreamInstruction(matrix, operator))
        else:
            rewritten.append(pikepdf.ContentStreamInstruction(operands, operator))
    # ``stream`` already contains the single outer ``q / reflection cm`` and
    # ``Q``.  Re-serialise the complete instruction list once; adding a new
    # wrapper here would apply the page reflection twice.
    adjusted = pikepdf.unparse_content_stream(rewritten)
    adjusted_expected = [
        _TextExpectation(
            text=item.text,
            start=item.start,
            end=item.end,
            source_matrix=item.source_matrix,
            output_matrix=tuple(
                value + (corrections[i][0] if j == 4 else corrections[i][1] if j == 5 else 0.0)
                for j, value in enumerate(item.output_matrix)
            ),
        )
        for i, item in enumerate(expected)
    ]
    return adjusted, adjusted_expected


def _verify_text_transforms(
    source_page: pymupdf.Page,
    output_page: pymupdf.Page,
    expected: list[_TextExpectation],
    width: float,
    height: float,
    axis: Axis,
    output_stream: bytes,
    reflection_matrix: tuple[float, float, float, float, float, float],
) -> None:
    """Strict post-write baseline and affine-transform QA for every run."""
    matrix_pdf = pikepdf.Pdf.new()
    matrices = [
        tuple(float(value) for value in operands)
        for operands, operator in pikepdf.parse_content_stream(
            pikepdf.Stream(matrix_pdf, output_stream)
        )
        if str(operator) == "Tm" and len(operands) == 6
    ]
    if len(matrices) != len(expected):
        raise VectorTextTransformError(
            f"Text-matrix count changed after vector mirror ({len(expected)} -> {len(matrices)})."
        )
    reflection = reflection_matrix
    for index, (want, matrix) in enumerate(zip(expected, matrices, strict=True)):
        effective = _matrix_multiply(reflection, matrix)
        wanted_effective = _matrix_multiply(reflection, want.output_matrix)
        if max(abs(actual - desired) for actual, desired in zip(matrix, want.output_matrix, strict=True)) > 1e-6 or max(
            abs(actual - desired) for actual, desired in zip(effective, wanted_effective, strict=True)
        ) > 1e-6:
            raise VectorTextTransformError(
                f"Text transform QA failed for run {index}: output matrix is not the reflected source matrix."
            )

    actual = _trace_runs(output_page)
    if len(actual) != len(expected):
        raise VectorTextTransformError(
            f"Text-run count changed after vector mirror ({len(expected)} -> {len(actual)})."
        )
    for index, (want, run) in enumerate(zip(expected, actual, strict=True)):
        got_text = _run_text(run)
        got_start = tuple(float(v) for v in run["chars"][0][2])
        got_end = _trace_end(run)
        if got_text != want.text or max(
            math.dist(got_start, want.start), math.dist(got_end, want.end)
        ) > _TEXT_TRANSFORM_TOLERANCE_PT:
            raise VectorTextTransformError(
                f"Text transform QA failed for run {index} {want.text!r}: "
                f"baseline differs from the reflected source by more than "
                f"{_TEXT_TRANSFORM_TOLERANCE_PT} pt."
            )


def _matrix_multiply(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Compose PDF affine matrices: result applies ``right`` then ``left``."""
    a, b, c, d, e, f = left
    A, B, C, D, E, F = right
    return (
        a * A + c * B, b * A + d * B,
        a * C + c * D, b * C + d * D,
        a * E + c * F + e, b * E + d * F + f,
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
