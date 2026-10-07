"""Assisted mirroring for mixed PDFs: vector text over a raster drawing.

A Revit view exported with raster processing arrives as one picture holding
the walls and linework, with the text left as real vector text, except for
any lettering that was drawn as part of the model (casework labels such as
``H*`` or ``H/O``). The vector route mirrors the picture correctly, but
lettering inside it would come out backwards; the raster route would turn
every perfectly good vector text item into pixels as well.

This module takes the third way, with a person in the loop:

1. :func:`find_picture_labels` runs OCR over each picture *only* and lists
   the lettering it finds. OCR on small plan labels is unreliable (``H/O``
   reads as ``OH`` or ``O/H``; stove burners read as ``QQ``), so these are
   suggestions: low-confidence finds start unticked.
2. The user confirms which finds are lettering and corrects their text.
3. :func:`mirror_mixed_pdf` erases the confirmed lettering from the
   picture, redraws it as real text at the same place and size, then runs
   the normal vector route with the picture treated as linework.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pikepdf
import pymupdf

from spejl.models import Axis, Document, Flag

# A find below this OCR confidence starts unticked in the review.
DEFAULT_MIN_CONFIDENCE = 0.9
# Pictures are read at no less than this resolution; plan labels are only a
# few points tall, and 150 dpi exports leave them ~10 px high.
_OCR_DPI = 300
# Reading directions the redraw supports, in Spejl's angle convention.
SUPPORTED_ANGLES = (0.0, 90.0, -90.0)
_REFERENCE_FONT = "helv"


@dataclass
class PictureLabel:
    """One piece of lettering found inside a picture, as the user reviews it."""

    page_index: int
    image_xref: int
    text: str
    angle_deg: float  # 0 left-to-right, 90 bottom-to-top, -90 top-to-bottom
    pixel_box: tuple[int, int, int, int]  # ink box in picture pixels, end-exclusive
    page_box: tuple[float, float, float, float]  # the same box in page points
    confidence: float
    include: bool = True
    ocr_text: str = field(default="", compare=False)


@dataclass(frozen=True)
class _Placement:
    page_index: int
    xref: int
    rect: pymupdf.Rect
    width: int
    height: int

    def to_page(self, px: float, py: float) -> tuple[float, float]:
        return (self.rect.x0 + px * self.rect.width / self.width,
                self.rect.y0 + py * self.rect.height / self.height)


def _placements(doc: pymupdf.Document) -> list[_Placement]:
    placements: list[_Placement] = []
    seen: set[int] = set()
    for page_index, page in enumerate(doc):
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref") or 0
            a, b, c, d, _e, _f = info["transform"]
            if xref <= 0:
                raise ValueError(
                    f"page {page_index + 1}: contains an inline picture, which assisted "
                    "mirroring cannot edit."
                )
            if abs(b) > 1e-6 or abs(c) > 1e-6 or a <= 0 or d <= 0:
                raise ValueError(
                    f"page {page_index + 1}: a picture is rotated or flipped on the page; "
                    "assisted mirroring supports upright pictures only."
                )
            if xref in seen:
                raise ValueError(
                    f"page {page_index + 1}: the same picture is placed more than once; "
                    "assisted mirroring cannot edit it safely."
                )
            seen.add(xref)
            placements.append(_Placement(page_index, xref, pymupdf.Rect(info["bbox"]),
                                         int(info["width"]), int(info["height"])))
    return placements


def picture_rgb(doc: pymupdf.Document, xref: int) -> np.ndarray:
    """The picture's pixels as an RGB array (alpha dropped)."""
    pix = pymupdf.Pixmap(doc, xref)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()


def _ink_box(gray: np.ndarray, box: tuple[float, float, float, float], pad: int = 2) -> tuple[int, int, int, int]:
    """Tighten an OCR box to the lettering's own dark pixels.

    Lines that merely pass through the OCR box (a cell border, a wall) are
    excluded: only dark components lying wholly inside the padded box count.
    """
    from scipy import ndimage

    h, w = gray.shape
    x0, y0, x1, y1 = (math.floor(box[0]) - pad, math.floor(box[1]) - pad,
                      math.ceil(box[2]) + pad, math.ceil(box[3]) + pad)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    margin = 12
    rx0, ry0, rx1, ry1 = max(x0 - margin, 0), max(y0 - margin, 0), min(x1 + margin, w), min(y1 + margin, h)
    labels, _count = ndimage.label(gray[ry0:ry1, rx0:rx1] < 128)
    keep = np.zeros_like(labels, dtype=bool)
    for index, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        cy0, cy1 = sl[0].start + ry0, sl[0].stop + ry0
        cx0, cx1 = sl[1].start + rx0, sl[1].stop + rx0
        if cx0 >= x0 and cy0 >= y0 and cx1 <= x1 and cy1 <= y1:
            keep |= labels == index
    ys, xs = np.where(keep)
    if len(xs) == 0:
        return (max(int(box[0]), 0), max(int(box[1]), 0), min(math.ceil(box[2]), w), min(math.ceil(box[3]), h))
    return (int(xs.min()) + rx0, int(ys.min()) + ry0, int(xs.max()) + rx0 + 1, int(ys.max()) + ry0 + 1)


def _looks_like_text(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def find_picture_labels(
    pdf_path: Path,
    backend=None,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[PictureLabel]:
    """OCR every picture in ``pdf_path`` and list the lettering found.

    ``include`` is pre-set only for confident, text-like, upright finds;
    everything else is listed unticked for the user to judge.
    """
    import cv2

    from spejl.detect.rotations import detect_all_orientations

    if backend is None:
        from spejl.detect.ocr import RapidOcrBackend

        backend = RapidOcrBackend()

    found: list[PictureLabel] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for placement in _placements(doc):
            rgb = picture_rgb(doc, placement.xref)
            gray = rgb.mean(axis=2)
            dpi = placement.width / max(placement.rect.width / 72.0, 1e-6)
            factor = max(1, math.ceil(_OCR_DPI / max(dpi, 1.0) - 1e-6))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if factor > 1:
                bgr = cv2.resize(bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            for det in detect_all_orientations(bgr, backend):
                text = det.text.strip()
                if not text:
                    continue
                box = tuple(v / factor for v in det.bbox)
                ink = _ink_box(gray, box)
                p0 = placement.to_page(ink[0], ink[1])
                p1 = placement.to_page(ink[2], ink[3])
                angle = float(det.angle_deg)
                found.append(PictureLabel(
                    page_index=placement.page_index,
                    image_xref=placement.xref,
                    text=text,
                    angle_deg=angle,
                    pixel_box=ink,
                    page_box=(p0[0], p0[1], p1[0], p1[1]),
                    confidence=float(det.conf),
                    include=(det.conf >= min_confidence and _looks_like_text(text)
                             and angle in SUPPORTED_ANGLES),
                    ocr_text=text,
                ))
    found.sort(key=lambda label: (label.page_index, label.page_box[1], label.page_box[0]))
    return found


def _rotate_param(angle_deg: float) -> int:
    if angle_deg == 0.0:
        return 0
    if angle_deg == 90.0:
        return 90
    if angle_deg == -90.0:
        return 270
    raise ValueError(f"reading direction {angle_deg:g}° is not supported for redrawn lettering")


def _rendered_ink(width: float, height: float, text: str, size: float, origin: pymupdf.Point,
                  rotate: int) -> tuple[float, float, float, float]:
    """Ink box of ``text`` drawn in the reference font, in page points."""
    tmp = pymupdf.open()
    page = tmp.new_page(width=width, height=height)
    page.insert_text(origin, text, fontsize=size, fontname=_REFERENCE_FONT, rotate=rotate)
    reach = size * (len(text) + 2)
    clip = pymupdf.Rect(origin.x - reach, origin.y - reach, origin.x + reach, origin.y + reach) & page.rect
    dpi = 1200
    pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=pymupdf.csGRAY, alpha=False)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
    ys, xs = np.where(a < 128)
    if len(xs) == 0:
        raise ValueError(f"could not measure {text!r} in the reference font")
    scale = 72 / dpi
    return (clip.x0 + xs.min() * scale, clip.y0 + ys.min() * scale,
            clip.x0 + (xs.max() + 1) * scale, clip.y0 + (ys.max() + 1) * scale)


def _fit_text(page: pymupdf.Page, label: PictureLabel) -> tuple[pymupdf.Point, float, int]:
    """Size and origin that put ``label.text`` over the original lettering."""
    rotate = _rotate_param(label.angle_deg)
    W, H = page.rect.width, page.rect.height
    tx0, ty0, tx1, ty1 = label.page_box
    probe = pymupdf.Point(W / 2, H / 2)
    ref = _rendered_ink(W, H, label.text, 10.0, probe, rotate)
    # Match the extent across the reading direction (the lettering's height).
    if rotate in (90, 270):
        size = 10.0 * (tx1 - tx0) / max(ref[2] - ref[0], 1e-6)
    else:
        size = 10.0 * (ty1 - ty0) / max(ref[3] - ref[1], 1e-6)
    size = max(round(size, 2), 1.0)
    ink = _rendered_ink(W, H, label.text, size, probe, rotate)
    origin = pymupdf.Point(probe.x + (tx0 + tx1) / 2 - (ink[0] + ink[2]) / 2,
                           probe.y + (ty0 + ty1) / 2 - (ink[1] + ink[3]) / 2)
    return origin, size, rotate


def _write_rgb(pdf: pikepdf.Pdf, xref: int, rgb: np.ndarray) -> None:
    """Replace a picture's pixels losslessly, keeping its object and mask."""
    image = pdf.get_object((xref, 0))
    image.write(zlib.compress(rgb.tobytes(), 6), filter=pikepdf.Name.FlateDecode)
    image.ColorSpace = pikepdf.Name.DeviceRGB
    image.BitsPerComponent = 8
    for key in ("/Decode", "/DecodeParms", "/Intent"):
        if key in image:
            del image[key]


def prepare_mixed_pdf(input_path: Path, output_path: Path, labels: list[PictureLabel]) -> list[PictureLabel]:
    """Write a copy of ``input_path`` with the confirmed lettering made real text.

    Returns the labels that were applied.
    """
    from spejl.erase.clean import erase_text

    chosen = [label for label in labels if label.include and label.text.strip()]
    for label in chosen:
        _rotate_param(label.angle_deg)  # refuse unsupported directions up front

    with TemporaryDirectory(prefix="spejl-mixed-") as temporary:
        erased_path = Path(temporary) / "erased.pdf"
        with pikepdf.open(str(input_path)) as pdf, pymupdf.open(str(input_path)) as doc:
            _placements(doc)  # same placement rules as the review
            for xref in sorted({label.image_xref for label in chosen}):
                rgb = picture_rgb(doc, xref)
                boxes = [label.pixel_box for label in chosen if label.image_xref == xref]
                erased = erase_text(rgb, [tuple(float(v) for v in box) for box in boxes], dilate_px=2).image
                _write_rgb(pdf, xref, np.ascontiguousarray(erased[:, :, :3]))
            pdf.save(str(erased_path))

        with pymupdf.open(str(erased_path)) as doc:
            for label in chosen:
                page = doc[label.page_index]
                origin, size, rotate = _fit_text(page, label)
                page.insert_text(origin, label.text.strip(), fontsize=size,
                                 fontname=_REFERENCE_FONT, rotate=rotate, color=(0, 0, 0))
            doc.save(str(output_path), garbage=3, deflate=True)
    return chosen


def mirror_mixed_pdf(
    input_path: Path,
    output_path: Path,
    axis: Axis,
    labels: list[PictureLabel],
) -> Document:
    """Mirror a mixed PDF after the user has confirmed its picture lettering."""
    from spejl.vector.pdf_mirror import mirror_pdf

    with TemporaryDirectory(prefix="spejl-mixed-") as temporary:
        prepared = Path(temporary) / "prepared.pdf"
        applied = prepare_mixed_pdf(input_path, prepared, labels)
        document = mirror_pdf(prepared, output_path, axis, pictures_as_linework=True)
    document.source = input_path
    for page in document.pages:
        redrawn = [label for label in applied if label.page_index == page.index]
        skipped = [label for label in labels if label.page_index == page.index and not label.include]
        page.flags.append(Flag(
            "picture-linework",
            "Walls and lines on this page are a picture and were mirrored as one. "
            "Any lettering inside it that was not confirmed in the review is mirrored too.",
            "warn",
        ))
        for label in redrawn:
            page.flags.append(Flag(
                "picture-text",
                f"'{label.text.strip()}' was lettering inside the picture; redrawn as real text"
                + (f" (OCR read '{label.ocr_text}')" if label.ocr_text and label.ocr_text != label.text.strip() else "")
                + ".",
                "info",
            ))
        if skipped:
            page.flags.append(Flag(
                "picture-text-skipped",
                f"{len(skipped)} OCR find(s) left as drawing: "
                + ", ".join(f"'{label.ocr_text or label.text}'" for label in skipped) + ".",
                "info",
            ))
    return document
