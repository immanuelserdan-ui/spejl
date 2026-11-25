"""Size-limited, print-scale JPEG export of a single floor-plan page."""

from io import BytesIO
import math

from PIL import Image, ImageChops, ImageOps
import pymupdf


from spejl.gui.jpeg_footer import with_footer


_LIMIT = 1_000_000  # Decimal MB, strictly less than the limit.
_PRINT_DPI = 300
_MARGINS = (18, 14, 18, 12)  # Afd 40 at print resolution.


def _minimum_dpi(page):
    # Keep at least 12 pixels per em when reducing native labels. Outlined
    # and image labels have no font metrics, so also enforce a 200 dpi floor.
    sizes = (span["size"] for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT
                                                      & ~pymupdf.TEXT_PRESERVE_IMAGES)["blocks"]
             for line in block.get("lines", ()) for span in line["spans"]
             if span["text"].strip() and span["size"] > 0)
    smallest = min(sizes, default=math.inf)
    return min(_PRINT_DPI, max(200, math.ceil(12 * 72 / smallest)))


def encode_plan_jpeg(page, quality, *, include_footer=False):
    """Return a complete under-limit JPEG, or fail before creating a file."""
    pix = page.get_pixmap(dpi=_PRINT_DPI, alpha=False)
    sheet = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    del pix
    bounds = ImageChops.invert(sheet).getbbox()
    content = sheet.crop(bounds) if bounds is not None else sheet
    del sheet
    minimum = _minimum_dpi(page)
    for dpi in dict.fromkeys([300, *(d for d in (275, 250, 225) if d > minimum), minimum]):
        scale = dpi / _PRINT_DPI
        image = content
        if dpi != _PRINT_DPI:
            size = tuple(max(1, round(n * scale)) for n in content.size)
            image = content.resize(size, Image.Resampling.LANCZOS)
        if bounds is not None:
            image = ImageOps.expand(image, tuple(round(n * scale) for n in _MARGINS), "white")
        if include_footer:
            image = with_footer(image, dpi, round(_MARGINS[3] * scale) if bounds else 0)
        # The normal encoding is deliberately first: already-small plans stay
        # byte-identical. Huffman optimization changes no decoded pixels.
        settings = [(quality, False)] if dpi == _PRINT_DPI else []
        settings.extend((q, True) for q in (quality, 90, 85, 80, 75))
        for q, optimize in settings:
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=q, optimize=optimize,
                       subsampling=0, dpi=(dpi, dpi))
            if buffer.tell() < _LIMIT:
                return buffer.getvalue()
    raise ValueError(f"Page {page.number + 1} cannot fit under 1 MB at the label-detail "
                     f"settings ({minimum} dpi minimum, JPEG quality 75 minimum).")
