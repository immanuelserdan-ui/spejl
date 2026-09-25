"""Format sniffing: decide Route A (vector) vs Route B (raster) *before*
touching pixels or OCR.

The decision is cheap and the payoff is large — see the build plan, §01.
Vector-native input (PDF today; SVG/AI can extend this later) always wins,
because it mirrors the actual glyph runs rather than reconstructing them.
"""

from __future__ import annotations

from pathlib import Path

from spejl.models import Route

_VECTOR_SUFFIXES = {".pdf"}
_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def sniff_route(path: Path) -> Route:
    """Pick a route from the file's extension (and, for ambiguous cases,
    its magic bytes) — never from a guess about what's inside it.
    """
    suffix = path.suffix.lower()
    if suffix in _VECTOR_SUFFIXES:
        return Route.VECTOR
    if suffix in _RASTER_SUFFIXES:
        return Route.RASTER

    # Extension-less or unfamiliar: fall back to magic bytes.
    with open(path, "rb") as f:
        # ISO 32000 permits the %PDF- header anywhere in the first 1024
        # bytes (to tolerate leading junk some generators/transports
        # prepend), not only at offset 0.
        head = f.read(1024)
    if b"%PDF-" in head:
        return Route.VECTOR
    if (
        head.startswith(b"\x89PNG")
        or head.startswith(b"\xff\xd8")
        or head.startswith(b"II*\x00")  # TIFF, little-endian
        or head.startswith(b"MM\x00*")  # TIFF, big-endian
        or head.startswith(b"BM")  # BMP
    ):
        return Route.RASTER

    raise ValueError(
        f"{path}: unrecognised format — expected a vector PDF or a raster "
        "image (PNG/JPEG/TIFF/BMP)."
    )


def validate_native_vector_pdf(path: Path) -> None:
    """Reject raster and image-bearing PDFs in the desktop editor.

    A PDF can contain full-page scans or a mixture of raster images and
    vector text. Spejl's general-purpose mirror engine can process those
    through its raster fallback, but that removes selectable text. The
    desktop editing workflow calls this before mirroring so its uploaded
    plans remain eligible for vector text editing.
    """
    if path.suffix.lower() != ".pdf":
        raise ValueError("Spejl accepts vector PDF files only.")

    import pymupdf

    try:
        with pymupdf.open(str(path)) as document:
            if document.page_count == 0:
                raise ValueError(f"{path.name}: the PDF has no pages.")
            for page_index, page in enumerate(document):
                if page.get_image_info():
                    raise ValueError(
                        f"{path.name}, page {page_index + 1}: contains embedded image content. "
                        "Scanned and mixed image/vector PDFs cannot retain editable text; "
                        "export an image-free vector PDF and try again."
                    )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{path.name}: could not read as a valid PDF ({exc}).") from exc
