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
        head = f.read(8)
    if head.startswith(b"%PDF-"):
        return Route.VECTOR
    if head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8"):
        return Route.RASTER

    raise ValueError(
        f"{path}: unrecognised format — expected a vector PDF or a raster "
        "image (PNG/JPEG/TIFF/BMP)."
    )
