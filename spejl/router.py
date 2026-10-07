"""Format sniffing: decide Route A (vector) vs Route B (raster) *before*
touching pixels or OCR.

The decision is cheap and the payoff is large — see the build plan, §01.
Vector-native input (PDF today; SVG/AI can extend this later) always wins,
because it mirrors the actual glyph runs rather than reconstructing them.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class InputProblem:
    """Why the desktop editor cannot mirror an uploaded file.

    ``badge`` is a few words for the source-pane badge and list markers;
    ``reason`` is the actionable explanation, and ``where`` names the file
    (and page) it applies to. ``kind`` is ``"mixed-image"`` for vector text
    over embedded pictures, which assisted mirroring can handle.
    """

    badge: str
    where: str
    reason: str
    kind: str = "other"

    @property
    def detail(self) -> str:
        return f"{self.where}: {self.reason}"


def native_vector_pdf_problem(path: Path) -> InputProblem | None:
    """Return why ``path`` cannot be mirrored as a native vector PDF, or None.

    A PDF can contain full-page scans or a mixture of raster images and
    vector text. Spejl's general-purpose mirror engine can process those
    through its raster fallback, but that removes selectable text and would
    reverse any lettering drawn inside the images. The desktop editor checks
    this on upload and again before mirroring, so plans are only accepted
    when they remain eligible for vector text editing.
    """
    if path.suffix.lower() != ".pdf":
        return InputProblem("Not a PDF", path.name, "Spejl accepts vector PDF files only.")

    import pymupdf

    try:
        with pymupdf.open(str(path)) as document:
            if document.page_count == 0:
                return InputProblem("Empty PDF", path.name, "the PDF has no pages.")
            for page_index, page in enumerate(document):
                images = page.get_image_info()
                if not images:
                    continue
                covered = sum(
                    (pymupdf.Rect(image["bbox"]) & page.rect).get_area() for image in images
                )
                share = min(covered / max(page.rect.get_area(), 1e-9), 1.0)
                pictures = f"{len(images)} picture{'s' if len(images) != 1 else ''}"
                return InputProblem(
                    "Mixed image PDF — re-export as vector",
                    f"{path.name}, page {page_index + 1}",
                    f"contains embedded image content ({pictures} covering {share:.0%} of the page) — "
                    + (
                        "the walls and linework are pictures, not vector lines. "
                        if share >= 0.5 else
                        "probably a logo or underlay; hide it. "
                    )
                    + "Re-export with Vector processing (in Revit, also avoid "
                    "transparency, shadows and shaded views).",
                    kind="mixed-image",
                )
    except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
        return InputProblem("Unreadable PDF", path.name, f"could not read as a valid PDF ({exc}).")
    return None


def validate_native_vector_pdf(path: Path) -> None:
    """Raise ``ValueError`` with :func:`native_vector_pdf_problem`'s detail."""
    problem = native_vector_pdf_problem(path)
    if problem is not None:
        raise ValueError(problem.detail)
