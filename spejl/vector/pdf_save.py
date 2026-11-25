"""Lossless serialization shared by PDF creation and editing paths."""

from pathlib import Path

import pymupdf


def save_compact_pdf(document: pymupdf.Document, output_path: Path) -> None:
    """Remove unreachable/duplicate objects and compress streams losslessly.

    No rendering, resampling, font substitution, page-box edits, or JPEG
    conversion takes place here. Existing drawing coordinates and PDF text
    remain intact; image pixels and font programs retain their original data.
    """
    document.save(
        str(output_path), garbage=4, deflate=True,
        deflate_images=True, deflate_fonts=True,
    )
