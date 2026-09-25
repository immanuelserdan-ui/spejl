"""router.py's format sniffing — extension first, magic bytes as the
fallback for extension-less or unfamiliar files. No prior test coverage
existed for this module at all before this file.
"""

from __future__ import annotations

import pytest

from spejl.models import Route
from spejl.router import sniff_route, validate_native_vector_pdf


def test_pdf_suffix_routes_to_vector(tmp_path):
    path = tmp_path / "plan.pdf"
    path.write_bytes(b"%PDF-1.4\n%%EOF")
    assert sniff_route(path) is Route.VECTOR


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"])
def test_known_raster_suffixes_route_to_raster(tmp_path, suffix):
    path = tmp_path / f"plan{suffix}"
    path.write_bytes(b"not a real image, but the suffix alone decides")
    assert sniff_route(path) is Route.RASTER


def test_extensionless_pdf_is_recognised_by_magic_bytes(tmp_path):
    path = tmp_path / "plan"
    path.write_bytes(b"%PDF-1.7\n%%EOF")
    assert sniff_route(path) is Route.VECTOR


def test_extensionless_png_is_recognised_by_magic_bytes(tmp_path):
    path = tmp_path / "plan"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    assert sniff_route(path) is Route.RASTER


def test_unrecognised_extensionless_file_is_reported_not_guessed(tmp_path):
    path = tmp_path / "plan"
    path.write_bytes(b"just some plain text, no known magic bytes here")
    with pytest.raises(ValueError, match="unrecognised format"):
        sniff_route(path)


def test_native_vector_pdf_passes_desktop_validation(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "vector-plan.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 50), "Room 1")
    doc.save(path)
    doc.close()

    validate_native_vector_pdf(path)


def test_pdf_with_embedded_image_is_rejected_for_desktop_editing(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "mixed-plan.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2, 2), 0)
    image.clear_with(255)
    page.insert_image(pymupdf.Rect(30, 30, 100, 100), pixmap=image)
    page.insert_text((40, 120), "Vector label")
    doc.save(path)
    doc.close()

    with pytest.raises(ValueError, match="embedded image content"):
        validate_native_vector_pdf(path)


def test_raster_file_is_rejected_by_desktop_validation(tmp_path):
    path = tmp_path / "plan.png"
    path.write_bytes(b"not relevant")

    with pytest.raises(ValueError, match="vector PDF files only"):
        validate_native_vector_pdf(path)
