"""router.py's format sniffing — extension first, magic bytes as the
fallback for extension-less or unfamiliar files. No prior test coverage
existed for this module at all before this file.
"""

from __future__ import annotations

import pytest

from spejl.models import Route
from spejl.router import sniff_route


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
