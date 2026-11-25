"""Lossless size regressions: compare reopened bytes, not save arguments."""

from pathlib import Path
from types import SimpleNamespace
import shutil

import pymupdf
import pytest

from spejl.models import Axis
from spejl.vector import pdf_mirror
from spejl.vector.pdf_save import save_compact_pdf


def assert_same_pdf(before, after):
    with pymupdf.open(before) as a, pymupdf.open(after) as b:
        assert len(a) == len(b)
        assert a.metadata == b.metadata
        for old, new in zip(a, b):
            for attr in ('rect', 'mediabox', 'cropbox', 'trimbox', 'bleedbox', 'artbox', 'rotation'):
                assert getattr(old, attr) == getattr(new, attr)
            assert old.get_text('words') == new.get_text('words')
            assert old.get_drawings() == new.get_drawings()
            # Print-resolution pixels must agree exactly, including colour.
            x, y = old.get_pixmap(dpi=300, alpha=True), new.get_pixmap(dpi=300, alpha=True)
            assert (x.width, x.height, x.samples) == (y.width, y.height, y.samples)
            for left, right in zip(old.get_images(full=True), new.get_images(full=True)):
                assert a.extract_image(left[0])['image'] == b.extract_image(right[0])['image']


def make_image_pdf(path, pages=1):
    with pymupdf.open() as drawing:
        page = drawing.new_page(width=400, height=300)
        page.draw_rect((20, 20, 370, 280), color=(0, 0, 0), width=0.25)
        page.draw_line((35, 70), (365, 70), color=(1, 0, 0), width=0.2)
        page.insert_text((50, 150), 'Stue 3200 mm', fontsize=10)
        png = page.get_pixmap(dpi=300, alpha=False).tobytes('png')
    with pymupdf.open() as pdf:
        for _ in range(pages):
            page = pdf.new_page(width=400, height=300)
            page.insert_image(page.rect, stream=png)
            page.insert_text((30, 290), 'Scale 1:100', fontsize=8)
        pdf.save(path)


def fake_raster(source, output, **kwargs):
    # Isolate PDF serialization from OCR; existing raster tests cover mirroring.
    shutil.copyfile(source, output)
    return SimpleNamespace(runs=[], document=SimpleNamespace(pages=[SimpleNamespace(flags=[])]))


def test_compaction_keeps_pixels_text_scale_and_page_settings(tmp_path):
    raw, compact = tmp_path/'raw.pdf', tmp_path/'compact.pdf'
    make_image_pdf(raw, pages=2)
    with pymupdf.open(raw) as doc:
        doc.set_metadata({'title': 'Drawing scale 1:100'})
        for page in doc:
            page.set_cropbox(pymupdf.Rect(10, 10, 390, 295))
            page.set_rotation(90)
            doc.xref_set_key(page.xref, 'UserUnit', '2')
        raw2 = tmp_path/'configured.pdf'
        doc.save(raw2)
        save_compact_pdf(doc, compact)
    assert raw2.stat().st_size > 1_000_000
    assert compact.stat().st_size < 100_000
    assert_same_pdf(raw2, compact)


@pytest.mark.parametrize('axis', list(Axis))
def test_vector_mirror_matches_old_save_with_embedded_font(tmp_path, monkeypatch, axis):
    source, old, compact = (tmp_path/n for n in ('source.pdf', 'old.pdf', 'compact.pdf'))
    with pymupdf.open() as doc:
        for _ in range(2):
            page = doc.new_page(width=400, height=300)
            writer = pymupdf.TextWriter(page.rect)
            writer.append((80, 120), 'Stue 3200', fontsize=12)
            writer.write_text(page)
            page.draw_rect((20, 20, 350, 280), width=0.25)
        doc.save(source)
    with monkeypatch.context() as patch:
        patch.setattr(pdf_mirror, 'save_compact_pdf', lambda doc, path: doc.save(path))
        pdf_mirror.mirror_pdf(source, old, axis)
    pdf_mirror.mirror_pdf(source, compact, axis)
    assert compact.stat().st_size < old.stat().st_size * 0.85
    assert_same_pdf(old, compact)


def test_image_route_compresses_full_resolution_pixels(tmp_path, monkeypatch):
    from spejl.raster import pipeline
    monkeypatch.setattr(pipeline, 'mirror_raster', fake_raster)
    source, old, compact = (tmp_path/n for n in ('source.pdf', 'old.pdf', 'compact.pdf'))
    make_image_pdf(source)
    with monkeypatch.context() as patch:
        patch.setattr(pdf_mirror, 'save_compact_pdf', lambda doc, path: doc.save(path))
        pdf_mirror.mirror_pdf(source, old)
    pdf_mirror.mirror_pdf(source, compact)
    assert old.stat().st_size > 1_000_000
    assert compact.stat().st_size < 100_000
    assert_same_pdf(old, compact)


def test_blank_page_fallback_preserves_other_pages_text_and_geometry(tmp_path, monkeypatch):
    from spejl.raster import pipeline
    source, expected, output = (tmp_path/n for n in ('source.pdf', 'expected.pdf', 'output.pdf'))
    with pymupdf.open() as doc:
        for index in range(3):
            page = doc.new_page(width=400+index*10, height=300)
            page.insert_text((80, 120), f'Page {index} 3200', fontsize=12)
            page.draw_rect((20, 20, 350, 280), width=0.25)
        doc.save(source)
    pdf_mirror.mirror_pdf(source, expected)
    seen = []
    def raster(source, output, **kwargs):
        seen.append(source.name)
        return fake_raster(source, output, **kwargs)
    monkeypatch.setattr(pipeline, 'mirror_raster', raster)
    checks = iter([True, False, True])
    monkeypatch.setattr(pdf_mirror, '_pixmap_has_ink', lambda pix: next(checks))
    result = pdf_mirror.mirror_pdf(source, output)
    assert seen == ['source-1.png']
    assert [p.index for p in result.pages] == [0, 1, 2]
    assert [bool(p.flags) for p in result.pages] == [False, True, False]
    assert output.stat().st_size < 100_000
    with pymupdf.open(expected) as old, pymupdf.open(output) as new:
        for index in (0, 2):
            assert old[index].get_text('words') == new[index].get_text('words')
            assert old[index].get_drawings() == new[index].get_drawings()
            assert old[index].get_pixmap(dpi=300).samples == new[index].get_pixmap(dpi=300).samples
            assert new[index].get_images() == []
        assert all(a.rect == b.rect for a, b in zip(old, new))
