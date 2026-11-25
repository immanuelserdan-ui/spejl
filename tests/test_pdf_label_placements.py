"""Saved-page regressions for relative text positions and reusable PDF forms."""

import pymupdf
import pytest

from spejl.models import Axis
from spejl.vector.pdf_mirror import mirror_pdf


def labels(page):
    return [
        (span["text"], span["origin"], line["dir"], span["size"])
        for block in page.get_text("dict")["blocks"]
        for line in block["lines"]
        for span in line["spans"]
    ]


# Compare rendered glyphs in baseline-relative windows. Rounding
# the starting pixel can move the mask by a pixel horizontally.
def ink(page, anchor, size, advance):
    rect = pymupdf.Rect(
        anchor[0] - 2, anchor[1] - size, anchor[0] + advance + 2, anchor[1] + size / 3
    )
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=rect, colorspace=pymupdf.csGRAY)
    return {(i % pix.width, i // pix.width) for i, v in enumerate(pix.samples) if v < 150}


@pytest.mark.parametrize("use_form", [False, True], ids=["relative-Td", "nested-reused-form"])
def test_readable_labels_at_reflected_baselines(tmp_path, use_form):
    source = tmp_path / "labels.pdf"
    target = tmp_path / "mirror.pdf"
    with pymupdf.open() as drawing:
        page = drawing.new_page(width=150, height=100)
        page.insert_font(fontname="helv")
        stream = drawing.get_new_xref()
        drawing.update_object(stream, "<<>>")
        drawing.update_stream(
            stream, b"BT /helv 10 Tf 13 70 Td (Room 12) Tj 7 -22 Td [(Level ) 0 (B)] TJ ET"
        )
        page.set_contents(stream)
        if use_form:
            with pymupdf.open() as sheet:
                page = sheet.new_page(width=500, height=350)
                # show_pdf_page creates a wrapper Form around a shared inner Form.
                for rect in [(16, 24, 166, 124), (140, 132, 440, 332)]:
                    page.show_pdf_page(pymupdf.Rect(rect), drawing, 0)
                sheet.save(source)
        else:
            drawing.save(source)

    result = mirror_pdf(source, target, axis=Axis.VERTICAL)
    with pymupdf.open(source) as original, pymupdf.open(target) as mirrored:
        before, after = labels(original[0]), labels(mirrored[0])
        assert [label[0] for label in before] == ["Room 12", "Level B"] * (2 if use_form else 1)
        assert result.pages[0].text_runs_mirrored == len(before)
        assert not mirrored[0].get_images()
        for (text, (x, y), _, size), (actual, origin, direction, out_size) in zip(
            before, after, strict=True
        ):
            advance = pymupdf.get_text_length(text, fontname="helv", fontsize=size)
            assert actual == text
            assert direction == pytest.approx((1, 0))
            assert out_size == pytest.approx(size)
            assert origin == pytest.approx((original[0].rect.width - x - advance, y), abs=0.05)
            src_ink, dst_ink = (
                ink(original[0], (x, y), size, advance),
                ink(mirrored[0], origin, size, advance),
            )
            assert src_ink
            assert max(
                len({(px + dx, py) for px, py in src_ink} & dst_ink) for dx in (-1, 0, 1)
            ) > 0.8 * len(src_ink)
