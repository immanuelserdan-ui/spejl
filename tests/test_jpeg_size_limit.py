from io import BytesIO

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageChops, ImageOps

from spejl.gui.jpeg_export import encode_plan_jpeg


@pytest.mark.parametrize(("quality", "footer"), [(95, False), (97, True)])
def test_large_page_keeps_label_detail_and_physical_margins(quality, footer):
    # Dense grey linework forces the encoder past its normal 300 dpi settings.
    with pymupdf.open() as doc:
        page = doc.new_page(width=500, height=550)
        pixels = np.random.default_rng(18).integers(0, 256, (2000, 2000), dtype=np.uint8)
        texture = BytesIO()
        Image.fromarray(pixels).save(texture, format="PNG")
        page.insert_image((10, 10, 490, 490), stream=texture.getvalue())
        page.insert_text((20, 520), "ROOM 204", fontsize=10)
        pix = page.get_pixmap(dpi=300, alpha=False)
        sheet = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        bounds = ImageChops.invert(sheet).getbbox()
        content = sheet.crop(bounds)
        normal = BytesIO()
        content.save(normal, format="JPEG", quality=quality, subsampling=0)
        assert len(normal.getvalue()) > 1_000_000
        data = encode_plan_jpeg(page, quality, include_footer=footer)
        assert len(data) < 1_000_000
        with Image.open(BytesIO(data)) as result:
            dpi = result.info["dpi"][0]
            assert 200 <= dpi < 300
            scale = dpi / 300
            target = content.resize(tuple(round(n * scale) for n in content.size),
                                    Image.Resampling.LANCZOS)
            reference = ImageOps.expand(target, tuple(round(n * scale) for n in (18, 14, 18, 12)),
                                        "white")
            if footer:
                from spejl.gui.jpeg_footer import with_footer
                reference = with_footer(reference, dpi, round(12 * scale))
            assert result.size == reference.size
            # Check the actual label against the uncompressed rendering at
            # output resolution, including stroke placement and contrast.
            label_box = (0, round((505-10)*dpi/72), round(100*dpi/72), result.height)
            expected = np.asarray(reference.crop(label_box).convert("L")) < 128
            observed = np.asarray(result.crop(label_box).convert("L")) < 128
            assert expected.sum() > 100
            assert (expected & observed).sum() / (expected | observed).sum() > .95


def test_rejects_oversize_page_without_shrinking_tiny_labels(monkeypatch):
    from spejl.gui import jpeg_export
    monkeypatch.setattr(jpeg_export, "_LIMIT", 100)
    with pymupdf.open() as doc:
        page = doc.new_page(width=100, height=60)
        page.insert_text((10, 30), "Small dimension", fontsize=2)
        with pytest.raises(ValueError, match="300 dpi minimum"):
            encode_plan_jpeg(page, 97)


@pytest.mark.parametrize(("size", "height", "letters"), [
    ((3043, 2382), 2547, (2286, 94, 3025, 146)),
    ((1562, 2573), 2676, (1170, 62, 1549, 89)),
])
def test_footer_matches_approved_proofs(size, height, letters):
    from spejl.gui.jpeg_footer import with_footer
    plan = Image.new("RGB", size, "white")
    image = with_footer(plan)
    assert image.size == (size[0], height)
    strip = image.crop((0, size[1], image.width, image.height))
    assert strip.convert("L").point(lambda v: 255 if v < 170 else 0).getbbox() == letters
