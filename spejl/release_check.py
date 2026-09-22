"""Opt-in packaged runtime diagnostic; writes only to its requested directory."""

from __future__ import annotations

import json
from pathlib import Path
import traceback


def run(directory: str) -> int:
    output = Path(directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "checks": []}
    try:
        import socket
        import ssl
        import cv2
        import numpy as np
        import pymupdf
        from fontTools.ttLib import TTFont
        from PIL import Image, ImageDraw, ImageFont
        from spejl.detect.ocr import RapidOcrBackend
        from spejl.raster.pipeline import mirror_raster
        from spejl.style.metrics import _FONT_CANDIDATES
        from spejl.vector.pdf_mirror import mirror_pdf

        report["checks"].append("runtime imports")
        source_pdf = output / "sample.pdf"
        result_pdf = output / "sample_mirrored.pdf"
        with pymupdf.open() as doc:
            page = doc.new_page(width=400, height=300)
            page.draw_rect(pymupdf.Rect(25, 25, 240, 260), width=2)
            page.insert_text((65, 130), "Stue 1234", fontsize=20)
            doc.save(source_pdf)
        mirror_pdf(source_pdf, result_pdf)
        with pymupdf.open(result_pdf) as doc:
            if "Stue 1234" not in doc[0].get_text():
                raise RuntimeError("Vector PDF output lost its label.")
        report["checks"].append("vector PDF mirroring")

        font_path = next(path for path in _FONT_CANDIDATES if Path(path).is_file())
        with TTFont(font_path) as font:
            if ord("S") not in font.getBestCmap():
                raise RuntimeError("Font does not contain test glyph.")
        image = Image.new("RGB", (1000, 650), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((50, 50, 800, 580), outline="black", width=3)
        draw.text((170, 240), "Stue 1234", font=ImageFont.truetype(font_path, 48), fill="black")
        source_png = output / "sample.png"
        result_png = output / "sample_mirrored.png"
        image.save(source_png)
        backend = RapidOcrBackend()
        readings = backend.detect_and_recognise(cv2.imread(str(source_png)))
        if not any("1234" in reading.text for reading in readings):
            raise RuntimeError("Bundled OCR could not read the fixture.")
        report["checks"].append("bundled OCR models and fonts")
        mirror_raster(source_png, result_png, backend=backend)
        rendered = cv2.imread(str(result_png))
        if rendered is None or rendered.shape[:2] != (650, 1000):
            raise RuntimeError("Raster output missing or wrong dimensions.")
        if not any("1234" in reading.text for reading in backend.detect_and_recognise(rendered)):
            raise RuntimeError("Mirrored raster label could not be read back.")
        report["checks"].append("raster mirroring and output OCR")
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
    (output / "self-test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["status"] == "passed" else 1
