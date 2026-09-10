"""OCR behind a protocol, so the engine can be swapped in one file.

The build plan's §07 note: the engine *will* change at least once —
Danish diacritic accuracy, packaging weight, or licence terms will force
it. Everything downstream depends on :class:`Detection`, never on a
vendor's return shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One text run as the detector saw it, in source-image pixels."""

    text: str
    quad: tuple[tuple[float, float], ...]  # 4 corners, source coordinates
    conf: float
    angle_deg: float = 0.0  # 0 = left-to-right, 90 = bottom-to-top

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.quad]
        ys = [p[1] for p in self.quad]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)


class OcrBackend(Protocol):
    """Detect-and-recognise over a BGR or grayscale numpy image."""

    def detect_and_recognise(self, image: np.ndarray) -> list[Detection]: ...


class RapidOcrBackend:
    """RapidOCR (PP-OCR models as ONNX) — the packaging-friendly default.

    Chosen over PaddleOCR because it freezes cleanly with PyInstaller and
    installs without ``paddlepaddle``; the recognition models are the same
    PP-OCR weights. A PaddleOcrBackend can implement the same protocol for
    accuracy comparisons without touching any caller.
    """

    def __init__(self, **kwargs) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._ocr = RapidOCR(**kwargs)

    def detect_and_recognise(self, image: np.ndarray) -> list[Detection]:
        raw, _elapse = self._ocr(image)
        if not raw:
            return []
        out: list[Detection] = []
        for quad, text, conf in raw:
            if not str(text).strip():
                continue
            out.append(
                Detection(
                    text=str(text).strip(),
                    quad=tuple((float(p[0]), float(p[1])) for p in quad),
                    conf=float(conf),
                )
            )
        return out
