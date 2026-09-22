"""Turn a plan file into a QPixmap for preview — vector or raster,
whichever route will actually mirror it.

Deliberately not routed through Qt's own image-format plugins for
raster loading: OpenCV is already a hard dependency of the pipeline
itself, so using it here means the preview always reads exactly the
formats the mirror will accept, with no dependency on which optional Qt
imageformats plugin happens to be installed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def bgr_to_qpixmap(image: np.ndarray) -> QPixmap:
    """A cv2 BGR (or grayscale) array, as Qt would draw it."""
    if image.ndim == 2:
        h, w = image.shape
        qimg = QImage(image.data, w, h, image.strides[0], QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())  # copy: detach from numpy's buffer

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def pdf_page_count(path: Path) -> int:
    """Return a PDF's page count for preview navigation only."""
    if path.suffix.lower() != ".pdf":
        return 1
    import pymupdf

    doc = pymupdf.open(str(path))
    try:
        return doc.page_count
    finally:
        doc.close()


def load_preview(path: Path, dpi: int = 150, page_index: int = 0) -> QPixmap | None:
    """Render any Spejl-supported file to a preview pixmap, or None."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pymupdf

        doc = pymupdf.open(str(path))
        try:
            if doc.page_count == 0:
                return None
            zoom = dpi / 72.0
            if not 0 <= page_index < doc.page_count:
                return None
            pix = doc[page_index].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
            fmt = QImage.Format.Format_RGBA8888 if pix.alpha else QImage.Format.Format_RGB888
            qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt)
            return QPixmap.fromImage(qimg.copy())
        finally:
            doc.close()

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return bgr_to_qpixmap(image)
