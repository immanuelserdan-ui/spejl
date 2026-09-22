"""Image writes must either create the requested file or raise."""
from pathlib import Path
import cv2
import numpy as np


def write_image(path: Path, image: np.ndarray) -> None:
    try:
        written = cv2.imwrite(str(path), image)
    except cv2.error as exc:
        raise OSError(f"Could not write image: {path}") from exc
    if not written:
        raise OSError(f"Could not write image: {path}")
