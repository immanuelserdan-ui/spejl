"""Read-only document navigation state for GUI previews.

The document session never participates in mirroring.  It only chooses which
existing PDF page to draw in the preview, so page navigation cannot affect the
engine's output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spejl.gui.imaging import pdf_page_count


@dataclass
class DocumentSession:
    path: Path | None = None
    page_count: int = 0
    page_index: int = 0

    def open(self, path: Path) -> None:
        self.path = path
        self.page_index = 0
        if path.suffix.lower() != ".pdf":
            self.page_count = 1
            return
        try:
            self.page_count = max(pdf_page_count(path), 1)
        except Exception:  # Preview failure is non-fatal; existing worker reports processing errors.
            self.page_count = 1

    @property
    def is_paginated(self) -> bool:
        return self.page_count > 1

    @property
    def label(self) -> str:
        if self.page_count <= 1:
            return "Single page"
        return f"Page {self.page_index + 1} of {self.page_count}"

    def move(self, offset: int) -> bool:
        next_index = min(max(self.page_index + offset, 0), max(self.page_count - 1, 0))
        if next_index == self.page_index:
            return False
        self.page_index = next_index
        return True
