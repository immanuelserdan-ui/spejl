"""State and filename rules for the desktop app's multi-plan workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re

from spejl.models import Document, Route


_ORIENTATION_FIELD = re.compile(r"(?<=-)([RS])(?=-V\d+(?:-|$))")


def mirrored_filename(source: Path) -> tuple[str, bool]:
    """Return the counterpart name and whether the R/S field was missing.

    Only the standalone orientation field immediately before the V-number is
    changed. A trailing revision such as R00 is therefore never modified.
    """
    stem, replacements = _ORIENTATION_FIELD.subn(
        lambda match: "S" if match.group(1) == "R" else "R",
        source.stem,
        count=1,
    )
    if replacements:
        return stem + source.suffix, False
    return source.stem + "_mirrored" + source.suffix, True


@dataclass
class BatchEntry:
    source: Path
    mirrored_name: str
    naming_warning: bool = False
    display_name: str | None = None
    uploaded_at: datetime = field(default_factory=datetime.now)
    status: str = "queued"
    output: Path | None = None
    document: Document | None = None
    route: Route | None = None
    error: str | None = None
    page_index: int = 0
    diff_overlay: Path | None = None
    overlap_findings: list[tuple[int, str]] = field(default_factory=list)
    overlaps_checked: bool = False
