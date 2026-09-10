"""Shared data contracts.

``models.py`` grows with the pipeline: Phase 1 (Route A, vector mirroring)
only needs :class:`Flag` and :class:`Document`. The richer OCR-facing
``TextRun`` / ``TextStyle`` contract described in the build plan lands in
Phase 2 (Route B) when there is real detection output to shape it around —
guessing those fields now, before a raster pipeline exists to populate
them, would just mean redefining them later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Axis(str, Enum):
    """Mirror axis, named the way an operator asks for it."""

    VERTICAL = "v"      # flip left/right — the common case
    HORIZONTAL = "h"    # flip top/bottom
    BOTH = "both"       # 180° turn


class Route(str, Enum):
    """Which pipeline actually processed the document."""

    VECTOR = "vector"   # Route A — lossless matrix mirroring
    RASTER = "raster"   # Route B — OCR + reconstruction (Phase 2+)


@dataclass(frozen=True)
class Flag:
    """A review-worthy event raised during processing."""

    code: str
    message: str
    severity: str = "info"  # "info" | "warn" | "error"


@dataclass
class PageResult:
    """Per-page bookkeeping for the sidecar report."""

    index: int
    width: float
    height: float
    text_runs_mirrored: int = 0
    flags: list[Flag] = field(default_factory=list)


@dataclass
class Document:
    """One mirror job, source to output, with enough detail to audit it."""

    source: Path
    output: Path
    axis: Axis
    route: Route
    pages: list[PageResult] = field(default_factory=list)

    def to_sidecar(self) -> dict:
        """The ``*.spejl.json`` payload — what changed, and how."""
        return {
            "source": str(self.source),
            "output": str(self.output),
            "axis": self.axis.value,
            "route": self.route.value,
            "spejl_version": __import__("spejl").__version__,
            "pages": [
                {
                    "index": p.index,
                    "width": p.width,
                    "height": p.height,
                    "text_runs_mirrored": p.text_runs_mirrored,
                    "flags": [
                        {"code": f.code, "message": f.message, "severity": f.severity}
                        for f in p.flags
                    ],
                }
                for p in self.pages
            ],
        }
