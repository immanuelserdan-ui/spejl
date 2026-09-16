"""Run a mirror job off the UI thread.

OCR-backed raster mirroring takes a few seconds — long enough that
doing it on Qt's main thread would freeze the window for the whole
job. The worker is a thin QThread wrapper around exactly the functions
the CLI calls (``mirror_pdf`` / ``mirror_raster``); it adds no pipeline
logic of its own.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from spejl.models import Axis, Document, Route
from spejl.qa.verify import VerifyReport
from spejl.router import sniff_route


class MirrorWorker(QThread):
    succeeded = Signal(object, object)  # Document, Route
    failed = Signal(str)

    def __init__(self, input_path: Path, output_path: Path, axis: Axis) -> None:
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.axis = axis

    def run(self) -> None:
        try:
            route = sniff_route(self.input_path)
        except ValueError as exc:
            self.failed.emit(str(exc))
            return

        try:
            document: Document
            if route is Route.VECTOR:
                from spejl.vector.pdf_mirror import mirror_pdf

                document = mirror_pdf(self.input_path, self.output_path, axis=self.axis)
            else:
                from spejl.raster.pipeline import mirror_raster

                document = mirror_raster(
                    self.input_path, self.output_path, axis=self.axis
                ).document
        except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
            self.failed.emit(str(exc))
            return

        self.succeeded.emit(document, route)


class VerifyWorker(QThread):
    """Runs qa/verify.py's own OCR-backed check off the UI thread, for
    the same reason MirrorWorker does — it re-detects text on the
    source image, which costs the same few seconds a mirror job's own
    detection pass does."""

    succeeded = Signal(object, object)  # VerifyReport, overlay Path
    failed = Signal(str)

    def __init__(self, source_path: Path, output_path: Path, axis: Axis, overlay_path: Path) -> None:
        super().__init__()
        self.source_path = source_path
        self.output_path = output_path
        self.axis = axis
        self.overlay_path = overlay_path

    def run(self) -> None:
        from spejl.qa.verify import render_diff_overlay, verify_mirror

        try:
            report: VerifyReport = verify_mirror(self.source_path, self.output_path, self.axis)
            # Rendered unconditionally, pass or fail — seeing that the
            # only red is where text sits is exactly what makes a
            # PASSING result trustworthy rather than just asserted.
            render_diff_overlay(self.source_path, self.output_path, self.axis, self.overlay_path)
        except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
            self.failed.emit(str(exc))
            return

        self.succeeded.emit(report, self.overlay_path)
