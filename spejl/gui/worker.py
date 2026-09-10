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
