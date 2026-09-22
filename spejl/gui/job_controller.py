"""UI-only orchestration for a protected mirror job.

This module intentionally does not inspect or transform document pixels, PDF
content, OCR results, or engine options.  Its job is limited to presenting
truthful lifecycle state around the existing :class:`MirrorWorker` call.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from spejl.gui.worker import MirrorWorker
from spejl.models import Axis


class MirrorJobController(QObject):
    """Own one mirror worker and expose safe, presentation-level state.

    Cancellation is deliberately cooperative: the protected engine is never
    interrupted midway through writing a document.  A cancel request prevents
    the completed temporary result from being presented, then returns control
    to the user as soon as the current engine call ends.
    """

    stage_changed = Signal(str)
    cancel_available_changed = Signal(bool)
    cancelled = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.worker: MirrorWorker | None = None
        self._cancel_requested = False

    @property
    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def start(self, input_path: Path, output_path: Path, axis: Axis) -> MirrorWorker:
        if self.is_running:
            raise RuntimeError("A mirror job is already running.")
        self._cancel_requested = False
        self.stage_changed.emit("Preparing plan…")
        self.worker = MirrorWorker(input_path, output_path, axis)
        self.worker.started.connect(self._on_worker_started)
        self.worker.succeeded.connect(self._on_worker_succeeded)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.start()
        return self.worker

    def request_cancel(self) -> None:
        if not self.is_running or self._cancel_requested:
            return
        self._cancel_requested = True
        self.cancel_available_changed.emit(False)
        self.stage_changed.emit("Finishing the current operation safely…")

    def _on_worker_started(self) -> None:
        self.stage_changed.emit("Mirroring plan…")
        self.cancel_available_changed.emit(True)

    def _on_worker_succeeded(self, *_unused: object) -> None:
        self.cancel_available_changed.emit(False)
        if self._cancel_requested:
            self.stage_changed.emit("Mirror cancelled. No result was opened.")
            self.cancelled.emit()
        else:
            self.stage_changed.emit("Validating mirrored result…")

    def _on_worker_failed(self, *_unused: object) -> None:
        self.cancel_available_changed.emit(False)
