"""Lightweight GUI regression coverage.

A full click-through (load a real plan, run the threaded mirror,
verify the preview and flags, save the result) is scripted manually in
the build session rather than kept as a slow, environment-sensitive
pytest — offscreen Qt platforms often lack real font/rendering support,
which makes a screenshot-diffing CI test brittle for the wrong reasons.
This file keeps the fast, deterministic part: the window constructs,
wires its signals, and its pure state-transition logic behaves — so a
refactor that breaks the wiring fails immediately, without needing a
real OCR run or a real display.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from spejl.models import Axis  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    from spejl.gui.main_window import MainWindow

    win = MainWindow()
    # Qt's isVisible() reflects actual on-screen visibility, which
    # requires the whole parent chain to have been shown — not just
    # `.show()` called on a child widget. show() the window itself so
    # child-widget visibility assertions behave the way they do for a
    # real running app, matching how MainWindow is actually used.
    win.show()
    yield win
    win.close()


def test_source_and_mirrored_panes_render_at_the_same_scale(window, tmp_path):
    """Regression: a QSplitter does not guarantee its two sides end up
    equal width (confirmed on real hardware: 571px vs 660px from an
    identical setSizes([1, 1]) call), and each preview pane used to
    independently fit its own pixmap to its own box — so the exact same
    drawing, at the exact same pixel dimensions, rendered at two
    different on-screen sizes. Nothing about that was wrong per-widget
    (each pane really did fit itself correctly); it only becomes a bug
    when the two are meant to be compared side by side. The fix makes
    both panes share one externally-computed scale instead of each
    computing its own, so this holds regardless of how the splitter
    ends up sized — including deliberately unequal here, not by
    accident, to prove the fix doesn't depend on the split being equal.
    """
    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    window._on_file_chosen(info["png"])
    # Same pixmap on both sides (source==mirrored dimensions is the
    # common, no-upscale case) with DELIBERATELY unequal pane widths.
    window._mirrored_view.set_pixmap_source(window._source_view._source)
    window._source_view.resize(400, 500)
    window._mirrored_view.resize(650, 500)
    window._sync_preview_scale()

    src_pixmap = window._source_view.pixmap()
    mir_pixmap = window._mirrored_view.pixmap()
    assert src_pixmap is not None and mir_pixmap is not None
    assert src_pixmap.size() == mir_pixmap.size(), (
        f"same drawing rendered at different sizes: "
        f"{src_pixmap.size()} vs {mir_pixmap.size()}"
    )
    # And specifically constrained by the NARROWER pane, not the wider
    # one — the whole point being neither pane independently decides.
    assert src_pixmap.width() <= 400


def test_mirror_button_actually_renders_its_accent_colour(window, tmp_path):
    """Regression for a real bug found in manual testing: the sidebar
    container had its own setStyleSheet() call for its background/
    border, which — silently, with no error and no visible sign beyond
    the widget just not being there — blocks Qt's style-sheet cascade
    from carrying the MAIN WINDOW's ID-selector rules (#mirrorButton,
    #saveButton, #routeBadge) down to that container's children at all.
    The button reported correct geometry, text, and enabled state the
    whole time; only the actual painted pixels were wrong.

    Property checks (isVisible, isEnabled, geometry) do NOT catch this
    class of bug — only sampling real rendered pixels does, which is
    why this test grabs the widget and checks its own paint output
    rather than asking Qt about its logical state.
    """
    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    window._on_file_chosen(info["png"])
    window._mirror_button.setEnabled(True)  # exercise the enabled (accent-blue) paint path
    window.resize(1172, 750)
    window.show()
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()

    pixmap = window._mirror_button.grab()
    image = pixmap.toImage()
    assert image.width() > 0 and image.height() > 0

    # Sample the centre pixel: accent blue (#0A6E8A) if rendering
    # correctly, or the sidebar's plain white/base background if the
    # cascade is broken again and the button silently isn't painting.
    centre = image.pixelColor(image.width() // 2, image.height() // 2)
    is_near_white = centre.red() > 230 and centre.green() > 230 and centre.blue() > 230
    assert not is_near_white, (
        f"mirror button centre pixel is {centre.getRgb()} — looks unpainted, "
        "not the accent-coloured button"
    )
    # And specifically: it should be recognisably the dark teal accent,
    # not just "not white" for some unrelated reason.
    assert centre.blue() > centre.red(), f"expected a blue-leaning accent, got {centre.getRgb()}"


def test_window_constructs_with_expected_widgets(window):
    assert window._mirror_button.isEnabled() is False  # no file chosen yet
    assert window._save_button.isEnabled() is False
    assert window._axis_buttons[Axis.VERTICAL].isChecked()  # sensible default
    assert not window._progress.isVisible()


def test_choosing_a_file_enables_mirror_and_shows_route(window, tmp_path):
    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    window._on_file_chosen(info["png"])

    assert window._input_path == info["png"]
    assert window._mirror_button.isEnabled()
    assert window._route_badge.isVisible()
    assert "Raster" in window._route_badge.text()


def test_choosing_a_pdf_shows_the_vector_route(window, tmp_path):
    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    window._on_file_chosen(info["pdf"])

    assert window._route_badge.isVisible()
    assert "Vector" in window._route_badge.text()


def test_choosing_a_second_file_resets_prior_results(window, tmp_path):
    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    window._save_button.setEnabled(True)  # simulate a completed prior mirror
    window._flags_list.addItem("stale flag from a previous run")

    window._on_file_chosen(info["png"])

    assert window._save_button.isEnabled() is False
    assert window._flags_list.count() == 0
    assert window._mirrored_view._source is None


def test_mirror_failure_is_shown_not_raised(window, tmp_path, qapp):
    """An unreadable file must surface via the failed signal and a
    dialog, never as an unhandled exception on the Qt event loop."""
    from PySide6.QtCore import QEventLoop, QTimer

    bad = tmp_path / "not_a_real_image.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)  # valid magic, garbage body
    window._on_file_chosen(bad)

    loop = QEventLoop()
    outcome = {}

    def on_fail(message: str) -> None:
        outcome["failed"] = message
        loop.quit()

    def on_ok(document, route) -> None:
        outcome["ok"] = True
        loop.quit()

    # Patch QMessageBox so the failure dialog doesn't block the test.
    import spejl.gui.main_window as mw

    original = mw.QMessageBox.critical
    mw.QMessageBox.critical = staticmethod(lambda *a, **k: None)
    try:
        window._on_mirror_clicked()
        window._worker.failed.connect(on_fail)
        window._worker.succeeded.connect(on_ok)
        QTimer.singleShot(15000, loop.quit)
        loop.exec()
    finally:
        mw.QMessageBox.critical = original

    assert "failed" in outcome, outcome
    assert window._mirror_button.isEnabled()  # re-enabled after failure


class _FakeRunningWorker:
    """Stands in for a MirrorWorker that's still executing — real
    thread timing is not something a test should have to win a race
    against to be deterministic; `isRunning()` is the one thing
    _on_mirror_clicked and _on_file_chosen actually check. `wait()` is
    a no-op so the window fixture's own teardown (closeEvent calls
    `self._worker.wait(2000)` for a genuinely running worker) doesn't
    fail on this stand-in once the test itself is done with it."""

    def isRunning(self) -> bool:  # noqa: N802 (matches QThread's own name)
        return True

    def wait(self, _timeout_ms: int) -> bool:
        return True


def test_a_click_while_a_job_is_running_does_not_spawn_a_new_worker(window, tmp_path):
    """Regression: nothing gated _on_mirror_clicked against an in-flight
    job, and _on_file_chosen unconditionally re-enabled the mirror
    button regardless of worker state — so dropping a second file while
    the first was still mirroring, then clicking Mirror again, created a
    SECOND MirrorWorker and overwrote self._worker with it while the
    first was still running in the background. Nothing then held a
    Python reference to that first, still-running QThread — a resource
    leak at best (two OCR pipelines racing for CPU when the user asked
    for one), a PySide6 use-after-free crash at worst if the orphaned
    wrapper got garbage-collected out from under its own live thread.
    """
    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    window._on_file_chosen(info["png"])

    running = _FakeRunningWorker()
    window._worker = running  # type: ignore[assignment]

    # A new file arriving while a job is (per the fake) still running
    # must not re-enable the button.
    window._on_file_chosen(info["png"])
    assert not window._mirror_button.isEnabled()

    # And a click that reaches the handler anyway must refuse to start
    # a second worker rather than replace self._worker.
    window._on_mirror_clicked()
    assert window._worker is running, "a second worker was created while the first was still 'running'"


def test_a_stale_jobs_result_does_not_overwrite_a_different_current_file(window, tmp_path):
    """The other half of the same fix: once a job's own file is no
    longer the one on screen (the user navigated away from it while it
    ran), its result must be silently discarded, not applied over
    whatever the CURRENT file's state is — otherwise a job the user has
    moved on from can still pop its output into view moments later.

    Needs a job that genuinely SUCCEEDS to be a real test of this: the
    bug lived in _on_mirror_succeeded overwriting `_output_path` and the
    preview, and the earlier, failure-only version of this test passed
    even against the pre-fix code, because a FAILED job's old handler
    never touched `_output_path` regardless of staleness — the assertion
    was trivially satisfied by the wrong mechanism. A real vector PDF
    (Route.VECTOR -> mirror_pdf) gets a genuine success fast, with none
    of the OCR-model-loading cost this file otherwise deliberately
    avoids (see its own module docstring).

    Polls `job_worker.isRunning()` via QCoreApplication.processEvents()
    rather than waiting on the worker's OWN succeeded/failed signal:
    that signal is already connected to _on_mirror_clicked's internal
    handlers BEFORE this test ever sees `job_worker`, and a fast vector
    mirror can finish and emit before a SECOND, test-local connection
    made after the fact ever gets attached — a real race, confirmed by
    this exact test hanging its full 15s safety timeout on a job that
    had actually already succeeded in well under a second. Polling
    observable state sidesteps needing to win that race at all.
    """
    from PySide6.QtCore import QCoreApplication

    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    window._on_file_chosen(info["pdf"])
    assert window._route_badge.property("route") == "vector"  # sanity: really Route.VECTOR

    window._on_mirror_clicked()
    job_worker = window._worker
    assert job_worker is not None

    # Simulate the user navigating away from the PDF WHILE that job is
    # still in flight for it — set synchronously, before polling below,
    # so this is deterministic rather than a race against the worker
    # thread's own timing.
    other = tmp_path / "other.pdf"
    other.write_bytes(info["pdf"].read_bytes())
    window._input_path = other

    for _ in range(300):  # up to ~3s
        QCoreApplication.processEvents()
        if not job_worker.isRunning():
            break
        job_worker.wait(10)
    assert not job_worker.isRunning(), "job never finished — test setup itself is broken"
    QCoreApplication.processEvents()  # let the now-queued succeeded/failed signal be delivered

    # The job was for the original PDF; the screen has since moved on
    # to `other`. Its result must not have been applied, but the button
    # must still end up enabled again now that nothing is running.
    assert window._input_path == other
    assert window._mirror_button.isEnabled()
    assert window._output_path is None
