"""Batch upload, counterpart naming, and selection behavior."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spejl.gui.batch_model import mirrored_filename


def test_desktop_upload_accepts_pdf_only(tmp_path):
    from spejl.gui.widgets import DropZone, SUPPORTED_SUFFIXES

    pdf = tmp_path / "vector.pdf"
    image = tmp_path / "scan.png"
    pdf.touch()
    image.touch()

    class Url:
        def __init__(self, path):
            self.path = path

        def toLocalFile(self):
            return str(self.path)

    class Mime:
        def urls(self):
            return [Url(pdf), Url(image)]

    class Event:
        def mimeData(self):
            return Mime()

    assert SUPPORTED_SUFFIXES == (".pdf",)
    assert DropZone._supported_paths(Event()) == [pdf]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("634-0001-0044-4418-T04-A00-S-V00-R00.pdf", "634-0001-0044-4418-T04-A00-R-V00-R00.pdf"),
        ("634-0001-0044-4426-T01-A00-R-V00-R00.pdf", "634-0001-0044-4426-T01-A00-S-V00-R00.pdf"),
        ("634-0001-0079-0035-T04-A00-R-V20-R00.png", "634-0001-0079-0035-T04-A00-S-V20-R00.png"),
    ],
)
def test_counterpart_name_swaps_only_orientation_field(source, expected):
    name, warning = mirrored_filename(Path(source))
    assert name == expected
    assert warning is False


def test_counterpart_name_does_not_modify_revision_code():
    name, warning = mirrored_filename(Path("634-T04-A00-V20-R00.pdf"))
    assert name == "634-T04-A00-V20-R00_mirrored.pdf"
    assert warning is True


def test_multiple_files_are_listed_and_selectable(qapp, tmp_path):
    from spejl.gui.main_window import MainWindow
    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    first = tmp_path / "634-T01-A00-R-V00-R00.pdf"
    second = tmp_path / "634-T02-A00-S-V20-R00.pdf"
    first.write_bytes(info["pdf"].read_bytes())
    second.write_bytes(info["pdf"].read_bytes())
    window = MainWindow()
    try:
        window._on_files_chosen([first, second, first])
        assert window._uploaded_list.count() == 2
        assert window._mirrored_list.count() == 2
        assert "-S-V00-" in window._mirrored_list.item(0).text()
        assert "-R-V20-" in window._mirrored_list.item(1).text()
        window._uploaded_list.setCurrentRow(1)
        assert window._input_path == second.resolve()
        assert window._source_filename_label.text() == second.name
        assert window._mirrored_list.currentRow() == 1
        window._mirrored_list.setCurrentRow(0)
        assert window._input_path == first.resolve()
        assert window._uploaded_list.currentRow() == 0
        assert not window._flags_list.isVisible()
        assert not window._verify_button.isVisible()
        assert not window._view_diff_button.isVisible()
        assert not window._save_button.isHidden()
    finally:
        window.close()


def test_page_navigation_and_text_selection_follow_selected_file(qapp, tmp_path):
    import pymupdf

    from spejl.gui.main_window import MainWindow
    from spejl.router import Route

    source = tmp_path / "634-T01-A00-R-V00-R00.pdf"
    output = tmp_path / "634-T01-A00-S-V00-R00.pdf"
    for destination, prefix in ((source, "Source"), (output, "Mirrored")):
        doc = pymupdf.open()
        for index, width in enumerate((400, 600)):
            page = doc.new_page(width=width, height=300)
            page.insert_text((40, 70), f"{prefix} page {index + 1}", fontsize=18)
        doc.save(destination)
        doc.close()

    window = MainWindow()
    try:
        window._on_file_chosen(source)
        entry = window._batch_entries[source.resolve()]
        entry.output = output
        entry.status = "completed"
        entry.route = Route.VECTOR
        window._select_batch_entry(source.resolve())
        assert window._page_label.text() == "Page 1 of 2"
        assert window._mirrored_view._text_regions

        window._next_page_button.click()
        assert window._page_label.text() == "Page 2 of 2"
        assert entry.page_index == 1
        assert window._mirrored_view._text_regions
        window._previous_page_button.click()
        assert window._page_label.text() == "Page 1 of 2"
        assert entry.page_index == 0
    finally:
        window.close()


def test_page_controls_navigate_between_single_page_uploaded_plans(qapp, tmp_path):
    import pymupdf

    from spejl.gui.main_window import MainWindow

    paths = [tmp_path / f"Plan-{index}-R-V00.pdf" for index in (1, 2, 3)]
    for index, path in enumerate(paths, start=1):
        doc = pymupdf.open()
        page = doc.new_page(width=400, height=300)
        page.insert_text((40, 70), f"Plan {index}", fontsize=18)
        doc.save(path)
        doc.close()

    window = MainWindow()
    try:
        window._on_files_chosen(paths)
        assert window._page_label.text() == "Plan 1 of 3"
        assert not window._previous_page_button.isEnabled()
        assert window._next_page_button.isEnabled()

        window._next_page_button.click()
        assert window._input_path == paths[1].resolve()
        assert window._page_label.text() == "Plan 2 of 3"
        assert window._uploaded_list.currentRow() == 1
        assert window._mirrored_list.currentRow() == 1

        window._previous_page_button.click()
        assert window._input_path == paths[0].resolve()
        assert window._page_label.text() == "Plan 1 of 3"
    finally:
        window.close()


def test_overlap_findings_are_shown_in_status_area(qapp, tmp_path):
    import pymupdf

    from spejl.gui.main_window import MainWindow
    from spejl.models import Route

    source = tmp_path / "634-T01-A00-R-V00-R00.pdf"
    output = tmp_path / "634-T01-A00-S-V00-R00.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    for x in (100, 150, 200):
        page.draw_line((x, 40), (x, 260), color=(0, 0, 0), width=2)
    for x, y, label in (
        (94, 90, "First overlap"),
        (144, 150, "Second overlap"),
        (194, 210, "Third overlap"),
    ):
        page.insert_text((x, y), label, fontsize=12)
    doc.save(source)
    doc.save(output)
    doc.close()

    window = MainWindow()
    try:
        window._on_file_chosen(source)
        entry = window._batch_entries[source.resolve()]
        entry.output = output
        entry.route = Route.VECTOR
        entry.status = "completed"
        window._output_path = output
        window._set_status_with_overlap("Plan ready.")
        assert "3 text/drawing overlap" in window._status_label.text()
        assert "First overlap" in window._status_label.text()
        assert "p.1: 'Third overlap'" in window._status_label.toolTip()
    finally:
        window.close()


def test_resolved_overlap_disappears_live_after_text_nudge(qapp, tmp_path):
    """Refreshing after an edit removes only the overlap the edit fixed."""
    import pymupdf

    from spejl.gui.main_window import MainWindow
    from spejl.models import Axis, Document, Route

    source = tmp_path / "634-T01-A00-R-V00-R00.pdf"
    output = tmp_path / "634-T01-A00-S-V00-R00.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.draw_line((100, 40), (100, 260), color=(0, 0, 0), width=2)
    page.draw_line((250, 40), (250, 260), color=(0, 0, 0), width=2)
    page.insert_text((94, 90), "Resolve me", fontsize=12)
    page.insert_text((244, 180), "Keep me", fontsize=12)
    doc.save(source)
    doc.close()
    output.write_bytes(source.read_bytes())

    window = MainWindow()
    try:
        window._on_file_chosen(source)
        entry = window._batch_entries[source.resolve()]
        entry.output = output
        entry.document = Document(source, output, Axis.VERTICAL, Route.VECTOR)
        entry.route = Route.VECTOR
        entry.status = "completed"
        window._select_batch_entry(source.resolve())

        window._set_status_with_overlap("Plan ready.", refresh=True)
        assert entry.overlap_findings == [(1, "Resolve me"), (1, "Keep me")]

        # The first visible text run moves left, clear of its vertical wall.
        window._on_preview_text_nudged([0], right=100, down=0, snap_to_text=False)

        assert entry.overlap_findings == [(1, "Keep me")]
        assert "1 text/drawing overlap" in window._status_label.text()
        assert "Resolve me" not in window._status_label.toolTip()
        assert "Keep me" in window._status_label.toolTip()
    finally:
        window.close()


def test_resolve_button_moves_every_flagged_run_three_points_right(qapp, tmp_path):
    import pymupdf

    from spejl.gui.main_window import MainWindow
    from spejl.models import Axis, Document, Route
    from spejl.vector.pdf_mirror import text_drawing_overlap_runs

    source = tmp_path / "634-T01-A00-R-V00-R00.pdf"
    doc = pymupdf.open()
    for y, label in ((80, "First flagged"), (180, "Second flagged")):
        page = doc.new_page(width=400, height=250)
        page.draw_line((42, 30), (42, 220), color=(0, 0, 0), width=2)
        page.insert_text((40, y), label, fontsize=12)
    doc.save(source)
    doc.close()

    window = MainWindow()
    try:
        window._on_file_chosen(source)
        entry = window._batch_entries[source.resolve()]
        output = Path(window._temp_dir.name) / "resolved-test.pdf"
        output.write_bytes(source.read_bytes())
        entry.output = output
        entry.document = Document(source, output, Axis.HORIZONTAL, Route.VECTOR)
        entry.route = Route.VECTOR
        entry.status = "completed"
        window._select_batch_entry(source.resolve())

        before = text_drawing_overlap_runs(output)
        assert [text for _page, _run, text in before] == ["First flagged", "Second flagged"]
        assert window._resolve_button.isEnabled()
        window._resolve_button.click()

        after = text_drawing_overlap_runs(output)
        assert after == []
        moved = pymupdf.open(output)
        try:
            for page_index, run_index, _text in before:
                old_doc = pymupdf.open(source)
                try:
                    old_runs = [
                        trace for trace in old_doc[page_index].get_texttrace()
                        if trace.get("type") == 0 and trace.get("chars")
                    ]
                    old_x = old_runs[run_index]["chars"][0][2][0]
                finally:
                    old_doc.close()
                new_runs = [
                    trace for trace in moved[page_index].get_texttrace()
                    if trace.get("type") == 0 and trace.get("chars")
                ]
                new_x = new_runs[run_index]["chars"][0][2][0]
                assert new_x == pytest.approx(old_x + 3.0, abs=0.05)
        finally:
            moved.close()
        assert "Moved 2 flagged text item(s) 3 pt to the right" in window._status_label.text()
        assert entry.overlap_findings == []
        assert not window._resolve_button.isEnabled()
    finally:
        window.close()


def test_clear_button_resets_session_and_removes_generated_work(qapp, tmp_path):
    import pymupdf

    from spejl.gui.main_window import MainWindow
    from spejl.models import Route

    source = tmp_path / "634-T01-A00-R-V00-R00.pdf"
    doc = pymupdf.open()
    doc.new_page(width=400, height=250)
    doc.save(source)
    doc.close()

    window = MainWindow()
    try:
        window._on_file_chosen(source)
        entry = window._batch_entries[source.resolve()]
        generated = Path(window._temp_dir.name) / "generated-result.pdf"
        generated.write_bytes(source.read_bytes())
        entry.output = generated
        entry.status = "completed"
        entry.route = Route.VECTOR
        window._select_batch_entry(source.resolve())
        window._status_label.setText("Loaded session data")

        window._clear_button.click()

        assert source.exists()
        assert not generated.exists()
        assert window._batch_entries == {}
        assert window._batch_order == []
        assert window._input_path is None
        assert window._output_path is None
        assert window._uploaded_list.count() == 0
        assert window._mirrored_list.count() == 0
        assert window._source_filename_label.text() == "No file selected"
        assert window._mirrored_filename_label.text() == "No file selected"
        assert window._status_label.text() == ""
        assert window._page_label.text() == "Single page"
        assert not window._mirror_button.isEnabled()
        assert not window._save_button.isEnabled()
        assert not window._resolve_button.isEnabled()
        assert "Drop plans here" in window._drop_zone._title.text()
    finally:
        window.close()


def test_zoom_renders_pdf_previews_at_higher_resolution(qapp, tmp_path):
    import pymupdf

    from spejl.gui.main_window import MainWindow

    source = tmp_path / "zoom.pdf"
    doc = pymupdf.open()
    doc.new_page(width=400, height=300)
    doc.save(source)
    doc.close()

    window = MainWindow()
    try:
        window._on_file_chosen(source)
        base = window._source_view.source_size()
        window._mirrored_zoom = 4.0
        window._reload_previews_for_zoom()
        zoomed = window._source_view.source_size()
        assert window._preview_dpi == 300
        assert abs(zoomed.width() - base.width() * 2) <= 1
        assert abs(zoomed.height() - base.height() * 2) <= 1
    finally:
        window.close()


def test_batch_processes_every_pdf_and_keeps_results(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QCoreApplication
    from spejl.gui.main_window import MainWindow
    from spejl.gui.batch_dialogs import MirrorSelectionDialog
    from spejl.qa import fixture_gen

    info = fixture_gen.generate(tmp_path, dpi=100)
    paths = []
    for name in ("634-T01-A00-R-V00-R00.pdf", "634-T02-A00-S-V20-R00.pdf"):
        path = tmp_path / name
        path.write_bytes(info["pdf"].read_bytes())
        paths.append(path)
    window = MainWindow()
    try:
        window._on_files_chosen(paths)
        monkeypatch.setattr(MirrorSelectionDialog, "exec", lambda self: 1)
        window._on_mirror_clicked()
        for _ in range(500):
            QCoreApplication.processEvents()
            worker = window._worker
            if not window._batch_queue and worker is not None and not worker.isRunning():
                QCoreApplication.processEvents()
                break
            if worker is not None:
                worker.wait(10)
        entries = [window._batch_entries[path.resolve()] for path in paths]
        assert [entry.status for entry in entries] == ["completed", "completed"]
        assert all(entry.output and entry.output.exists() for entry in entries)
        assert window._mirrored_list.count() == 2
        assert all(window._mirrored_list.item(i).text().startswith("✓") for i in range(2))
    finally:
        window.close()


def test_save_as_uses_swapped_name_and_writes_jpeg(qapp, tmp_path, monkeypatch):
    from spejl.gui.main_window import MainWindow
    from spejl.gui.batch_dialogs import JpegExportDialog
    from spejl.qa import fixture_gen
    import spejl.gui.main_window as main_window_module

    info = fixture_gen.generate(tmp_path, dpi=100)
    source = tmp_path / "634-T01-A00-R-V00-R00.pdf"
    source.write_bytes(info["pdf"].read_bytes())
    monkeypatch.setattr(JpegExportDialog, "exec", lambda self: 1)
    monkeypatch.setattr(main_window_module.QFileDialog, "getExistingDirectory", lambda *_: str(tmp_path))
    window = MainWindow()
    try:
        window._on_file_chosen(source)
        entry = window._batch_entries[source.resolve()]
        entry.output = info["pdf"]
        entry.status = "completed"
        window._select_batch_entry(source.resolve())
        window._on_save_clicked()
        source_jpeg = tmp_path / "634-T01-A00-R-V00-R00.jpg"
        mirror_jpeg = tmp_path / "634-T01-A00-S-V00-R00.jpg"
        assert source_jpeg.read_bytes().startswith(b"\xff\xd8")
        assert mirror_jpeg.read_bytes().startswith(b"\xff\xd8")
    finally:
        window.close()


def test_mirror_dialog_returns_only_checked_uploads(qapp, tmp_path):
    import pymupdf
    from PySide6.QtCore import Qt
    from spejl.gui.batch_dialogs import MirrorSelectionDialog
    from spejl.gui.batch_model import BatchEntry

    paths = [tmp_path / f"plan-{index}.pdf" for index in range(2)]
    for path in paths:
        with pymupdf.open() as doc:
            doc.new_page()
            doc.save(path)
    dialog = MirrorSelectionDialog([BatchEntry(path, path.name) for path in paths])
    try:
        dialog.list.item(1).setCheckState(Qt.CheckState.Unchecked)
        assert dialog.selected_paths() == [paths[0]]
    finally:
        dialog.close()


def test_only_checked_plan_is_mirrored(qapp, tmp_path, monkeypatch):
    import pymupdf
    from PySide6.QtCore import QCoreApplication, Qt
    from spejl.gui.batch_dialogs import MirrorSelectionDialog
    from spejl.gui.main_window import MainWindow

    paths = [tmp_path / f"plan-{index}-R-V00.pdf" for index in range(2)]
    for path in paths:
        with pymupdf.open() as doc:
            page = doc.new_page(width=200, height=200)
            page.insert_text((20, 50), "Room")
            doc.save(path)

    def accept_first(dialog):
        dialog.list.item(1).setCheckState(Qt.CheckState.Unchecked)
        return 1

    monkeypatch.setattr(MirrorSelectionDialog, "exec", accept_first)
    window = MainWindow()
    try:
        window._on_files_chosen(paths)
        window._on_mirror_clicked()
        for _ in range(400):
            QCoreApplication.processEvents()
            if window._worker is not None and not window._worker.isRunning():
                QCoreApplication.processEvents()
                break
            if window._worker is not None:
                window._worker.wait(10)
        first, second = (window._batch_entries[path.resolve()] for path in paths)
        assert first.status == "completed" and first.output is not None
        assert second.status == "queued" and second.output is None
    finally:
        window.close()


def test_source_only_jpeg_export_without_mirroring(qapp, tmp_path, monkeypatch):
    import pymupdf
    from spejl.gui.batch_dialogs import JpegExportDialog
    from spejl.gui.main_window import MainWindow
    import spejl.gui.main_window as main_window_module

    source = tmp_path / "source-R-V00.pdf"
    with pymupdf.open() as doc:
        doc.new_page(width=200, height=300)
        doc.new_page(width=300, height=200)
        doc.save(source)
    export_dir = tmp_path / "export"
    monkeypatch.setattr(JpegExportDialog, "exec", lambda self: 1)
    monkeypatch.setattr(main_window_module.QFileDialog, "getExistingDirectory", lambda *_: str(export_dir))
    window = MainWindow()
    try:
        window._on_file_chosen(source)
        assert window._save_button.isEnabled()
        window._on_save_clicked()
        assert (export_dir / "source-R-V00_p01.jpg").read_bytes().startswith(b"\xff\xd8")
        assert (export_dir / "source-R-V00_p02.jpg").read_bytes().startswith(b"\xff\xd8")
        window._on_save_clicked()
        assert (export_dir / "source-R-V00_p01_2.jpg").exists()
    finally:
        window.close()


def test_jpeg_dialog_can_choose_source_or_mirror_independently(qapp, tmp_path):
    import pymupdf
    from spejl.gui.batch_dialogs import JpegExportDialog
    from spejl.gui.batch_model import BatchEntry

    source = tmp_path / "plan-R-V00.pdf"
    mirrored = tmp_path / "plan-S-V00.pdf"
    for path in (source, mirrored):
        with pymupdf.open() as doc:
            doc.new_page(width=200, height=200)
            doc.save(path)
    entry = BatchEntry(source, mirrored.name)
    entry.output = mirrored
    dialog = JpegExportDialog([entry])
    try:
        assert dialog.selected_items() == [(source, "source"), (source, "mirrored")]
        dialog._choices[0][2].setChecked(False)
        assert dialog.selected_items() == [(source, "mirrored")]
    finally:
        dialog.close()


def test_remove_uploaded_keeps_source_and_rename_changes_display_only(qapp, tmp_path, monkeypatch):
    import pymupdf
    from spejl.gui.main_window import MainWindow
    import spejl.gui.main_window as main_window_module

    source = tmp_path / "plan-R-V00.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(source)
    window = MainWindow()
    try:
        window._on_file_chosen(source)
        monkeypatch.setattr(main_window_module.QInputDialog, "getText", lambda *_args, **_kwargs: ("Kitchen plan", True))
        window._rename_uploaded(source.resolve())
        assert window._uploaded_list.item(0).text() == "Kitchen plan"
        assert window._source_filename_label.text() == "Kitchen plan"
        assert source.name == "plan-R-V00.pdf"
        window._remove_uploaded(source.resolve())
        assert source.exists()
        assert window._uploaded_list.count() == 0
    finally:
        window.close()
