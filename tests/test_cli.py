"""cli.py's error handling — no prior test coverage existed for this
module at all before this file. Covers the gaps a full-package review
found: a directory passed as input, and a mis-named/corrupt file that
passes route-sniffing on its extension alone but fails when the chosen
route actually tries to open it.
"""

from __future__ import annotations

from typer.testing import CliRunner

from spejl.cli import app

runner = CliRunner()


def test_a_directory_argument_is_rejected_cleanly_not_a_raw_traceback(tmp_path):
    """Regression: `exists=True` alone leaves Click's own `dir_okay`
    default (True), so a directory passes argument validation and used
    to crash inside router.sniff_route's magic-byte fallback with
    IsADirectoryError/PermissionError — not the ValueError the old
    handler caught. `dir_okay=False` rejects it at parse time instead,
    before any of this project's own code ever runs.
    """
    result = runner.invoke(app, ["mirror", str(tmp_path)])
    assert result.exit_code != 0
    assert "directory" in result.output.lower()


def test_a_nonexistent_path_is_rejected_cleanly(tmp_path):
    result = runner.invoke(app, ["mirror", str(tmp_path / "does_not_exist.pdf")])
    assert result.exit_code != 0


def test_an_unrecognised_format_gives_a_clean_message_not_a_traceback(tmp_path):
    src = tmp_path / "plan.xyz"
    src.write_bytes(b"nothing recognisable here")
    result = runner.invoke(app, ["mirror", str(src)])
    assert result.exit_code == 1
    assert "unrecognised format" in result.output
    assert "Traceback" not in result.output


def test_a_corrupt_pdf_gives_a_clean_message_not_a_raw_traceback(tmp_path):
    """Regression: sniff_route trusts a recognised .pdf suffix over
    content — a corrupt or mis-named file reaches mirror_pdf, which
    raises pymupdf.FileDataError (a RuntimeError subclass), not the
    ValueError the old handler caught around sniff_route alone. Must
    surface as the same clean, red one-line message, not a raw
    traceback the user has to interpret themselves.
    """
    src = tmp_path / "plan.pdf"
    src.write_bytes(b"this is not a real PDF file, just text with a .pdf name")
    result = runner.invoke(app, ["mirror", str(src)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
