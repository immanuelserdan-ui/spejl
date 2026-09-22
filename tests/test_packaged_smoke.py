"""The release check must not accept error dialogs or premature exits."""

import json
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

from tools import smoke_packaged_app


@pytest.mark.parametrize(
    ("status", "returncode", "expected_error"),
    [
        (None, 0, "without confirming"),
        ("ready", 1, "exited with code"),
        ("wrong-pid", 0, "Invalid startup handshake"),
        ("ready", 0, None),
    ],
)
def test_requires_readiness_and_clean_exit(monkeypatch, status, returncode, expected_error):
    process = Mock(pid=1234)
    process.wait.return_value = returncode
    process.poll.return_value = returncode

    def launch(*args, env, **kwargs):
        if status:
            Path(env["SPEJL_SMOKE_TEST_REPORT"]).write_text(
                json.dumps({"status": "ready", "pid": 1234 if status == "ready" else 999}),
                encoding="utf-8",
            )
        return process

    monkeypatch.setattr(smoke_packaged_app.subprocess, "Popen", launch)
    if expected_error:
        with pytest.raises(RuntimeError, match=expected_error):
            smoke_packaged_app.check_launch(Path("Spejl.exe"), 1)
    else:
        smoke_packaged_app.check_launch(Path("Spejl.exe"), 1)
    process.terminate.assert_not_called()


def test_stuck_startup_is_rejected_and_terminated(monkeypatch):
    process = Mock(pid=1234)
    process.wait.side_effect = [subprocess.TimeoutExpired("Spejl.exe", 1), 1]
    process.poll.return_value = None
    monkeypatch.setattr(smoke_packaged_app.subprocess, "Popen", Mock(return_value=process))
    with pytest.raises(RuntimeError, match="Timed out"):
        smoke_packaged_app.check_launch(Path("Spejl.exe"), 1)
    process.terminate.assert_called_once()
