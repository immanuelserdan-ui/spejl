"""Check that a frozen Spejl reaches its main window and exits cleanly.

Usage: python tools/smoke_packaged_app.py "Spejl-app/Spejl.exe"
The startup handshake prevents splash screens and error dialogs from passing.
This check does not exercise floor-plan mirroring.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def check_launch(executable: Path, timeout: float) -> None:
    with tempfile.TemporaryDirectory(prefix="spejl-smoke-") as directory:
        report = Path(directory) / "ready.json"
        env = dict(os.environ, SPEJL_SMOKE_TEST_REPORT=str(report))
        process = subprocess.Popen([str(executable)], cwd=executable.parent, env=env)
        try:
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError("Timed out before Spejl completed its main-window check.") from None
            if returncode != 0:
                raise RuntimeError(f"Spejl exited with code {returncode}.")
            if not report.is_file():
                raise RuntimeError("Spejl exited without confirming its main window was ready.")
            result = json.loads(report.read_text(encoding="utf-8"))
            if result.get("status") != "ready" or result.get("pid") != process.pid:
                raise RuntimeError(f"Invalid startup handshake: {result!r}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"Executable not found: {executable}")
    if args.timeout <= 0:
        parser.error("Timeout must be positive.")
    try:
        check_launch(executable, args.timeout)
    except (RuntimeError, OSError, ValueError) as error:
        raise SystemExit(f"FAIL: {error}") from error
    print("PASS: Spejl completed its splash transition, showed its main window, and exited cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
