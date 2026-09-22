"""PyInstaller entry point for the packaged desktop app.

A tiny top-level script rather than pointing PyInstaller at
``spejl/gui/__main__.py`` directly: PyInstaller's import analysis walks
from whatever file it's given, and a top-level launcher keeps that walk
unambiguous relative to the package layout, independent of how the
build is invoked (this file, `spejl.spec`, or a future CI job).
"""

from __future__ import annotations

import sys

from spejl.gui.__main__ import main

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--self-test":
        from spejl.release_check import run

        sys.exit(run(sys.argv[2]))
    sys.exit(main())
