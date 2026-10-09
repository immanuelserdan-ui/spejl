"""Suite-wide test settings."""

from __future__ import annotations

import os

# A MainWindow checks GitHub for a newer Spejl after its splash closes. A dev
# checkout reports version 0.1.0, so any published release counts as newer
# and the app opens a modal "update available" box -- which nothing closes in
# a headless run, hanging the suite. Set before any test module imports the GUI.
os.environ["SPEJL_NO_UPDATE_CHECK"] = "1"
