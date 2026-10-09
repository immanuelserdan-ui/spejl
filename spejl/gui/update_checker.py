"""Background checks for newer public Spejl installers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import logging
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

from PySide6.QtCore import QObject, QThread, Signal, Slot

from spejl import __version__

REPOSITORY = "immanuelserdan-ui/spejl"
RELEASES_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
# Environment variable that turns the startup check off (value "1").
UPDATE_CHECK_OPT_OUT = "SPEJL_NO_UPDATE_CHECK"


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    release_url: str
    download_url: str | None


def _version_key(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", value.lstrip("vV"))
    return tuple(int(number) for number in numbers) or (0,)


def find_update(payload: dict, current_version: str = __version__) -> UpdateInfo | None:
    """Return release information only when it is newer than this build."""
    tag = str(payload.get("tag_name", "")).strip()
    if not tag or _version_key(tag) <= _version_key(current_version):
        return None
    assets = payload.get("assets") or []
    installer = next(
        (
            asset.get("browser_download_url")
            for asset in assets
            if str(asset.get("name", "")).lower().endswith(".exe")
        ),
        None,
    )
    return UpdateInfo(
        version=tag.lstrip("vV"),
        release_url=str(payload.get("html_url") or "https://github.com/" + REPOSITORY + "/releases"),
        download_url=installer,
    )


def fetch_update() -> UpdateInfo | None:
    request = Request(
        RELEASES_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "Spejl-updater"},
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed GitHub endpoint
        payload = json.loads(response.read().decode("utf-8"))
    return find_update(payload)


class UpdateCheckWorker(QObject):
    finished = Signal(object)

    @Slot()
    def run(self) -> None:
        # Retry transient startup/network failures off the GUI thread.
        for attempt in range(3):
            if QThread.currentThread().isInterruptionRequested():
                break
            try:
                self.finished.emit(fetch_update())
                return
            except (OSError, URLError, ValueError, KeyError, TypeError) as error:
                logging.getLogger(__name__).warning(
                    "Spejl update check attempt %s failed: %s", attempt + 1, error
                )
                if attempt < 2:
                    sleep(2 ** attempt)
        self.finished.emit(None)
