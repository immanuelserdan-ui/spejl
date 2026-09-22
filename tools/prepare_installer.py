"""Stage a frozen app with license notices and a hash manifest."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys

RUNTIME_PACKAGES = (
    "annotated-doc", "colorama", "flatbuffers", "fonttools", "lxml",
    "markdown-it-py", "mdurl", "numpy", "onnxruntime", "opencv-python",
    "packaging", "pikepdf", "pillow", "protobuf", "pyclipper", "Pygments",
    "pymupdf", "PySide6", "PySide6_Addons", "PySide6_Essentials", "PyYAML",
    "RapidFuzz", "rapidocr-onnxruntime", "rich", "scipy", "shapely",
    "shellingham", "shiboken6", "six", "typer", "typing_extensions",
)


def stage(source: Path, destination: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise SystemExit(f"Staging folder must be new: {destination}")
    required = (
        "Spejl.exe", "_internal/python312.dll", "_internal/_socket.pyd",
        "_internal/vcruntime140.dll", "_internal/vcruntime140_1.dll",
        "_internal/msvcp140.dll", "_internal/rapidocr_onnxruntime/config.yaml",
        "_internal/libssl-3-x64.dll", "_internal/libcrypto-3-x64.dll",
        "_internal/PySide6/plugins/platforms/qwindows.dll",
        "_internal/spejl/lexicon/da_dk.json",
    )
    for relative in required:
        if not (source / relative).is_file():
            raise SystemExit(f"Incomplete frozen app: {relative}")
    if len(list((source / "_internal/rapidocr_onnxruntime").rglob("*.onnx"))) < 3:
        raise SystemExit("OCR model files are missing.")
    shutil.copytree(source, destination)
    shutil.copy2(root / "spejl/gui/assets/spejl-icon.ico", destination)
    shutil.copy2(root / "installer/GETTING-STARTED.txt", destination)
    licenses = destination / "licenses"
    licenses.mkdir()
    notices = [
        "Spejl third-party components", "",
        "This inventory does not replace the component license terms.",
        "Dependency metadata and available license texts accompany this file.",
        "PyMuPDF/MuPDF use AGPL/commercial licensing; PySide6/Qt include LGPL",
        "and other licensed components. Review applicable distribution terms.",
        "Microsoft runtime DLLs are the redistributable files supplied with PySide6.",
        "",
    ]
    for name in RUNTIME_PACKAGES:
        dist = metadata.distribution(name)
        notices.append(f"{dist.metadata['Name']} {dist.version}")
        folder = licenses / name
        folder.mkdir()
        (folder / "METADATA.txt").write_text(dist.read_text("METADATA") or "", encoding="utf-8")
        for entry in dist.files or []:
            # Include nested license directories and vendor notices. Never
            # copy installed code or absolute local paths into the notices.
            if any(part.lower().startswith(("license", "copying", "notice", "copyright"))
                   for part in entry.parts):
                src = Path(dist.locate_file(entry))
                if src.is_file() and ".." not in entry.parts:
                    target = folder / Path(*entry.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, target)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.exists():
        shutil.copy2(python_license, licenses / "Python-LICENSE.txt")
    notices.append(f"Python {sys.version.split()[0]}")
    (destination / "THIRD-PARTY-NOTICES.txt").write_text("\n".join(notices) + "\n", encoding="utf-8")
    manifest = {
        path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob("*")) if path.is_file()
    }
    (destination / "payload-sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Staged {len(manifest)} files in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    stage(args.source, args.destination)
