# PyInstaller build spec for the Spejl desktop app.
#
# Build with:  pyinstaller spejl.spec --noconfirm
# Output:      dist/Spejl/Spejl.exe  (one-folder build — see the note
#              in the build session's summary for why not --onefile)
#
# Three things a naive `pyinstaller app_launcher.py` would have missed,
# each of which is a silent-failure-at-runtime, not a build error:
#
#   1. RapidOCR's ONNX model weights and config.yaml are package DATA,
#      not Python source — PyInstaller's import analysis walks code,
#      not a package's non-.py files, so they never got collected
#      without being listed explicitly. Missing them doesn't fail the
#      build; RapidOCR() just raises FileNotFoundError the first time
#      someone tries to mirror a raster image.
#   2. spejl/lexicon/da_dk.json is read via `Path(__file__).with_name(...)`
#      at runtime (spejl/lexicon/snap.py) — same category of problem,
#      same silent-until-you-click-Mirror failure mode.
#   3. PySide6's Qt plugins (platform integration, image codecs) need
#      PyInstaller's own hook to be picked up; the hook ships with
#      PyInstaller itself and fires automatically once `PySide6` is
#      imported anywhere in the analysed code, so no extra datas line
#      is needed for it — noted here because it's the one dependency
#      in this list that *didn't* need manual handling, which is easy
#      to assume incorrectly extends to the other two.

import os

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

rapidocr_datas = collect_data_files("rapidocr_onnxruntime", includes=["**/*.onnx", "**/*.yaml"])

a = Analysis(
    ["app_launcher.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("spejl/lexicon/da_dk.json", "spejl/lexicon"),
        *rapidocr_datas,
    ],
    hiddenimports=[
        "spejl.vector.pdf_mirror",
        "spejl.raster.pipeline",
        # qa/self_correct.py imports this dynamically inside the glyph
        # coverage check, so PyInstaller cannot discover it from the AST.
        "fontTools.ttLib",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Codex's document tooling prepends its own Poppler/native-runtime folders
# to PATH.  PyInstaller's dependency scanner can otherwise mistake those
# unrelated ICU and Universal CRT DLLs for application dependencies and
# bundle them at the app root.  They then shadow Windows' compatible system
# DLLs and QtWidgets fails at startup with "procedure could not be found".
# Spejl does not use that runtime, so exclude anything resolved from it.
_CODEX_RUNTIME_MARKERS = ("\\.cache\\codex-runtimes\\", "\\.codex\\tmp\\")
a.binaries = [
    binary for binary in a.binaries
    if not any(marker in str(binary[1]).lower() for marker in _CODEX_RUNTIME_MARKERS)
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Spejl-debug",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-compressing onnxruntime's DLLs is a common source
                # of false-positive AV flags on a freshly built exe;
                # not worth the smaller download for an internal tool.
    console=True,   # diagnostic build
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Spejl-debug",
)
