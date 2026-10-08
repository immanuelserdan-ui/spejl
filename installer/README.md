# Spejl Windows installer

Build with Inno Setup 6.7.3 or newer. The output is a single offline installer
for Windows 10 22H2 / Windows 11 on Intel/AMD x64, installed per user.

From the repository root, use the Python environment containing Spejl's
`raster`, `gui`, and `dev` dependencies plus PyInstaller:

```powershell
& ./.venv-spejl-release/Scripts/python.exe -m PyInstaller spejl.spec --noconfirm --clean --workpath build-installer --distpath dist-installer
& ./.venv-spejl-release/Scripts/python.exe tools/prepare_installer.py dist-installer/Spejl installer/payload
& ./work/installer-tools/InnoSetup/ISCC.exe installer/Spejl.iss
```

Staging requires a new directory so obsolete DLLs cannot enter a release.
For another location use `/DPayloadDir=<absolute-path>` and
`/DReleaseDir=<absolute-path>` as compiler arguments before the script.
The default release folder is `releases/0.1.0`.

Before distributing, run `tools/smoke_packaged_app.py` on the installed
executable, and run that executable with `--self-test <writable-directory>`.
Require exit code 0 and `status: passed` in `self-test.json`. Test with Python
environment variables removed and PATH containing only Windows directories.
Use `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /NOICONS /TASKS= /DIR=<test-folder>`
for an isolated installation test; uninstall that test copy afterward.

The installer supplies shortcuts, an Apps entry, and an uninstaller. It does
not bundle user drawings, development environments, or build logs. App-local
Microsoft runtime DLLs come from the PySide6 wheel. Dependencies' license
notices accompany the installed app. Code signing requires the distributor's
own certificate; this build is unsigned.

Compiler: https://jrsoftware.org/isdl.php
Portable compiler setup: https://jrsoftware.org/ishelp/topic_technotes.htm
App-local runtime deployment: https://learn.microsoft.com/en-us/cpp/windows/deployment-in-visual-cpp
