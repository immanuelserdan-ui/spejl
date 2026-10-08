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
own certificate; see "Code signing" below.

Compiler: https://jrsoftware.org/isdl.php
Portable compiler setup: https://jrsoftware.org/ishelp/topic_technotes.htm
App-local runtime deployment: https://learn.microsoft.com/en-us/cpp/windows/deployment-in-visual-cpp

## Code signing

The release workflow (`.github/workflows/release.yml`) signs `Spejl.exe` (before
the self-test and launch check, so the signed program is what gets tested) and
the `Setup.exe` installer, then prints who signed each file. It picks the method
from the repository's settings (**Settings → Secrets and variables → Actions**);
with neither configured, releases stay unsigned exactly as before.

**Option A — a certificate file (.pfx).** For a certificate from the company's
own certificate authority (trusted on company PCs only) or any exportable
code-signing certificate.

| Kind | Name | Value |
|---|---|---|
| Secret | `SIGNING_CERT_PFX_BASE64` | The .pfx file as Base64: `[Convert]::ToBase64String([IO.File]::ReadAllBytes("cert.pfx"))` |
| Secret | `SIGNING_CERT_PASSWORD` | The .pfx password |
| Variable (optional) | `SIGNING_TIMESTAMP_URL` | Timestamp server; defaults to `http://timestamp.digicert.com` |

`tools/sign_windows.ps1` does the signing and also works by hand on a PC.

**Option B — Azure Artifact Signing** (formerly Trusted Signing; publicly
trusted, about USD 10/month). Create a Signing Account and a Certificate Profile
in Azure, and an app registration with the *Artifact Signing Certificate Profile
Signer* role on the profile.

| Kind | Name | Value |
|---|---|---|
| Secret | `AZURE_TENANT_ID` | Directory (tenant) ID of the app registration |
| Secret | `AZURE_CLIENT_ID` | Application (client) ID |
| Secret | `AZURE_CLIENT_SECRET` | A client secret of the app registration |
| Variable | `AZURE_SIGNING_ENDPOINT` | The account's region endpoint, e.g. `https://weu.codesigning.azure.net/` |
| Variable | `AZURE_SIGNING_ACCOUNT` | Signing Account name |
| Variable | `AZURE_SIGNING_PROFILE` | Certificate Profile name |

If both are configured, Option A is used. The uninstaller Inno Setup writes
(`unins000.exe`) is not signed.
