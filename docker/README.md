# Spejl in Docker

Docker is useful for the command-line PDF/image pipeline, batch jobs and a
consistent Linux test environment. It packages Python 3.12, native libraries,
OCR models and fonts. The desktop editor and Windows installer continue to be
the interface for interactive editing; the container runs `spejl mirror`.
This is an optional batch/development workflow: it requires Docker Engine or
Docker Desktop and extra image storage, and does not change OCR accuracy.
Rendering uses Liberation Sans/DejaVu on Linux, so font substitution can differ
from the Windows desktop environment.

## Build and mirror a plan

Install Docker with Linux containers and Compose v2, then run from the repo root:

```sh
docker compose build spejl
```

Create `workspace` and put your source plan in it. On PowerShell:

```powershell
New-Item -ItemType Directory -Force workspace | Out-Null
Copy-Item 'C:\path\to\plan.pdf' workspace\plan.pdf
docker compose run --rm spejl mirror /work/plan.pdf --axis v -o /work/plan_mirrored.pdf
```

On Linux, set the container user to your host UID/GID so files written into the
bind mount belong to you:

```sh
mkdir -p workspace
export SPEJL_UID=$(id -u) SPEJL_GID=$(id -g)
cp /path/to/plan.pdf workspace/plan.pdf
docker compose run --rm spejl mirror /work/plan.pdf --axis v -o /work/plan_mirrored.pdf
```

The output and `plan_mirrored.pdf.spejl.json` appear in `workspace` on the host.
PNG/JPEG inputs use the same command. `--axis h` mirrors top/bottom, and
`--axis both` mirrors both axes. Flags in the sidecar need review as described
in the main README; containerization does not change the mirroring algorithm.

To use an existing folder instead, set `SPEJL_WORKDIR` to its absolute path.
For example, `$env:SPEJL_WORKDIR = (Resolve-Path 'C:\Plans').Path` in PowerShell.
All command arguments use the corresponding `/work/...` container path.

For a batch in PowerShell:

```powershell
Get-ChildItem workspace -Filter *.pdf |
    Where-Object BaseName -NotLike '*_mirrored' |
    ForEach-Object {
        docker compose run --rm spejl mirror "/work/$($_.Name)" --axis v
        if ($LASTEXITCODE -ne 0) { throw "Mirroring failed: $($_.Name)" }
    }
```

## Validate the images

```sh
docker compose run --rm spejl --help
docker compose run --rm --entrypoint python spejl /opt/spejl-smoke.py
docker compose --profile test build tests
docker compose --profile test run --rm tests
```

The smoke check runs real vector/raster/OCR/mixed-PDF processing, validates the
installed package and writes a CLI output plus JSON sidecar through the bind
mount. It uses temporary synthetic plans and removes them afterward. The
separate test image includes Qt and dev dependencies for the existing
regression suite. To include the slower OCR fixture tests, override its command:

```sh
docker compose --profile test run --rm tests -q -p no:cacheprovider --basetemp=/tmp/pytest
```

Compose runs containers without network access, with a read-only root filesystem,
a writable temporary directory, and no extra capabilities. Only the selected
work folder is mounted. Builds require network access to fetch dependencies;
runtime OCR uses the models already included in the image. No ports or Docker
socket are exposed.

The Python base image is pinned by digest; Dependabot proposes base updates.
Python dependency ranges follow `pyproject.toml`, so rebuilds can resolve newer
versions. The runtime image omits PySide6 and test tools. The existing licensing
section in the main README also applies when distributing container images.
This setup is validated on Linux/amd64 using Docker Desktop and CI.
