"""Exercise the installed container package, offline and as a non-root user."""

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

from spejl.release_check import run


def main() -> None:
    assert os.getuid() != 0, "The runtime should run as a non-root user."
    assert importlib.util.find_spec("PySide6") is None, "Qt belongs only in the test image."
    opencv = {
        dist.metadata["Name"].lower()
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"].lower().startswith("opencv-")
    }
    assert opencv == {"opencv-python"}, f"Overlapping OpenCV wheels: {opencv}"

    # Use the mounted work directory to prove output and sidecar writes work.
    # The existing self-test exercises vector, raster/OCR and mixed PDF paths.
    with tempfile.TemporaryDirectory(prefix="spejl-smoke-", dir=Path.cwd()) as directory:
        output = Path(directory)
        result = run(directory)
        report = json.loads((output / "self-test.json").read_text(encoding="utf-8"))
        print(json.dumps(report, indent=2))
        if result:
            raise SystemExit(result)
        destination = output / "cli_mirrored.pdf"
        subprocess.run(
            ["spejl", "mirror", str(output / "sample.pdf"), "--axis", "v", "-o", str(destination)],
            check=True,
        )
        assert destination.is_file()
        sidecar = json.loads(destination.with_suffix(".pdf.spejl.json").read_text(encoding="utf-8"))
        assert sidecar, "The CLI must also write its JSON sidecar."
    print("Container smoke checks passed.")


if __name__ == "__main__":
    main()
