"""``spejl mirror`` — the real interface; a future GUI is a client of it.

See build plan §06: the CLI ships before and independent of any GUI, so
batch runs over a whole unit-type folder work from day one.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from spejl.models import Axis, Route
from spejl.router import sniff_route

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _main() -> None:
    """Spejl — mirror floor plans without mirroring their text.

    (This callback exists only to stop Typer collapsing a single-command
    app into a bare top-level command — `inspect` and `batch` join
    `mirror` here in later phases, per the build plan's roadmap.)
    """


@app.command()
def mirror(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Source plan: vector PDF today."
    ),
    output: Path = typer.Option(None, "-o", "--output", help="Output path. Defaults to <name>_mirrored<ext>."),
    axis: Axis = typer.Option(Axis.VERTICAL, "--axis", help="v = left/right, h = top/bottom, both = 180°."),
) -> None:
    """Mirror a floor plan without mirroring its text."""
    try:
        route = sniff_route(input_path)

        if output is None:
            output = input_path.with_name(f"{input_path.stem}_mirrored{input_path.suffix}")

        if route is Route.VECTOR:
            from spejl.vector.pdf_mirror import mirror_pdf

            doc = mirror_pdf(input_path, output, axis=axis)
        else:
            from spejl.raster.pipeline import mirror_raster

            doc = mirror_raster(input_path, output, axis=axis).document

        sidecar = output.with_suffix(output.suffix + ".spejl.json")
        sidecar.write_text(json.dumps(doc.to_sidecar(), indent=2), encoding="utf-8")
    except (ValueError, OSError, RuntimeError) as exc:
        # Every realistic failure this command can hit on real input, in
        # one place rather than two narrower try/excepts that left gaps
        # between them: an unrecognised format or a sheet too
        # low-resolution to mirror honestly (both ValueError, the two
        # cases this used to catch); a corrupt or mis-named PDF pymupdf
        # itself refuses to open (pymupdf.FileDataError, a RuntimeError
        # subclass — sniff_route trusts a recognised suffix over
        # content, so a raster file saved with a .pdf extension reaches
        # this, not a route-sniffing ValueError); a permission or
        # missing-directory error writing the output file or its
        # sidecar (OSError). All surface as the same clean, red one-line
        # message this command already gave a bad input format, instead
        # of a raw Python traceback.
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    total_runs = sum(p.text_runs_mirrored for p in doc.pages)
    total_flags = sum(len(p.flags) for p in doc.pages)
    # ASCII only: Windows consoles default to cp1252, which cannot encode
    # the obvious tick/arrow glyphs and would crash on the success path.
    typer.secho(
        f"OK  {input_path.name} -> {output.name}  "
        f"({len(doc.pages)} page(s), {total_runs} text run(s) mirrored"
        + (f", {total_flags} flag(s) - see {sidecar.name}" if total_flags else "")
        + ")",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
