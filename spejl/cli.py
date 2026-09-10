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
    input_path: Path = typer.Argument(..., exists=True, help="Source plan: vector PDF today."),
    output: Path = typer.Option(None, "-o", "--output", help="Output path. Defaults to <name>_mirrored<ext>."),
    axis: Axis = typer.Option(Axis.VERTICAL, "--axis", help="v = left/right, h = top/bottom, both = 180°."),
) -> None:
    """Mirror a floor plan without mirroring its text."""
    try:
        route = sniff_route(input_path)
    except ValueError as exc:  # unrecognised format — an existing file we
        # still can't route (bad extension, magic bytes match nothing)
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if output is None:
        output = input_path.with_name(f"{input_path.stem}_mirrored{input_path.suffix}")

    if route is Route.VECTOR:
        from spejl.vector.pdf_mirror import mirror_pdf

        doc = mirror_pdf(input_path, output, axis=axis)
    else:
        from spejl.raster.pipeline import mirror_raster

        try:
            doc = mirror_raster(input_path, output, axis=axis).document
        except ValueError as exc:  # resolution too low to mirror honestly
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc

    sidecar = output.with_suffix(output.suffix + ".spejl.json")
    sidecar.write_text(json.dumps(doc.to_sidecar(), indent=2), encoding="utf-8")

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
