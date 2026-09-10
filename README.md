# Spejl

Mirror floor plans without mirroring the text.

A straight `cv2.flip` (or its vector equivalent) correctly reflects a
plan's geometry — walls, doors, the kitchen run — but reverses every
glyph in every room label and dimension. Spejl mirrors the geometry and
independently re-anchors each text run, upright and correctly
positioned, at its mirrored spot.

Full technical blueprint: [`docs/build-plan.html`](docs/build-plan.html)
(open in a browser — it's the design doc this implementation follows).

## Status

**Phase 1 (Route A — lossless vector mirroring) is built and tested.**
Given a vector PDF (the common case for CAD-exported plans), Spejl
reads each glyph run's exact position and font, discards it, mirrors
the page's vector geometry, and re-inserts every string upright at its
mirrored, ISO-convention-correct anchor — no OCR, no reconstruction,
no loss.

Phase 2+ (Route B — OCR-based mirroring for raster PNG/JPEG input, the
case a hand-photographed or exported floor plan image needs) is
scoped in the build plan but not yet implemented.

## Install

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

## Use

```bash
spejl mirror plan.pdf --axis v          # left/right mirror (default)
spejl mirror plan.pdf --axis h -o out.pdf
```

Writes `plan_mirrored.pdf` plus a `plan_mirrored.pdf.spejl.json`
sidecar recording what moved, in what direction, and any flags raised
along the way (see `spejl/models.py::Document.to_sidecar`).

## Test

```bash
pytest -v
```

Seven tests cover the golden two-room-unit fixture's three hard cases
simultaneously: geometry reflects, a horizontal room label stays
left-to-right, and a vertical dimension stays bottom-to-top — see
`tests/test_pdf_mirror.py` and build plan §10 for the acceptance gates
this is meant to satisfy at full pipeline scale.

## Layout

```
spejl/
├── models.py           Axis, Route, Document, Flag — the shared contract
├── router.py            sniff format → Route A (vector) vs Route B (raster)
├── vector/pdf_mirror.py Route A: page-geometry reconstruction + per-run
│                        text mirroring (the build plan's §05 math, live)
└── cli.py                `spejl mirror ...`
tests/test_pdf_mirror.py
docs/build-plan.html      the full blueprint (stack decision, roadmap,
                           edge cases, acceptance gates)
demo/                      a tiny synthetic plan.pdf / plan_mirrored.pdf
                           pair, generated for a CLI smoke test
```
