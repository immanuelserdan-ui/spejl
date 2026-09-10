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

**Phase 2 detection is spiked and measured.** Route B's riskiest
assumption — that OCR can read small rotated dimension text and Danish
diacritics off a plan — is settled. Against the golden fixture, triple-
pass detection plus the lexicon reaches **100% text recall, 100%
character accuracy and 100% orientation accuracy at both 150 and 300
dpi**, with mean anchor error 1.7 px. `pytest` enforces those as gates.

Still to build for a working raster pipeline: erase + line repair (S5),
the flip (S6, trivial), and re-render (S7).

### Measuring detection yourself

```bash
python -m spejl.qa.fixture_gen --dpi 150 300     # fixture + ground truth
python -m spejl.qa.score_ocr   --dpi 150 300     # the numbers
```

The fixture is drawn as vector and its ground truth is extracted from
that same vector source, so annotation and image can never drift apart.
`--single-pass` shows what the rotated passes buy: without them, all five
vertical dimensions are mis-oriented and `2105` reads as `105` at 0.99
confidence.

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
