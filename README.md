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

**Route A handles vector PDFs and preserves image-bearing pages through a raster fallback.**
Given a vector PDF (the common case for CAD-exported plans), Spejl
reads each glyph run's exact position and font, discards it, mirrors
the page's vector geometry, and re-inserts every string upright at its
mirrored, ISO-convention-correct anchor — no OCR on vector-only pages,
with font-substitution and diagonal-angle limitations. Pages containing scans or embedded images are rendered at 300 dpi and processed through Route B; the result warns that those pages lose vector editability.

**Route B (raster PNG/JPEG, S1–S7) is built end-to-end and passing its
own round-trip test:** mirror the golden fixture, then OCR the *output*
and confirm every string reads back correctly at its mirrored position.
The golden-fixture tests enforce text recall, character accuracy, orientation and geometry checks, including mirroring twice. These are fixture measurements, not a guarantee for arbitrary drawings — see `tests/test_raster_pipeline.py`.

**The drawing outranks the type.** A mirrored sheet carries two kinds of
content and they are not equally recoverable: a text run is
reconstructed — Spejl knows the string, the measured style and the
mirrored anchor, and can redraw it at any time — while linework is not
reconstructed at all, only flipped. So the two are composited in that
order, paper then type then geometry, and a re-rendered label that lands
on a wall, an arc or a dimension line is drawn *underneath* it rather
than through it. A mis-anchored label is then a legible mistake sitting
under an intact plan instead of a silent hole in one. The cost is that
such a label reads poorly or not at all, so a run the geometry mostly
covers is flagged `hidden-behind-linework` for review rather than
passing quietly. See `tests/test_drawing_z_order.py`.

**Reverse-out text erases correctly.** Local "paper" for the erase mask
is read from the ring immediately outside each text box, not the box's
own interior — the interior is exactly the one region guaranteed to
contain the glyph, so for text knocked out lighter than its own
background (white room-name lettering on a filled panel), the interior's
own light end just IS the glyph. The old "darker than the light end"
test never fired for that case: the panel got erased as "not paper" and
the white lettering was judged paper and left standing whole and
unmirrored. Masking by absolute deviation from the ring's own reading,
in either direction, fixes both the ordinary case and this one with a
single test. See `tests/test_erase_clean.py`.

**S8 — the pre-save QA gate.** Every stage above enforces its own piece
of the pipeline's invariants at the point of doing the work; S8 is the
independent check that the *finished, fully-composited* sheet actually
satisfies them, run automatically before every save (`qa_gate=True` by
default on `mirror_raster`). Five structural/confidence rules, checked against the real
output bytes rather than assumed from upstream stages having run
correctly:

* **Spatial integrity** — every drawing pixel on the source survives,
  mirrored, in the output. No corrector exists for this one, on
  purpose: if geometry is missing, the one thing this stage knows about
  what should be there is already gone from every input it has, so
  there is nothing safe to invent — a violation here refuses the save
  (`QAGateFailure`) rather than ship a wrong picture.
* **Layer ordering** — every fully-solid drawing pixel still holds the
  pre-text plate's own value. This one is cheaply self-healing: the
  plate is authoritative geometry, so a violation is fixed by
  restoring from it.
* **Text fidelity** — a run's rendered cap height must land in an
  evidence-based band of its measured target (not just under a
  ceiling: S7's own shrink-to-fit guard has no floor, so a run can be
  "successfully" shrunk into illegibility and never get flagged today),
  and every character a run's corrected string needs must have a real
  glyph in the resolved font — checked against the font's own cmap
  table via `fontTools`, not by rendering and looking for ink (a
  missing glyph renders as a visible fallback box with the same pixel
  count as a real one, confirmed on a CJK and a Thai codepoint neither
  system font on this project's dev machine actually covers). Both are
  correctable within a few bounded, local retries — widen the target,
  or swap to the next candidate font that covers the text.

Bounded, not "loop until it passes": every correction is a
deterministic, mechanical retry over a small fixed set of
alternatives, so either the page settles within a couple of passes or
the violation is structural and no amount of further looping helps —
confirmed catching two real, previously-silent issues in the course of
building it (a protected region's own deliberately-unflipped footprint
misread as missing geometry; the height-lock ceiling first copying
`OVERFLOW_TOLERANCE`, the threshold for when to START shrinking, not
the one for when a render is actually a problem — the golden fixture's
own `2900`/`3200` land at a legitimate 105%, no shrink ever attempted).
The structural gate originally added about 4% to end-to-end mirror time on the golden fixture — under
40ms of the ~1.8s a full raster mirror already costs, dominated by OCR.
An additional output OCR pass now checks the intended wording before save. Missing, different or low-confidence readings produce `rendered-text-unconfirmed` review warnings; saving a flagged result does not certify its wording. The Verify button checks saved output wording and every differing pixel outside padded text exclusions. It reports uncertainty rather than claiming that all geometry or text is guaranteed correct. OCR agreement is not independent ground truth. See `spejl/qa/self_correct.py`, `spejl/qa/verify.py` and `tests/test_scan_regressions.py`.

```bash
spejl mirror plan.png --axis v      # same CLI as the vector path
```

Command line:
`python -m spejl.cli mirror <file>` routes automatically — vector PDF
takes Route A, raster PNG/JPEG takes Route B.

**Desktop app is built.** A drop zone, axis picker, and side-by-side
before/after preview over the same `mirror_pdf` / `mirror_raster` the
CLI calls — no separate implementation to drift out of sync. Mirroring
runs on a background thread so the window stays responsive during OCR.

```bash
python -m spejl.gui       # or: spejl-app, once installed
```

**Packaged as a standalone .exe** — no Python install needed to run it:

```bash
pip install pyinstaller
pyinstaller spejl.spec --noconfirm
```

Produces `dist/Spejl/Spejl.exe` (~375 MB — PySide6 + OpenCV + the OCR
model weights). Not `--onefile`: that would unpack ~375 MB to a temp
folder on every launch, so this ships as a folder with the exe inside
it, the standard shape for a Windows desktop app of this size (a
Desktop shortcut gives the double-click experience without that
penalty). `spejl.spec` documents the two things a naive
`pyinstaller app_launcher.py` would have silently gotten wrong:
RapidOCR's ONNX model weights and `spejl/lexicon/da_dk.json` are
package *data*, not Python source, so PyInstaller's import analysis
never finds them on its own — the build would succeed and the app
would open, then fail the moment someone actually tried to mirror a
raster image. Verified by running the real OCR pipeline (both routes,
against the golden fixture) from inside a frozen build before
shipping — see the build session's log for the console-mode proof.

One licensing note before distributing this *outside* the team:
PyMuPDF (Route A) is dual-licensed AGPL-3.0 / commercial. Fine bundled
into an internal tool; a commercial license (or a swap to the BSD/
Apache-licensed `pypdfium2`) is needed before shipping the exe to
anyone outside the organisation.

Reviewed and code-reviewed before this build: 21 candidate bugs found
across 4 parallel review passes plus empirical stress-testing, 17
confirmed and fixed (each with a regression test), 4 checked and
deliberately left as documented, currently-unreachable edge cases. The
most serious was silent — every erase fill was landing ~3% darker than
true paper due to a colour-quantisation bug, invisible at a glance but
visible as faint ghost text at zoom. See `git log` for the full
review-and-fix commit.

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
pip install -e ".[raster,gui,dev]"
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
├── models.py            Axis, Route, Document, Flag — the shared contract
├── router.py             sniff format → Route A (vector) vs Route B (raster)
├── transform/mirror.py   the mirror math BOTH routes share: point/bbox
│                         reflection, the readability-canonicalisation
│                         angle rule (build plan §05), image flip
├── vector/pdf_mirror.py  Route A: page-geometry reconstruction + per-run
│                         text mirroring
├── detect/
│   ├── ocr.py             OcrBackend protocol + RapidOCR adapter
│   └── rotations.py       triple-pass detection, NMS + containment merge
├── lexicon/               Danish plan vocabulary + three-tier snap (S3)
├── style/metrics.py       measure ink/paper/size/tracking from pixels (S4)
├── erase/clean.py         local-paper erase + line-pixel repair (S5)
├── render/text.py         supersampled re-render behind the drawing layer (S7)
├── raster/pipeline.py     wires S1–S8 together — the Route B entry point
├── qa/
│   ├── self_correct.py    S8: pre-save gate — check, correct, refuse (see above)
│   ├── fixture_gen.py     golden fixture with free ground truth (vector-drawn)
│   ├── metrics.py         recall / char accuracy / anchor error scoring
│   └── score_ocr.py       `python -m spejl.qa.score_ocr` — the go/no-go report
├── gui/
│   ├── main_window.py     the window: drop zone, axis picker, before/after
│   ├── worker.py          runs mirror_pdf/mirror_raster off the UI thread
│   ├── widgets.py         DropZone, ScaledImageLabel
│   └── imaging.py         PDF/raster -> QPixmap for preview
└── cli.py                 `spejl mirror ...` (routes A vs B automatically)
tests/
├── test_pdf_mirror.py      Route A
├── test_transform.py       shared mirror math
├── test_lexicon.py         Danish diacritic + dimension-plausibility snap
├── test_detect_ocr.py      detection gates (marked slow — loads OCR models)
├── test_raster_pipeline.py Route B round-trip: mirror -> OCR the output -> score
├── test_drawing_z_order.py geometry stays in front of, and intact under, the type
├── test_erase_clean.py, test_style_metrics.py, test_render_text.py,
│   test_rotations_merge.py, test_protected_regions.py, test_qa_metrics.py
│                          edge cases from the code review pass
└── test_gui.py             window construction and state-transition wiring
docs/build-plan.html       the full blueprint (stack decision, roadmap,
                            edge cases, acceptance gates)
demo/                       before/after pairs and the app screenshot
```
