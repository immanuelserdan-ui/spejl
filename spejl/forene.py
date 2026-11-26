"""Offline floor import, stair analysis and single-sheet composition in PDF points."""
from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
import logging
import math
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
import os

import cv2
import numpy as np
from PIL import Image, ImageOps
import pymupdf as pdf

LOG = logging.getLogger(__name__)
REFERENCE_GAP = 1.5  # Percent of a typical plan width, measured between exterior walls.
REFERENCE_DATUM = 0.56
FOOTER = Path(__file__).parent / "gui" / "assets" / "2d-footer.png"


@dataclass(frozen=True)
class Floor:
    name: str
    source: bytes  # Immutable single-page PDF snapshot shared by preview and export.
    bounds: tuple[float, float, float, float]
    detected: float | None
    manual: float | None = None
    scale: float = 1.0
    stair_edge: tuple[float, float, bool] | None = None  # Datum, tread pitch / length, orientation.
    shared_edge: bool = False
    turn: float | None = None

    @property
    def datum(self):
        if self.manual is not None:
            return self.manual
        if self.stair_edge is not None:
            return self.stair_edge[0]
        if self.detected is not None:
            return self.detected
        return REFERENCE_DATUM


def match_landings(floors):
    """Mark groups with compatible labelled tread edges."""
    edges = [f.stair_edge for f in floors if f.stair_edge is not None]
    shared = (len(edges) >= 2 and
              len({edge[2] for edge in edges}) == 1 and
              max(e[1] for e in edges) <= 1.25 * min(e[1] for e in edges))
    return [replace(f, shared_edge=shared and f.stair_edge is not None) for f in floors]


def unconfirmed_floors(floors):
    """Multiple floors need corresponding tread edges or explicit manual anchors."""
    if len(floors) < 2:
        return []
    return [i for i, floor in enumerate(floors, 1)
            if floor.manual is None and not floor.shared_edge]


def _tread_bank(gray):
    """Find overlapping, evenly spaced thin tread lines, in either orientation."""
    h, w = gray.shape
    ink = (gray < 165).astype(np.uint8)
    minimum = max(10, round(w * .045))
    runs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, minimum), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(runs)
    segments = sorted((y + height // 2, x, width) for x, y, width, height, _ in stats[1:]
                      if minimum <= width <= w * .38 and height <= max(3, min(h*.015, width*.12)))
    banks = []
    for i, (y, x, width) in enumerate(segments):
        aligned = [yy for yy, xx, ww in segments[i+1:]
                   if yy - y <= h * .55 and abs(xx-x) <= width * .18
                   and abs(ww-width) <= width * .25]
        for second in aligned:
            step = second - y
            if step < 3 or step > min(width * .75, h * .09):
                continue
            count, last = 2, second
            for yy in aligned:
                if yy <= last + step * .7:
                    continue
                if abs((yy-last) - step) > max(1.5, step * .18):
                    break
                count, last = count + 1, yy
            if count >= 4:
                banks.append((count, int(width), int(y), int(x), int(last)))
    return banks


def _winder_turn(gray):
    """Locate a horizontal return flight by its fan, divider and straight treads."""
    segments = cv2.createLineSegmentDetector().detect(gray)[0]
    if segments is None:
        return None
    h, w = gray.shape
    diagonals = []
    for x, y, u, v in segments.reshape(-1, 4).tolist():
        if x > u:
            x, y, u, v = u, v, x, y
        width = u - x
        if max(8, w*.025) < width < w*.35 and .35*width < abs(v-y) < 2.5*width:
            diagonals.append((x, y, u, v, (v-y)/width))
    turns = []
    for x, y, u, v, slope in diagonals:
        if slope <= 0:
            continue
        for a, b, c, d, other in diagonals:
            if other >= 0:
                continue
            span = min(u-x, c-a)
            cx = (b-y+slope*x-other*a)/(slope-other)
            cy = y+slope*(cx-x)
            for direction, ends in ((1, (u,c)), (-1, (x,a))):
                if max(abs(cx-end) for end in ends) > .40*span:
                    continue
                # Require a thin horizontal divider extending into a bank of
                # treads. A roof hatch or door swing alone is insufficient.
                left, right = sorted((round(cx+direction*.25*span), round(cx+direction*1.9*span)))
                band = max(2, round(.13*span))
                row = round(cy)
                if left < 0 or right >= w or row-3*band < 0 or row+3*band >= h:
                    continue
                strip = gray[:, left:right]
                rows = np.flatnonzero(np.mean(strip[row-band:row+band+1] < 165, axis=1) > .8)
                if not len(rows):
                    continue
                # Treads cross a probe strip on both sides of the divider.
                for side in (-1, 1):
                    lo, hi = sorted((row+side*2*band, row+side*3*band))
                    dark = np.mean(strip[lo:hi] < 180, axis=0) > .85
                    edges = np.flatnonzero(np.diff(np.r_[False, dark, False]))
                    centers = [(s+e)/2 for s,e in zip(edges[::2], edges[1::2])
                               if e-s <= .15*span]
                    gaps = np.diff(centers)
                    if not any(.12*span < p < .8*span and abs(p-q) < .22*p
                               for p,q in zip(gaps, gaps[1:])):
                        break
                else:
                    turns.append(row-band+float(np.median(rows)))
    if turns and max(turns)-min(turns) < max(4, h*.015):
        return median(turns)/h
    return None


def _turn_datum(gray):
    h, w = gray.shape
    # Detect line segments at a consistent plan resolution. Upsampled
    # raster strokes can split into several edges at the analysis resolution.
    ratio = min(1., 800 / max(h, w))
    turn = _winder_turn(cv2.resize(gray, None, fx=ratio, fy=ratio,
                                   interpolation=cv2.INTER_AREA)) if ratio < 1 else _winder_turn(gray)
    return turn


def stair_datum(gray):
    h, w = gray.shape
    candidates = []
    for rotated, plane in ((False, gray), (True, gray.T.copy())):
        for count, width, y, x, last in _tread_bank(plane):
            top, left, right = (x, y, last) if rotated else (y, x, x + width)
            candidates.append((count, width, top / h, left / w))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))
    count, _, anchor, _ = candidates[0]
    if any(n >= count - 1 and abs(a-anchor) > .08 for n, _, a, _ in candidates):
        return None
    return anchor


def _labelled_stair_edge(page, bounds):
    """Recover a flight's upper edge from repeated vector tread strokes near a stair label."""
    labels = [((w[0]+w[2])/2, (w[1]+w[3])/2) for w in page.get_text("words")
              if w[4].casefold() in {"op", "ned", "up", "down"}]
    if not labels:
        return None
    groups = {}
    for path in page.get_drawings():
        if path["type"] not in {"s", "fs"}:
            continue
        for item in path["items"]:
            if item[0] != "l":
                continue
            a, b = item[1:3]
            vertical = abs(a.x-b.x) <= .02
            if not vertical and abs(a.y-b.y) > .02:
                continue
            start, end = sorted((a.y, b.y) if vertical else (a.x, b.x))
            length = end-start
            if not .035*bounds.width <= length <= .40*bounds.width:
                continue
            pos = a.x if vertical else a.y
            # Revit can paint each tread twice. Count distinct strokes only.
            key = vertical, round(start, 2), round(end, 2)
            groups.setdefault(key, {})[round(pos, 2)] = (start, pos)
    candidates = []
    for (vertical, start, end), strokes in groups.items():
        positions = sorted(strokes)
        if len(positions) < 4:
            continue
        pitch = median(np.diff(positions))
        length = end-start
        if not .06*length <= pitch <= .8*length:
            continue
        runs = [[]]
        for pos in positions:
            if runs[-1] and abs(pos-runs[-1][-1]-pitch) > pitch*.15:
                runs.append([])
            runs[-1].append(pos)
        for run in runs:
            if len(run) < 4:
                continue
            rect = (pdf.Rect(run[0], start, run[-1], end) if vertical else
                    pdf.Rect(start, run[0], end, run[-1]))
            distance = min(math.hypot(max(rect.x0-x, 0, x-rect.x1),
                                      max(rect.y0-y, 0, y-rect.y1)) for x, y in labels)
            if distance > length or not bounds.contains(rect):
                continue
            top = median(strokes[x][0] for x in run) if vertical else strokes[run[0]][1]
            candidates.append((distance/length, (top-bounds.y0)/bounds.height,
                               pitch/length, vertical, rect))
    if not candidates:
        return None
    nearest = min(candidates, key=lambda c: c[0])
    flights = [nearest]
    for c in candidates:
        r, n = c[4], nearest[4]
        # A return stair has two adjacent runs. Its UP and DOWN annotations
        # can be closest to opposite runs on successive floors.
        overlap = min(r.x1, n.x1) - max(r.x0, n.x0)
        gap = max(0., r.y0-n.y1, n.y0-r.y1)
        if (c[3] == nearest[3] and abs(c[2]-nearest[2]) < .15*nearest[2]
                and overlap >= .7*max(r.width, n.width)
                and gap <= .3*min(r.height, n.height)):
            flights.append(c)
        elif c[0] < nearest[0]+.2 and abs(c[1]-nearest[1]) > .015:
            return None
    upper = min(flights, key=lambda c: c[1])
    return upper[1:4]


def _analyze(doc, name):
    page = doc[0]
    page.remove_rotation()
    rect = page.rect
    zoom = min(4, 1600 / max(rect.width, rect.height))
    pix = page.get_pixmap(matrix=pdf.Matrix(zoom, zoom), colorspace=pdf.csGRAY, alpha=False)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    ys, xs = np.nonzero(gray < 243)
    if not len(xs):
        raise ValueError(f"{name} contains no visible drawing.")
    # One analysis pixel of bleed keeps antialiased outlines inside the sheet.
    x0, y0 = max(0, xs.min()-1), max(0, ys.min()-1)
    x1, y1 = min(pix.width, xs.max()+2), min(pix.height, ys.max()+2)
    bounds = pdf.Rect(x0/zoom, y0/zoom, x1/zoom, y1/zoom) & rect
    crop = gray[y0:y1, x0:x1]
    turn = _turn_datum(crop)
    return Floor(name, doc.tobytes(garbage=3, deflate=True), tuple(bounds),
                 turn if turn is not None else stair_datum(crop), turn=turn,
                 stair_edge=_labelled_stair_edge(page, bounds))


def import_floors(path: Path):
    """Expand one PDF or one drawing image. Fail the whole file on a bad page."""
    try:
        if path.suffix.lower() == ".pdf":
            with pdf.open(path) as source:
                if not source.is_pdf or source.needs_pass or not source.page_count:
                    raise ValueError("Expected an unlocked PDF with at least one page.")
                floors = []
                for i in range(source.page_count):
                    with pdf.open() as page:
                        page.insert_pdf(source, from_page=i, to_page=i)
                        label = path.name if len(source) == 1 else f"{path.name} · p{i+1}"
                        floors.append(_analyze(page, label))
        else:
            with Image.open(path) as raw, pdf.open() as doc:
                if getattr(raw, "n_frames", 1) != 1:
                    raise ValueError("Export each image frame as a separate drawing.")
                dpi = raw.info.get("dpi", (96, 96))
                dx, dy = [v if math.isfinite(v) and v > 1 else 96 for v in dpi]
                if raw.getexif().get(274, 1) in (5, 6, 7, 8):
                    dx, dy = dy, dx
                img = ImageOps.exif_transpose(raw).convert("RGBA")
                flat = Image.new("RGB", img.size, "white")
                flat.paste(img, mask=img.getchannel("A"))
                data = BytesIO()
                flat.save(data, "PNG")
                page = doc.new_page(width=72*img.width/dx, height=72*img.height/dy)
                page.insert_image(page.rect, stream=data.getvalue(), keep_proportion=False)
                floors = [_analyze(doc, path.name)]
    except Exception as exc:
        LOG.warning("Forene import failed for %s: %s", path, exc)
        raise ValueError(f"{path.name}: {exc}") from exc
    LOG.info("Imported %s (%d floors)", path.name, len(floors))
    return floors


def placement(floors, gap_percent=REFERENCE_GAP):
    if not floors:
        raise ValueError("Insert a floor plan first.")
    sizes = [(pdf.Rect(f.bounds).width*f.scale, pdf.Rect(f.bounds).height*f.scale)
             for f in floors]
    gap = median(w for w, _ in sizes)*gap_percent/100
    datum = max(h*f.datum for f, (_, h) in zip(floors, sizes))
    x = 0.
    rects = []
    for floor, (width, height) in zip(floors, sizes):
        y = datum - height*floor.datum
        rects.append(pdf.Rect(x, y, x+width, y+height))
        x += width + gap
    return rects, datum


def compose(floors, gap_percent, footer):
    rects, _ = placement(floors, gap_percent)
    width, height = rects[-1].x1, max(r.y1 for r in rects)
    footer_rect = None
    if footer:
        with Image.open(FOOTER) as image:
            fw = width / 4
            fh = fw * image.height / image.width
        inset = min(2.16, width / 20)
        footer_rect = pdf.Rect(width-fw-inset, height+6, width-inset, height+6+fh)
        height = footer_rect.y1 + inset
    with pdf.open() as out:
        page = out.new_page(width=width, height=height)
        for f, target in zip(floors, rects):
            with pdf.open(stream=f.source, filetype="pdf") as src:
                page.show_pdf_page(target, src, 0, clip=pdf.Rect(f.bounds))
        if footer_rect:
            page.insert_image(footer_rect, filename=str(FOOTER))
        return out.tobytes(garbage=3, deflate=True)


def export_sheet(sheet, folder, fmt):
    """Publish complete bytes without overwriting an existing or concurrent export."""
    if fmt == "jpg":
        with pdf.open(stream=sheet, filetype="pdf") as doc:
            page = doc[0]
            size = [math.ceil(v * 300/72) for v in (page.rect.width, page.rect.height)]
            if max(size) > 65500 or math.prod(size) > 80_000_000:
                raise ValueError("Sheet exceeds the 300 dpi JPEG limit. Choose PDF or reduce scale.")
            sheet = page.get_pixmap(dpi=300).tobytes("jpeg", jpg_quality=95)
    elif fmt != "pdf":
        raise ValueError("Choose JPEG or PDF.")
    with NamedTemporaryFile(dir=folder, suffix=".tmp", delete=False) as temp:
        staged = Path(temp.name)
    try:
        staged.write_bytes(sheet)
        index = 1
        while True:
            suffix = "" if index == 1 else f"_{index}"
            target = Path(folder) / f"Forene{suffix}.{fmt}"
            try:
                if os.name == "nt":
                    staged.rename(target)
                else:
                    os.link(staged, target)
                LOG.info("Exported %s", target)
                return target
            except FileExistsError:
                index += 1
    finally:
        staged.unlink(missing_ok=True)
