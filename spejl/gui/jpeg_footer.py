"""Compose the company footer on a cropped 300-dpi plan or a scaled copy."""
from functools import lru_cache
from pathlib import Path
from PIL import Image


@lru_cache(maxsize=1)
def _artwork():
    with Image.open(Path(__file__).parent / "assets" / "2d-footer.png") as image:
        return image.copy()


def with_footer(plan, dpi=300, bottom_margin=12):
    """Append white space below the plan, counting its existing bottom margin."""
    artwork = _artwork()
    width = max(1, round(plan.width / 4))
    height = max(1, round(width * artwork.height / artwork.width))
    scale = dpi / 300
    inset = round(9 * scale)
    band = max(round(86 * scale), height + round(48 * scale))
    lettering = artwork.resize((width, height), Image.Resampling.LANCZOS)
    ink = lettering.point(lambda value: 255 if value < 170 else 0).getbbox()
    if ink:
        band += ink[3] - ink[1]
    canvas = Image.new("RGB", (plan.width, plan.height + max(0, band - bottom_margin)), "white")
    canvas.paste(plan, (0, 0))
    canvas.paste(lettering,
                 (plan.width - inset - width, canvas.height - inset - height))
    return canvas
