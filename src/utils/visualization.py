"""
src/utils/visualization.py
Image grid, comparison, and annotation utilities.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont


def make_image_grid(
    images: List[Image.Image],
    cols: Optional[int] = None,
    rows: Optional[int] = None,
    padding: int = 4,
    bg_color: Tuple[int, int, int] = (20, 20, 20),
) -> Image.Image:
    """
    Arrange PIL images into a grid.

    Parameters
    ----------
    images:
        List of PIL images (can be different sizes; will be resized to match first).
    cols:
        Number of columns. Inferred from rows if not given.
    rows:
        Number of rows. Inferred from cols if not given.
    padding:
        Pixel gap between images.
    bg_color:
        Background colour (RGB tuple).
    """
    if not images:
        raise ValueError("No images to arrange.")

    n = len(images)
    if cols is None and rows is None:
        cols = math.ceil(math.sqrt(n))
    if cols is None:
        cols = math.ceil(n / rows)
    if rows is None:
        rows = math.ceil(n / cols)

    # Normalise all images to the same size
    w, h = images[0].size
    images = [img.resize((w, h), Image.LANCZOS) for img in images]

    grid_w = cols * w + (cols + 1) * padding
    grid_h = rows * h + (rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), bg_color)

    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        x = padding + col * (w + padding)
        y = padding + row * (h + padding)
        grid.paste(img, (x, y))

    return grid


def annotate_image(
    image: Image.Image,
    label: str,
    position: str = "bottom",
    font_size: int = 18,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 160),
) -> Image.Image:
    """Overlay a text label onto an image with a translucent background."""
    img = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = 6

    if position == "bottom":
        x = (img.width - text_w) // 2
        y = img.height - text_h - pad * 2
    elif position == "top":
        x = (img.width - text_w) // 2
        y = pad
    else:
        x, y = position  # tuple (x, y)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [x - pad, y - pad, x + text_w + pad, y + text_h + pad],
        fill=bg_color,
    )
    img = Image.alpha_composite(img, overlay)
    ImageDraw.Draw(img).text((x, y), label, fill=text_color, font=font)
    return img.convert("RGB")


def make_comparison_strip(
    images: List[Image.Image],
    labels: Optional[List[str]] = None,
    padding: int = 6,
    bg_color: Tuple[int, int, int] = (15, 15, 15),
) -> Image.Image:
    """
    Create a horizontal strip comparing multiple images side-by-side.

    Parameters
    ----------
    images:
        Images to compare.
    labels:
        Optional text labels overlaid on each image.
    """
    if labels and len(labels) != len(images):
        raise ValueError("labels length must match images length")

    if labels:
        images = [annotate_image(img, lbl) for img, lbl in zip(images, labels)]

    w, h = images[0].size
    images = [img.resize((w, h), Image.LANCZOS) for img in images]

    n = len(images)
    strip_w = n * w + (n + 1) * padding
    strip_h = h + 2 * padding
    strip = Image.new("RGB", (strip_w, strip_h), bg_color)

    for i, img in enumerate(images):
        x = padding + i * (w + padding)
        strip.paste(img, (x, padding))

    return strip


def save_comparison(
    before: Image.Image,
    after: Image.Image,
    path: Union[str, Path],
    labels: Tuple[str, str] = ("Before", "After"),
) -> None:
    """Save a two-image before/after comparison."""
    strip = make_comparison_strip([before, after], labels=list(labels))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    strip.save(str(path))
