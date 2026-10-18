"""16:9 LinkedIn article cover — dollar index that is not rolling over."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dollar-index-cover.png"

W, H = 1920, 1080
SCALE = 2
SW, SH = W * SCALE, H * SCALE

BG = (10, 14, 18)
TEXT = (244, 246, 248)
AMBER = (224, 122, 61)
TEAL = (56, 168, 164)
BLUE = (76, 139, 245)

HN = "/System/Library/Fonts/HelveticaNeue.ttc"


def _hn(size: int, *, index: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(HN, size * SCALE, index=index)
    except OSError:
        return ImageFont.load_default()


def _glow(size, cx, cy, radius, color, alpha) -> Image.Image:
    w, h = size
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
    a = np.clip(1.0 - d, 0, 1) ** 2.2
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    arr[..., 3] = (a * alpha).astype(np.uint8)
    return Image.fromarray(arr, "RGBA").filter(ImageFilter.GaussianBlur(radius=int(80 * SCALE / 2)))


def _grain(size, amount: int = 12) -> Image.Image:
    w, h = size
    noise = np.random.default_rng(3).integers(0, amount, (h, w), dtype=np.uint8)
    arr = np.dstack([noise, noise, noise, np.full((h, w), 22, dtype=np.uint8)])
    return Image.fromarray(arr, "RGBA")


def _line_width(text: str, font, word_gap: int) -> int:
    words = text.split()
    widths = [font.getbbox(w)[2] - font.getbbox(w)[0] for w in words]
    return sum(widths) + word_gap * max(0, len(words) - 1)


def _draw_words(draw, x, y, text, font, fill, word_gap) -> None:
    for word in text.split():
        draw.text((x, y), word, font=font, fill=fill)
        x += (font.getbbox(word)[2] - font.getbbox(word)[0]) + word_gap


def _center_words(draw, y, text, font, fill, word_gap) -> None:
    tw = _line_width(text, font, word_gap)
    _draw_words(draw, (SW - tw) // 2, y, text, font, fill, word_gap)


def build_cover() -> Image.Image:
    base = Image.new("RGB", (SW, SH), BG).convert("RGBA")
    base = Image.alpha_composite(base, _glow((SW, SH), SW * 0.72, SH * 0.42, SW * 0.38, AMBER, 50))
    base = Image.alpha_composite(base, _glow((SW, SH), SW * 0.22, SH * 0.70, SW * 0.28, TEAL, 28))

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    n = 64
    xs = np.linspace(int(SW * 0.06), int(SW * 0.94), n)
    t = np.linspace(0, 1, n)
    # Sideways-to-higher path: the decline that did not arrive.
    ys = SH * (0.72 - 0.18 * t + 0.035 * np.sin(7 * np.pi * t) * (1 - 0.4 * t) - 0.06 * (t**1.6))
    pts = [(int(x), int(y)) for x, y in zip(xs, ys)]
    draw.line(pts, fill=(*AMBER, 220), width=8 * SCALE)
    # Faint euro-weighted peer that did not rally enough to sink DXY
    ys2 = SH * (0.58 + 0.04 * np.sin(3 * np.pi * t) + 0.02 * t)
    pts2 = [(int(x), int(y)) for x, y in zip(xs, ys2)]
    draw.line(pts2, fill=(*BLUE, 90), width=3 * SCALE)
    base.alpha_composite(overlay.filter(ImageFilter.GaussianBlur(radius=max(1, SCALE // 2))))
    base = Image.alpha_composite(base, _grain((SW, SH)))

    draw = ImageDraw.Draw(base)
    thin = _hn(72, index=12)
    bold = _hn(78, index=1)
    gap = 10 * SCALE
    word_gap = 22 * SCALE
    line1, line2 = "Why the Dollar", "Isn't Falling"
    h1 = thin.getbbox("Why")[3] - thin.getbbox("Why")[1]
    h2 = bold.getbbox("Isn't")[3] - bold.getbbox("Isn't")[1]
    block = h1 + gap + h2
    y0 = (SH - block) // 2 - 10 * SCALE
    _center_words(draw, y0, line1, thin, TEXT, word_gap)
    _center_words(draw, y0 + h1 + gap, line2, bold, TEXT, word_gap)

    cap = _hn(16, index=10)
    _center_words(draw, int(SH * 0.86), "ICE DXY  ·  14 Oct CPI  ·  18-month high", cap, (*AMBER, 210), 12 * SCALE)
    return base.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)


def main() -> None:
    cover = build_cover()
    cover.save(OUT, quality=96, subsampling=0)
    print(f"Wrote {OUT} ({cover.size[0]}x{cover.size[1]})")


if __name__ == "__main__":
    main()
