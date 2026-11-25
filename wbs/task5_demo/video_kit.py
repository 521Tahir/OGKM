"""Drawing and timing helpers for the demo video.

There is no video editor on this box, so the film is drawn frame by frame with
PIL and piped to ffmpeg. These are the pieces every scene uses: the palette, a
word-wrapping text renderer that can colour individual spans, eased animation,
and the small chrome (chapter strip, lower third) that makes the result look
like a film rather than a slideshow.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30

NAVY = (11, 31, 58)
NAVY2 = (18, 46, 85)
INK = (20, 34, 51)
PAPER = (246, 248, 251)
WHITE = (255, 255, 255)
MUTE = (138, 155, 175)
TEAL = (0, 163, 161)
AMBER = (242, 169, 0)
RED = (216, 69, 69)
GREEN = (46, 174, 110)
LINE = (38, 60, 92)

_TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    # a size animated from 0 is a legitimate thing for a scene to ask for and an
    # error for FreeType, so the floor lives here rather than at every call site
    size = max(6, int(size))
    key = (kind, size)
    if key not in _CACHE:
        name = {"bold": "DejaVuSans-Bold.ttf", "regular": "DejaVuSans.ttf",
                "mono": "DejaVuSansMono.ttf", "monob": "DejaVuSansMono-Bold.ttf",
                "italic": "DejaVuSans-Oblique.ttf"}[kind]
        _CACHE[key] = ImageFont.truetype(str(_TTF / name), size)
    return _CACHE[key]


# ── easing ───────────────────────────────────────────────────────────────
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def ease(x: float) -> float:
    """Smooth in and out; the standard cosine ease."""
    return 0.5 - 0.5 * math.cos(math.pi * clamp(x))


def ease_out(x: float) -> float:
    return 1 - (1 - clamp(x)) ** 3


def seg(t: float, start: float, dur: float) -> float:
    """Progress of a sub-animation that starts at `start` and lasts `dur`."""
    return clamp((t - start) / dur) if dur > 0 else 1.0


# ── canvas ───────────────────────────────────────────────────────────────
def canvas(dark: bool = True) -> Image.Image:
    im = Image.new("RGB", (W, H), NAVY if dark else PAPER)
    if dark:                                   # a very slight vertical lift
        a = np.linspace(0, 26, H, dtype=np.float32)[:, None, None]
        arr = np.asarray(im, dtype=np.float32) + a
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return im


def fade(im: Image.Image, k: float) -> Image.Image:
    """k=1 full image, k=0 black."""
    if k >= 0.999:
        return im
    return Image.blend(Image.new("RGB", im.size, (0, 0, 0)), im, clamp(k))


def rounded(d: ImageDraw.ImageDraw, box, r, fill=None, outline=None, width=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


# ── text ─────────────────────────────────────────────────────────────────
def measure(d: ImageDraw.ImageDraw, s: str, f) -> int:
    return int(d.textlength(s, font=f))


def wrap(d, s: str, f, width: int) -> list[str]:
    out, line = [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if measure(d, trial, f) <= width or not line:
            line = trial
        else:
            out.append(line)
            line = word
    if line:
        out.append(line)
    return out


def pill(d, cx, cy, text: str, f, fill, outline, fg=WHITE, pad_x=22, pad_y=13,
         width=3, k: float = 1.0):
    """A rounded label box centred on (cx, cy), sized to its text.

    Graph nodes in this film carry place names and predicate names of very
    different lengths, so the box is measured from the string rather than
    given a fixed width - a node that clips its own label is worse than no
    diagram at all.
    """
    w = measure(d, text, f) + pad_x * 2
    h = f.size + pad_y * 2
    box = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
    if k < 1.0:
        cw, ch = int(w * k) // 2, int(h * k) // 2
        box = (cx - cw, cy - ch, cx + cw, cy + ch)
    rounded(d, box, min(16, max(6, int(h * k) // 2)), fill=fill, outline=outline,
            width=width)
    if k > 0.55:
        draw_text(d, (cx, cy + 1), text, f, fg, anchor="mm")
    return box


def edge(d, a, b, col, width=3, k: float = 1.0, label: str = "", lf=None):
    """A line from a to b, drawn k of the way, with an optional mid-label."""
    x0, y0 = a
    x1, y1 = b
    xe, ye = x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
    d.line((x0, y0, xe, ye), fill=col, width=width)
    if label and k > 0.8 and lf is not None:
        draw_text(d, ((x0 + x1) // 2, (y0 + y1) // 2 - 4), label, lf, col, anchor="mm")


def fit_font(d, s: str, kind: str, size: int, width: int, floor: int = 12):
    """The largest `kind` font at or below `size` whose rendering of `s` fits
    `width` pixels. A one-line label that overflows its card looks like a bug
    to an audience, so labels that can grow with the data ask for this."""
    f = font(kind, size)
    while size > floor and measure(d, s, f) > width:
        size -= 1
        f = font(kind, size)
    return f


def draw_text(d, xy, s: str, f, fill=WHITE, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def draw_block(d, x, y, s: str, f, width: int, fill=WHITE, leading=1.35,
               reveal: float = 1.0, spans: dict | None = None,
               strike: set | None = None) -> int:
    """Wrapped paragraph with optional typewriter reveal and per-span colour.

    `spans` maps a literal substring to a colour; every word of that substring
    is drawn in it, which is how the film marks a claim the graph cannot back.
    """
    lines = wrap(d, s, f, width)
    lh = int(f.size * leading)
    shown = int(len(s) * clamp(reveal))
    seen = 0
    for i, line in enumerate(lines):
        cx = x
        for word in line.split(" "):
            if seen >= shown:
                return y + len(lines) * lh
            frag = word[: max(0, shown - seen)]
            col = fill
            if spans:
                for key, c in spans.items():
                    if word.strip(".,:;()\"'") and word.strip(".,:;()\"'") in key.split():
                        col = c
            d.text((cx, y + i * lh), frag, font=f, fill=col)
            wlen = measure(d, word + " ", f)
            if strike and word.strip(".,:;()\"'") in strike and len(frag) == len(word):
                yy = y + i * lh + int(f.size * 0.62)
                d.line((cx, yy, cx + measure(d, word, f), yy), fill=RED, width=3)
            cx += wlen
            seen += len(word) + 1
    return y + len(lines) * lh


# ── chrome ───────────────────────────────────────────────────────────────
def chapter(d, label: str, idx: int, total: int, k: float = 1.0):
    """Top-left chapter label and the progress ticks on the right."""
    d.rectangle((0, 0, W, 6), fill=TEAL)
    draw_text(d, (72, 44), label.upper(), font("bold", 22), TEAL)
    x0 = W - 72 - (total * 26 - 10)
    for i in range(total):
        c = TEAL if i < idx else (44, 68, 104)
        d.rounded_rectangle((x0 + i * 26, 48, x0 + i * 26 + 16, 54), 3, fill=c)


def lower_third(d, title: str, sub: str = "", y: int = H - 190, accent=TEAL):
    d.rectangle((72, y, 76, y + 86), fill=accent)
    draw_text(d, (100, y - 2), title, font("bold", 40), WHITE)
    if sub:
        draw_text(d, (100, y + 50), sub, font("regular", 24), MUTE)


def card(im: Image.Image, d, box, title: str, accent, tag: str = "",
         dark: bool = True):
    x0, y0, x1, y1 = box
    rounded(d, box, 14, fill=(16, 40, 72) if dark else WHITE,
            outline=(32, 58, 96) if dark else (222, 228, 236), width=2)
    d.rounded_rectangle((x0, y0, x1, y0 + 46), 14, fill=accent)
    d.rectangle((x0, y0 + 30, x1, y0 + 46), fill=accent)
    draw_text(d, (x0 + 20, y0 + 10), title, font("bold", 23), WHITE)
    if tag:
        draw_text(d, (x1 - 20, y0 + 12), tag, font("bold", 20), WHITE, anchor="ra")
    return x0 + 20, y0 + 66, x1 - 20


def counter(d, xy, value: float, suffix: str, label: str, f_size=86,
            color=TEAL, fmt="{:.0f}"):
    x, y = xy
    draw_text(d, (x, y), fmt.format(value) + suffix, font("bold", f_size), color)
    draw_text(d, (x, y + f_size + 6), label, font("regular", 24), MUTE)
