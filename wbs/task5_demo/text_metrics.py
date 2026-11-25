"""Measured text metrics, shared by the deck generator and the deck checker.

There is no PowerPoint renderer on this host, so a slide's layout has to be
computed from an estimate of how text wraps. The first version of that estimate
was a characters-per-line guess, which is wrong in both directions: it breaks
"Jeju Folklore & Natural History Museum" too early and a column of digits too
late.

This measures instead. PIL wraps each paragraph with DejaVu Sans at the run's
own point size, rendered at 4x so integer rounding is not the error term.
DejaVu Sans is wider than the deck's Segoe UI at every size, so a layout that
fits here fits there, and the error only ever makes a box too generous.
"""
from __future__ import annotations

from pathlib import Path

from pptx.util import Emu, Inches, Pt

_FONTS: dict[tuple[str, int], object] = {}
EMU_PER_PT = Inches(1).emu / 72.0


def _font(size_pt: float, mono: bool = False):
    import matplotlib
    from PIL import ImageFont
    px = max(6, int(round(size_pt * 4)))
    key = ("mono" if mono else "sans", px)
    if key not in _FONTS:
        name = "DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf"
        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
        _FONTS[key] = ImageFont.truetype(str(ttf), px)
    return _FONTS[key]


def lines(text: str, width_pt: float, size_pt: float, mono: bool = False) -> int:
    """How many lines `text` takes when wrapped to `width_pt`."""
    f = _font(size_pt, mono)
    budget = max(8.0, width_pt) * 4
    n = 0
    for para in str(text).split("\n"):
        words = para.split()
        if not words:
            n += 1
            continue
        line, breaks = "", 0
        for w in words:
            trial = f"{line} {w}".strip()
            if f.getlength(trial) <= budget or not line:
                line = trial
            else:
                breaks += 1
                line = w
        n += breaks + 1
    return n


def height_pt(text: str, width_pt: float, size_pt: float, leading: float = 1.22,
              mono: bool = False) -> float:
    return lines(text, width_pt, size_pt, mono) * size_pt * leading


def height(text, width_emu, size_pt: float, leading: float = 1.22,
           mono: bool = False) -> Emu:
    """Wrapped height in EMU for a box `width_emu` wide."""
    return Emu(int(height_pt(text, width_emu / EMU_PER_PT, size_pt, leading,
                             mono) * EMU_PER_PT))


def fits(text, width_emu, height_emu, size_pt: float, leading: float = 1.22,
         mono: bool = False) -> bool:
    return height(text, width_emu, size_pt, leading, mono) <= height_emu


def shrink_to_fit(text, width_emu, height_emu, size_pt: float, floor: float = 8.0,
                  leading: float = 1.22, mono: bool = False) -> float:
    """The largest size at or below `size_pt` whose wrapped text fits the box."""
    s = float(size_pt)
    while s > floor and not fits(text, width_emu, height_emu, s, leading, mono):
        s -= 0.5
    return s
