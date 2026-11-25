"""Geometry and overflow check for the generated deck.

There is no PowerPoint renderer on this host, so a slide that overflows its
box looks fine to the generator and wrong to the audience. This reads the
saved .pptx back and reports two things per slide:

  * out of bounds - any shape whose box leaves the 13.333 x 7.5 inch slide;
  * estimated spill - a text frame whose wrapped text, measured with the same
    character-width estimate make_slides.py lays out with, runs past something
    that would actually show: the bottom of the panel it sits in, the top of
    the next shape under it, or the bottom of the slide.

A text box may legitimately be shorter than its text, because PowerPoint does
not clip: the generator sizes many boxes loosely and lets the text flow into
empty space. Only a spill into occupied space or off the slide is a defect, so
that is what this counts.

Line breaking is measured with real font metrics rather than a
characters-per-line guess: PIL wraps each paragraph with DejaVu Sans at the
run's own point size. DejaVu is wider than the deck's Segoe UI at every size,
so the measurement errs towards reporting a spill that will not happen and
never towards missing one.

  python -m wbs.task5_demo.check_deck
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "wbs" / "deliverables" / "presentation" / "Ontology_Business_Plan.pptx"
W, H = Inches(13.333), Inches(7.5)
SLACK = Inches(0.06)          # a shape may sit this far outside before it counts


_FONTS: dict[int, object] = {}


def _font(size_pt: float):
    """DejaVu Sans at `size_pt`, rendered at 4x so rounding is not the error."""
    import matplotlib
    from PIL import ImageFont
    px = max(6, int(round(size_pt * 4)))
    if px not in _FONTS:
        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        _FONTS[px] = ImageFont.truetype(str(ttf), px)
    return _FONTS[px]


def wrapped_lines(text: str, w_pt: float, size_pt: float) -> int:
    f = _font(size_pt)
    budget = w_pt * 4                       # same 4x scale as the font
    n = 0
    for para in text.split("\n"):
        words, line = para.split(), ""
        if not words:
            n += 1
            continue
        k = 0
        for wd in words:
            trial = f"{line} {wd}".strip()
            if f.getlength(trial) <= budget or not line:
                line = trial
            else:
                k += 1
                line = wd
        n += k + 1
    return n


def est_height(text: str, w_emu: int, size: float) -> float:
    """Wrapped height in EMU, from measured line breaks."""
    w_pt = w_emu / Inches(1).emu * 72
    return Pt(wrapped_lines(text, max(8.0, w_pt), size) * size * 1.22).emu


def main() -> None:
    if not DECK.exists():
        sys.exit(f"{DECK} not found - run make_slides first")
    prs = Presentation(str(DECK))
    oob, over = [], []
    for i, sl in enumerate(prs.slides, 1):
        shapes = [sh for sh in sl.shapes
                  if sh.left is not None and sh.top is not None]
        # panels: filled rectangles big enough to be a card, which a text box
        # inside is expected to stay within
        panels = [sh for sh in shapes
                  if (sh.width or 0) > Inches(0.8) and (sh.height or 0) > Inches(0.4)
                  and sh.shape_type is not None and not sh.has_text_frame
                  or (sh.shape_type is not None and sh.has_text_frame
                      and not sh.text_frame.text.strip()
                      and (sh.height or 0) > Inches(0.4))]
        for sh in shapes:
            r, b = sh.left + (sh.width or 0), sh.top + (sh.height or 0)
            if (sh.left < -SLACK or sh.top < -SLACK
                    or r > W + SLACK or b > H + SLACK):
                oob.append((i, sh.shape_type, round(sh.left / 914400, 2),
                            round(sh.top / 914400, 2), round(r / 914400, 2),
                            round(b / 914400, 2)))
            if not sh.has_text_frame or not sh.text_frame.text.strip():
                continue
            tf = sh.text_frame
            need = 0.0
            for para in tf.paragraphs:
                sz = max([(rr.font.size.pt if rr.font.size else 18)
                          for rr in para.runs] or [18])
                txt = "".join(rr.text for rr in para.runs)
                need += est_height(txt, max(1, (sh.width or 0)
                                            - tf.margin_left - tf.margin_right), sz)
                need += Pt(para.space_after.pt if para.space_after else 0).emu
            bottom = sh.top + tf.margin_top + need
            # how far down may this text go before it hits something?
            limit = H - Inches(0.1)
            for pa in panels:
                if (pa.left - SLACK <= sh.left and pa.top - SLACK <= sh.top
                        and pa.left + (pa.width or 0) + SLACK >= r
                        and pa.top + (pa.height or 0) > sh.top):
                    limit = min(limit, pa.top + (pa.height or 0) - Inches(0.02))
            for ot in shapes:
                if ot is sh or not ot.has_text_frame:
                    continue
                if not ot.text_frame.text.strip():
                    continue
                # a shape below this one, horizontally overlapping it
                if (ot.top > sh.top + Pt(4).emu
                        and ot.left < r and ot.left + (ot.width or 0) > sh.left):
                    limit = min(limit, ot.top)
            if bottom > limit + Pt(4).emu:
                over.append((i, round((bottom - limit) / 914400, 2),
                             round(need / 914400, 2),
                             tf.text[:70].replace("\n", " / ")))
    print(f"{len(prs.slides._sldIdLst)} slides")
    print(f"out of bounds:        {len(oob)}")
    for x in oob[:12]:
        print(f"   slide {x[0]}: {x[1]} box ({x[2]},{x[3]})-({x[4]},{x[5]}) in")
    print(f"estimated spills:     {len(over)}")
    for x in sorted(over, key=lambda y: -y[1])[:16]:
        print(f"   slide {x[0]}: spills {x[1]}\" (text needs {x[2]}\") | {x[3]}")
    if oob or over:
        sys.exit(1)


if __name__ == "__main__":
    main()
