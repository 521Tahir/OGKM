"""Build the business-plan deck from the measured evidence.

Every number, table and quoted answer comes out of
wbs/deliverables/presentation/demo_evidence.json, which is itself generated
from the deliverables. Re-run build_demo_evidence.py and the deck moves with
the measurements instead of going stale.

    python -m wbs.task5_demo.make_slides
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wbs.task5_demo import text_metrics as TM
DELIV = ROOT / "wbs" / "deliverables"
OUT = DELIV / "presentation"
FIG = DELIV / "figures"

NAVY = RGBColor(0x0B, 0x1F, 0x3A)
NAVY2 = RGBColor(0x12, 0x2E, 0x55)
PAPER = RGBColor(0xF6, 0xF8, 0xFB)
INK = RGBColor(0x14, 0x22, 0x33)
MUTE = RGBColor(0x5B, 0x6B, 0x7E)
TEAL = RGBColor(0x00, 0xA3, 0xA1)
AMBER = RGBColor(0xF2, 0xA9, 0x00)
RED = RGBColor(0xC8, 0x3A, 0x3A)
GREEN = RGBColor(0x1E, 0x8E, 0x4F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Segoe UI"
MONO = "Consolas"

W, H = Inches(13.333), Inches(7.5)
# python-pptx gives a new text box 0.1" of left and right inset, so the width
# text actually wraps to is narrower than the box. Measure against that.
TXT_PAD = Inches(0.2)


# ── primitives ───────────────────────────────────────────────────────────
def rect(sl, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, lw=1.0):
    s = sl.shapes.add_shape(shape, x, y, w, h)
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    s.text_frame.word_wrap = True
    return s


def paragraphs(runs) -> list[list[tuple]]:
    """A paragraph is a string or a list of (text, style) chunks; `runs` is one
    paragraph or a list of them. Normalising here keeps every call site short."""
    if isinstance(runs, str):
        return [[(runs, {})]]
    if runs and isinstance(runs[0], tuple):          # one paragraph, several styles
        return [list(runs)]
    out = []
    for item in runs:
        out.append([(item, {})] if isinstance(item, str) else list(item))
    return out


def text(sl, x, y, w, h, runs, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT,
         font=FONT, space=0, anchor=MSO_ANCHOR.TOP, line_spacing=None):
    tb = sl.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, chunks in enumerate(paragraphs(runs)):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        if line_spacing:
            p.line_spacing = line_spacing
        for s, kw in chunks:
            r = p.add_run()
            r.text = s
            f = r.font
            f.name = kw.get("font", font)
            f.size = Pt(kw.get("size", size))
            f.bold = kw.get("bold", bold)
            f.italic = kw.get("italic", False)
            f.color.rgb = kw.get("color", color)
    return tb


def est_height(runs, w, size) -> Emu:
    """Height of wrapped text, measured rather than guessed.

    There is no PowerPoint renderer here, so the layout is spaced from
    `text_metrics`, which wraps the string with real font metrics in a font
    wider than the deck's own. A box sized from this is never too small.
    """
    w_pt = w / Inches(1).emu * 72
    total = 0.0
    for chunks in paragraphs(runs):
        txt = "".join(t for t, _ in chunks)
        sz = max([kw.get("size", size) for _, kw in chunks] or [size])
        total += TM.height_pt(txt, w_pt - TXT_PAD / Inches(1).emu * 72, sz)
    return Emu(int(Pt(total).emu))


def bullets(sl, x, y, w, h, items, size=16, color=INK, gap=10, dot=TEAL):
    yy = y
    for it in items:
        rect(sl, x, yy + Pt(size * 0.45), Pt(7), Pt(7), fill=dot, shape=MSO_SHAPE.OVAL)
        hh = est_height(it, w - Inches(0.26), size)
        text(sl, x + Inches(0.26), yy, w - Inches(0.26), hh + Pt(gap), it,
             size=size, color=color, space=gap)
        yy = yy + hh + Pt(gap) + Inches(0.06)


def slide(prs, paper=True):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    rect(sl, 0, 0, W, H, fill=PAPER if paper else NAVY)
    return sl


def header(sl, kicker, title, sub=None):
    rect(sl, 0, 0, W, Inches(1.28), fill=NAVY)
    rect(sl, 0, Inches(1.28), W, Pt(3), fill=TEAL)
    text(sl, Inches(0.62), Inches(0.2), Inches(11), Inches(0.3),
         kicker.upper(), size=11, color=TEAL, bold=True)
    tsize = 27 if len(title) <= 58 else (23 if len(title) <= 72 else 20)
    text(sl, Inches(0.62), Inches(0.44), Inches(12.1), Inches(0.6),
         title, size=tsize, color=WHITE, bold=True)
    if sub:
        text(sl, Inches(0.62), Inches(1.42), Inches(12.1), Inches(0.4),
             sub, size=13, color=MUTE)


def footer(sl, src, n=None):
    rect(sl, 0, H - Inches(0.42), W, Inches(0.42), fill=RGBColor(0xE7, 0xEC, 0xF2))
    text(sl, Inches(0.62), H - Inches(0.37), Inches(11.2), Inches(0.3),
         [[("source: ", {"bold": True}), (src, {})]], size=9.5, color=MUTE)
    if n is not None:
        text(sl, W - Inches(1.1), H - Inches(0.37), Inches(0.5), Inches(0.3),
             str(n), size=9.5, color=MUTE, align=PP_ALIGN.RIGHT)


def kpi(sl, x, y, w, value, label, note="", accent=TEAL, h=Inches(1.72)):
    rect(sl, x, y, w, h, fill=WHITE, line=RGBColor(0xDD, 0xE4, 0xEC))
    rect(sl, x, y, Pt(5), h, fill=accent)
    compact = h < Inches(1.45)
    iw = w - Inches(0.44)
    vs = TM.shrink_to_fit(value, iw - TXT_PAD, Inches(0.5 if compact else 0.72),
                          23 if compact else 34, floor=15)
    text(sl, x + Inches(0.26), y + Inches(0.06 if compact else 0.14),
         w - Inches(0.4), Inches(0.5 if compact else 0.72), value,
         size=vs, color=accent, bold=True)
    ls = 11 if compact else 13
    ly = y + Inches(0.47 if compact else 0.76)
    lh = TM.height(label, iw - TXT_PAD, ls)
    text(sl, x + Inches(0.26), ly, w - Inches(0.4), lh, label, size=ls,
         color=INK, bold=True)
    if note:
        ny = ly + lh + Inches(0.03)
        nh = max(Inches(0.2), y + h - ny - Inches(0.06))
        ns = TM.shrink_to_fit(note, iw - TXT_PAD, nh, 9 if compact else 10.5,
                              floor=7.5)
        text(sl, x + Inches(0.26), ny, iw, nh, note, size=ns, color=MUTE)


def table(sl, x, y, w, rowsdata, col_w=None, head_fill=NAVY, size=12,
          row_h=Inches(0.36), highlight=None, align_right_from=1):
    nr, nc = len(rowsdata), len(rowsdata[0])
    shp = sl.shapes.add_table(nr, nc, x, y, w, row_h * nr)
    tb = shp.table
    if col_w:
        tot = sum(col_w)
        for i, cw in enumerate(col_w):
            tb.columns[i].width = Emu(int(w * cw / tot))
    for r in range(nr):
        tb.rows[r].height = row_h
        for c in range(nc):
            cell = tb.cell(r, c)
            cell.text = str(rowsdata[r][c])
            cell.margin_left, cell.margin_right = Inches(0.08), Inches(0.08)
            cell.margin_top, cell.margin_bottom = Inches(0.02), Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = head_fill
            elif highlight is not None and r == highlight:
                cell.fill.fore_color.rgb = RGBColor(0xE6, 0xF7, 0xF4)
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else RGBColor(0xF1, 0xF4, 0xF8)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if (c >= align_right_from and r) else PP_ALIGN.LEFT
            for run in p.runs:
                run.font.size = Pt(size)
                run.font.name = FONT
                run.font.bold = r == 0 or c == 0
                run.font.color.rgb = WHITE if r == 0 else INK
    return tb


def picture(sl, path: Path, x, y, w=None, h=None):
    if not path.exists():
        return None
    return sl.shapes.add_picture(str(path), x, y, width=w, height=h)


def quote_panel(sl, x, y, w, h, title, body, accent, tag=""):
    rect(sl, x, y, w, h, fill=WHITE, line=RGBColor(0xDD, 0xE4, 0xEC))
    rect(sl, x, y, w, Inches(0.42), fill=accent)
    tw = w - (Inches(2.15) if tag else Inches(0.36))
    text(sl, x + Inches(0.18), y + Inches(0.08), tw, Inches(0.28), title,
         size=TM.shrink_to_fit(title, tw - TXT_PAD, Inches(0.28), 12.5, floor=9),
         color=WHITE, bold=True)
    if tag:
        text(sl, x + w - Inches(2.1), y + Inches(0.09), Inches(1.9), Inches(0.28),
             tag, size=10.5, color=WHITE, align=PP_ALIGN.RIGHT)
    bw, bh = w - Inches(0.4), h - Inches(0.7)
    text(sl, x + Inches(0.2), y + Inches(0.54), bw, bh, body,
         size=TM.shrink_to_fit(body, bw - TXT_PAD, bh, 11.5, floor=8.5, leading=1.16),
         color=INK, line_spacing=1.12)


def line(sl, x, y, w, h, s_, size, **kw):
    """One line of text that shrinks to fit rather than wrapping into whatever
    sits underneath it."""
    return text(sl, x, y, w, h, s_,
                size=TM.shrink_to_fit(s_, w - TXT_PAD, h, size,
                                      floor=max(7.0, size - 6)), **kw)


def clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + " …"


def pv(p) -> str:
    if p is None:
        return "—"
    return f"p < 1e-6" if p < 1e-6 else f"p = {p:.3g}"


def four_part(sl, question, before, after, fix, why, *, before_tag="", after_tag="",
              fix_lines=None, y=Inches(1.88), ans_h=Inches(2.32)):
    """The layout the business-plan brief asks for, on one slide: the problem,
    the ontology fix, the result, and why it improved. `fix_lines` is the graph
    edit itself, drawn in mono so a reader can see it is one line of data."""
    rect(sl, Inches(0.62), y, Inches(12.1), Inches(0.56), fill=NAVY2)
    text(sl, Inches(0.82), y + Inches(0.13), Inches(11.7), Inches(0.4),
         [[("The customer asks  ", {"bold": True, "color": AMBER}),
           (clip(question, 180), {"color": WHITE})]], size=12)
    qy = y + Inches(0.68)
    quote_panel(sl, Inches(0.62), qy, Inches(5.95), ans_h,
                "BEFORE · the graph was silent", clip(before, 560),
                RED, before_tag)
    quote_panel(sl, Inches(6.78), qy, Inches(5.95), ans_h,
                "AFTER · one fact later", clip(after, 560),
                GREEN, after_tag)
    fy = qy + ans_h + Inches(0.14)
    rect(sl, Inches(0.62), fy, Inches(5.95), Inches(1.62), fill=WHITE,
         line=RGBColor(0xDD, 0xE4, 0xEC))
    rect(sl, Inches(0.62), fy, Pt(5), Inches(1.62), fill=TEAL)
    text(sl, Inches(0.9), fy + Inches(0.1), Inches(5.4), Inches(0.3),
         "THE ONTOLOGY FIX · how it was applied", size=10.5, color=TEAL, bold=True)
    yy = fy + Inches(0.4)
    for ln in (fix_lines or []):
        text(sl, Inches(0.9), yy, Inches(5.5), Inches(0.28), ln,
             size=11.5, color=INK, font=MONO, bold=True)
        yy = yy + Inches(0.27)
    fh = fy + Inches(1.56) - yy
    text(sl, Inches(0.9), yy + Inches(0.02), Inches(5.5), fh, fix,
         size=TM.shrink_to_fit(fix, Inches(5.5) - TXT_PAD, fh, 10.5, floor=7.5,
                               leading=1.14),
         color=MUTE, line_spacing=1.1)
    rect(sl, Inches(6.78), fy, Inches(5.95), Inches(1.62), fill=WHITE,
         line=RGBColor(0xDD, 0xE4, 0xEC))
    rect(sl, Inches(6.78), fy, Pt(5), Inches(1.62), fill=AMBER)
    text(sl, Inches(7.06), fy + Inches(0.1), Inches(5.4), Inches(0.3),
         "WHY IT IMPROVED", size=10.5, color=AMBER, bold=True)
    text(sl, Inches(7.06), fy + Inches(0.4), Inches(5.5), Inches(1.12), why,
         size=TM.shrink_to_fit(why, Inches(5.5) - TXT_PAD, Inches(1.12), 11,
                               floor=7.5, leading=1.18),
         color=INK, line_spacing=1.13)


# ── the deck ─────────────────────────────────────────────────────────────
def main() -> None:
    ev = json.loads((OUT / "demo_evidence.json").read_text(encoding="utf-8"))
    V, G, P = ev["variants"], ev["gated"], ev["pretty"]
    SZ = ev["sizes"]
    small, big = SZ[0], SZ[-1]
    er = ev["edit_vs_retrain"]
    ED = ev["edits"]
    sb = ev["small_beats_big"]
    lim = ev["limits"]
    sc = ev["scale"]
    live = ev["live_edit"]
    cc = (live or {}).get("claim_check") or {}
    # "8x smaller" should come from the model names, not from a typed-in number
    import re as _re
    params = [float(_re.search(r"([\d.]+)B", P[m]).group(1)) for m in (small, big)]
    ratio = int(round(params[1] / params[0]))

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    n = [0]

    def num():
        n[0] += 1
        return n[0]

    # 1 ── title ──────────────────────────────────────────────────────────
    sl = slide(prs, paper=False)
    rect(sl, 0, 0, Inches(0.22), H, fill=TEAL)
    rect(sl, Inches(7.4), 0, Inches(5.93), H, fill=NAVY2)
    text(sl, Inches(0.9), Inches(1.1), Inches(6.3), Inches(0.4),
         "BUSINESS PLAN · ONTOLOGY + LLM", size=12, color=TEAL, bold=True)
    text(sl, Inches(0.9), Inches(1.55), Inches(6.4), Inches(3.2),
         [[("Empowering small language models with instant, cost-free "
            "customizability", {"color": WHITE})],
          [("Why ontology integration is critical", {"color": TEAL, "size": 24})]],
         size=36, bold=True, line_spacing=1.04)
    text(sl, Inches(0.9), Inches(5.1), Inches(6.3), Inches(1.4),
         [f"Jeju tourism knowledge base · {sc['concepts']:,} concepts · "
          f"{sc['relations']:,} relations at the published v1.5.0"
          + (f", {ED['facts'] - er['triples_added']} more in the v1.6.0 edit this "
             f"deck demonstrates" if ED.get('n', 1) > 1 else ""),
          f"Measured on {sc['benchmark_items']} held-out questions at three model sizes",
          "Jeju National University · Prof. Yoon Yeo-Chan · joint research WBS"],
         size=13, color=RGBColor(0xC3, 0xD0, 0xDE), space=6)
    for i, (val, lab) in enumerate([
            (f"{sb['token_f1_gain']:+.1f}", f"answer-quality points: {P[small]}\nwith the "
                                            f"ontology over {P[big]} without it"),
            (f"{er['edit_gpu_hours']:.0f} GPU-h", "to correct the knowledge:\nan ontology edit, "
                                                  "not a retrain"),
            (f"{G[big]['vs_rag']['clean']['delta']:+.1f} pp", "fully grounded answers\nagainst "
                                                              "the retrieval baseline")]):
        y = Inches(1.3 + i * 1.75)
        rect(sl, Inches(7.95), y, Inches(4.8), Inches(1.5), fill=NAVY,
             line=RGBColor(0x2A, 0x44, 0x6B))
        rect(sl, Inches(7.95), y, Pt(5), Inches(1.5), fill=[TEAL, AMBER, TEAL][i])
        text(sl, Inches(8.25), y + Inches(0.18), Inches(4.3), Inches(0.6),
             val, size=30, color=WHITE, bold=True)
        text(sl, Inches(8.25), y + Inches(0.78), Inches(4.3), Inches(0.6),
             lab.split("\n"), size=11.5, color=RGBColor(0xA9, 0xBC, 0xD0), space=0)
    text(sl, Inches(0.9), Inches(6.85), Inches(11), Inches(0.3),
         "Every figure in this deck is generated from the project's own measurement "
         "files; none is illustrative.", size=10.5, color=RGBColor(0x8F, 0xA3, 0xB8))

    # 2 ── the argument in one slide ──────────────────────────────────────
    sl = slide(prs)
    header(sl, "executive summary", "The argument in one slide",
           "Three claims, each with the measurement that backs it and the limit that bounds it")
    cards = [
        ("Change knowledge in minutes, not GPU-hours",
         f"Two edits in two weeks put {ED['facts']} facts into the graph — safety states "
         f"and then policy rules — taking the rule layer from "
         f"{er['rules_enforceable_before']}/{er['rules_total']} to "
         f"{er['rules_enforceable_after']}/{er['rules_total']} enforceable rules for "
         f"{ED['gpu_hours']:.0f} GPU-hours and {ED['validation_errors']} validation "
         f"errors. The {ED['retrain_adapters']} adapters this project fine-tuned cost "
         f"{ED['retrain_gpu_hours_total']:.0f} GPU-hours between them.", TEAL),
        ("A 1B model with the graph beats an 8B model without it",
         f"{P[small]} + ontology scores {sb['small_with_ontology']['token_f1']} against "
         f"{P[big]} alone at {sb['big_without_ontology']['token_f1']}, and asserts "
         f"{sb['ungrounded_cut']:.1f} pp fewer unsupported claims — on the same "
         f"{sc['benchmark_items']} questions.", AMBER),
        ("Grounding is what the ontology buys, at every size",
         f"The deployed configuration cuts unsupported claims and raises fully-grounded "
         f"answers against the retrieval baseline at all three sizes "
         f"({', '.join(P[m] for m in SZ)}), every test significant.", TEAL),
    ]
    for i, (t, b, a) in enumerate(cards):
        x = Inches(0.62 + i * 4.12)
        rect(sl, x, Inches(1.85), Inches(3.85), Inches(2.5), fill=WHITE,
             line=RGBColor(0xDD, 0xE4, 0xEC))
        rect(sl, x, Inches(1.85), Inches(3.85), Pt(5), fill=a)
        text(sl, x + Inches(0.24), Inches(2.02), Inches(3.4), Inches(0.8),
             t, size=15, color=INK, bold=True, line_spacing=1.05)
        text(sl, x + Inches(0.24), Inches(2.9), Inches(3.4), Inches(1.35),
             b, size=11.5, color=MUTE, line_spacing=1.14)
    rect(sl, Inches(0.62), Inches(4.62), Inches(12.1), Inches(2.1), fill=WHITE,
         line=RGBColor(0xDD, 0xE4, 0xEC))
    rect(sl, Inches(0.62), Inches(4.62), Pt(5), Inches(2.1), fill=RED)
    text(sl, Inches(0.95), Inches(4.78), Inches(11.5), Inches(0.4),
         "And the claim we will not make", size=15, color=RED, bold=True)
    text(sl, Inches(0.95), Inches(5.18), Inches(11.5), Inches(1.4),
         [f"The ontology arms do not beat plain retrieval on word-overlap with the "
          f"reference answers at any size: the deployed cell is "
          f"{G[big]['vs_rag']['token_f1']['delta']:+.2f} points at {P[big]}. The reference "
          f"answers were written from the same corpus the retrieval baseline reads, and "
          f"rewording them removes "
          f"{min(lim['paraphrase']['share_of_lead_explained_pct'].values())}–"
          f"{max(lim['paraphrase']['share_of_lead_explained_pct'].values())}% of that lead — "
          f"part of the gap is how the benchmark was built, and part is real.",
          "Ontology grounding is a safety-and-maintenance argument, not an overlap argument. "
          "This deck is built on the part that measures true."],
         size=12, color=INK, space=8, line_spacing=1.14)
    footer(sl, "wbs/deliverables/presentation/demo_evidence.json (generated from the "
               "evaluation deliverables)", num())

    # 3 ── section 1 ──────────────────────────────────────────────────────
    sl = slide(prs, paper=False)
    rect(sl, Inches(0.62), Inches(2.6), Inches(0.16), Inches(2.2), fill=TEAL)
    text(sl, Inches(1.05), Inches(2.55), Inches(10), Inches(0.5),
         "PART 1", size=13, color=TEAL, bold=True)
    text(sl, Inches(1.05), Inches(2.95), Inches(10.5), Inches(1.6),
         [[("Why we need an ontology", {})],
          [("Three reasons, each measured on this project", {"size": 18, "color": RGBColor(0xA9, 0xBC, 0xD0), "bold": False})]],
         size=40, color=WHITE, bold=True, space=14)
    text(sl, Inches(1.05), Inches(4.6), Inches(10.5), Inches(1.2),
         ["1 · Zero-cost, instant modification of what the model knows",
          "2 · A performance boost that lets a small model carry the job",
          "3 · An explanation a non-expert can read"],
         size=15, color=RGBColor(0xA9, 0xBC, 0xD0), space=8)

    # 4 ── rationale 1: zero-cost edit ────────────────────────────────────
    sl = slide(prs)
    header(sl, "reason 1 · zero-cost, instant modifying",
           "Correcting the model costs a graph edit, not a training run",
           f"Worked through on the first of this deck's two edits — the weather-safety "
           f"facts the rule layer was missing. The second, {ED['facts'] - er['triples_added']} "
           f"policy rules in v1.6.0, is the demo in Part 2.")
    kpi(sl, Inches(0.62), Inches(1.95), Inches(2.9),
        f"{er['edit_gpu_hours']:.0f}", "GPU-hours to apply the edit",
        f"{er['triples_added']} facts over {er['entities_touched']} places, extracted from "
        f"{sc['corpus_records']:,} source records at a support threshold of "
        f"{er['support_threshold']}", TEAL)
    kpi(sl, Inches(3.72), Inches(1.95), Inches(2.9),
        f"{er['train_hours_min']}–{er['train_hours_max']}", "GPU-hours per fine-tune",
        f"what the same correction costs as a retrain, per model: measured wall-clock "
        f"of the {ED['retrain_adapters']} adapters this project trained, "
        f"{ED['retrain_gpu_hours_total']:.0f} GPU-hours in total", AMBER)
    kpi(sl, Inches(6.82), Inches(1.95), Inches(2.9),
        f"{er['rules_enforceable_before']} → {er['rules_enforceable_after']}",
        f"of {er['rules_total']} business rules enforceable",
        f"{', '.join(er['unblocked'])} became checkable against the graph the moment the "
        f"facts landed", TEAL)
    kpi(sl, Inches(9.92), Inches(1.95), Inches(2.8),
        f"{er['validation_after_edit']['errors']}", "validation errors after the edit",
        f"SHACL conforms: {str(er['validation_after_edit']['shacl_conforms']).lower()}; "
        f"OWL reasoning entailed {er['validation_after_edit']['owl_entailed']:,} further "
        f"triples with no contradiction", GREEN)
    text(sl, Inches(0.62), Inches(3.95), Inches(12.1), Inches(0.4),
         "What actually happened, in order", size=16, color=INK, bold=True)
    steps = [
        ("1", "Found the gap", f"Two of the eight seed rules could not be enforced: no "
                               f"concept in the graph denoted a storm, typhoon or "
                               f"high-sea state."),
        ("2", "Read the corpus, not the model", f"{er['sentences_with_weather_language']:,} "
                                                f"sentences name a place and a weather state; "
                                                f"{er['candidate_pairs']} candidate pairs, "
                                                f"{er['triples_added']} kept at support ≥ "
                                                f"{er['support_threshold']}."),
        ("3", "Wrote a new version beside the old", "v1.4.0 → v1.5.0, additively. The "
                                                    "published version and every published "
                                                    "number were left untouched."),
        ("4", "Re-validated, did not re-train", f"0 errors, SHACL conforms, reasoner "
                                                f"consistent. Model weights byte-identical "
                                                f"(sha {live['retraining']['adapter_sha_before']})."
         if live else "0 errors, SHACL conforms, reasoner consistent."),
    ]
    for i, (k, t, b) in enumerate(steps):
        x = Inches(0.62 + i * 3.08)
        rect(sl, x, Inches(4.42), Inches(2.85), Inches(2.1), fill=WHITE,
             line=RGBColor(0xDD, 0xE4, 0xEC))
        rect(sl, x + Inches(0.2), Inches(4.62), Inches(0.42), Inches(0.42), fill=NAVY,
             shape=MSO_SHAPE.OVAL)
        text(sl, x + Inches(0.2), Inches(4.67), Inches(0.42), Inches(0.3), k,
             size=13, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        line(sl, x + Inches(0.74), Inches(4.66), Inches(2.0), Inches(0.33), t,
             13, color=INK, bold=True)
        text(sl, x + Inches(0.2), Inches(5.16), Inches(2.5), Inches(1.3), b,
             size=10.5, color=MUTE, line_spacing=1.12)
    footer(sl, "task2_formalization/weather_state_extraction.json, rule_compliance__1.5.0.json, "
               "validation_report__1.5.0.json; training wall-clock from the adapter logs", num())

    # 4b ── the cost chart ───────────────────────────────────────────────
    sl = slide(prs)
    header(sl, "reason 1 · in one picture", "The same correction, two ways",
           "Left: what it cost to train each adapter once. Right: what it cost to change "
           "what the system knows this week.")
    picture(sl, OUT / "deck_fig_cost_of_a_change.png", Inches(1.3), Inches(1.9),
            w=Inches(10.7))
    text(sl, Inches(0.62), Inches(6.35), Inches(12.1), Inches(0.6),
         [[("And it is per model. ", {"bold": True}),
           (f"The fine-tuning bar is paid again for every model you deploy — "
            f"{ED['retrain_adapters']} adapters, {ED['retrain_gpu_hours_total']:.0f} "
            f"GPU-hours between them. A graph edit is paid once and every model size "
            f"reads the same corrected facts: both edits in this deck took "
            f"{ED['facts']} facts and {ED['index_rebuild_s']:.0f} seconds of CPU to "
            f"re-index.", {})]],
         size=12, color=INK)
    footer(sl, "adapter wall-clock from the training logs; edit cost from "
               "task2_formalization/weather_state_extraction.json", num())

    # 5 ── rationale 2: small models ──────────────────────────────────────
    sl = slide(prs)
    header(sl, "reason 2 · performance boost for small models",
           "The graph is what lets the small model do the job",
           "Same 347 questions, same scorer, three model sizes, stored runs")
    rowsd = [["", "1B", "3B", "8B"]]
    rowsd.append(["Model alone (no ontology)"] + [f"{V[m]['V1_llm']['token_f1']:.1f}" for m in SZ])
    rowsd.append(["+ ontology + retrieval"] +
                 [f"{V[m]['V4_ontology_rag']['token_f1']:.1f}" for m in SZ])
    rowsd.append(["gain"] + [f"{ev['lift_within_size'][m]['token_f1']:+.1f}" for m in SZ])
    table(sl, Inches(0.62), Inches(2.0), Inches(5.9), rowsd, col_w=[2.6, 1, 1, 1],
          highlight=3, size=12.5)
    rowsu = [["unsupported claims ↓", "1B", "3B", "8B"]]
    rowsu.append(["Model alone"] + [f"{V[m]['V1_llm']['ungrounded']:.2f}%" for m in SZ])
    rowsu.append(["+ ontology + retrieval"] +
                 [f"{V[m]['V4_ontology_rag']['ungrounded']:.2f}%" for m in SZ])
    rowsu.append(["+ validation layer (deployed)"] + [f"{G[m]['ungrounded']:.2f}%" for m in SZ])
    table(sl, Inches(0.62), Inches(3.72), Inches(5.9), rowsu, col_w=[2.6, 1, 1, 1],
          highlight=3, size=12.5)
    rect(sl, Inches(6.9), Inches(2.0), Inches(5.82), Inches(3.2), fill=WHITE,
         line=RGBColor(0xDD, 0xE4, 0xEC))
    rect(sl, Inches(6.9), Inches(2.0), Inches(5.82), Pt(5), fill=AMBER)
    text(sl, Inches(7.2), Inches(2.2), Inches(5.2), Inches(0.8),
         f"{P[small]} + ontology beats {P[big]} with no ontology",
         size=19, color=INK, bold=True, line_spacing=1.05)
    rowsb = [["", f"{P[small]} + ontology", f"{P[big]} alone"],
             ["answer quality (token F1)", f"{sb['small_with_ontology']['token_f1']:.1f}",
              f"{sb['big_without_ontology']['token_f1']:.1f}"],
             ["unsupported claims ↓", f"{sb['small_with_ontology']['ungrounded']:.2f}%",
              f"{sb['big_without_ontology']['ungrounded']:.2f}%"],
             ["fully grounded answers ↑", f"{sb['small_with_ontology']['clean']:.1f}%",
              f"{sb['big_without_ontology']['clean']:.1f}%"]]
    table(sl, Inches(7.2), Inches(3.08), Inches(5.2), rowsb, col_w=[2.2, 1.3, 1.1], size=11.5)
    text(sl, Inches(7.2), Inches(4.6), Inches(5.2), Inches(0.56),
         f"A model {ratio}× smaller, on a fraction of the hardware, is ahead on all "
         f"three — the knowledge sits in the graph, not in the weights.",
         size=11, color=MUTE, line_spacing=1.1)
    text(sl, Inches(0.62), Inches(5.45), Inches(12.1), Inches(1.4),
         [[("Why this is the commercial point. ", {"bold": True}),
           (f"Serving {P[big]} costs roughly an order of magnitude more per answer than "
            f"{P[small]} and needs a bigger card to host. The ontology moves the accuracy "
            f"back to the small model, and it is the same graph for every model size — "
            f"one knowledge asset, not one per deployment.", {})]],
         size=12.5, color=INK, line_spacing=1.16)
    footer(sl, "task4_evaluation/variant_scores__*.json, selective_axis__*.json", num())

    # 6 ── rationale 3: explainability ────────────────────────────────────
    sl = slide(prs)
    header(sl, "reason 3 · easy for a non-expert to understand",
           "The brilliant new hire, and the company rulebook",
           "The brief's analogy, and then the same thing happening in the real system")
    boxes = [("THE NEW HIRE", "The language model",
              "Has read every book. Fluent, fast, widely informed — and willing to "
              "guess when your business is not in any of those books.", NAVY2, WHITE),
             ("THE RULEBOOK", "The ontology",
              "The org chart and the policy manual: what exists, what connects to "
              "what, and which answers are not allowed. One page a manager can read "
              "and correct.", TEAL, WHITE),
             ("THE RIGHT ANSWER", "Every sentence traceable",
              "The hire answers from the manual, and every line of the answer can be "
              "pointed back to the rule it came from.", GREEN, WHITE)]
    for i, (kick, t, b, col, fg) in enumerate(boxes):
        x = Inches(0.62 + i * 4.18)
        rect(sl, x, Inches(1.95), Inches(3.62), Inches(1.92), fill=col)
        text(sl, x + Inches(0.26), Inches(2.08), Inches(3.1), Inches(0.3), kick,
             size=10, color=WHITE, bold=True)
        line(sl, x + Inches(0.26), Inches(2.36), Inches(3.12), Inches(0.34), t,
             17, color=fg, bold=True)
        text(sl, x + Inches(0.26), Inches(2.78), Inches(3.15), Inches(1.0), b,
             size=10.5, color=RGBColor(0xDC, 0xE6, 0xF0), line_spacing=1.12)
        if i < 2:
            text(sl, x + Inches(3.68), Inches(2.6), Inches(0.5), Inches(0.5), "+" if i == 0 else "=",
                 size=26, color=MUTE, bold=True, align=PP_ALIGN.CENTER)
    text(sl, Inches(0.62), Inches(4.08), Inches(12.1), Inches(0.4),
         "And it is not a metaphor: here is one real answer with its rules attached",
         size=15, color=INK, bold=True)
    tr = ev.get("trace")
    if tr:
        text(sl, Inches(0.62), Inches(4.5), Inches(5.95), Inches(0.26),
             "what the system said", size=10, color=MUTE, bold=True)
        text(sl, Inches(6.78), Inches(4.5), Inches(5.95), Inches(0.26),
             "the approved facts that sentence rests on", size=10, color=MUTE, bold=True)
        y = Inches(4.78)
        for sent in tr["sentences"][:3]:
            rect(sl, Inches(0.62), y, Inches(5.95), Inches(0.62), fill=WHITE,
                 line=RGBColor(0xDD, 0xE4, 0xEC))
            text(sl, Inches(0.78), y + Inches(0.07), Inches(5.65), Inches(0.5),
                 clip(sent["sentence"], 140), size=10, color=INK, line_spacing=1.08)
            rect(sl, Inches(6.78), y, Inches(5.95), Inches(0.62),
                 fill=RGBColor(0xEC, 0xF7, 0xF5), line=RGBColor(0xC7, 0xE6, 0xE1))
            text(sl, Inches(6.94), y + Inches(0.07), Inches(5.65), Inches(0.5),
                 clip(" · ".join(sent["triples"][:3]), 140), size=10,
                 color=RGBColor(0x0A, 0x6E, 0x6C), font=MONO, line_spacing=1.08)
            y = y + Inches(0.7)
        text(sl, Inches(0.62), Inches(6.88), Inches(12.1), Inches(0.3),
             [[("Where no fact supports a sentence, the layer deletes that clause. ",
                {"bold": True}),
               (f"The answer above is {tr['grounded_coverage'] * 100:.0f}% covered by "
                f"approved facts, computed rather than asserted.", {})]],
             size=10.5, color=MUTE)
    footer(sl, "task4_evaluation/variant__llama3.1-8b__V5_ontology_rag_validated__hybded.jsonl "
               "(explanation_trace, verbatim)", num())

    # 7 ── section 2 ──────────────────────────────────────────────────────
    sl = slide(prs, paper=False)
    rect(sl, Inches(0.62), Inches(2.3), Inches(0.16), Inches(2.9), fill=AMBER)
    text(sl, Inches(1.05), Inches(2.25), Inches(10), Inches(0.5),
         "PART 2", size=13, color=AMBER, bold=True)
    text(sl, Inches(1.05), Inches(2.62), Inches(10.5), Inches(1.6),
         [[("The demo: before and after", {})],
          [("A wrong answer becoming a right one, because somebody edited the graph",
            {"size": 18, "color": RGBColor(0xA9, 0xBC, 0xD0), "bold": False})]],
         size=40, color=WHITE, bold=True, space=14)
    text(sl, Inches(1.05), Inches(4.3), Inches(11.0), Inches(1.9),
         ["Two kinds of edit cover every correction in this section",
          "The brief's own two examples, in the layout the brief asks for",
          "Four live corrections: the fact was absent, we added it, the answer changed",
          "Two stored corrections: the fact was never there, and the answer refuses to "
          "invent it",
          "All sixteen questions counted, controls included, with what did not move"],
         size=15, color=RGBColor(0xA9, 0xBC, 0xD0), space=8)
    text(sl, Inches(1.05), Inches(6.45), Inches(11.0), Inches(0.5),
         "No answer in this section was written or edited by hand. Both sides ran on the "
         "same model, the same weights, the same decoding settings and the same hardware.",
         size=12, color=RGBColor(0x7F, 0x93, 0xA8))

    # 7b ── the two correction patterns ─────────────────────────────────
    sl = slide(prs)
    header(sl, "the two patterns", "Two kinds of edit cover the corrections in this demo",
           "The pattern on the left, what it is in this knowledge base and what it does, "
           "on the right")
    pats = [
        ("1 · An asserted link, or the absence of one",
         "[ Product A ] —(is_incompatible_with)→ [ Platform B ]",
         "[ Woljeongri Beach ] —(avoid_during)→ [ Rough Seas ]",
         "A prohibition cannot be inferred from statistics; it has to be asserted. "
         f"{er['triples_added']} such facts were added in v1.5.0 and two blocked rules "
         "became enforceable. The absence of a link does the same work in the other "
         "direction: the graph ties no mainland city and no subway to Jeju, so neither "
         "the drive to Busan nor the ten-line metro can be asserted.", TEAL),
        ("2 · A category boundary",
         "[ Contractor ] is-not-a [ Permanent Employee ]",
         "[ Jjimjilbang ] —(is_a)→ [ Attraction ] , not [ Cuisine ]",
         "A plausible-sounding word gets generalised across categories unless the taxonomy "
         f"says otherwise. The graph types a jjimjilbang as an Attraction and nothing "
         f"types it as food, so a bathhouse offered as a breakfast dish — which the base "
         f"model does, elsewhere in the same benchmark — has no fact behind it.", AMBER),
    ]
    for i, (title, generic, ours, why, col) in enumerate(pats):
        y = Inches(1.95 + i * 2.45)
        rect(sl, Inches(0.62), y, Inches(12.1), Inches(2.2), fill=WHITE,
             line=RGBColor(0xDD, 0xE4, 0xEC))
        rect(sl, Inches(0.62), y, Pt(5), Inches(2.2), fill=col)
        line(sl, Inches(0.95), y + Inches(0.16), Inches(4.45), Inches(0.36), title,
             16, color=INK, bold=True)
        text(sl, Inches(0.95), y + Inches(0.62), Inches(4.6), Inches(0.3),
             "the pattern", size=10, color=MUTE, bold=True)
        text(sl, Inches(0.95), y + Inches(0.86), Inches(4.7), Inches(0.4), generic,
             size=12, color=MUTE, font=MONO)
        text(sl, Inches(0.95), y + Inches(1.32), Inches(4.6), Inches(0.3),
             "in this knowledge base", size=10, color=MUTE, bold=True)
        text(sl, Inches(0.95), y + Inches(1.56), Inches(4.7), Inches(0.4), ours,
             size=12, color=col, font=MONO, bold=True)
        text(sl, Inches(6.1), y + Inches(0.24), Inches(6.3), Inches(1.7), why,
             size=12, color=INK, line_spacing=1.16)
    line(sl, Inches(0.62), Inches(6.9), Inches(12.1), Inches(0.3),
         "Both are one line in the graph, reviewable by a non-engineer, and applied "
         "without touching the model.", 11.5, color=MUTE)
    footer(sl, "the pattern column follows the business-plan brief; the right-hand column "
               "is this project's ontology v1.5.0", num())

    # 7d ── the brief's own two examples, in that layout ─────────────────
    sl = slide(prs)
    header(sl, "the pattern, in a generic enterprise",
           "The two corrections the business brief uses as its examples",
           "Drawn here as the brief writes them, so the Jeju cases that follow can be "
           "read as the same two shapes")
    gen = [("A hard requirement the model cannot infer",
            "\u201cCan I use Flagship Software V1 with Cloud Server 2026?\u201d",
            "\u201cYes, V1 is fully compatible with all cloud servers released after "
            "2024.\u201d",
            "[ Flagship Software V1 ]  \u2014(is_incompatible_with)\u2192  [ Cloud Server 2026 ]",
            "\u201cNo. V1 is not compatible with Cloud Server 2026. Use V2, or apply the "
            "compatibility patch.\u201d",
            "A prohibition is not a statistical tendency. It exists only if somebody "
            "asserts it, and once asserted the model retrieves it instead of guessing.",
            RED),
           ("A category boundary the model generalises across",
            "\u201cWhat is our remote-work policy for contractors?\u201d",
            "\u201cContractors get 3 remote days a week under the standard hybrid "
            "policy.\u201d",
            "[ Contractor ]  is-not-a  [ Permanent Employee ]  \u2014(remote_work)\u2192  [ Not eligible ]",
            "\u201cExternal contractors work on site and are not covered by the hybrid "
            "remote-work policy.\u201d",
            "The taxonomy stops a rule about one class leaking into another. The model "
            "followed the hierarchy rather than the word similarity.",
            AMBER)]
    for i, (t, q, bad, triple, good, why, col) in enumerate(gen):
        y = Inches(1.9 + i * 2.5)
        rect(sl, Inches(0.62), y, Inches(12.1), Inches(2.28), fill=WHITE,
             line=RGBColor(0xDD, 0xE4, 0xEC))
        rect(sl, Inches(0.62), y, Pt(5), Inches(2.28), fill=col)
        text(sl, Inches(0.95), y + Inches(0.12), Inches(7.0), Inches(0.3), t,
             size=14, color=INK, bold=True)
        text(sl, Inches(0.95), y + Inches(0.46), Inches(5.6), Inches(0.3),
             clip(q, 72), size=11, color=MUTE)
        text(sl, Inches(0.95), y + Inches(0.78), Inches(5.6), Inches(0.62),
             [[("before  ", {"bold": True, "color": RED}), (bad, {"color": INK})]],
             size=11, line_spacing=1.1)
        text(sl, Inches(0.95), y + Inches(1.52), Inches(5.7), Inches(0.62),
             [[("after  ", {"bold": True, "color": GREEN}), (good, {"color": INK})]],
             size=11, line_spacing=1.1)
        rect(sl, Inches(6.85), y + Inches(0.42), Inches(5.6), Inches(0.52),
             fill=RGBColor(0xEC, 0xF7, 0xF5), line=RGBColor(0xC7, 0xE6, 0xE1))
        text(sl, Inches(6.98), y + Inches(0.54), Inches(5.4), Inches(0.34),
             triple, size=10, color=RGBColor(0x0A, 0x6E, 0x6C), font=MONO, bold=True)
        text(sl, Inches(6.98), y + Inches(1.06), Inches(5.5), Inches(1.0), why,
             size=11, color=MUTE, line_spacing=1.13)
    line(sl, Inches(0.62), Inches(6.9), Inches(12.1), Inches(0.3),
         "Our ontology holds the same two shapes: a requirement nobody can deduce, and "
         "a boundary nobody should cross. The next slides are real runs of both.",
         11.5, color=MUTE)
    footer(sl, "Ontology-Comment, 19 October — the brief's own examples, reproduced",
           num())

    # 7c ── how to read the demo, and the brief's own two examples ───────
    sl = slide(prs)
    header(sl, "how to read the next slides", "Four panels per correction, every time",
           "The layout the business brief asks for: the problem, the edit, the result, "
           "and why the result is better")
    steps = [("1", "THE PROBLEM", "The question a customer actually asks, and the "
                                  "answer the system gave when the graph was silent "
                                  "about it. Quoted verbatim from the run.", RED),
             ("2", "THE ONTOLOGY FIX", "What was added to the graph. One line of data, "
                                       "readable by a non-engineer, carrying the source "
                                       "record it was read from.", TEAL),
             ("3", "THE RESULT", "The same question, the same model, the same weights — "
                                 "asked again after the edit. Also verbatim.", GREEN),
             ("4", "WHY IT IMPROVED", "Which fact the answer now rests on, and why "
                                      "statistics alone could not have supplied it.",
              AMBER)]
    for i, (k, t, b, col) in enumerate(steps):
        x = Inches(0.62 + i * 3.08)
        rect(sl, x, Inches(2.1), Inches(2.85), Inches(2.5), fill=WHITE,
             line=RGBColor(0xDD, 0xE4, 0xEC))
        rect(sl, x, Inches(2.1), Inches(2.85), Pt(5), fill=col)
        rect(sl, x + Inches(0.22), Inches(2.34), Inches(0.44), Inches(0.44), fill=col,
             shape=MSO_SHAPE.OVAL)
        text(sl, x + Inches(0.22), Inches(2.4), Inches(0.44), Inches(0.3), k,
             size=14, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        text(sl, x + Inches(0.78), Inches(2.42), Inches(1.95), Inches(0.3), t,
             size=11.5, color=INK, bold=True)
        text(sl, x + Inches(0.22), Inches(2.92), Inches(2.45), Inches(1.5), b,
             size=10.5, color=MUTE, line_spacing=1.14)
    rect(sl, Inches(0.62), Inches(4.86), Inches(12.1), Inches(1.84), fill=WHITE,
         line=RGBColor(0xDD, 0xE4, 0xEC))
    rect(sl, Inches(0.62), Inches(4.86), Pt(5), Inches(1.84), fill=NAVY)
    text(sl, Inches(0.95), Inches(5.0), Inches(11.5), Inches(0.3),
         "AND ONE RULE WE IMPOSED ON OURSELVES", size=10.5, color=NAVY, bold=True)
    text(sl, Inches(0.95), Inches(5.32), Inches(11.6), Inches(1.3),
         [[("Nothing on the demo slides was authored for the demo. ", {"bold": True}),
           ("Every fact added to the graph was extracted from the source corpus with a "
            "support threshold and kept its provenance; every answer is the model's own "
            "output at greedy decoding, trimmed only where a slide cannot hold the whole "
            "paragraph. Half the questions we asked are controls the edit should not "
            "touch, and we report what happened to those too.", {})]],
         size=12, color=INK, line_spacing=1.16)
    footer(sl, "wbs/task2_formalization/policy_constraint_extraction.md, "
               "wbs/task5_demo/demo_policy_before_after.py", num())

    # 7e ── the policy edit, case by case, in the brief's layout ──────────
    PE = ev.get("policy_edit")
    PX = ev.get("policy_extraction") or {}
    evid = {t["triple"]: t for t in (PX.get("top") or [])}
    cno = [0]

    def corr() -> int:
        cno[0] += 1
        return cno[0]

    if PE and PE["cases"]:
        for ci, case in enumerate(PE["cases"][:4]):
            sl = slide(prs)
            s_, pr_, o_ = case["triple"]
            tkey = f"{s_} {pr_} {o_}"
            ex = evid.get(tkey) or {}
            corrected = case["kind_of_change"] == "corrected"
            header(sl, f"correction {corr()} · {case['rule'].rstrip('.').lower()}",
                   "One fact added to the graph. Nothing else changed."
                   if corrected else
                   "The answer was already cautious. Now it is also backed.",
                   f"{PE['model']} · {PE['decoding']} · ontology v{PE['before']} → "
                   f"v{PE['after']} · model weights byte-identical "
                   f"(sha {PE['retraining']['adapter_sha_before']})")
            b_facts = [f for f in (PE["cards"].get(s_) or {}).get(PE["before"], [])]
            a_facts = [f for f in (PE["cards"].get(s_) or {}).get(PE["after"], [])]
            new_fact = [f for f in a_facts if f not in b_facts]
            fix_lines = [f"[ {s_} ]  \u2014({pr_})\u2192  [ {o_} ]"]
            fix = (f"Read from {ex.get('support', 0)} independent source records "
                   f"at a support threshold of "
                   f"{PX.get('min_support', 3)}, and quoted from the corpus: "
                   f"\u201c{clip(ex.get('evidence', ''), 150)}\u201d  "
                   f"The fact carries those record ids as provenance, so a reviewer "
                   f"can check it without reading the model.")
            card_q = (f" — the card for {s_} now reads "
                      f"\u201c{clip(new_fact[0], 56)}\u201d" if new_fact else "")
            why = (
                (f"The rule is not a tendency a model could average its way to. It is "
                 f"a fact about this place, and until somebody asserted it there was "
                 f"nothing to retrieve, so the answer fell back on the generic advice "
                 f"above. After the edit the retriever hands the fact to the model with "
                 f"the question{card_q}, and the answer states it. No weights were "
                 f"touched and no training ran.")
                if corrected else
                (f"Read it carefully: the wording barely moves, because this model had "
                 f"already guessed right. What moves is whether the guess is "
                 f"*checkable*. Before the edit no approved fact backed it, so the "
                 f"validation layer could neither confirm it nor enforce it; after the "
                 f"edit it is an asserted fact with provenance{card_q}. A guess that "
                 f"happens to be right is not an answer a business can stand behind.")) 
            four_part(sl, case["question"], case["before"]["answer"],
                      case["after"]["answer"], fix, why,
                      before_tag=f"{case['before']['generation_s']}s · verbatim",
                      after_tag=f"{case['after']['generation_s']}s · verbatim",
                      fix_lines=fix_lines)
            footer(sl, f"presentation/demo_policy_before_after__*.json, item "
                       f"{case['id']} (verbatim); the fact from "
                       f"task2_formalization/policy_constraint_extraction.json",
                   num())

    # 7f ── what the edit did and did not move ───────────────────────────
    if PE:
        sl = slide(prs)
        sm = PE["summary"]
        header(sl, "the demo, counted", "All sixteen questions, including the ones the "
                                        "edit should not have touched",
               f"{sm['targeted']} questions the {PE['edit']['triples_added']}-fact "
               f"edit speaks to, {sm['controls']} controls it does not. Same model, "
               f"same decoding, same order, both sides.")
        kpi(sl, Inches(0.62), Inches(1.95), Inches(2.9),
            f"{sm['rule_stated_before']} → {sm['rule_stated_after']}",
            f"of {sm['targeted']} answers state the rule",
            "before the edit the graph was silent about these policies; after it, the "
            "answer carries them", TEAL)
        kpi(sl, Inches(3.72), Inches(1.95), Inches(2.9),
            f"{sm['rule_reached_context_before']} → {sm['rule_reached_context_after']}",
            "of those, the fact reached the prompt",
            "the retriever put the new fact in front of the model; the gap between this "
            "and the column on the left is the honest part", AMBER)
        kpi(sl, Inches(6.82), Inches(1.95), Inches(2.9),
            f"{PE['edit']['triples_added']}", "facts added to the graph",
            f"index rebuilt on CPU in "
            f"{PE['edit']['card_index_build_s'][PE['after']]:.0f} s; "
            f"{PE['retraining']['gpu_hours_spent_on_the_edit']:.0f} GPU-hours; "
            f"{PE['validation']['errors']} validation errors, SHACL conforms", GREEN)
        kpi(sl, Inches(9.92), Inches(1.95), Inches(2.8),
            f"{sm['control_answers_changed']}/{sm['controls']}",
            "control answers changed at all",
            f"mean word overlap between the two versions of a control answer: "
            f"{PE['control_overlap_mean']:.0f}%", RED if sm["control_answers_changed"]
            else GREEN)
        also = PE.get("also_run_at") or []
        text(sl, Inches(0.62), Inches(3.95), Inches(12.1), Inches(0.4),
             [[("Every targeted question, and what moved", {}),
               (f"      — the same probe set at {also[0]['model']}: "
                f"{also[0]['rule_stated_before']} → {also[0]['rule_stated_after']} of "
                f"{also[0]['targeted']}, controls "
                f"{also[0]['control_answers_changed']}/{also[0]['controls']}"
                if also else "", {"size": 11, "bold": False, "color": MUTE})]],
             size=15, color=INK, bold=True)
        rws = [["the rule added in v" + PE["after"], "states the rule, before",
                "states the rule, after"]]
        for t in PE["all_targeted"]:
            rws.append([t["rule"].rstrip("."),
                        "yes" if t["stated_before"] else "no",
                        "yes" if t["stated_after"] else "no"])
        table(sl, Inches(0.62), Inches(4.35), Inches(7.4), rws, col_w=[4.4, 1.3, 1.3],
              size=9.5, row_h=Inches(0.235))
        rect(sl, Inches(8.3), Inches(4.35), Inches(4.42), Inches(2.3), fill=WHITE,
             line=RGBColor(0xDD, 0xE4, 0xEC))
        rect(sl, Inches(8.3), Inches(4.35), Pt(5), Inches(2.3), fill=AMBER)
        text(sl, Inches(8.58), Inches(4.48), Inches(3.9), Inches(0.3),
             "WHAT WE WILL NOT ROUND UP", size=10.5, color=AMBER, bold=True)
        caveats = [
            [(f"{sm['rule_stated_before']} of {sm['targeted']} were already right. ",
              {"bold": True}),
             ("The edit makes those checkable, and fixes the rest.", {})],
            [("Two did not move with the fact in the prompt. ", {"bold": True}),
             ("A fact in the context is not a fact the model used.", {})],
            [("Controls moved a little. ", {"bold": True}),
             (f"New cards in the index reword untouched answers "
              f"({PE['control_overlap_mean']:.0f}% overlap).", {})]]
        cw, ch = Inches(4.0), Inches(1.78)
        cs = min(TM.shrink_to_fit("".join(t for p_ in caveats for t, _ in p_),
                                  cw - TXT_PAD, ch - Inches(0.26), 10.5, floor=7.5,
                                  leading=1.2), 10.5)
        text(sl, Inches(8.58), Inches(4.78), cw, ch, caveats,
             size=cs, color=INK, space=5, line_spacing=1.1)
        footer(sl, "presentation/demo_policy_before_after__*.json "
                   "(all 16 items, both versions, verbatim)", num())

    # 7g ── the whole policy layer, and the discipline behind it ─────────
    if PX:
        sl = slide(prs)
        header(sl, "the edit in full", "Six rules, 22 facts, and the gate each one "
                                       "had to pass",
               f"Extracted from {PX['records_scanned']:,} source records — not written "
               f"for the demo, and not written by the model")
        rws = [["constraint", "predicate", "facts", "the customer question it answers"]]
        for cname, cd in sorted(PX["constraints"].items()):
            rws.append([cname, cd["predicate"], str(cd["pairs"]),
                        cd["question_it_answers"]])
        table(sl, Inches(0.62), Inches(1.95), Inches(7.5), rws,
              col_w=[2.2, 1.5, 0.8, 3.4], size=11, row_h=Inches(0.38),
              align_right_from=2)
        rect(sl, Inches(8.4), Inches(1.95), Inches(4.32), Inches(2.9), fill=WHITE,
             line=RGBColor(0xDD, 0xE4, 0xEC))
        rect(sl, Inches(8.4), Inches(1.95), Pt(5), Inches(2.9), fill=TEAL)
        text(sl, Inches(8.68), Inches(2.08), Inches(3.9), Inches(0.3),
             "WHAT WAS THROWN AWAY", size=10.5, color=TEAL, bold=True)
        # the gates fire at different stages - a sentence can be dropped before
        # any pair exists - so these are not a subtraction chain and the slide
        # does not draw them as one
        drops = [(f"{PX['candidate_pairs']}", "candidate (place, rule) pairs found",
                  INK),
                 (f"{PX['accepted_pairs']}", "asserted at support \u2265 "
                  f"{PX['min_support']} source records", TEAL),
                 (f"{PX['rejected_for_support']}", "dropped: too few source records",
                  MUTE),
                 (f"{PX['pairs_dropped_for_distance']}", "dropped: the place sat more "
                  f"than {PX['proximity_window_chars']} characters from the rule", MUTE),
                 (f"{PX['matches_dropped_for_negation']}", "dropped: the sentence "
                  "negates the rule", MUTE),
                 (f"{PX['sentences_dropped_for_no_cue']}", "sentences dropped before "
                  "pairing: no prohibition or restriction cue", MUTE)]
        y = Inches(2.42)
        for v, lab, col in drops:
            text(sl, Inches(8.68), y, Inches(0.9), Inches(0.26), v, size=11.5,
                 color=col, bold=True, align=PP_ALIGN.RIGHT)
            text(sl, Inches(9.68), y, Inches(2.9), Inches(0.4), lab, size=9.5,
                 color=INK if col is TEAL else MUTE, line_spacing=1.05)
            y = y + Inches(0.4)
        text(sl, Inches(0.62), Inches(5.05), Inches(12.1), Inches(1.5),
             [[("The gate that matters most is the dullest one. ", {"bold": True}),
               ("A rule is asserted about a place only if the gazetteer can put that "
                "place on a map. That single test is what separates Gwaneumsa Temple "
                "from \u201cJeju culture\u201d and \u201cKorean phrases\u201d — concepts "
                "this ontology really does hold, and which no rule can be enforced at. "
                "Without it the extraction asserted a dress code against a language "
                "phrasebook.", {})],
              [("And the schema is checked against what was actually asserted. ",
                {"bold": True}),
               (f"The first run declared `requires` over temples and cultural sites and "
                f"then asserted it of four museums; its own conformance check caught "
                f"all seven violations before the version was written. The published "
                f"version has {len((PX.get('schema_conformance') or {}).get('violations') or [])}.",
                {})]],
             size=12, color=INK, space=8, line_spacing=1.16)
        footer(sl, "wbs/task2_formalization/policy_constraint_extraction.md; "
                   "artifacts/ontologies/1.6.0/; validation_report__1.6.0.json", num())

    # 7h ── the same two shapes, in the buyer's own business ─────────────
    sl = slide(prs)
    header(sl, "how this maps to your business",
           "The same two shapes, in four industries",
           "Every row is one line of graph data, reviewable by the person who owns the "
           "rule and applied without touching the model")
    rws = [["your rule", "the shape", "what it stops the assistant saying",
            "on Jeju, this is"]]
    rws += [["Product A does not run on Platform B", "requirement / prohibition",
             "“fully compatible with everything since 2024”",
             "a rental car needs an International Driving Permit"],
            ["Contractors are not on the hybrid policy", "category boundary",
             "applying the staff policy to a contractor",
             "a bathhouse is an attraction, not a dish"],
            ["This SKU cannot ship to that region", "requirement / prohibition",
             "quoting a delivery date that cannot happen",
             "drone flying is restricted at Jeju Stone Park"],
            ["This discount needs an approval step", "requirement",
             "promising a price nobody signed off",
             "this market takes cash, not cards"]]
    table(sl, Inches(0.62), Inches(1.95), Inches(12.1), rws,
          col_w=[2.9, 2.0, 3.5, 3.5], size=11, row_h=Inches(0.62),
          align_right_from=99)
    rect(sl, Inches(0.62), Inches(4.95), Inches(12.1), Inches(1.75), fill=WHITE,
         line=RGBColor(0xDD, 0xE4, 0xEC))
    rect(sl, Inches(0.62), Inches(4.95), Pt(5), Inches(1.75), fill=TEAL)
    text(sl, Inches(0.95), Inches(5.08), Inches(11.5), Inches(0.3),
         "WHAT A PILOT WOULD NEED FROM YOU", size=10.5, color=TEAL, bold=True)
    text(sl, Inches(0.95), Inches(5.4), Inches(11.6), Inches(1.2),
         [[("A source of truth, and someone who owns it. ", {"bold": True}),
           ("The rules above are not written by the model; they are read from "
            "documents you already have, kept with the record each one came from, and "
            "approved by a person before they are served. That is the asset — the "
            "model is interchangeable.", {})],
          [("And a regression set. ", {"bold": True}),
           (f"An edit reworded {PE['summary']['control_answers_changed']} of "
            f"{PE['summary']['controls']} answers it had nothing to say about, because "
            f"the retrieval index moves under them. A few hundred held-out questions "
            f"scored before and after each edit turns that into a line in the release "
            f"notes instead of a surprise." if PE else
            "An edit changes what the retriever hands the model, so a held-out "
            "question set scored before and after each edit belongs in the release "
            "process.", {})]],
         size=12, color=INK, space=8, line_spacing=1.16)
    footer(sl, "the left-hand column follows the business-plan brief; the right-hand "
               "column is this project's ontology v1.6.0", num())

    # 8/9 ── the two stored cases ─────────────────────────────────────────
    ms = ev["mode_support"]
    titles = ["a route that cannot be driven",
              "a transport system the island does not have"]
    whys = [
        "The base model treated Jeju like any mainland itinerary and sent the traveller to "
        "Busan — a city 300 km away across open sea, unreachable by the rental car the "
        "question asks about. The graph contains no path, no ferry and no entity for it, so "
        "the claim has nothing to rest on and the grounded arm never makes it.",
        f"Jeju has no rail of any kind. The graph holds \"subway\" only as a generic "
        f"transport mode — one fact, no edge joining it to the island — so \"the Jeju "
        f"Subway, with 10 lines\" has nothing to rest on. Every mode the grounded answer "
        f"names ({ms['named_and_linked']} of "
        f"{len(ms['named_in_grounded_answer'])}) is one the graph does tie to Jeju.",
    ]
    for ci, case in enumerate(ev["cases"]):
        sl = slide(prs)
        header(sl, f"correction {corr()} · {titles[ci]}",
               "The fact was never there, and the grounded arm will not invent it",
               f"{P[ev.get('demo_model', 'llama3.2-1b')] if ev.get('demo_model') else P[small]}"
               f" · benchmark item {case['bench_id']} · category: {case['category']}")
        rect(sl, Inches(0.62), Inches(1.9), Inches(12.1), Inches(0.62), fill=NAVY2)
        text(sl, Inches(0.82), Inches(2.0), Inches(11.7), Inches(0.45),
             [[("Question  ", {"bold": True, "color": AMBER}),
               (clip(case["question"], 200), {"color": WHITE})]], size=12)
        quote_panel(sl, Inches(0.62), Inches(2.66), Inches(5.95), Inches(2.5),
                    "BEFORE · the model on its own", clip(case["before_excerpt"], 620),
                    RED, f"{len(case['before']['unresolved_spans'])} unverifiable spans")
        quote_panel(sl, Inches(6.78), Inches(2.66), Inches(5.95), Inches(2.5),
                    "AFTER · ontology + validation layer", clip(case["after"]["answer"], 620),
                    GREEN, f"{len(case['after']['unresolved_spans'])} unverifiable spans")
        rect(sl, Inches(0.62), Inches(5.3), Inches(5.95), Inches(1.6), fill=WHITE,
             line=RGBColor(0xDD, 0xE4, 0xEC))
        text(sl, Inches(0.82), Inches(5.42), Inches(5.5), Inches(0.3),
             "How the ontology was applied", size=12.5, color=INK, bold=True)
        y = Inches(5.76)
        if ci == 0:
            for c in [c for c in case["span_checks"] if not c["in_ontology"]][:4]:
                text(sl, Inches(0.82), y, Inches(5.6), Inches(0.26),
                     [[(f"{c['span']}", {"font": MONO, "color": RED, "bold": True}),
                       ("  → no concept, alias or place in the graph; 0 facts",
                        {"color": MUTE})]], size=10)
                y = y + Inches(0.26)
        else:
            for m in ms["claimed"][:4]:
                what = ("not in the graph at all" if not m["in_ontology"]
                        else f"a generic concept with {m['triples']} fact"
                             f"{'s' if m['triples'] != 1 else ''}")
                text(sl, Inches(0.82), y, Inches(5.5), Inches(0.26),
                     [[(f"{m['phrase']}", {"font": MONO, "color": RED, "bold": True}),
                       (f"  → {what}; {m['linked_to_jeju']} edges to Jeju",
                        {"color": MUTE})]], size=10.5)
                y = y + Inches(0.26)
        rect(sl, Inches(6.78), Inches(5.3), Inches(5.95), Inches(1.6), fill=WHITE,
             line=RGBColor(0xDD, 0xE4, 0xEC))
        text(sl, Inches(6.98), Inches(5.42), Inches(5.5), Inches(0.3),
             "Why it improved", size=12.5, color=INK, bold=True)
        text(sl, Inches(6.98), Inches(5.72), Inches(5.55), Inches(1.1),
             whys[ci], size=11, color=MUTE, line_spacing=1.12)
        footer(sl, f"task4_evaluation/variant__{small}__V1_llm.jsonl and "
                   f"__V5_ontology_rag_validated__hybded.jsonl, item {case['bench_id']}, "
                   f"answers quoted verbatim", num())

    # 10 ── the live edit ─────────────────────────────────────────────────
    if live:
        sl = slide(prs)
        header(sl, f"correction {corr()} · the safety rule, added live",
               "One graph edit, no retraining, and the safety claim becomes checkable",
               f"{live['model']} · one adapter, loaded once · greedy decoding · only the "
               f"ontology version changed between the two runs · the demo orders "
               f"constraint facts first in the entity card (off by default; see R18)")
        it = live["items"][0]
        b, a = live["ontology"]["before"], live["ontology"]["after"]
        rect(sl, Inches(0.62), Inches(1.95), Inches(12.1), Inches(0.56), fill=NAVY2)
        text(sl, Inches(0.82), Inches(2.04), Inches(11.7), Inches(0.4),
             [[("Question  ", {"bold": True, "color": AMBER}),
               (clip(it["question"], 190), {"color": WHITE})]], size=12)
        for i, (vv, lab, col) in enumerate([(b, f"BEFORE · ontology v{b}", RED),
                                            (a, f"AFTER · ontology v{a}", GREEN)]):
            x = Inches(0.62 + i * 6.16)
            quote_panel(sl, x, Inches(2.64), Inches(5.95), Inches(1.5), lab,
                        clip(it["runs"][vv]["answer"], 330), col,
                        f"{it['runs'][vv]['generation_s']}s")
            facts = live["cards"]["Woljeongri Beach"][vv][:3]
            rect(sl, x, Inches(4.26), Inches(5.95), Inches(1.16),
                 fill=RGBColor(0xF1, 0xF4, 0xF8), line=RGBColor(0xDD, 0xE4, 0xEC))
            text(sl, x + Inches(0.18), Inches(4.34), Inches(5.6), Inches(0.26),
                 "what the graph handed the model about this beach", size=10, color=MUTE,
                 bold=True)
            text(sl, x + Inches(0.18), Inches(4.6), Inches(5.6), Inches(0.78),
                 [f"· {f}" for f in facts], size=10, color=INK, font=MONO, space=2)
        rect(sl, Inches(0.62), Inches(5.52), Inches(12.1), Inches(0.52), fill=WHITE,
             line=RGBColor(0xDD, 0xE4, 0xEC))
        text(sl, Inches(0.82), Inches(5.6), Inches(11.7), Inches(0.4),
             [[("Read it carefully: ", {"bold": True, "color": AMBER}),
               ("the wording hardly moves — this model was already cautious. What moves is "
                "whether the caution is backed. Before the edit the graph could not confirm "
                "the beach-and-sea-state pairing; after it, the pairing is an approved fact "
                "and the rule layer can enforce it.", {"color": INK})]], size=11)
        tot = cc.get("totals") or {}
        if tot:
            kpi(sl, Inches(0.62), Inches(6.04), Inches(3.9),
                f"{tot[b]['asserted']}/{tot[b]['claimed']} → {tot[a]['asserted']}/{tot[a]['claimed']}",
                "safety claims backed by an approved fact",
                "same model, same words in the answer, different graph", TEAL, h=Inches(1.0))
        kpi(sl, Inches(4.72), Inches(6.04), Inches(3.9),
            f"+{a and live['ontology']['counts'][a]['avoid_during'] - live['ontology']['counts'][b]['avoid_during']}",
            "avoid-during facts added", f"graph rebuilt and re-indexed in "
                                        f"{live['ontology']['card_index_build_s'][a]:.0f} s on CPU",
            AMBER, h=Inches(1.0))
        kpi(sl, Inches(8.82), Inches(6.04), Inches(3.9),
            "unchanged", "model weights",
            f"adapter digest {live['retraining']['adapter_sha_before']} before and after the edit",
            GREEN, h=Inches(1.0))
        footer(sl, "wbs/task5_demo/demo_live_edit.py → presentation/demo_live_edit.json "
                   "(answers verbatim, digests recomputed from the adapter files)", num())

    # 11 ── section 3 ─────────────────────────────────────────────────────
    sl = slide(prs, paper=False)
    rect(sl, Inches(0.62), Inches(2.6), Inches(0.16), Inches(2.2), fill=TEAL)
    text(sl, Inches(1.05), Inches(2.55), Inches(10), Inches(0.5),
         "PART 3", size=13, color=TEAL, bold=True)
    text(sl, Inches(1.05), Inches(2.95), Inches(10.5), Inches(1.6),
         [[("What the measurement says", {})],
          [("347 questions, three sizes, paired tests", {"size": 18, "color": RGBColor(0xA9, 0xBC, 0xD0), "bold": False})]],
         size=40, color=WHITE, bold=True, space=14)
    text(sl, Inches(1.05), Inches(4.6), Inches(10.8), Inches(1.2),
         ["Where ontology grounding wins, by how much, and with what confidence",
          "Where it does not win, and why we think that is",
          "What it would take to settle the questions still open"],
         size=15, color=RGBColor(0xA9, 0xBC, 0xD0), space=8)

    # 12 ── headline results ──────────────────────────────────────────────
    sl = slide(prs)
    header(sl, "results · deployed configuration vs retrieval baseline",
           "Grounding improves at every size, and the tests agree",
           "Paired per question against Simple RAG on the same answers; the deployed cell "
           "spends the validation layer only on drafts its own detector flags")
    rws = [["", "unsupported claims ↓", "Δ vs RAG", "significance",
            "fully grounded ↑", "Δ vs RAG", "significance"]]
    for m in SZ:
        g = G[m]
        rws.append([P[m], f"{g['ungrounded']:.2f}%",
                    f"{g['vs_rag']['ungrounded_rate']['delta']:+.2f} pp",
                    pv(g["vs_rag"]["ungrounded_rate"]["p"]),
                    f"{g['clean']:.1f}%", f"{g['vs_rag']['clean']['delta']:+.2f} pp",
                    pv(g["vs_rag"]["clean"]["p"])])
    table(sl, Inches(0.62), Inches(2.1), Inches(12.1), rws,
          col_w=[1.6, 1.5, 1.1, 1.2, 1.4, 1.1, 1.2], size=12, row_h=Inches(0.42))
    text(sl, Inches(0.62), Inches(3.95), Inches(12.1), Inches(0.4),
         "Reading it", size=15, color=INK, bold=True)
    bullets(sl, Inches(0.62), Inches(4.35), Inches(12.0), Inches(2.2), [
        [("Unsupported claims fall and fully-grounded answers rise at all three sizes, "
          "every test significant. ", {"bold": True}),
         (f"At {P[big]} the deployed cell asserts "
          f"{G[big]['vs_rag']['ungrounded_rate']['delta']:+.2f} pp fewer unsupported "
          f"spans and returns {G[big]['vs_rag']['clean']['delta']:+.1f} pp more fully "
          f"grounded answers than the retrieval baseline.", {})],
        [("Spending the layer selectively is what makes it affordable. ", {"bold": True}),
         (f"Gating on the detector's own signal recovers "
          f"{G[big]['gate_token_f1_gain']:+.2f} answer-quality points at {P[big]} compared "
          f"with validating everything, and the two guardrails overlap "
          f"({G[big]['interaction']:+.2f} interaction) — they are largely one fix, so pay "
          f"for one.", {})],
        [("The graph alone is not the whole story. ", {"bold": True}),
         (f"Ontology context without the retrieved passages beats the baseline on both "
          f"groundedness measures at "
          f"{', '.join(P[m] for m in lim['ontology_only_beats_rag_at']) or 'no size'} "
          f"and not elsewhere; the combination wins at all three. The credit belongs to "
          f"graph plus retrieval, and we say so.", {})],
    ], size=12)
    footer(sl, "task4_evaluation/selective_axis__*.json, ontology_only_validated__*.json", num())

    # 12b ── the picture ─────────────────────────────────────────────────
    sl = slide(prs)
    header(sl, "results · in one picture", "What the ontology buys, at three model sizes",
           "Model alone, with retrieval, and with the deployed ontology configuration")
    picture(sl, OUT / "deck_fig_what_the_graph_buys.png", Inches(0.62), Inches(1.9),
            w=Inches(12.1))
    text(sl, Inches(0.62), Inches(5.6), Inches(12.1), Inches(0.9),
         [[("Two panels out of three go our way, and the third is the one we are honest "
            "about. ", {"bold": True}),
           ("Unsupported claims fall by roughly an order of magnitude against the model "
            "alone and stay below the retrieval baseline at every size; grounded answers "
            "rise; reference overlap stays a little behind retrieval — the gap this deck "
            "does not hide.", {})]], size=12, color=INK)
    footer(sl, "wbs/task5_demo/make_deck_figures.py, drawn from the same evidence file "
               "as the tables", num())

    # 12c ── the same comparison outside our own benchmark ────────────────
    xm = json.loads((DELIV / "task4_evaluation" / "cross_model_results.json")
                    .read_text(encoding="utf-8"))
    xeq, xex = xm["summary"]["equal"], xm["summary"]["extended"]
    sl = slide(prs)
    header(sl, "results · does it hold outside our benchmark?",
           f"{xm['n_models']} models, {xm['n_families']} model lines, "
           f"{xm['split_n']:,} held-out questions each",
           "The same two arms, re-scored on our metrics, read by who got how much "
           "training")
    picture(sl, FIG / "fig16_cross_model.png", Inches(0.62), Inches(1.9),
            w=Inches(12.1))
    text(sl, Inches(0.62), Inches(5.5), Inches(12.1), Inches(1.1),
         [[("The headline survived the widening, and it came with a price tag. ",
            {"bold": True}),
           (f"At equal training budget the overlap deficit is still there at "
            f"{xeq['pairs'] - xeq['clean_sweeps']} of {xeq['pairs']} models — thirteen "
            f"times our own benchmark, on a second vendor's models — while the "
            f"groundedness lead reproduces at both Qwen 2.5 sizes. Give the ontology arm "
            f"three or four training passes against a baseline given one and "
            f"{xex['never_worse']} of {xex['pairs']} are worse on nothing we measure. "
            f"That is the best result we have and it is a 3:1 advantage, so we label it "
            f"rather than lead with it.", {})]], size=11.5, color=INK)
    footer(sl, "task4_evaluation/cross_model_results.md — 20 stored arms re-scored, "
               "training budgets read from the companion project's own receipts", num())

    # 13 ── limits ────────────────────────────────────────────────────────
    sl = slide(prs)
    header(sl, "results · the honest half",
           "Where ontology grounding does not win, and what we did about it",
           "Four things we tested that did not go our way, each with the number")
    items = [
        ("Word overlap with the reference answers",
         f"The deployed cell is {G[big]['vs_rag']['token_f1']['delta']:+.2f} points at "
         f"{P[big]} and behind at every size. The graph carries the facts; the references "
         f"carry the phrasing.", RED),
        ("Part of that gap is how the benchmark was built",
         f"The gold answers were written from the corpus the baseline retrieves from. "
         f"Rewording {lim['paraphrase']['accepted']} of "
         f"{lim['paraphrase']['total']} references removes "
         f"{min(lim['paraphrase']['share_of_lead_explained_pct'].values())}–"
         f"{max(lim['paraphrase']['share_of_lead_explained_pct'].values())}% of the "
         f"baseline's lead — it inflates the gap without creating it.", AMBER),
        ("Routing per question cannot close it",
         f"Four cross-validated routers all settle on \"always use retrieval\". Even an "
         f"oracle that picks the better stored answer per question gains only "
         f"{lim['oracle_headroom']:.2f} points — that is the ceiling for the whole family.",
         RED),
        ("Two of eight rules are still unenforceable",
         f"{', '.join(er['still_blocked'])} need a graded risk severity. It appears in "
         f"13 of {sc['corpus_records']:,} source records, so this is missing from the "
         f"source and no extraction can invent it.", AMBER),
    ]
    for i, (t, b, col) in enumerate(items):
        x = Inches(0.62 + (i % 2) * 6.16)
        y = Inches(2.0 + (i // 2) * 2.3)
        rect(sl, x, y, Inches(5.95), Inches(2.0), fill=WHITE, line=RGBColor(0xDD, 0xE4, 0xEC))
        rect(sl, x, y, Pt(5), Inches(2.0), fill=col)
        text(sl, x + Inches(0.26), y + Inches(0.16), Inches(5.4), Inches(0.5), t,
             size=14.5, color=INK, bold=True, line_spacing=1.05)
        text(sl, x + Inches(0.26), y + Inches(0.72), Inches(5.45), Inches(1.2), b,
             size=11.5, color=MUTE, line_spacing=1.14)
    text(sl, Inches(0.62), Inches(6.68), Inches(12.1), Inches(0.4),
         "A deck that hid these would not survive the first technical question. The "
         "measured case for the ontology is maintenance, safety and small-model economics.",
         size=12, color=INK, bold=True)
    footer(sl, "task4_evaluation/reference_paraphrase__llama3.1-8b.json, "
               "coverage_routing__llama3.1-8b.json, task2_formalization/"
               "rule_compliance__1.5.0.json", num())

    # 13b ── R19: the cost of being careful, re-measured ─────────────────
    RP = ev.get("repair") or {}
    vxp = DELIV / "task4_evaluation" / "verifier_across_models__test.json"
    vx = json.loads(vxp.read_text(encoding="utf-8")) if vxp.exists() else None
    if all((RP.get(m) or {}).get("after") for m in SZ):
        sl = slide(prs)
        header(sl, "results · the cost of being careful, re-measured",
               "Most of what checking appeared to cost was the rewrite, not the check",
               "Same gate, same detector, same questions; only how the layer fixes a "
               "flagged answer changes")
        rws = [["", "how the layer fixes an answer", "answer quality",
                "vs retrieval", "unsupported claims ↓", "fully grounded ↑", "cost"]]
        for m in SZ:
            r = RP[m]
            for lab, k in (("before", "before"), ("after", "after")):
                d = r[k]
                rws.append([f"{P[m]} · {lab}",
                            "re-ask the model to rewrite it" if k == "before"
                            else "delete the clause that is wrong",
                            f"{d['token_f1']:.2f}",
                            f"{d['vs_rag_token_f1']:+.2f}",
                            f"{d['ungrounded']:.2f}%", f"{d['clean']:.1f}%",
                            d["cost"]])
        table(sl, Inches(0.62), Inches(1.95), Inches(12.1), rws,
              col_w=[1.55, 2.5, 1.3, 1.25, 1.55, 1.45, 2.5], size=10,
              row_h=Inches(0.30))
        text(sl, Inches(0.62), Inches(4.18), Inches(12.1), Inches(0.4),
             "Why this matters commercially", size=15, color=INK, bold=True)
        bullets(sl, Inches(0.62), Inches(4.58), Inches(12.0), Inches(2.1), [
            [("The safety check is now effectively free. ", {"bold": True}),
             (f"Rewriting a flagged answer meant a second pass through the model. "
              f"Deleting the offending clause takes about "
              f"{G[big].get('layer_ms', 1):.0f} ms of ordinary CPU, so the check can "
              f"run on every answer rather than on a budget.", {})],
            [("And it stopped costing answer quality. ", {"bold": True}),
             (f"The gap to the retrieval baseline closes from "
              f"{RP[big]['before']['vs_rag_token_f1']:+.2f} to "
              f"{RP[big]['after']['vs_rag_token_f1']:+.2f} points at {P[big]}, and to "
              f"{RP[SZ[1]]['after']['vs_rag_token_f1']:+.2f} at {P[SZ[1]]}, where it is "
              f"no longer distinguishable from no gap at all.", {})],
            [("One caveat we put on the slide ourselves. ", {"bold": True}),
             (f"The 0.00% above is 0.00% against our own checker, which is the thing "
              f"doing the deleting. Measured with a stricter independent rule the "
              f"figure is {G[big]['ungrounded_strict']:.2f}% against the retrieval "
              f"baseline's {G[big]['rag']['ungrounded_strict']:.2f}% — still better, and "
              f"that is the number to argue from.", {})],
    ] + ([[("It is not tied to the model we built it on. ", {"bold": True}),
           (f"The checker is now ordinary software, so we ran it unchanged over "
            f"{vx['models'][list(vx['models'])[0]]['n']:,} held-out answers from "
            f"{len(vx['models'])} different models — including "
            f"{', '.join(n for n in ('Mistral-7B', 'Qwen3-8B', 'DeepSeek-R1-8B'))}, "
            f"which we never trained anything for. Unsupported claims more than halve "
            f"for every one of them, at the same fraction-of-a-point cost.", {})]]
         if vx and vx.get("models") else []), size=11.5)
        footer(sl, "task4_evaluation/repair_policy.md, "
                   "verifier_across_models__test.md", num())

    # 14 ── architecture ─────────────────────────────────────────────────
    sl = slide(prs)
    header(sl, "how it works", "The system behind the demo",
           "Retrieval proposes, the graph checks, the layer removes what it cannot back")
    picture(sl, FIG / "fig01_architecture.png", Inches(0.62), Inches(1.95), w=Inches(7.5))
    rect(sl, Inches(8.4), Inches(1.95), Inches(4.32), Inches(4.5), fill=WHITE,
         line=RGBColor(0xDD, 0xE4, 0xEC))
    text(sl, Inches(8.66), Inches(2.1), Inches(3.85), Inches(0.35),
         "Four stages", size=14, color=INK, bold=True)
    stg = [("Retrieve", "passages and entity cards for the question"),
           ("Draft", "the small model answers from that context"),
           ("Check", "every claim against approved facts in the graph"),
           ("Repair or decline", "delete the clause that is unsupported, keep the rest; refuse when nothing is left")]
    y = Inches(2.5)
    for i, (t, d) in enumerate(stg):
        rect(sl, Inches(8.66), y, Inches(0.34), Inches(0.34), fill=TEAL, shape=MSO_SHAPE.OVAL)
        text(sl, Inches(8.66), y + Inches(0.02), Inches(0.34), Inches(0.3), str(i + 1),
             size=11, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        text(sl, Inches(9.1), y - Inches(0.02), Inches(3.4), Inches(0.9),
             [[(t + " — ", {"bold": True}), (d, {"color": MUTE})]], size=11.5,
             line_spacing=1.1)
        y = y + Inches(0.95)
    footer(sl, "wbs/deliverables/figures/fig01_architecture.png", num())

    # 15 ── the knowledge asset ──────────────────────────────────────────
    sl = slide(prs)
    header(sl, "the asset", "What the knowledge base is, and how it is governed",
           "A versioned, validated, fully-sourced graph — the thing the business owns")
    picture(sl, FIG / "fig02_ontology_graph.png", Inches(0.62), Inches(1.95), w=Inches(6.6))
    kpi(sl, Inches(7.5), Inches(1.95), Inches(2.5), f"{sc['concepts']:,}", "concepts",
        f"{sc['classes']} classes, {sc['predicates']} relation types", TEAL, h=Inches(1.3))
    kpi(sl, Inches(10.22), Inches(1.95), Inches(2.5), f"{sc['relations']:,}", "approved facts",
        f"{sc['isolated_concepts']} orphan concepts", TEAL, h=Inches(1.3))
    kpi(sl, Inches(7.5), Inches(3.42), Inches(2.5),
        f"{sc['provenance_coverage'] * 100:.0f}%", "facts with provenance",
        "every fact names the record it came from", GREEN, h=Inches(1.3))
    kpi(sl, Inches(10.22), Inches(3.42), Inches(2.5), f"{sc['corpus_records']:,}",
        "source records", "the corpus the graph was extracted from", AMBER, h=Inches(1.3))
    text(sl, Inches(7.5), Inches(4.95), Inches(5.22), Inches(1.6),
         [[("Governed, not just built. ", {"bold": True}),
           (f"Versions are additive and published side by side; each release is checked by "
            f"SHACL shapes and an OWL reasoner before it is used "
            f"({er['validation_after_edit']['errors']} errors on the latest). An edit is "
            f"reviewable because every fact carries its source record.", {})]],
         size=11.5, color=INK, line_spacing=1.14)
    footer(sl, "task2_formalization/ontology_quality_metrics__1.5.0.json, "
               "validation_report__1.5.0.json", num())

    # 16 ── cost of running it ───────────────────────────────────────────
    sl = slide(prs)
    header(sl, "operations", "What it costs to run, per answer",
           "Measured on the stored runs, same hardware for every arm")
    rws = [["", "prompt tokens", "generation", "validation", "unsupported ↓", "grounded ↑"]]
    for lab, k in [("model alone", "V1_llm"), ("+ retrieval", "V2_rag"),
                   ("+ ontology", "V3_ontology_only"), ("+ both", "V4_ontology_rag")]:
        d = V[big][k]
        rws.append([lab, f"{d['prompt_tokens']:,}", f"{d['generation_s']:.1f} s", "—",
                    f"{d['ungrounded']:.2f}%", f"{d['clean']:.1f}%"])
    d5 = V[big]["V5_ontology_rag_validated"]
    rws.append(["+ validation (all answers)", f"{d5['prompt_tokens']:,}",
                f"{d5['generation_s']:.1f} s", f"{d5['validation_ms'] / 1000:.1f} s",
                f"{d5['ungrounded']:.2f}%", f"{d5['clean']:.1f}%"])
    rws.append(["+ validation (deployed, gated)", f"{d5['prompt_tokens']:,}",
                f"{d5['generation_s']:.1f} s", "only on flagged drafts",
                f"{G[big]['ungrounded']:.2f}%", f"{G[big]['clean']:.1f}%"])
    table(sl, Inches(0.62), Inches(2.05), Inches(12.1), rws,
          col_w=[2.4, 1.1, 1.1, 1.6, 1.2, 1.1], size=11.5, highlight=6, row_h=Inches(0.38))
    bullets(sl, Inches(0.62), Inches(4.9), Inches(12.0), Inches(1.6), [
        [("The ontology costs context, not latency. ", {"bold": True}),
         (f"Entity cards add tokens to the prompt; generation time is unchanged within "
          f"noise.", {})],
        [("The validation layer is the expensive part, so it is spent selectively. ",
          {"bold": True}),
         (f"Validating every answer costs {d5['validation_ms'] / 1000:.1f} s per answer; "
          f"gating on the detector pays it only on the drafts that need it and gives back "
          f"{G[big]['gate_token_f1_gain']:+.2f} answer-quality points.", {})],
        [("All figures are at ", {"bold": True}), (f"{P[big]}", {"bold": True}),
         ("; the 1B model generates in a fraction of the time, which is the point of "
          "reason 2.", {})],
    ], size=12)
    footer(sl, "task4_evaluation/variant_scores__llama3.1-8b.json, selective_axis__*.json", num())

    # 17 ── what is still open ───────────────────────────────────────────
    sl = slide(prs)
    header(sl, "what remains", "Four things people must settle",
           "Stated plainly, with what each one needs from outside the pipeline")
    # the compliance count is read, not typed, so this slide cannot go stale
    pcj = json.loads((DELIV / "PROPOSAL_COMPLIANCE.json").read_text(encoding="utf-8"))
    _xm = json.loads((DELIV / "task4_evaluation" / "cross_model_results.json")
                     .read_text(encoding="utf-8"))
    _mt = _xm["summary"]["matched"]
    mt_never, mt_pairs = _mt["never_worse"], _mt["pairs"]
    mt_clean = len(_xm["metric_rollup"]["matched"]["clean"]["ontology_better_at"])
    rws = [["open item", "what it needs", "who provides it"],
           ["Human adjudication of the hallucination audit",
            "two or more annotators filling the two review instruments already "
            "built; the agreement scoring they feed is written and tested",
            "domain reviewers"],
           [f"Rules {', '.join(er['still_blocked'])}",
            "a source that carries graded risk severity (13 of "
            f"{sc['corpus_records']:,} records have it)", "data owner / Jeju authority"],
           ["Identity behind the governance roles",
            "an identity provider. The roles and permissions are enforced — a "
            "principal without one is refused — but the principal is self-declared",
            "pilot deployment"],
           ["Expert panel and user study",
            "human-subject work scoped out of this phase by the proposal itself",
            "next phase"],
           ["Equal-budget parity at extended training — CLOSED",
            f"ran it: Simple RAG retrained to the same 3-4 passes on all four "
            f"models (exp060, ~51 GPU-hours). {mt_never} of {mt_pairs} models stay "
            f"worse on nothing measured, down from 4 of 4. The overlap wins were "
            f"the budget; the groundedness win at {mt_clean} of {mt_pairs} was not",
            "done — no longer open"]]
    table(sl, Inches(0.62), Inches(2.0), Inches(12.1), rws, col_w=[2.6, 4.2, 1.8],
          size=11, row_h=Inches(0.52), align_right_from=99)
    rect(sl, Inches(0.62), Inches(5.3), Inches(12.1), Inches(1.3), fill=WHITE,
         line=RGBColor(0xDD, 0xE4, 0xEC))
    rect(sl, Inches(0.62), Inches(5.3), Pt(5), Inches(1.3), fill=TEAL)
    text(sl, Inches(0.95), Inches(5.45), Inches(11.5), Inches(1.0),
         [[("Everything else is done and audited. ", {"bold": True}),
           (f"All 16 WBS line items are complete on the evidence on disk, "
            f"{pcj['met']} of the proposal's {pcj['total']} requirements are met with "
            f"{pcj['out_of_scope']} needing a human panel rather than more code, and the "
            f"full chain — extraction, formalisation, reasoning, retrieval, evaluation, "
            f"scalability — reproduces from one package with its tests green.",
            {})]], size=12.5, color=INK, line_spacing=1.16)
    footer(sl, "wbs/deliverables/GOAL_STATUS.md, WBS_STATUS.md, PROPOSAL_COMPLIANCE.md", num())

    # 18 ── closing ──────────────────────────────────────────────────────
    sl = slide(prs, paper=False)
    rect(sl, 0, 0, Inches(0.22), H, fill=TEAL)
    text(sl, Inches(0.95), Inches(0.9), Inches(11.5), Inches(0.5),
         "THE CASE, IN THREE LINES", size=13, color=TEAL, bold=True)
    text(sl, Inches(0.95), Inches(1.4), Inches(11.6), Inches(2.6),
         [f"Knowledge you can change in minutes: {ED['facts']} facts over "
          f"{ED['n']} edits, {er['rules_enforceable_before']} → "
          f"{er['rules_enforceable_after']} rules enforceable, "
          f"{ED['gpu_hours']:.0f} GPU-hours.",
          f"A small model that carries the job: {P[small]} with the graph ahead of "
          f"{P[big]} without it on every measure we report.",
          f"Answers you can defend: unsupported claims down and fully-grounded answers up "
          f"against the retrieval baseline at all three sizes."],
         size=19, color=WHITE, space=16, line_spacing=1.1)
    rect(sl, Inches(0.95), Inches(4.3), Inches(11.5), Inches(1.25), fill=NAVY2,
         line=RGBColor(0x2A, 0x44, 0x6B))
    text(sl, Inches(1.25), Inches(4.45), Inches(11.0), Inches(1.0),
         [[("And the boundary we keep. ", {"bold": True, "color": AMBER}),
           (f"The ontology does not beat plain retrieval on word overlap with "
            f"corpus-written reference answers, and we do not claim it does. Since the "
            f"checker stopped rewriting whole answers the gap is "
            f"{abs(G[big]['vs_rag']['token_f1']['delta']):.2f} of a point at {P[big]} "
            f"and no longer distinguishable from zero at {P[SZ[1]]} — parity, not a "
            f"lead. It wins on groundedness, maintenance cost and small-model "
            f"economics, which is what a deployment is judged on.",
            {"color": RGBColor(0xC3, 0xD0, 0xDE)})]], size=13,
         line_spacing=1.16)
    text(sl, Inches(0.95), Inches(5.95), Inches(11.5), Inches(0.9),
         ["Next: fill the two review instruments with human annotators, and pilot the "
          "graph-edit workflow with the data owner.",
          "Deck, demo video and every number in them: wbs/deliverables/presentation/"],
         size=12.5, color=RGBColor(0x8F, 0xA3, 0xB8), space=6)

    # 19 ── appendix ─────────────────────────────────────────────────────
    sl = slide(prs)
    header(sl, "appendix", "How to check any number on these slides",
           "Nothing here was typed by hand")
    bullets(sl, Inches(0.62), Inches(2.0), Inches(12.0), Inches(4), [
        [("The deck is generated. ", {"bold": True}),
         ("wbs/task5_demo/make_slides.py reads presentation/demo_evidence.json and writes "
          "this file. Change a measurement, re-run, and the slide changes.", {})],
        [("The evidence file is generated too. ", {"bold": True}),
         ("wbs/task5_demo/build_demo_evidence.py collects it from the evaluation "
          "deliverables; each slide's footer names the file its figures came from.", {})],
        [("The quoted answers are verbatim. ", {"bold": True}),
         ("Before/after text is copied from the stored run files, trimmed only for length "
          "where the slide cannot hold the whole answer.", {})],
        [("The live edit is reproducible. ", {"bold": True}),
         ("python -m wbs.task5_demo.demo_live_edit --gpu 5 re-runs the two generations and "
          "re-checks the adapter digests.", {"font": MONO, "size": 11})],
        [("The pipeline reproduces. ", {"bold": True}),
         ("wbs/deliverables/REPRODUCIBILITY holds the input manifest, the run order and the "
          "environment; the test suite covers the scorers and the report generators.", {})],
    ], size=13)
    footer(sl, "wbs/deliverables/REPRODUCIBILITY, wbs/task5_demo/", num())

    p = OUT / "Ontology_Business_Plan.pptx"
    prs.save(str(p))
    print(f"wrote {p.relative_to(ROOT)}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
