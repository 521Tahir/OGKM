"""
Slide deck explaining the 200-question multi-hop benchmark: why it was needed,
how it was built, and how each arm is applied to it.

  python -m experiments.exp025_multihop.make_slides
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

E25 = ROOT / "experiments" / "exp025_multihop" / "outputs"
OUTPPT = ROOT / "docs" / "Jeju_Multihop_Dataset_Slides.pptx"

NAVY = RGBColor(0x14, 0x2C, 0x47)
BLUE = RGBColor(0x1F, 0x6F, 0xB2)
TEAL = RGBColor(0x0E, 0x8C, 0x8C)
RED = RGBColor(0xB5, 0x35, 0x35)
GREEN = RGBColor(0x1E, 0x7D, 0x4F)
GREY = RGBColor(0x5A, 0x63, 0x6E)
LIGHT = RGBColor(0xEF, 0xF3, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = Inches(13.333), Inches(7.5)


def deck() -> Presentation:
    p = Presentation()
    p.slide_width, p.slide_height = W, H
    return p


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def box(slide, x, y, w, h, fill=None, line=None, line_w=1.0):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.adjustments[0] = 0.06
    if fill:
        s.fill.solid(); s.fill.fore_color.rgb = fill
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = line; s.line.width = Pt(line_w)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=1.0):
    """runs: list of (text, size, bold, color) or list of such lists (paragraphs)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    paras = runs if isinstance(runs[0], list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        for t, size, bold, color in para:
            r = p.add_run(); r.text = t
            r.font.size = Pt(size); r.font.bold = bold
            r.font.color.rgb = color; r.font.name = "Calibri"
    return tb


def mono(slide, x, y, w, h, lines, size=12, fill=LIGHT, color=NAVY):
    box(slide, x, y, w, h, fill=fill)
    tb = slide.shapes.add_textbox(x + Inches(0.15), y + Inches(0.1),
                                  w - Inches(0.3), h - Inches(0.2))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Emu(0)
    for i, (txt, c, bold) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.08
        r = p.add_run(); r.text = txt
        r.font.size = Pt(size); r.font.name = "Consolas"
        r.font.color.rgb = c or color; r.font.bold = bold
    return tb


def header(slide, title, kicker=None):
    bar = slide.shapes.add_shape(1, Emu(0), Emu(0), W, Inches(0.09))
    bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background()
    bar.shadow.inherit = False
    y = Inches(0.42)
    if kicker:
        text(slide, Inches(0.7), y, Inches(12), Inches(0.3),
             [(kicker.upper(), 12, True, TEAL)])
        y = Inches(0.78)
    text(slide, Inches(0.7), y, Inches(12), Inches(0.6), [(title, 30, True, NAVY)])
    return Inches(1.55) if kicker else Inches(1.35)


def footer(slide, n, note=""):
    text(slide, Inches(0.7), Inches(7.02), Inches(10), Inches(0.3),
         [(note, 10, False, GREY)])
    text(slide, Inches(12.0), Inches(7.02), Inches(0.7), Inches(0.3),
         [(str(n), 10, False, GREY)], align=PP_ALIGN.RIGHT)


def table(slide, x, y, w, rows, col_w, header_row=True, size=12, row_h=0.36):
    """Manual table: crisper than pptx's default styling."""
    yy = y
    for ri, row in enumerate(rows):
        h = Inches(row_h)
        if ri == 0 and header_row:
            box(slide, x, yy, w, h, fill=NAVY)
        elif ri % 2 == 0:
            box(slide, x, yy, w, h, fill=LIGHT)
        xx = x
        for ci, cell in enumerate(row):
            cw = Inches(col_w[ci])
            val, color, bold = cell if isinstance(cell, tuple) else (cell, None, False)
            c = WHITE if (ri == 0 and header_row) else (color or NAVY)
            b = True if (ri == 0 and header_row) else bold
            tb = slide.shapes.add_textbox(xx + Inches(0.1), yy, cw - Inches(0.15), h)
            tf = tb.text_frame; tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = tf.margin_right = Emu(0)
            tf.margin_top = tf.margin_bottom = Emu(0)
            p = tf.paragraphs[0]
            r = p.add_run(); r.text = str(val)
            r.font.size = Pt(size); r.font.bold = b; r.font.color.rgb = c
            r.font.name = "Calibri"
            xx += cw
        yy += h
    return yy


def build() -> None:
    d25 = json.loads((E25 / "bench_results.json").read_text())
    a25 = d25["aggregates"]
    prs = deck()
    n = 0

    # ── 1 TITLE ──────────────────────────────────────────────────────────
    s = blank(prs)
    bg = s.shapes.add_shape(1, Emu(0), Emu(0), W, H)
    bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
    bg.shadow.inherit = False
    text(s, Inches(1.0), Inches(2.3), Inches(11.3), Inches(1.0),
         [("Building a Multi-Hop Benchmark", 44, True, WHITE)])
    text(s, Inches(1.0), Inches(3.25), Inches(11.3), Inches(0.8),
         [("for Jeju Tourism QA", 44, True, RGBColor(0x7E, 0xC8, 0xE3))])
    bar = s.shapes.add_shape(1, Inches(1.0), Inches(4.35), Inches(1.6), Inches(0.05))
    bar.fill.solid(); bar.fill.fore_color.rgb = TEAL; bar.line.fill.background()
    bar.shadow.inherit = False
    text(s, Inches(1.0), Inches(4.75), Inches(11), Inches(0.9),
         [[("200 questions  ·  100 multi-hop + 100 single-hop  ·  gazetteer-grounded",
            18, False, RGBColor(0xC9, 0xD6, 0xE3))],
          [("How the dataset was created, and how Simple RAG and Ontology are tested on it",
            15, False, RGBColor(0x9F, 0xB2, 0xC6))]], spacing=1.5)
    n += 1

    # ── 2 WHY A NEW DATASET ──────────────────────────────────────────────
    s = blank(prs); y = header(s, "Why we needed a new dataset", "motivation")
    text(s, Inches(0.7), y, Inches(12), Inches(0.5),
         [("The existing 4,607-item test split cannot test multi-hop reasoning.",
           17, True, NAVY)])
    table(s, Inches(0.7), y + Inches(0.65), Inches(6.0), [
        ["Question type in test split", "Count"],
        ["Itinerary / planning requests", "1,541"],
        ["Other free-form requests", "2,687"],
        [("wh-questions (what/where/when)", None, True), ("379", None, True)],
        [("Labelled multi-hop", RED, True), ("0", RED, True)],
    ], [4.2, 1.8], row_h=0.42)
    box(s, Inches(7.1), y + Inches(0.65), Inches(5.5), Inches(2.1), fill=LIGHT)
    text(s, Inches(7.35), y + Inches(0.85), Inches(5.1), Inches(1.8),
         [[("A typical test item:", 13, True, NAVY)],
          [("“Recommend a 5-day Jeju itinerary for seniors "
            "travelling in autumn, prioritising relaxation, cultural "
            "sites and wellness.”", 13, False, GREY)],
          [("Free-form generation — not a factual question with a "
            "checkable answer.", 12, True, RED)]], spacing=1.3)
    text(s, Inches(0.7), Inches(5.55), Inches(12), Inches(0.9),
         [[("Multi-hop chaining is exactly where a knowledge graph should win, "
            "if it wins anywhere.", 16, True, TEAL)],
          [("So the benchmark had to be constructed from scratch.", 15, False, NAVY)]],
         spacing=1.4)
    footer(s, n := n + 1, "Experiment 2 · motivation")

    # ── 3 WHAT IS MULTI-HOP ──────────────────────────────────────────────
    s = blank(prs); y = header(s, "What counts as a multi-hop question", "definition")
    text(s, Inches(0.7), y, Inches(12), Inches(0.4),
         [("A question that cannot be answered by a single lookup — the answer "
           "requires composing two facts.", 16, False, NAVY)])
    yy = y + Inches(0.7)
    text(s, Inches(0.7), yy, Inches(12), Inches(0.4),
         [[("“Which museum is nearest to Jeju's UNESCO-listed lava tube, "
            "and what is it noted for?”", 17, True, NAVY)]])
    yy += Inches(0.75)
    chain = [("Jeju's UNESCO-listed\nlava tube", BLUE, "the question names\nonly a description"),
             ("Manjanggul Cave", TEAL, "HOP 1\nresolve the description"),
             ("Jeju Haenyeo Museum", TEAL, "HOP 2\nnearest museum, 6 km"),
             ("“women divers museum”", GREEN, "attribute\nlookup")]
    x = Inches(0.7)
    for i, (label, col, note) in enumerate(chain):
        box(s, x, yy, Inches(2.6), Inches(0.95), fill=None, line=col, line_w=2.0)
        text(s, x + Inches(0.12), yy + Inches(0.12), Inches(2.36), Inches(0.75),
             [(label, 13, True, col)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + Inches(0.12), yy + Inches(1.05), Inches(2.36), Inches(0.7),
             [(note, 11, False, GREY)], align=PP_ALIGN.CENTER)
        if i < 3:
            text(s, x + Inches(2.62), yy + Inches(0.2), Inches(0.5), Inches(0.5),
                 [("→", 22, True, GREY)], align=PP_ALIGN.CENTER)
        x += Inches(3.12)
    box(s, Inches(0.7), Inches(5.5), Inches(11.9), Inches(1.0), fill=LIGHT)
    text(s, Inches(1.0), Inches(5.7), Inches(11.4), Inches(0.7),
         [[("Design rule: the bridge entity is never named in the question.",
            16, True, RED)],
          [("If the question said “nearest to Manjanggul Cave” it would collapse "
            "to a single hop.", 14, False, NAVY)]], spacing=1.35)
    footer(s, n := n + 1, "Experiment 2 · definition")

    # ── 4 THE FAIRNESS PROBLEM ───────────────────────────────────────────
    s = blank(prs); y = header(s, "The critical question: where does the gold answer come from?",
                               "methodology")
    text(s, Inches(0.7), y, Inches(12), Inches(0.4),
         [("Whichever source we ground the answers in decides the winner before "
           "a single model runs.", 16, True, NAVY)])
    yy = y + Inches(0.65)
    cards = [
        ("The project ontology", RED, "REJECTED",
         ["Gold retrievable by the ontology arm and", "by nothing else → it wins by",
          "construction.", "", "Also: 59.9% of type-checkable",
          "relations are factually wrong."]),
        ("The 46k training corpus", RED, "REJECTED",
         ["This is the RAG arm's own index → gold", "drawn from it favours RAG.", "",
          "Also: the corpus is itself LLM-generated,", "so a co-occurrence filter removed",
          "almost nothing."]),
        ("The curated gazetteer", GREEN, "USED",
         ["69 real Jeju places with real", "coordinates, categories and notes.", "",
          "Neither arm indexes it directly →", "neutral to both.", "",
          "And it is factually correct."]),
    ]
    x = Inches(0.7)
    for title, col, tag, lines in cards:
        box(s, x, yy, Inches(3.85), Inches(3.3), fill=None, line=col, line_w=2.0)
        tagb = box(s, x, yy, Inches(3.85), Inches(0.42), fill=col)
        text(s, x, yy + Inches(0.06), Inches(3.85), Inches(0.3),
             [(tag, 12, True, WHITE)], align=PP_ALIGN.CENTER)
        text(s, x + Inches(0.2), yy + Inches(0.58), Inches(3.45), Inches(0.4),
             [(title, 16, True, NAVY)])
        text(s, x + Inches(0.2), yy + Inches(1.05), Inches(3.45), Inches(2.1),
             [[(l, 12, False, GREY)] for l in lines], spacing=1.12)
        x += Inches(4.03)
    text(s, Inches(0.7), Inches(6.35), Inches(12), Inches(0.5),
         [("Every gold answer is computed by haversine over real coordinates — "
           "auditable, not asserted.", 15, True, TEAL)])
    footer(s, n := n + 1, "Experiment 2 · source selection")

    # ── 5 THE GAZETTEER ──────────────────────────────────────────────────
    s = blank(prs); y = header(s, "The source we used: the curated gazetteer", "source")
    text(s, Inches(0.7), y, Inches(12), Inches(0.4),
         [("data/jeju_full/jeju_gazetteer.json — the only curated, non-synthetic "
           "artefact in the project.", 15, False, NAVY)])
    mono(s, Inches(0.7), y + Inches(0.6), Inches(6.1), Inches(1.5), [
        ('{"name":     "Manjanggul Cave",', NAVY, True),
        (' "category": "NaturalAttraction",', None, False),
        (' "lat": 33.5285, "lon": 126.7717,', None, False),
        (' "note":     "UNESCO lava tube"}', TEAL, True)], size=13)
    table(s, Inches(7.1), y + Inches(0.6), Inches(5.5), [
        ["Contents", "Count"],
        ["Locations (with lat/lon)", "69"],
        ["Cuisine entries", "15"],
        ["Activities", "15"],
        [("Jeju-only anchor names", None, True), ("111", None, True)],
    ], [3.8, 1.7], row_h=0.38)
    text(s, Inches(0.7), Inches(4.3), Inches(12), Inches(0.4),
         [("13 location categories provide the type vocabulary the templates use:",
           14, True, NAVY)])
    text(s, Inches(0.7), Inches(4.75), Inches(12), Inches(0.9),
         [[("NaturalAttraction 13 · Beach 9 · Trail 9 · Region 8 · "
            "Attraction 7 · Museum 6 · Island 4", 14, False, GREY)],
          [("Mountain 3 · CulturalAttraction 3 · City 2 · Market 2 · "
            "ReligiousSite 2 · Transportation 1", 14, False, GREY)]], spacing=1.3)
    box(s, Inches(0.7), Inches(5.85), Inches(11.9), Inches(0.85), fill=LIGHT)
    text(s, Inches(1.0), Inches(6.05), Inches(11.4), Inches(0.5),
         [("Because every entity comes from a Jeju-only gazetteer, all 200 questions "
           "are Jeju-specific by construction.", 15, True, GREEN)])
    footer(s, n := n + 1, "Experiment 2 · source")

    # ── 6 PIPELINE ───────────────────────────────────────────────────────
    s = blank(prs); y = header(s, "How the 200 questions were created", "pipeline")
    steps = [
        ("1", "Select anchor", "Pick a gazetteer entity\nas the question's anchor"),
        ("2", "Compute hop 1", "Haversine over real lat/lon\n→ nearest entity of a\ntarget category"),
        ("3", "Compute hop 2", "Look up that bridge\nentity's category, note\nor nearest city"),
        ("4", "Verbalise", "Fill a template; the\nbridge is never named\nin the question"),
        ("5", "Verify", "Jeju guard, blocklist,\nde-duplication,\nbridge-leak check"),
    ]
    x = Inches(0.55)
    for num, title, body in steps:
        box(s, x, y + Inches(0.3), Inches(2.3), Inches(2.5), fill=None, line=BLUE, line_w=1.5)
        circ = s.shapes.add_shape(9, x + Inches(0.9), y + Inches(0.05), Inches(0.5), Inches(0.5))
        circ.fill.solid(); circ.fill.fore_color.rgb = BLUE; circ.line.fill.background()
        circ.shadow.inherit = False
        text(s, x + Inches(0.9), y + Inches(0.13), Inches(0.5), Inches(0.35),
             [(num, 16, True, WHITE)], align=PP_ALIGN.CENTER)
        text(s, x + Inches(0.15), y + Inches(0.75), Inches(2.0), Inches(0.4),
             [(title, 15, True, NAVY)], align=PP_ALIGN.CENTER)
        text(s, x + Inches(0.15), y + Inches(1.25), Inches(2.0), Inches(1.4),
             [(body, 12, False, GREY)], align=PP_ALIGN.CENTER)
        if num != "5":
            text(s, x + Inches(2.32), y + Inches(1.25), Inches(0.4), Inches(0.4),
                 [("→", 20, True, GREY)], align=PP_ALIGN.CENTER)
        x += Inches(2.53)
    box(s, Inches(0.55), Inches(5.3), Inches(12.2), Inches(1.35), fill=LIGHT)
    text(s, Inches(0.85), Inches(5.5), Inches(11.6), Inches(1.0),
         [[("Every item carries an explicit evidence chain, so any answer can be audited:",
            14, True, NAVY)],
          [("hop 1: UNESCO-listed lava tube = Manjanggul Cave   |   "
            "hop 2: nearest Museum = Jeju Haenyeo Museum (6.0 km)   |   "
            "attr: note = women divers museum", 12, False, TEAL)]], spacing=1.4)
    footer(s, n := n + 1, "Experiment 2 · construction")

    # ── 7 TEMPLATES ──────────────────────────────────────────────────────
    s = blank(prs); y = header(s, "The seven question templates", "composition")
    table(s, Inches(0.7), y, Inches(11.9), [
        ["Template", "Hops", "n", "Pattern"],
        [("T1  proximity → attribute", TEAL, True), "2", "25",
         "Which <category> lies closest to X, and what is it known for?"],
        [("T2  region → attraction", TEAL, True), "2", "25",
         "X sits closest to which town, and which natural attraction is near it?"],
        [("T3  descriptor → proximity", TEAL, True), "2", "25",
         "Which <category> is nearest to <definite description>, and what is it noted for?"],
        [("T4  trail → city", TEAL, True), "2", "25",
         "If I hike the trail closest to X, which city would I then be nearest to?"],
        [("S1  what-is", BLUE, True), "1", "34",
         "What kind of place is X, and what is it known for?"],
        [("S2  which-city", BLUE, True), "1", "33",
         "Which of Jeju's cities is X closest to?"],
        [("S3  which-region", BLUE, True), "1", "33",
         "Setting aside the two cities, which town or region is X nearest to?"],
    ], [3.1, 0.7, 0.6, 7.5], size=12, row_h=0.46)
    box(s, Inches(0.7), Inches(5.55), Inches(5.8), Inches(1.0), fill=LIGHT)
    text(s, Inches(1.0), Inches(5.72), Inches(5.3), Inches(0.7),
         [[("100 multi-hop  +  100 single-hop  =  200", 17, True, NAVY)],
          [("T3 is the hardest: hop 1 is a definite description, not a name.",
            12, False, GREY)]], spacing=1.35)
    box(s, Inches(6.8), Inches(5.55), Inches(5.8), Inches(1.0), fill=LIGHT)
    text(s, Inches(7.1), Inches(5.72), Inches(5.3), Inches(0.7),
         [[("Pool available: 480 multi-hop / 199 single-hop", 15, True, NAVY)],
          [("Sampled round-robin per template so no template dominates.",
            12, False, GREY)]], spacing=1.35)
    footer(s, n := n + 1, "Experiment 2 · templates")

    # ── 8 QC ─────────────────────────────────────────────────────────────
    s = blank(prs); y = header(s, "Quality control — all 200 items", "validation")
    checks = [("Jeju guard", "src/utils/jeju_guard.py on every question", "0 failures"),
              ("Non-Jeju leakage", "blocklist regex over question and answer", "0 mentions"),
              ("Duplicate questions", "exact string match after normalisation", "0 duplicates"),
              ("Bridge leakage", "bridge entity must not appear in the question",
               "0 leaked")]
    yy = y + Inches(0.1)
    for name, method, result in checks:
        box(s, Inches(0.7), yy, Inches(11.9), Inches(0.9), fill=LIGHT)
        tick = s.shapes.add_shape(9, Inches(0.95), yy + Inches(0.22), Inches(0.45), Inches(0.45))
        tick.fill.solid(); tick.fill.fore_color.rgb = GREEN; tick.line.fill.background()
        tick.shadow.inherit = False
        text(s, Inches(0.95), yy + Inches(0.29), Inches(0.45), Inches(0.35),
             [("✓", 16, True, WHITE)], align=PP_ALIGN.CENTER)
        text(s, Inches(1.6), yy + Inches(0.15), Inches(3.0), Inches(0.35),
             [(name, 16, True, NAVY)])
        text(s, Inches(1.6), yy + Inches(0.5), Inches(6.5), Inches(0.35),
             [(method, 12, False, GREY)])
        text(s, Inches(9.3), yy + Inches(0.26), Inches(3.0), Inches(0.4),
             [(result, 17, True, GREEN)], align=PP_ALIGN.RIGHT)
        yy += Inches(1.05)
    text(s, Inches(0.7), Inches(6.35), Inches(12), Inches(0.5),
         [("Jeju-specificity is guaranteed twice over: by construction (gazetteer-only "
           "entities) and by post-hoc check.", 15, True, TEAL)])
    footer(s, n := n + 1, "Experiment 2 · quality control")

    # ── 9 WORKED EXAMPLE ─────────────────────────────────────────────────
    s = blank(prs); y = header(s, "A finished item, end to end", "example")
    mono(s, Inches(0.7), y, Inches(11.9), Inches(1.15), [
        ("QUESTION   Which museum on Jeju is nearest to Jeju's UNESCO-listed lava tube,",
         NAVY, True),
        ("           and what is it noted for?", NAVY, True)], size=14)
    mono(s, Inches(0.7), y + Inches(1.3), Inches(11.9), Inches(1.15), [
        ("GOLD       Jeju's UNESCO-listed lava tube is Manjanggul Cave. The nearest museum",
         GREEN, True),
        ("           is Jeju Haenyeo Museum, roughly 6 km away — women divers museum.",
         GREEN, True)], size=14)
    mono(s, Inches(0.7), y + Inches(2.6), Inches(11.9), Inches(1.5), [
        ("EVIDENCE   hop 1 : Jeju's UNESCO-listed lava tube = Manjanggul Cave", TEAL, False),
        ("           hop 2 : nearest Museum = Jeju Haenyeo Museum (6.0 km, haversine)",
         TEAL, False),
        ("           attr  : note = “women divers museum”", TEAL, False)], size=13)
    text(s, Inches(0.7), Inches(6.2), Inches(12), Inches(0.6),
         [[("record_id 900022  ·  template T3  ·  hops 2", 13, True, NAVY)],
          [("The model must resolve the description before it can take the second hop.",
            13, False, GREY)]], spacing=1.35)
    footer(s, n := n + 1, "Experiment 2 · worked example")

    # ── 10 HOW THE TWO ARMS ARE APPLIED ──────────────────────────────────
    s = blank(prs); y = header(s, "How each arm is tested on the same 200 questions",
                               "experimental setup")
    text(s, Inches(0.7), y, Inches(12), Inches(0.4),
         [("Identical model, adapter, prompt template and decoding. "
           "Only the context block differs.", 15, True, NAVY)])
    yy = y + Inches(0.6)
    for title, col, steps, stat, x0 in [
        ("Simple RAG", BLUE,
         ["Embed the question with BGE",
          "FAISS → top-5 nearest train passages",
          "Drop exact-duplicate questions",
          "Truncate each to 900 chars, number [1]…[5]",
          "Header: “Retrieved reference passages:”"],
         "2,527 chars  ·  4.8 passages", Inches(0.7)),
        ("Ontology", TEAL,
         ["Extract question content tokens",
          "Score all 9,208 triples by token overlap",
          "Weight informative predicates, cap is_a",
          "Add up to 6 temporal facts",
          "Header: “Authoritative ontology facts:”"],
         "853 chars  ·  16.7 facts", Inches(6.8)),
    ]:
        box(s, x0, yy, Inches(5.8), Inches(3.5), fill=None, line=col, line_w=2.0)
        hb = box(s, x0, yy, Inches(5.8), Inches(0.5), fill=col)
        text(s, x0, yy + Inches(0.08), Inches(5.8), Inches(0.35),
             [(title, 18, True, WHITE)], align=PP_ALIGN.CENTER)
        ty = yy + Inches(0.7)
        for i, st in enumerate(steps, 1):
            text(s, x0 + Inches(0.3), ty, Inches(5.2), Inches(0.4),
                 [(f"{i}.  ", 13, True, col), (st, 13, False, NAVY)])
            ty += Inches(0.42)
        box(s, x0 + Inches(0.3), yy + Inches(2.85), Inches(5.2), Inches(0.45), fill=LIGHT)
        text(s, x0 + Inches(0.3), yy + Inches(2.95), Inches(5.2), Inches(0.3),
             [("measured on the benchmark:  ", 11, False, GREY), (stat, 11, True, NAVY)],
             align=PP_ALIGN.CENTER)
    box(s, Inches(0.7), Inches(6.0), Inches(11.9), Inches(0.85), fill=LIGHT)
    text(s, Inches(1.0), Inches(6.15), Inches(11.4), Inches(0.6),
         [[("Both arms then feed the same LoRA-adapted model, greedy decoding, "
            "max 512 tokens.", 14, True, NAVY)]])
    footer(s, n := n + 1, "Experiment 2 · how the arms are applied")

    # ── 11 THE MULTI-HOP MISMATCH ────────────────────────────────────────
    s = blank(prs); y = header(s, "Why multi-hop is hard for each arm", "analysis")
    for title, col, body, x0 in [
        ("Simple RAG", BLUE,
         ["Retrieves passages similar to the WHOLE question.",
          "",
          "But a two-hop question names only the start of",
          "the chain — so retrieved passages describe the",
          "anchor, not the bridge or the endpoint.",
          "",
          "It cannot compute “nearest museum”; it can only",
          "surface text that happens to mention both."], Inches(0.7)),
        ("Ontology", TEAL,
         ["A two-hop question is literally two edges in the",
          "graph — in principle the ideal structure.",
          "",
          "But the v1 retriever does NOT traverse the graph.",
          "It scores each triple independently against the",
          "question, so the second hop is retrieved only if",
          "the bridge entity happens to share tokens with",
          "the question."], Inches(6.8)),
    ]:
        box(s, x0, y, Inches(5.8), Inches(3.5), fill=None, line=col, line_w=2.0)
        hb = box(s, x0, y, Inches(5.8), Inches(0.5), fill=col)
        text(s, x0, y + Inches(0.08), Inches(5.8), Inches(0.35),
             [(title, 18, True, WHITE)], align=PP_ALIGN.CENTER)
        text(s, x0 + Inches(0.3), y + Inches(0.75), Inches(5.2), Inches(2.6),
             [[(b, 13, False, NAVY)] for b in body], spacing=1.15)
    box(s, Inches(0.7), Inches(5.5), Inches(11.9), Inches(1.2), fill=None, line=RED, line_w=2.0)
    text(s, Inches(1.0), Inches(5.68), Inches(11.4), Inches(0.9),
         [[("Known limitation, stated in the report:", 14, True, RED)],
          [("A genuine multi-hop retriever — seed on question entities, then expand "
            "along edges — is the obvious next step and is not yet implemented.",
            14, False, NAVY)]], spacing=1.35)
    footer(s, n := n + 1, "Experiment 2 · analysis")

    # ── 12 RESULTS ───────────────────────────────────────────────────────
    s = blank(prs); y = header(s, "What the benchmark showed", "results")
    rows = [["Condition", "BERT-F1", "METEOR", "COMET", "Key-token", "Resp. Acc."]]
    for arm, lab, col in [("rag", "Simple RAG", None), ("onto", "Ontology", TEAL)]:
        r = a25.get(f"multihop__MACRO__{arm}")
        rows.append([(lab, col, col is not None)] +
                    [(f"{r[k]:.4f}", col, col is not None)
                     for k in ["bertf1", "meteor", "comet", "keytok", "judge"]])
    r_, o_ = a25["multihop__MACRO__rag"], a25["multihop__MACRO__onto"]
    rows.append([("Δ (onto − rag)", GREY, True)] +
                [((f"+{o_[k]-r_[k]:.4f}" if o_[k] >= r_[k] else f"−{r_[k]-o_[k]:.4f}"),
                  GREEN if o_[k] >= r_[k] else RED, True)
                 for k in ["bertf1", "meteor", "comet", "keytok", "judge"]])
    text(s, Inches(0.7), y, Inches(12), Inches(0.35),
         [("Multi-hop subset, macro-average over four models (100 questions)",
           15, True, NAVY)])
    table(s, Inches(0.7), y + Inches(0.5), Inches(11.9), rows,
          [2.6, 1.86, 1.86, 1.86, 1.86, 1.86], size=13, row_h=0.5)
    box(s, Inches(0.7), Inches(4.3), Inches(5.8), Inches(1.5), fill=None, line=GREEN, line_w=2.0)
    text(s, Inches(1.0), Inches(4.5), Inches(5.2), Inches(1.1),
         [[("Ontology ahead — shared adapter", 17, True, GREEN)],
          [("BERT-F1, METEOR and COMET favour the ontology in "
            "all 12 model × subset cells. Measured with both arms on ONE "
            "adapter; see the correction below.", 13, False, NAVY)]], spacing=1.25)
    box(s, Inches(6.8), Inches(4.3), Inches(5.8), Inches(1.5), fill=None, line=RED, line_w=2.0)
    text(s, Inches(7.1), Inches(4.5), Inches(5.2), Inches(1.1),
         [[("But read it carefully", 17, True, RED)],
          [("Effect sizes are small (+0.004 to +0.033). Key-token and "
            "Response Accuracy show no reliable ontology gain. The two metric "
            "families disagree.", 13, False, NAVY)]], spacing=1.25)
    text(s, Inches(0.7), Inches(6.1), Inches(11.9), Inches(0.7),
         [[("Correction (exp037): with each arm on its OWN adapter, the overlap "
            "metrics reverse on the larger models.", 14, True, RED)],
          [("Response Accuracy is what holds: +0.0765 (1B), +0.0730 (3B), "
            "+0.0540 (Qwen3-8B), +0.0505 (Llama-3.1-8B), significant on t and "
            "Wilcoxon, no significant loss on any model. A shared adapter "
            "flatters the ontology here; the \u201cwins 5/5 on all six "
            "models\u201d claim is withdrawn. An ontology supplies facts, not "
            "the reference\u2019s phrasing \u2014 so judged correctness rises "
            "while n-gram overlap falls.", 12, False, GREY)]], spacing=1.2)
    footer(s, n := n + 1, "Experiment 2 · results")

    # ── 13 SUMMARY ───────────────────────────────────────────────────────
    s = blank(prs)
    bg = s.shapes.add_shape(1, Emu(0), Emu(0), W, H)
    bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
    bg.shadow.inherit = False
    text(s, Inches(0.9), Inches(0.75), Inches(11.5), Inches(0.7),
         [("Dataset summary", 34, True, WHITE)])
    bar = s.shapes.add_shape(1, Inches(0.9), Inches(1.55), Inches(1.4), Inches(0.05))
    bar.fill.solid(); bar.fill.fore_color.rgb = TEAL; bar.line.fill.background()
    bar.shadow.inherit = False
    facts = [("200", "questions total"), ("100 / 100", "multi-hop / single-hop"),
             ("7", "templates"), ("69", "real Jeju places"),
             ("100%", "Jeju-specific"), ("0", "QC failures")]
    x, yy = Inches(0.9), Inches(2.0)
    for i, (big, small) in enumerate(facts):
        if i == 3:
            x, yy = Inches(0.9), Inches(3.5)
        box(s, x, yy, Inches(3.7), Inches(1.25), fill=RGBColor(0x1E, 0x3D, 0x5E))
        text(s, x + Inches(0.25), yy + Inches(0.12), Inches(3.2), Inches(0.6),
             [(big, 28, True, RGBColor(0x7E, 0xC8, 0xE3))])
        text(s, x + Inches(0.25), yy + Inches(0.75), Inches(3.2), Inches(0.4),
             [(small, 13, False, RGBColor(0xC9, 0xD6, 0xE3))])
        x += Inches(3.9)
    text(s, Inches(0.9), Inches(5.3), Inches(11.5), Inches(1.4),
         [[("Every answer computed by haversine over real coordinates — auditable, "
            "not asserted.", 16, True, WHITE)],
          [("Deliberately grounded in neither the ontology nor the RAG corpus, so "
            "neither arm is handed the answer.", 15, False, RGBColor(0x9F, 0xB2, 0xC6))],
          [("experiments/exp025_multihop/outputs/jeju_bench_200.jsonl", 13, False,
            RGBColor(0x7E, 0xC8, 0xE3))]], spacing=1.5)
    footer(s, n := n + 1)

    OUTPPT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPPT))
    print(f"wrote {OUTPPT}  ({OUTPPT.stat().st_size/1024:.0f} KB, {len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
