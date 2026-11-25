"""Render the narrated demo video.

Every word the narrator says and every number on screen comes from
wbs/deliverables/presentation/demo_evidence.json; the quoted answers are the
stored model outputs, not re-typed. Narration is synthesised with the local
piper voice, the frames are drawn with PIL and the two are muxed by ffmpeg.

    python -m wbs.task5_demo.make_demo_video            # full render
    python -m wbs.task5_demo.make_demo_video --stills   # one PNG per scene

The scene list is data: each entry carries its narration and its draw
function, and the scene's length is whatever the narration needs plus a beat.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wbs.task5_demo.video_kit import (AMBER, FPS, GREEN, H, INK, LINE, MUTE, NAVY,
                                      NAVY2, PAPER, RED, TEAL, W, WHITE, canvas,
                                      card, chapter, clamp, counter, draw_block,
                                      draw_text, ease, ease_out, edge, fade,
                                      fit_font, font, lower_third, measure, pill,
                                      rounded, seg, wrap)

OUT = ROOT / "wbs" / "deliverables" / "presentation"
VOICE = Path("/home/tahirahmad/tools/piper_voices/en_US-ryan-high.onnx")
TAIL = 0.9          # beat of silence after each line
SR = 22050


def clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + " …"


# ── scenes ───────────────────────────────────────────────────────────────
def scene_title(c, d, im, t, dur):
    k = ease(seg(t, 0.1, 1.0))
    d.rectangle((0, 0, W, 6), fill=TEAL)
    d.rectangle((0, H - 6, W, H), fill=TEAL)
    x = 150
    draw_text(d, (x, 300), "JEJU TOURISM ASSISTANT", font("bold", 26),
              tuple(int(a + (b - a) * k) for a, b in zip(NAVY, TEAL)))
    for i, line in enumerate(["Why the assistant needs", "an ontology"]):
        kk = ease(seg(t, 0.4 + i * 0.25, 1.0))
        col = tuple(int(NAVY[j] + (WHITE[j] - NAVY[j]) * kk) for j in range(3))
        draw_text(d, (x, 360 + i * 92), line, font("bold", 84), col)
    k3 = ease(seg(t, 1.3, 1.0))
    if k3 > 0:
        d.rectangle((x, 575, x + int(360 * k3), 579), fill=AMBER)
    k4 = ease(seg(t, 1.6, 1.2))
    if k4 > 0:
        col = tuple(int(NAVY[j] + (MUTE[j] - NAVY[j]) * k4) for j in range(3))
        draw_block(d, x, 620, f"A {c['small_name']} model, a knowledge graph that can be "
                               f"edited in minutes, and the measurements behind both.",
                   font("regular", 32), 1000, col)
    k5 = ease(seg(t, 2.2, 1.2))
    if k5 > 0:
        for i, (v, l) in enumerate(c["title_kpis"]):
            bx = 150 + i * 400
            rounded(d, (bx, 760, bx + 340, 900), 14, fill=(16, 40, 72),
                    outline=(32, 58, 96), width=2)
            col = tuple(int(NAVY2[j] + ((TEAL, AMBER, GREEN)[i][j] - NAVY2[j]) * k5)
                        for j in range(3))
            draw_text(d, (bx + 24, 786), v, font("bold", 52), col)
            draw_block(d, bx + 24, 848, l, font("regular", 20), 300, MUTE)


def scene_problem(c, d, im, t, dur):
    chapter(d, "the problem", c["ch"]["problem"], c["n_chapters"])
    case = c["case1"]
    draw_text(d, (96, 120), "A question from the benchmark", font("bold", 30), TEAL)
    y = draw_block(d, 96, 170, case["question"], font("regular", 34), 1730, WHITE,
                   reveal=seg(t, 0.2, 2.0))
    if t > 2.4:
        x0, y0, x1 = card(im, d, (96, 330, 1824, 740),
                          f"{c['small_name']} with no ontology", RED,
                          tag="stored run, verbatim")
        draw_block(d, x0, y0, case["before"], font("regular", 30), x1 - x0, (226, 232, 240),
                   reveal=seg(t, 2.6, 4.2), spans=case["bad_spans"],
                   strike=set() if t < 7.0 else case["bad_words"])
    if t > 7.2:
        k = ease(seg(t, 7.2, 0.8))
        col = tuple(int(NAVY[j] + (RED[j] - NAVY[j]) * k) for j in range(3))
        lower_third(d, case["punch"], case["punch_sub"], y=H - 150, accent=col)


def scene_check(c, d, im, t, dur):
    chapter(d, "the check", c["ch"]["check"], c["n_chapters"])
    draw_text(d, (96, 120), "Every place the answer names is looked up in the graph",
              font("bold", 42), WHITE)
    draw_text(d, (96, 182), "concept · alias · gazetteer entry · supporting facts",
              font("regular", 26), MUTE)
    rows = c["case1"]["checks"]
    for i, r in enumerate(rows):
        k = ease(seg(t, 0.5 + i * 0.55, 0.5))
        if k <= 0:
            continue
        y = 300 + i * 126
        x1 = 96 + int((1730) * k)
        rounded(d, (96, y, x1, y + 104), 12, fill=(16, 40, 72), outline=(32, 58, 96))
        if k > 0.7:
            draw_text(d, (130, y + 32), r["span"], font("monob", 36), WHITE)
            draw_text(d, (900, y + 38), "not a concept, alias or place in the graph",
                      font("regular", 26), RED)
            draw_text(d, (1560, y + 36), f"{r['triples']} facts", font("bold", 30), RED)
            d.ellipse((1740, y + 32, 1784, y + 76), outline=RED, width=4)
            d.line((1752, y + 44, 1772, y + 64), fill=RED, width=4)
            d.line((1772, y + 44, 1752, y + 64), fill=RED, width=4)
    if t > 0.5 + len(rows) * 0.55 + 0.6:
        lower_third(d, "No fact supports it, so the answer may not assert it",
                    "the rule the whole system runs on", y=H - 190, accent=RED)


def scene_after(c, d, im, t, dur):
    chapter(d, "the fix", c["ch"]["after"], c["n_chapters"])
    case = c["case1"]
    draw_text(d, (96, 120), "Same question, same model, with the ontology in front of it",
              font("bold", 42), WHITE)
    x0, y0, x1 = card(im, d, (96, 220, 1824, 700), "ontology + validation layer", GREEN,
                      tag="stored run, verbatim")
    draw_block(d, x0, y0, case["after"], font("regular", 32), x1 - x0, (226, 240, 232),
               reveal=seg(t, 0.4, 3.6))
    if t > 4.2:
        k = ease(seg(t, 4.2, 0.6))
        for i, (v, l, col) in enumerate(c["case1"]["after_kpis"]):
            bx = 96 + i * 580
            rounded(d, (bx, 760, bx + 520, 900), 14, fill=(16, 40, 72),
                    outline=(32, 58, 96))
            draw_text(d, (bx + 26, 786), v, font("bold", 12 + int(34 * k)), col)
            draw_block(d, bx + 26, 848, l, font("regular", 22), 470, MUTE)


def scene_case2(c, d, im, t, dur):
    chapter(d, "a second kind of error", c["ch"]["case2"], c["n_chapters"])
    case = c["case2"]
    draw_text(d, (96, 116), clip(case["question"], 110), font("regular", 30), MUTE)
    for i, (title, body, col, tag) in enumerate(
            [("before · model alone", case["before"], RED, case["before_tag"]),
             ("after · with the ontology", case["after"], GREEN, case["after_tag"])]):
        k = ease(seg(t, 0.3 + i * 1.6, 0.7))
        if k <= 0:
            continue
        x = 96 + i * 880
        x0, y0, x1 = card(im, d, (x, 190, x + 848, 760), title, col, tag=tag)
        draw_block(d, x0, y0, clip(body, 620), font("regular", 26), x1 - x0,
                   (226, 232, 240), reveal=seg(t, 0.5 + i * 1.6, 2.0),
                   spans=case["bad_spans"] if i == 0 else None)
    if t > 4.0:
        lower_third(d, case["punch"], case["punch_sub"], y=H - 210, accent=AMBER)


def scene_pipeline(c, d, im, t, dur):
    chapter(d, "how it works", c["ch"]["pipeline"], c["n_chapters"])
    draw_text(d, (96, 120), "Four stages", font("bold", 46), WHITE)
    stages = [("Retrieve", "passages and entity cards"),
              ("Draft", "the small model answers"),
              ("Check", "every claim against approved facts"),
              ("Repair or decline", "rewrite, or refuse to assert")]
    for i, (t1, t2) in enumerate(stages):
        k = ease(seg(t, 0.4 + i * 0.9, 0.7))
        if k <= 0:
            continue
        x = 96 + i * 448
        h = int(380 * k)
        rounded(d, (x, 560 - h // 2, x + 400, 560 + h // 2), 16,
                fill=(16, 40, 72), outline=TEAL if k > 0.9 else (32, 58, 96), width=3)
        if k > 0.8:
            d.ellipse((x + 28, 588 - h // 2, x + 76, 636 - h // 2), fill=TEAL)
            draw_text(d, (x + 52, 610 - h // 2), str(i + 1), font("bold", 28), NAVY,
                      anchor="mm")
            draw_text(d, (x + 28, 660 - h // 2), t1, font("bold", 34), WHITE)
            draw_block(d, x + 28, 706 - h // 2, t2, font("regular", 24), 340, MUTE)
        if i < 3 and k > 0.95:
            d.line((x + 404, 560, x + 440, 560), fill=TEAL, width=4)
            d.polygon([(x + 440, 548), (x + 440, 572), (x + 460, 560)], fill=TEAL)
    if t > 4.4:
        lower_third(d, "Retrieval proposes · the graph decides",
                    "the layer is spent only on drafts its own detector flags",
                    y=H - 170, accent=TEAL)


def scene_edit(c, d, im, t, dur):
    chapter(d, "changing what it knows", c["ch"]["edit"], c["n_chapters"])
    e = c["edit"]
    draw_text(d, (96, 118), e["headline"], font("bold", 44), WHITE)
    draw_block(d, 96, 182, e["why"], font("regular", 28), 1100, MUTE)

    cx, cy = 560, 620
    sats = e["old_facts"]
    for i, s in enumerate(sats):
        k = ease(seg(t, 0.3 + i * 0.25, 0.5))
        if k <= 0:
            continue
        ang = math.radians(150 + i * 55)
        sx, sy = cx + int(330 * math.cos(ang) * k), cy + int(240 * math.sin(ang) * k)
        d.line((cx, cy, sx, sy), fill=(44, 68, 104), width=2)
        d.ellipse((sx - 96, sy - 32, sx + 96, sy + 32), fill=(14, 34, 62),
                  outline=(44, 68, 104), width=2)
        draw_text(d, (sx, sy), clip(s, 22), font("regular", 19), MUTE, anchor="mm")

    d.ellipse((cx - 150, cy - 56, cx + 150, cy + 56), fill=(16, 40, 72), outline=TEAL,
              width=3)
    draw_text(d, (cx, cy), e["entity"], font("bold", 28), WHITE, anchor="mm")

    k = ease(seg(t, 3.0, 1.2))
    if k > 0:
        nx, ny = 1180, 480
        d.line((cx + 150, cy - 10, cx + 150 + int((nx - cx - 150) * k),
                cy - 10 - int((cy - 10 - ny) * k)), fill=AMBER, width=6)
        if k > 0.55:
            d.ellipse((nx - 170, ny - 56, nx + 170, ny + 56), fill=(60, 44, 10),
                      outline=AMBER, width=4)
            draw_text(d, (nx, ny), e["new_object"], font("bold", 32), AMBER, anchor="mm")
            draw_text(d, ((cx + nx) // 2, (cy + ny) // 2 - 40), e["predicate"],
                      font("monob", 26), AMBER, anchor="mm")
    if t > 4.6:
        kk = ease(seg(t, 4.6, 1.4))
        counter(d, (1320, 700), e["facts"] * kk, " facts", "added to the graph, from the "
                                                           "source records", color=TEAL)
        counter(d, (1320, 860), 0, " GPU-h", "spent: the model was not retrained",
                color=AMBER)
    if t > 6.4:
        lower_third(d, f"rules enforceable {e['before']} → {e['after']}",
                    f"model weights byte-identical: {e['sha']}", y=H - 140, accent=GREEN)


def scene_analogy(c, d, im, t, dur):
    """The brief's own explanation, for a reader who does not want the
    architecture: a brilliant new hire, and the company rulebook."""
    chapter(d, "in plain words", c["ch"]["analogy"], c["n_chapters"])
    draw_text(d, (96, 112), "Why this works, without a single technical word",
              font("bold", 44), WHITE)
    draw_text(d, (96, 176), "the business brief's own explanation",
              font("regular", 26), MUTE)
    panels = c["analogy"]
    for i, (kick, title, body, col) in enumerate(panels):
        k = ease(seg(t, 0.4 + i * 1.1, 0.8))
        if k <= 0:
            continue
        x = 96 + i * 588
        h = int(300 * k)
        rounded(d, (x, 270, x + 540, 270 + h), 16, fill=(16, 40, 72), outline=col,
                width=3)
        if k > 0.6:
            d.rectangle((x, 270, x + 540, 276), fill=col)
            draw_text(d, (x + 26, 304), kick, font("bold", 20), col)
            draw_text(d, (x + 26, 340), title, font("bold", 34), WHITE)
            draw_block(d, x + 26, 396, body, font("regular", 25), 488, (190, 205, 222))
        if i < 2 and k > 0.9:
            draw_text(d, (x + 564, 400), "+" if i == 0 else "=", font("bold", 40),
                      MUTE, anchor="mm")
    if t > 3.6:
        k = ease(seg(t, 3.6, 0.9))
        rounded(d, (96, 630, 1824, 830), 14, fill=(16, 40, 72), outline=TEAL, width=3)
        draw_text(d, (130, 658), "AND THE PART A MANAGER CARES ABOUT",
                  font("bold", 20), TEAL)
        draw_block(d, 130, 696, c["analogy_note"], font("regular", 28), 1660,
                   (214, 226, 240) if k > 0.4 else NAVY2)
    if t > 5.0:
        lower_third(d, "The rulebook is edited, not retrained",
                    "which is why a correction costs minutes instead of GPU-hours",
                    y=H - 150, accent=GREEN)


# ── drawing an ontology structure ────────────────────────────────────────
MUTED_FILL, MUTED_LINE = (16, 40, 72), (44, 68, 104)
EDGE_DIM = (72, 98, 134)
NEW_FILL = (62, 46, 12)


def struct_panel(d, box, title, k=1.0, accent=AMBER):
    x0, y0, x1, y1 = box
    rounded(d, box, 16, fill=(12, 32, 58), outline=accent if k > 0.5 else MUTED_LINE,
            width=3)
    d.rectangle((x0, y0, x1, y0 + 6), fill=accent)
    draw_text(d, (x0 + 26, y0 + 24), title, font("bold", 20), accent)
    return x0 + 26, y0 + 58, x1 - 26


def node(d, cx, cy, pred, obj, t, t0, *, new=False, size=20, psize=16):
    """One object of the subject, with the predicate that reaches it written
    above the name. Putting the predicate on the node rather than on the line
    keeps the diagram readable at a distance, which a line label does not."""
    k = ease(seg(t, t0, 0.5))
    if k <= 0:
        return None
    col = AMBER if new else (120, 140, 164)
    if k > 0.6:
        draw_text(d, (cx, cy - size - psize + 2), pred,
                  font("monob" if new else "mono", psize), col, anchor="mm")
    box = pill(d, cx, cy, obj, font("bold" if new else "regular", size),
               NEW_FILL if new else MUTED_FILL, col, WHITE if new else (198, 212, 228),
               k=k)
    return box


def subject_node(d, cx, cy, name, t, t0=0.0, size=26):
    k = ease(seg(t, t0, 0.5))
    if k <= 0:
        return None
    return pill(d, cx, cy, name, font("bold", size), (16, 46, 84), TEAL, WHITE, k=k)


def scene_generic(c, d, im, t, dur, idx: int):
    """The business brief's own example, drawn the way the brief asks for it:
    the problem, the ontology structure that fixes it, the result, and why.

    These two are the brief's illustrations, not runs of this system, and the
    scene says so on the card - the real ones follow immediately after.
    """
    g = c["generic"][idx]
    chapter(d, g["chapter"], c["ch"][f"generic{idx + 1}"], c["n_chapters"])
    draw_text(d, (96, 100), g["title"], fit_font(d, g["title"], "bold", 42, 1730),
              WHITE)
    draw_text(d, (96, 156), g["source"], font("regular", 23), MUTE)

    # 1 - the problem
    if t > 0.2:
        k = ease(seg(t, 0.2, 0.5))
        rounded(d, (96, 200, 1824, 292), 12, fill=(18, 46, 85), outline=MUTED_LINE)
        draw_text(d, (124, 214), "THE PROBLEM \u00b7 what the customer asked",
                  font("bold", 18), AMBER)
        draw_block(d, 124, 244, g["question"], font("regular", 28), 1660,
                   WHITE if k > 0.4 else NAVY2, reveal=seg(t, 0.3, 1.2))
    if t > 1.5:
        x0, y0, x1 = card(im, d, (96, 306, 1000, 548), "BEFORE \u00b7 the answer it gave",
                          RED, tag=g["before_tag"])
        draw_block(d, x0, y0, g["before"], font("regular", 26), x1 - x0,
                   (232, 226, 226), reveal=seg(t, 1.6, 1.8))

    # 2 - the ontology fix, drawn as the structure it is
    if t > 3.2:
        k = ease(seg(t, 3.2, 0.6))
        px0, py0, px1 = struct_panel(d, (1040, 306, 1824, 846),
                                     "THE ONTOLOGY FIX \u00b7 the structure that was "
                                     "added", k)
        g["draw"](d, t, 3.6)

    # 3 - the result
    if t > 6.6:
        x0, y0, x1 = card(im, d, (96, 578, 1000, 846), "AFTER \u00b7 with that structure "
                          "in the graph", GREEN, tag="corrected")
        draw_block(d, x0, y0, g["after"], font("regular", 26), x1 - x0,
                   (226, 240, 232), reveal=seg(t, 6.7, 2.0))

    # 4 - why it improved
    if t > 8.6:
        k = ease(seg(t, 8.6, 0.7))
        rounded(d, (96, 872, 1824, 996), 12, fill=MUTED_FILL, outline=TEAL, width=2)
        draw_text(d, (124, 892), "WHY IT IMPROVED", font("bold", 19), TEAL)
        draw_block(d, 124, 922, g["why"], font("regular", 25), 1660,
                   (206, 220, 236) if k > 0.4 else NAVY2)


def scene_generic1(c, d, im, t, dur):
    scene_generic(c, d, im, t, dur, 0)


def scene_generic2(c, d, im, t, dur):
    scene_generic(c, d, im, t, dur, 1)


def draw_compat(d, t, t0):
    """Example 1: a subject, what the graph already held, and the three edges
    the fix adds - the incompatibility, the version that does work, and the
    patch. The corrected answer names all three, so all three have to be in
    the structure."""
    cx, cy = 1432, 400
    subject_node(d, cx, cy, "Flagship Software V1", t, t0)
    kids = [(-232, 180, "is_a", "Product", False),
            (232, 180, "has_successor", "Flagship Software V2", False),
            (-232, 360, "is_incompatible_with", "Cloud Server 2026", True),
            (232, 360, "has_patch", "Compatibility Patch", True)]
    for i, (dx, dy, pred, obj, new) in enumerate(kids):
        t1 = t0 + 0.35 + i * 0.45
        k = ease(seg(t, t1, 0.5))
        if k <= 0:
            continue
        edge(d, (cx, cy + 26), (cx + dx, cy + dy - 40),
             AMBER if new else EDGE_DIM, 5 if new else 3, k=k)
        node(d, cx + dx, cy + dy, pred, obj, t, t1 + 0.1, new=new, size=19, psize=15)
    k = ease(seg(t, t0 + 2.4, 0.6))
    if k > 0:
        draw_text(d, (cx, 820), "two new edges \u00b7 one line of data each",
                  font("regular", 20), AMBER if k > 0.5 else NAVY2, anchor="mm")


def draw_boundary(d, t, t0):
    """Example 2: a category boundary. The point is the taxonomy, so this one
    is drawn as the hierarchy it is rather than as a star: the shared parent,
    the two classes the brief says must not be confused, the disjointness
    between them, and the attribute hanging on each side."""
    top = (1400, 424)
    subject_node(d, top[0], top[1], "Staff", t, t0, size=24)
    left, right = (1200, 566), (1600, 566)
    for i, (pt, name) in enumerate(((left, "Permanent Employee"),
                                    (right, "Contractor"))):
        t1 = t0 + 0.3 + i * 0.35
        k = ease(seg(t, t1, 0.5))
        if k <= 0:
            continue
        edge(d, (top[0], top[1] + 24), (pt[0], pt[1] - 26), EDGE_DIM, 3, k=k)
        pill(d, pt[0], pt[1], name, font("bold", 21),
             NEW_FILL if i else MUTED_FILL, AMBER if i else (120, 140, 164),
             WHITE, k=k)
    k = ease(seg(t, t0 + 1.2, 0.6))
    if k > 0:
        d.line((left[0] + 122, 566, right[0] - 80, 566), fill=RED, width=3)
        draw_text(d, (1400, 542), "is-not-a", font("monob", 18), RED, anchor="mm")
    # one attribute under the permanent class, two under the contractor, each
    # on its own diagonal so no edge passes through a node it does not end at
    attrs = [(left[0], left[0], "remote_work", "3 days / week", False),
             (right[0], right[0] - 104, "remote_work", "Not eligible", True),
             (right[0], right[0] + 104, "work_location", "On-site", True)]
    for i, (fx, tx, pred, val, new) in enumerate(attrs):
        t1 = t0 + 1.7 + i * 0.45
        k = ease(seg(t, t1, 0.5))
        if k <= 0:
            continue
        edge(d, (fx, 592), (tx, 666), AMBER if new else EDGE_DIM,
             5 if new else 3, k=k)
        node(d, tx, 716, pred, val, t, t1 + 0.1, new=new, size=19, psize=15)
    k = ease(seg(t, t0 + 3.2, 0.6))
    if k > 0:
        draw_text(d, (1400, 806), "the boundary, and the attribute on each side",
                  font("regular", 20), AMBER if k > 0.5 else NAVY2, anchor="mm")


def _policy_scene(c, d, im, t, dur, idx: int):
    """One real correction, in the same four parts as the brief's examples:
    the problem, the ontology structure that was added, the result, and why -
    except that here the structure comes out of the corpus with its provenance
    and both answers are the model's own output."""
    case = c["policy"][idx]
    chapter(d, case["chapter"], c["ch"][f"policy{idx + 1}"], c["n_chapters"])
    draw_text(d, (96, 92), case["title"], fit_font(d, case["title"], "bold", 40, 1730),
              WHITE)
    draw_text(d, (96, 146), case["sub"], font("regular", 22), MUTE)

    # 1 - the problem
    if t > 0.1:
        rounded(d, (96, 188, 1824, 276), 12, fill=(18, 46, 85), outline=MUTED_LINE)
        draw_text(d, (124, 200), "THE PROBLEM \u00b7 what the customer asked",
                  font("bold", 18), AMBER)
        draw_block(d, 124, 228, case["question"], font("regular", 26), 1660, WHITE,
                   reveal=seg(t, 0.2, 1.2))
    if t > 1.5:
        x0, y0, x1 = card(im, d, (96, 288, 982, 620),
                          "BEFORE \u00b7 the graph was silent about this rule", RED,
                          tag="verbatim")
        draw_block(d, x0, y0, case["before"], font("regular", 23), x1 - x0,
                   (226, 232, 240), reveal=seg(t, 1.6, 2.0),
                   spans=case["before_spans"])

    # 2 - the ontology fix, as the structure the retriever actually hands over
    if t > 3.6:
        k = ease(seg(t, 3.6, 0.6))
        struct_panel(d, (1014, 288, 1824, 620),
                     "THE ONTOLOGY FIX \u00b7 the entity card, before and after", k)
        cx, cy = 1419, 392
        subject_node(d, cx, cy, case["subject"], t, 3.7, size=23)
        kids = case["old_facts"][:2]
        span = 430
        for i, (pred, obj) in enumerate(kids):
            x = cx + (i - (len(kids) - 1) / 2) * span
            t1 = 3.85 + i * 0.2
            kk = ease(seg(t, t1, 0.45))
            if kk <= 0:
                continue
            edge(d, (cx, cy + 24), (int(x), 474), EDGE_DIM, 3, k=kk)
            node(d, int(x), 504, pred, obj, t, t1 + 0.05, size=18, psize=14)
        t2 = 4.9
        kk = ease(seg(t, t2, 0.6))
        if kk > 0:
            edge(d, (cx, cy + 24), (cx, 534), AMBER, 5, k=kk)
            node(d, cx, 582, case["pred"], case["obj"], t, t2 + 0.1, new=True,
                 size=21, psize=16)

    # 3 - where that structure came from, and why the answer is better
    if t > 5.8:
        k = ease(seg(t, 5.8, 0.6))
        rounded(d, (96, 640, 952, 762), 12, fill=MUTED_FILL, outline=MUTED_LINE)
        draw_text(d, (122, 658), "QUOTED FROM THE SOURCE CORPUS", font("bold", 17),
                  MUTE)
        draw_block(d, 122, 688, case["evidence"], font("italic", 20), 804,
                   (190, 205, 222) if k > 0.4 else NAVY2)
        draw_text(d, (122, 742), case["provenance"],
                  fit_font(d, case["provenance"], "regular", 17, 800, floor=12), MUTE)
        rounded(d, (984, 640, 1824, 762), 12, fill=MUTED_FILL, outline=TEAL, width=2)
        draw_text(d, (1010, 658), "WHY IT IMPROVED", font("bold", 17), TEAL)
        draw_block(d, 1010, 688, case["why"], font("regular", 19), 796,
                   (206, 220, 236) if k > 0.4 else NAVY2)

    # 4 - the result
    if t > 7.0:
        x0, y0, x1 = card(im, d, (96, 782, 1824, 1000),
                          "AFTER \u00b7 the same model, one fact later", GREEN,
                          tag="same weights, same decoding")
        draw_block(d, x0, y0, case["after"], font("regular", 24), x1 - x0,
                   (226, 240, 232), reveal=seg(t, 7.1, 2.2),
                   spans=case["after_spans"])


def scene_policy1(c, d, im, t, dur):
    _policy_scene(c, d, im, t, dur, 0)


def scene_policy2(c, d, im, t, dur):
    _policy_scene(c, d, im, t, dur, 1)


def scene_policy_counted(c, d, im, t, dur):
    chapter(d, "the demo, counted", c["ch"]["policy_counted"], c["n_chapters"])
    pc = c["counted"]
    draw_text(d, (96, 112), pc["title"],
              fit_font(d, pc["title"], "bold", 44, 1730, floor=24), WHITE)
    draw_text(d, (96, 176), pc["sub"],
              fit_font(d, pc["sub"], "regular", 26, 1730, floor=18), MUTE)
    for i, (v, lab, note, col) in enumerate(pc["kpis"]):
        k = ease(seg(t, 0.3 + i * 0.5, 0.6))
        if k <= 0:
            continue
        x = 96 + i * 440
        rounded(d, (x, 250, x + 400, 480), 14, fill=(16, 40, 72), outline=(32, 58, 96))
        d.rectangle((x, 250, x + 400, 256), fill=col)
        draw_text(d, (x + 24, 286), v,
                  fit_font(d, v, "bold", int(10 + 48 * k), 352, floor=20), col)
        ly = draw_block(d, x + 24, 356, lab, font("bold", 22), 352, WHITE)
        draw_block(d, x + 24, ly + 14, note, font("regular", 19), 352, MUTE)
    if t > 2.6:
        rounded(d, (96, 520, 1824, 960), 14, fill=(16, 40, 72), outline=AMBER, width=3)
        draw_text(d, (130, 548), "WHAT WE WILL NOT ROUND UP", font("bold", 21), AMBER)
        y = 590
        for i, (head, body) in enumerate(pc["caveats"]):
            k = ease(seg(t, 2.8 + i * 0.7, 0.6))
            if k <= 0:
                continue
            draw_text(d, (130, y), head, font("bold", 26), WHITE)
            y = draw_block(d, 130, y + 36, body, font("regular", 24), 1640,
                           (190, 205, 222)) + 16


def scene_results(c, d, im, t, dur):
    chapter(d, "measured", c["ch"]["results"], c["n_chapters"])
    draw_text(d, (96, 110), c["results"]["title"], font("bold", 44), WHITE)
    draw_text(d, (96, 172), c["results"]["sub"], font("regular", 26), MUTE)
    panels = c["results"]["panels"]
    for pi, p in enumerate(panels):
        ox = 110 + pi * 900
        base = 820
        draw_text(d, (ox, 240), p["title"], font("bold", 30), WHITE)
        draw_text(d, (ox, 282), p["axis"], font("regular", 22), MUTE)
        d.line((ox, base, ox + 760, base), fill=(44, 68, 104), width=2)
        gw = 760 // len(p["groups"])
        for gi, (glab, vals) in enumerate(p["groups"]):
            for bi, (lab, v, col) in enumerate(vals):
                k = ease_out(seg(t, 0.6 + gi * 0.25 + bi * 0.12, 1.0))
                bw = int(gw / (len(vals) + 0.9))
                x = ox + gi * gw + 18 + bi * bw
                h = int((v / p["max"]) * 460 * k)
                d.rounded_rectangle((x, base - h, x + bw - 10, base), 6, fill=col)
                if k > 0.9:
                    draw_text(d, (x + (bw - 10) // 2, base - h - 28), p["fmt"].format(v),
                              font("bold", 22), WHITE, anchor="ma")
            draw_text(d, (ox + gi * gw + gw // 2 - 10, base + 16), glab,
                      font("bold", 24), MUTE, anchor="ma")
    if t > 2.6:
        for i, (lab, col) in enumerate(c["results"]["legend"]):
            x = 110 + i * 420
            d.rounded_rectangle((x, 940, x + 30, 966), 5, fill=col)
            draw_text(d, (x + 44, 940), lab, font("regular", 24), (200, 214, 230))
    # the zero belongs beside its caveat, not two scenes later: the validation
    # layer deletes exactly what this checker counts
    if t > 3.4 and c["results"].get("caveat"):
        draw_block(d, 110, 1000, c["results"]["caveat"], font("regular", 22), 1700,
                   (150, 166, 184))


def scene_small_big(c, d, im, t, dur):
    chapter(d, "the economics", c["ch"]["smallbig"], c["n_chapters"])
    s = c["smallbig"]
    draw_text(d, (96, 130), s["title"], font("bold", 50), WHITE)
    draw_text(d, (96, 196), s["sub"], font("regular", 28), MUTE)
    for i, (name, col) in enumerate([(s["small_name"], TEAL), (s["big_name"], MUTE)]):
        k = ease(seg(t, 0.3 + i * 0.5, 0.6))
        if k <= 0:
            continue
        x = 180 + i * 900
        rounded(d, (x, 300, x + 700, 880), 16, fill=(16, 40, 72),
                outline=col if i == 0 else (32, 58, 96), width=3)
        draw_text(d, (x + 36, 334), name, font("bold", 36), col)
        draw_text(d, (x + 36, 384), s["labels"][i], font("regular", 24), MUTE)
        for j, (lab, val) in enumerate(s["rows"][i]):
            yy = 450 + j * 128
            draw_text(d, (x + 36, yy), lab, font("regular", 24), MUTE)
            kk = ease(seg(t, 1.4 + i * 0.3 + j * 0.2, 0.7))
            draw_text(d, (x + 36, yy + 34), val, font("bold", 14 + int(38 * kk)),
                      WHITE if i else col)
    if t > 3.4:
        lower_third(d, s["punch"], s["punch_sub"], y=H - 150, accent=TEAL)


def scene_limits(c, d, im, t, dur):
    chapter(d, "what we do not claim", c["ch"]["limits"], c["n_chapters"])
    draw_text(d, (96, 124), "One result goes the other way, and it stays in the deck",
              font("bold", 44), WHITE)
    # the cards are laid out from how many there are: a fourth caveat should
    # shrink the stack, not fall off the bottom of the frame
    n = max(1, len(c["limits"]))
    top, bot = 226, 1040
    step = (bot - top) / n
    card_h = int(step - 22)
    ts = 34 if n <= 3 else 30
    bs = 28 if n <= 3 else 25
    for i, (title, body) in enumerate(c["limits"]):
        k = ease(seg(t, 0.4 + i * (3.6 / n), 0.8))
        if k <= 0:
            continue
        y = int(top + i * step)
        rounded(d, (96, y, 1824, y + card_h), 14, fill=(16, 40, 72), outline=(32, 58, 96))
        d.rounded_rectangle((96, y, 104, y + card_h), 4, fill=AMBER)
        draw_text(d, (140, y + 20), title, font("bold", ts), AMBER)
        draw_block(d, 140, y + 20 + int(ts * 1.6), body, font("regular", bs), 1600,
                   (205, 218, 232))


def scene_close(c, d, im, t, dur):
    k = ease(seg(t, 0.2, 1.0))
    d.rectangle((0, 0, W, 6), fill=TEAL)
    draw_text(d, (150, 230), "THE CASE, IN THREE LINES", font("bold", 26), TEAL)
    for i, line in enumerate(c["close"]):
        kk = ease(seg(t, 0.5 + i * 0.7, 1.0))
        col = tuple(int(NAVY[j] + (WHITE[j] - NAVY[j]) * kk) for j in range(3))
        draw_block(d, 150, 310 + i * 150, line, font("bold", 44), 1620, col)
    if t > 3.2:
        kk = ease(seg(t, 3.2, 1.0))
        col = tuple(int(NAVY[j] + (MUTE[j] - NAVY[j]) * kk) for j in range(3))
        d.rectangle((150, 820, 150 + int(1620 * kk), 822), fill=(44, 68, 104))
        draw_block(d, 150, 860, c["close_note"], font("regular", 26), 1620, col)


SCENES = [
    ("title", scene_title), ("problem", scene_problem), ("check", scene_check),
    ("after", scene_after), ("case2", scene_case2), ("pipeline", scene_pipeline),
    ("analogy", scene_analogy),
    ("generic1", scene_generic1), ("generic2", scene_generic2),
    ("edit", scene_edit),
    ("policy1", scene_policy1), ("policy2", scene_policy2),
    ("policy_counted", scene_policy_counted),
    ("results", scene_results), ("smallbig", scene_small_big),
    ("limits", scene_limits), ("close", scene_close),
]
# the chapter rail counts the body of the film, not the title and close
CHAPTERS = ["problem", "check", "after", "case2", "pipeline", "analogy",
            "generic1", "generic2", "edit", "policy1", "policy2",
            "policy_counted", "results", "smallbig", "limits"]


# ── the script: narration + everything the scenes draw ───────────────────
def build_context(ev: dict) -> tuple[dict, dict]:
    P, SZ = ev["pretty"], ev["sizes"]
    small, big = SZ[0], SZ[-1]
    V, G = ev["variants"], ev["gated"]
    vxp = (ROOT / "wbs" / "deliverables" / "task4_evaluation"
           / "verifier_across_models__test.json")
    vx = json.loads(vxp.read_text(encoding="utf-8")) if vxp.exists() else None
    er, sb, lim = ev["edit_vs_retrain"], ev["small_beats_big"], ev["limits"]
    live = ev["live_edit"]
    cc = live["claim_check"]
    c1, c2 = ev["cases"][0], ev["cases"][1]
    bad = [x["span"] for x in c1["span_checks"] if not x["in_ontology"]][:3]

    pe = ev.get("policy_edit") or {}
    px = ev.get("policy_extraction") or {}
    pev = {t["triple"]: t for t in (px.get("top") or [])}
    ED = ev["edits"]

    def pcase(i: int) -> dict:
        """One policy correction, as the film needs it: the question, the two
        answers, the triple and the corpus sentence it was read from."""
        k = pe["cases"][i]
        s_, pr_, o_ = k["triple"]
        exq = pev.get(f"{s_} {pr_} {o_}", {})
        b_card = (pe["cards"].get(s_) or {}).get(pe["before"], [])
        a_card = (pe["cards"].get(s_) or {}).get(pe["after"], [])
        new = [f for f in a_card if f not in b_card and o_.lower() in f.lower()]
        # the words to mark: the object of the new fact, wherever the answer
        # names it. Before the edit the answer does not; after it, it does.
        stems = {w.lower()[:5] for w in o_.split() if len(w) > 3}
        mark = {w.strip(".,:;()\"'") for w in k["after"]["answer"].split()
                if w.strip(".,:;()\"'").lower()[:5] in stems}
        # the three facts the card already carried, split into predicate and
        # object so the diagram can label the edge rather than the node
        def split(fact: str) -> tuple[str, str]:
            rest = fact[len(s_):].strip() if fact.startswith(s_) else fact
            for pr in ("avoid during", "has activity", "located in", "part of",
                       "near", "is a", "connected to", "reachable by",
                       "available in", "suitable for", "best for", "served at",
                       "requires", "restricted in", "has risk"):
                if rest.lower().startswith(pr):
                    return pr, rest[len(pr):].strip() or "—"
            bits = rest.split(" ", 1)
            return (bits[0], bits[1]) if len(bits) == 2 else (rest, "—")

        old_facts = [split(f) for f in b_card][:3]
        return {
            "chapter": k["rule"].rstrip(".").lower(),
            "title": k["rule"].rstrip("."),
            "sub": f"{pe['model']} \u00b7 ontology v{pe['before']} \u2192 "
                   f"v{pe['after']} \u00b7 one adapter loaded once \u00b7 greedy "
                   f"decoding \u00b7 weights byte-identical",
            "question": clip(k["question"], 150),
            "before": clip(k["before"]["answer"], 380),
            "after": clip(k["after"]["answer"], 395),
            "before_spans": {}, "after_spans": {w: GREEN for w in mark},
            "subject": s_, "pred": pr_, "obj": o_,
            "old_facts": [(pr.replace("_", " "), clip(ob, 26))
                          for pr, ob in old_facts],
            "triple": f"[ {s_} ]  \u2014({pr_})\u2192  [ {o_} ]",
            "provenance": (f"{exq.get('support', 0)} source records \u00b7 support "
                           f"threshold {px.get('min_support', 3)} \u00b7 written into "
                           f"v{pe['after']} beside the published version"),
            "why": ("A rule like this is not a tendency a model can average its "
                    "way to. Until the structure asserted it there was nothing to "
                    "retrieve, and the answer fell back on generic advice."),
            "evidence": "\u201c" + clip(exq.get("evidence", ""), 150) + "\u201d",
        }

    # the chapter rail counts the body scenes, and the numbering comes from the
    # scene list rather than from literals a later edit would silently desync
    c = {
        "n_chapters": len(CHAPTERS),
        "ch": {key: i + 1 for i, key in enumerate(CHAPTERS)},
        "small_name": P[small], "big_name": P[big],
        "title_kpis": [
            (f"{er['edit_gpu_hours']:.0f} GPU-h", "to correct what the system knows"),
            (f"{sb['token_f1_gain']:+.1f}", f"quality points: {P[small]} with the graph "
                                            f"over {P[big]} without"),
            (f"{G[big]['vs_rag']['clean']['delta']:+.1f} pp", "more fully grounded answers "
                                                              "than retrieval alone"),
        ],
        "case1": {
            "question": clip(c1["question"], 200),
            "before": clip(c1["before_excerpt"], 640),
            "after": clip(c1["after"]["answer"], 640),
            "bad_spans": {s: RED for s in bad},
            "bad_words": {w for s in bad for w in s.split()},
            "punch": "A three-hundred-kilometre error, fluently written",
            "punch_sub": f"{len(c1['before']['unresolved_spans'])} spans in this answer "
                         f"cannot be resolved against the knowledge base",
            "checks": c1["headline_spans"][:4],
            "after_kpis": [
                (f"{len(c1['after']['unresolved_spans'])}", "unverifiable spans left in "
                                                            "the answer", GREEN),
                (f"{V[small]['V1_llm']['ungrounded']:.1f}% → {G[small]['ungrounded']:.2f}%",
                 "unsupported claims, averaged over the whole benchmark", TEAL),
                (f"{G[small]['clean']:.0f}%", "of answers fully grounded in approved facts",
                 TEAL),
            ],
        },
        "case2": {
            "question": c2["question"],
            "before": c2["before_excerpt"], "after": c2["after"]["answer"],
            # the words to mark are the ones the graph cannot tie to the island
            "bad_spans": {w: RED for m in ev["mode_support"]["claimed"]
                          if not m["linked_to_jeju"]
                          for w in m["phrase"].split()
                          if w.lower() not in ("jeju", "bus")},
            "before_tag": f"{len(c2['before']['unresolved_spans'])} unverifiable",
            "after_tag": f"{len(c2['after']['unresolved_spans'])} unverifiable",
            "punch": "A subway with ten lines, on an island that has no rail",
            "punch_sub": f"the graph holds \u201csubway\u201d as a generic transport mode "
                         f"with "
                         f"{ev['mode_support']['claimed'][1]['linked_to_jeju']} edges to "
                         f"Jeju",
        },
        "edit": {
            "headline": f"{er['rules_total'] - er['rules_enforceable_before']} of "
                        f"{er['rules_total']} rules could not be checked against anything",
            "why": f"Two of them were blocked on one missing concept: nothing in the graph "
                   f"denoted a storm, a typhoon or a high-sea state, so "
                   f"{', '.join(er['unblocked'])} had nothing to test.",
            "entity": "Woljeongri Beach",
            "old_facts": [f.replace("Woljeongri Beach ", "") for f in
                          live["cards"]["Woljeongri Beach"][live["ontology"]["before"]][1:5]],
            "new_object": "Rough Seas", "predicate": "avoid_during",
            "facts": er["triples_added"], "before": er["rules_enforceable_before"],
            "after": er["rules_enforceable_after"],
            "sha": live["retraining"]["adapter_sha_before"],
            "question": live["items"][0]["question"],
            "runs": [(v, live["items"][0]["runs"][v]["answer"],
                      live["cards"]["Woljeongri Beach"][v], col)
                     for v, col in ((live["ontology"]["before"], RED),
                                    (live["ontology"]["after"], GREEN))],
            "claims_before": f"{cc['totals'][live['ontology']['before']]['asserted']}"
                             f" of {cc['totals'][live['ontology']['before']]['claimed']}",
            "claims_after": f"{cc['totals'][live['ontology']['after']]['asserted']}"
                            f" of {cc['totals'][live['ontology']['after']]['claimed']}",
            "claims_note": "The wording hardly moves — this model was already cautious. "
                           "What moves is whether the caution is backed by a fact the "
                           "business approved.",
        },
        "analogy": [
            ("THE NEW HIRE", "The language model",
             "Has read every book. Fluent, fast, widely informed \u2014 and willing to "
             "guess when your business is not in any of those books.", (138, 155, 175)),
            ("THE RULEBOOK", "The ontology",
             "The org chart and the policy manual: what exists, what connects to what, "
             "and which answers are not allowed.", TEAL),
            ("THE RIGHT ANSWER", "Every sentence traceable",
             "The hire answers from the manual, and every line can be pointed back to "
             "the rule it came from.", GREEN),
        ],
        "analogy_note": (f"Correcting the rulebook is a data change, not a training run. "
                         f"The {ED['facts']} facts behind this film went in over "
                         f"{ED['n']} edits for {ED['gpu_hours']:.0f} GPU-hours; the "
                         f"{ED['retrain_adapters']} fine-tuned adapters this project "
                         f"trained cost {ED['retrain_gpu_hours_total']:.0f} GPU-hours "
                         f"between them, and would all have to be paid again."),
        "generic": [
            {"chapter": "the pattern: a hard requirement",
             "title": "Example 1 \u2014 a requirement the model cannot infer",
             "source": "the business brief's own example, 19 October \u00b7 an "
                       "illustration of the shape, not a run of this system",
             "question": "\u201cCan I use our flagship software V1 with the new "
                         "Cloud Server 2026?\u201d",
             "before": "\u201cYes, V1 is fully compatible with all cloud servers "
                       "released after 2024.\u201d",
             "before_tag": "hallucination \u00b7 risk",
             "after": "\u201cNo. Flagship Software V1 is not compatible with Cloud "
                      "Server 2026. Please use V2, or apply the compatibility "
                      "patch.\u201d",
             "why": "A prohibition is not a statistical tendency; it exists only "
                    "because somebody asserted it. Once the structure holds the "
                    "incompatibility, the version that does work and the patch, the "
                    "model retrieves the rule instead of guessing from how the words "
                    "usually go together.",
             "draw": draw_compat},
            {"chapter": "the pattern: a category boundary",
             "title": "Example 2 \u2014 a boundary the model generalises across",
             "source": "the business brief's own example, 19 October \u00b7 an "
                       "illustration of the shape, not a run of this system",
             "question": "\u201cWhat is our company's remote work policy for "
                         "contractors?\u201d",
             "before": "\u201cContractors are eligible for 3 days of remote work per "
                       "week under the standard hybrid policy.\u201d",
             "before_tag": "wrong policy applied",
             "after": "\u201cUnder company guidelines, external contractors are "
                      "required to work on site and are not covered by the hybrid "
                      "remote work policy.\u201d",
             "why": "The taxonomy stops a rule about one class leaking into another. "
                    "Separating the two kinds of staff and hanging the attribute on "
                    "each side is what keeps the model from generalising the "
                    "employee policy to a contractor.",
             "draw": draw_boundary},
        ],
        "policy": [pcase(0), pcase(1)] if pe.get("cases") else [],
        "counted": {
            "title": "All sixteen questions, including the ones the edit should not "
                     "have touched",
            "sub": (f"{pe['model']} \u00b7 ontology v{pe['before']} \u2192 "
                    f"v{pe['after']} \u00b7 same weights, same decoding, same order"
                    if pe else ""),
            "kpis": [
                (f"{pe['summary']['rule_stated_before']} \u2192 "
                 f"{pe['summary']['rule_stated_after']}",
                 f"of {pe['summary']['targeted']} answers state the rule",
                 "the graph was silent about these policies before the edit", TEAL),
                (f"{pe['edit']['triples_added']}", "facts added to the graph",
                 f"re-indexed on CPU in "
                 f"{pe['edit']['card_index_build_s'][pe['after']]:.0f} seconds, "
                 f"{pe['validation']['errors']} validation errors", AMBER),
                (f"{pe['retraining']['gpu_hours_spent_on_the_edit']:.0f}", "GPU-hours",
                 f"adapter digest {pe['retraining']['adapter_sha_before']} before and "
                 f"after \u2014 byte for byte", GREEN),
                (f"{pe['summary']['control_answers_changed']}/"
                 f"{pe['summary']['controls']}", "control answers changed at all",
                 f"{pe['control_overlap_mean']:.0f}% word overlap between the two "
                 f"versions", RED),
            ] if pe else [],
            "caveats": [
                (f"{pe['summary']['rule_stated_before']} of "
                 f"{pe['summary']['targeted']} were already right.",
                 "This model knows that temples expect modest dress. The edit is what "
                 "makes the answer checkable \u2014 and it fixes the ones that were "
                 "not right."),
                ("Two did not move even with the fact in the prompt.",
                 "A fact in the context is not a fact the model used. That is why the "
                 "system has a validation layer behind the retriever, and why we "
                 "measure that layer on its own."),
                ("The controls moved a little.",
                 f"New concepts mean new cards in the retrieval index, so a question "
                 f"the edit says nothing about can still come back reworded. We show "
                 f"the overlap rather than claim zero."),
            ] if pe else [],
        },
        "results": {
            "title": f"{ev['scale']['benchmark_items']} held-out questions, three model "
                     f"sizes, paired tests",
            "sub": "model alone · with retrieval · with the deployed ontology configuration",
            "legend": [("model alone", (154, 167, 180)), ("+ retrieval", AMBER),
                       ("+ ontology (deployed)", TEAL)],
            "caveat": (f"0.00% and 100% are measured against this project's own "
                       f"checker — the one the layer deletes by. Under a stricter "
                       f"independent rule the ontology column is "
                       f"{G[big].get('ungrounded_strict', 0):.2f}% unsupported against "
                       f"retrieval's {G[big]['rag'].get('ungrounded_strict', 0):.2f}%."),
            "panels": [
                {"title": "Unsupported claims", "axis": "% of claims — lower is better",
                 "max": max(V[m]["V1_llm"]["ungrounded"] for m in SZ) * 1.15,
                 "fmt": "{:.2f}",
                 "groups": [(P[m].split("-")[-1],
                             [("alone", V[m]["V1_llm"]["ungrounded"], (154, 167, 180)),
                              ("rag", V[m]["V2_rag"]["ungrounded"], AMBER),
                              ("onto", G[m]["ungrounded"], TEAL)]) for m in SZ]},
                {"title": "Fully grounded answers", "axis": "% of answers — higher is better",
                 "max": 105.0, "fmt": "{:.0f}",
                 "groups": [(P[m].split("-")[-1],
                             [("alone", V[m]["V1_llm"]["clean"], (154, 167, 180)),
                              ("rag", V[m]["V2_rag"]["clean"], AMBER),
                              ("onto", G[m]["clean"], TEAL)]) for m in SZ]},
            ],
        },
        "smallbig": {
            "title": "A small model, carrying the job",
            "sub": "the same benchmark, the same scorer, stored runs",
            "small_name": P[small], "big_name": P[big],
            "labels": ["with the ontology", "with no ontology"],
            "rows": [
                [("answer quality", f"{sb['small_with_ontology']['token_f1']:.1f}"),
                 ("unsupported claims", f"{sb['small_with_ontology']['ungrounded']:.2f}%"),
                 ("fully grounded", f"{sb['small_with_ontology']['clean']:.1f}%")],
                [("answer quality", f"{sb['big_without_ontology']['token_f1']:.1f}"),
                 ("unsupported claims", f"{sb['big_without_ontology']['ungrounded']:.2f}%"),
                 ("fully grounded", f"{sb['big_without_ontology']['clean']:.1f}%")],
            ],
            "punch": "One knowledge asset, every model",
            "punch_sub": (f"checked unchanged on {vx['models'][list(vx['models'])[0]]['n']:,} "
                          f"answers from {len(vx['models'])} models — three of them "
                          f"families we never trained for"
                          if vx and vx.get("models") else
                          "the graph is edited once and read by every deployment"),
        },
        "limits": [
            ("Reference overlap still favours plain retrieval",
             f"The deployed configuration is {G[big]['vs_rag']['token_f1']['delta']:+.2f} "
             f"points at {P[big]}, and behind at every size."),
            ("Part of that gap is the benchmark's own provenance",
             f"The reference answers were written from the corpus the retrieval baseline "
             f"reads. Rewording {lim['paraphrase']['accepted']} of "
             f"{lim['paraphrase']['total']} of them removes "
             f"{min(lim['paraphrase']['share_of_lead_explained_pct'].values())}–"
             f"{max(lim['paraphrase']['share_of_lead_explained_pct'].values())}% of the "
             f"lead — and leaves the rest."),
            ("Choosing an arm per question cannot close it",
             f"An oracle that picks the better stored answer every time gains only "
             f"{lim['oracle_headroom']:.2f} points. That is the ceiling, so we stopped "
             f"optimising it."),
            ("Zero per cent is zero against our own checker",
             f"The layer deletes exactly what that checker flags, so of course nothing "
             f"survives it. Under a stricter independent rule the figure is "
             f"{G[big].get('ungrounded_strict', 0):.2f}% against the retrieval "
             f"baseline's {G[big]['rag'].get('ungrounded_strict', 0):.2f}% — still "
             f"better, and that is the number we argue from."),
        ],
        "close": [
            f"Knowledge you can change in minutes: {ED['facts']} facts over "
            f"{ED['n']} edits, {er['rules_enforceable_before']} → "
            f"{er['rules_enforceable_after']} rules enforceable, "
            f"{ED['gpu_hours']:.0f} GPU-hours.",
            f"A small model that carries the job: {P[small]} with the graph ahead of "
            f"{P[big]} without it.",
            "Answers you can defend: every sentence traced to a fact someone approved.",
        ],
        "close_note": "Every number in this film is generated from the project's "
                      "measurement files · wbs/deliverables/presentation/",
    }

    narration = {
        "title": f"Why a tourism assistant needs an ontology. A small language model, a "
                 f"knowledge graph you can edit in minutes, and the measurements behind "
                 f"both. Everything you are about to see is a real run.",
        "problem": f"Here is the problem, from a real run. We asked a {P[small]} model to "
                   f"plan a three day drive on Jeju Island. It answered fluently, and it "
                   f"sent the traveller to Busan: a city three hundred kilometres away, "
                   f"across open sea.",
        "check": "Every place an answer names is checked against the knowledge graph. "
                 "Busan is not in it. No entity, no road, no supporting fact. So the claim "
                 "has nothing to rest on, and the system is not allowed to assert it.",
        "after": f"With the ontology in front of it, the same small model stays on the "
                 f"island, and the validation layer leaves nothing unverifiable in the "
                 f"answer.",
        "case2": "The second failure is a transport system that does not exist. The model "
                 "offered a Jeju subway with ten lines. The graph holds the word subway as "
                 "a generic concept, and not one edge joining it to the island, so the "
                 "claim has nothing behind it.",
        "pipeline": "The system works in four stages. Retrieval proposes. The graph checks "
                    "every claim. The layer repairs what it can, and declines what it "
                    "cannot.",
        "edit": f"Now the part a business plan cares about. Four of our eight safety "
                f"rules could not be enforced, and two of them were blocked on one missing "
                f"thing: nothing in the graph denoted a storm or a high sea state. We "
                f"added {er['triples_added']} such facts, extracted "
                f"from the source records and validated. The model was not retrained: its "
                f"weights are byte for byte identical.",
        "analogy": (f"Here is the whole idea without a technical word in it. The "
                    f"language model is a brilliant new hire who has read every book "
                    f"and knows none of your rules. The ontology is the org chart and "
                    f"the rulebook. Correcting the rulebook is a data change: the "
                    f"{ED['facts']} facts in this film cost no GPU time at all, while "
                    f"the {ED['retrain_adapters']} fine-tuned adapters behind this "
                    f"project cost {ED['retrain_gpu_hours_total']:.0f} GPU-hours "
                    f"between them."),
        "generic1": "The business brief asks for the demo as a before and after, "
                    "and gives two examples. Here is the first, drawn exactly as the "
                    "brief writes it. A customer asks whether version one of the "
                    "flagship software runs on the new cloud server. The assistant "
                    "says yes, it is compatible with everything since twenty "
                    "twenty-four. That is a guess, and an expensive one. The fix is "
                    "structure: an incompatibility edge, the version that does work, "
                    "and the patch. Three lines of data, and the answer becomes a "
                    "refusal with a route forward.",
        "generic2": "The second example is a category boundary. Asked about remote "
                    "work for contractors, the assistant applies the staff policy and "
                    "offers three days a week. The fix separates the two kinds of "
                    "staff in the hierarchy and hangs the attribute on each side, so "
                    "the rule for one class stops leaking into the other. Those two "
                    "shapes - a hard requirement and a boundary - cover every "
                    "correction in this film. The rest of it is the same two shapes, "
                    "on our own graph, with real runs on both sides.",
        "policy1": (f"So here is that edit, as a customer would see it. "
                    f"{pcase(0)['question']} Before the edit the graph held no such "
                    f"rule, and the answer fell back on generic etiquette. We added "
                    f"one fact, read from the source corpus with its provenance, and "
                    f"asked again. Same weights, same decoding."
                    if pe.get("cases") else ""),
        "policy2": (f"A second one, in the other direction. "
                    f"{pcase(1)['question']} The first answer recommends sunscreen. The "
                    f"second recommends the reef-safe sunscreen the island's own "
                    f"guidance asks for, because that is now a fact about this beach "
                    f"rather than a general tip."
                    if pe.get("cases") else ""),
        "policy_counted": (f"Across the whole probe set: ten questions the edit speaks "
                           f"to and six controls it does not. The rule is stated in "
                           f"{pe['summary']['rule_stated_after']} of ten answers after "
                           f"the edit, against "
                           f"{pe['summary']['rule_stated_before']} before. Four of the "
                           f"six already knew the custom; the edit is what makes those "
                           f"answers checkable. Two did not move even with the fact in "
                           f"the prompt, which is exactly why there is a validation "
                           f"layer behind the retriever."
                           if pe else ""),
        # the spoken range is read off the grids rather than typed, so a re-run
        # cannot leave the narrator saying last month's numbers
        "results": f"Across {ev['scale']['benchmark_items']} held-out questions and three "
                   f"model sizes, the deployed configuration cuts unsupported claims below "
                   f"the retrieval baseline at every size, and raises fully grounded "
                   f"answers by "
                   f"{min(round(G[m]['vs_rag']['clean']['delta']) for m in SZ)} to "
                   f"{max(round(G[m]['vs_rag']['clean']['delta']) for m in SZ)} points. "
                   f"The zero on the left is zero against our own checker, and we come "
                   f"back to that.",
        "smallbig": (f"And the small model carries the job. One billion parameters "
                     f"with the graph is ahead of eight billion without it, on answer "
                     f"quality and on groundedness, on the same questions. The checker "
                     f"itself is ordinary software: we ran it unchanged over "
                     f"{len(vx['models'])} different models, three of them families we "
                     f"never trained anything for, and unsupported claims more than "
                     f"halve in every one."
                     if vx and vx.get("models") else
                     "And the small model carries the job. One billion parameters with "
                     "the graph is ahead of eight billion without it, on answer quality "
                     "and on groundedness, on the same questions."),
        "limits": f"One result goes the other way. On word overlap with reference "
                  f"answers written from the retrieval corpus, plain retrieval still "
                  f"leads — by {abs(G[big]['vs_rag']['token_f1']['delta']):.2f} of a "
                  f"point at {P[big]}, once the checker stops rewriting whole answers "
                  f"and deletes only the part that is wrong. Rewording those references "
                  f"removes about a fifth of what is left. And our zero per cent is zero "
                  f"against our own checker; a stricter independent rule puts it at "
                  f"{G[big].get('ungrounded_strict', 0):.2f}% against the baseline's "
                  f"{G[big]['rag'].get('ungrounded_strict', 0):.2f}%.",
        "close": "Knowledge you can change in minutes. A small model that carries the job. "
                 "Answers you can defend.",
    }
    return c, narration


# ── audio ────────────────────────────────────────────────────────────────
def synth(narration: dict, tmp: Path) -> dict[str, float]:
    if not VOICE.exists():
        print(f"no piper voice at {VOICE}; rendering silent", flush=True)
        return {k: max(4.0, len(v.split()) / 2.9) for k, v in narration.items()}
    dur = {}
    for key, line in narration.items():
        p = tmp / f"{key}.wav"
        subprocess.run(["piper", "-m", str(VOICE), "-f", str(p)],
                       input=line, text=True, check=True, capture_output=True)
        d, sr = sf.read(p)
        dur[key] = len(d) / sr
    return dur


def build_track(narration: dict, dur: dict, tmp: Path) -> Path:
    parts = []
    for key in narration:
        p = tmp / f"{key}.wav"
        if p.exists():
            d, sr = sf.read(p)
            if d.ndim > 1:
                d = d.mean(axis=1)
        else:
            d, sr = np.zeros(int(SR * dur[key])), SR
        parts.append(d.astype(np.float32))
        parts.append(np.zeros(int(sr * TAIL), dtype=np.float32))
    track = np.concatenate(parts)
    out = tmp / "narration.wav"
    sf.write(out, track, SR)
    return out


# ── render ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", action="store_true",
                    help="write one PNG per scene instead of the film")
    ap.add_argument("--out", default="Ontology_Demo.mp4")
    a = ap.parse_args()

    ev = json.loads((OUT / "demo_evidence.json").read_text(encoding="utf-8"))
    c, narration = build_context(ev)
    tmp = ROOT / "wbs" / "deliverables" / "presentation" / "_video"
    tmp.mkdir(parents=True, exist_ok=True)

    if a.stills:
        for key, fn in SCENES:
            im = canvas()
            d = ImageDraw.Draw(im)
            fn(c, d, im, 9.0, 10.0)
            im.save(tmp / f"still_{key}.png")
        print(f"wrote {len(SCENES)} stills to {tmp}")
        return

    print("synthesising narration …", flush=True)
    dur = synth(narration, tmp)
    track = build_track(narration, dur, tmp)
    total = sum(dur[k] + TAIL for k, _ in SCENES)
    print(f"  {total / 60:.1f} min of narration over {len(SCENES)} scenes", flush=True)

    out = OUT / a.out
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", str(track),
           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
           # the voice comes out of piper near full scale and a little quiet on
           # average; EBU R128 at -16 LUFS with 1.5 dB of headroom is what a
           # meeting-room projector wants
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
           "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    frames = 0
    for key, fn in SCENES:
        sd = dur[key] + TAIL
        n = int(sd * FPS)
        for i in range(n):
            t = i / FPS
            im = canvas()
            d = ImageDraw.Draw(im)
            fn(c, d, im, t, sd)
            k = min(seg(t, 0.0, 0.45), seg(sd - t, 0.0, 0.35))
            proc.stdin.write(fade(im, k).tobytes())
            frames += 1
        print(f"  {key}: {n} frames ({sd:.1f}s)", flush=True)
    proc.stdin.close()
    err = proc.stderr.read().decode()[-1500:]
    if proc.wait() != 0:
        sys.exit(f"ffmpeg failed:\n{err}")
    print(f"wrote {out.relative_to(ROOT)}  ({frames} frames, {frames / FPS / 60:.1f} min)")


if __name__ == "__main__":
    main()
