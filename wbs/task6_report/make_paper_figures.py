"""WBS Task 6 - the two figures the paper needs: ontology structure, and a
qualitative RAG-vs-ontology comparison.

Figure 19 is the whole ontology as a *structure*: the class taxonomy on the
left, the schema-level relation graph on the right, both read from the Neo4j
snapshot so the figure and the database cannot disagree. The instance-level
pictures (every concept, every relation) are figures 17 and 18; this one is the
schema, which is what a paper can print at column width and a reader can follow.

Figure 20 is the qualitative comparison: for four real benchmark questions, the
prompt, the reference answer, what the Simple-RAG arm generated and what the
ontology arm generated, with the entities that decide correctness marked.

A word about how the cases were chosen, because it is the part that is easy to
get wrong. They are drawn from the scored generations, filtered to cases where
the ontology arm's judge score is >= 0.9 and the RAG arm's is <= 0.2. That is a
deliberately favourable slice and the figure says so on its face: across the
1,200 scored (model, question) pairs the reverse case - RAG right, ontology
wrong - happens 42 times against the ontology's 58. Four hand-picked rows are an
illustration of *how* the ontology arm differs, not evidence of how often it
wins; the per-metric tables carry that.

  python -m wbs.task6_report.make_paper_figures
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "wbs" / "deliverables" / "figures"
STORE = ROOT / "wbs" / "deliverables" / "task3_storage"
EXP = ROOT / "experiments" / "exp024_rag_vs_ontology" / "outputs"

plt.rcParams.update({
    "font.size": 9, "savefig.bbox": "tight", "figure.dpi": 150,
    "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

INK = "#15181d"
MUTED = "#5b6471"
LINE = "#c9d0da"
NAVY = "#1f4e79"
GREEN = "#1f7a4d"
RED = "#b3261e"
PALETTE = ["#1f4e79", "#2e7d5b", "#c98a2b", "#8e44ad", "#b03a4a", "#2b7a8c",
           "#6b7b2a", "#a0522d", "#4a69bd", "#7f8c8d"]

MODELS = ["deepseek-r1-8b", "llama3.1-8b", "llama3.2-1b", "llama3.2-3b",
          "mistral-7b", "qwen3-8b"]


# --------------------------------------------------------------- loading


def jsonl(path: Path) -> dict:
    return {json.loads(line)["record_id"]: json.loads(line)
            for line in path.open(encoding="utf-8") if line.strip()}


def ontology_structure() -> dict:
    """Classes, counts, taxonomy and schema-level edges, from the Neo4j read."""
    schema = json.loads((STORE / "ontology_schema_neo4j.json")
                        .read_text(encoding="utf-8"))
    snap = json.loads((STORE / "ontology_from_neo4j.json")
                      .read_text(encoding="utf-8"))
    counts = Counter(c.get("concept_class") or "Concept"
                     for c in snap["concepts"])

    # Class-to-class edges, aggregated over the endpoint table the schema read
    # produced. `is_a` and `subclass_of` are the taxonomy, drawn separately.
    taxonomy_preds = {"IS_A", "SUBCLASS_OF"}
    pair = Counter()
    pred_total = Counter()
    for e in schema["endpoint_classes"]:
        if e["type"] in taxonomy_preds:
            continue
        s, o = e["subject_class"], e["object_class"]
        if not s or not o:
            continue
        pair[(s, o, e["type"])] += e["edges"]
        pred_total[e["type"]] += e["edges"]

    return {
        "version": snap["version"],
        "engine": snap["meta"]["engine"],
        "concepts": snap["meta"]["nodes"],
        "relations": snap["meta"]["relationships"],
        "counts": counts,
        "hierarchy": schema["hierarchy"],
        "hierarchy_depth": schema["hierarchy_depth"],
        "pair": pair,
        "pred_total": pred_total,
        "relationship_types": schema["relationship_types"],
    }


# ------------------------------------------------- figure 19: the structure


TOP_LINKS = 70          # class-pair links drawn in panel (b)


def _taxonomy_groups(hier: list[dict], counts: Counter) -> list[dict]:
    """Classes grouped under their taxonomy root, for the ring ordering.

    Siblings end up adjacent, so the dense traffic inside a branch becomes a
    short arc near the rim instead of another chord through the middle.
    """
    children = {h["parent"]: list(h["children"]) for h in hier}
    seen_as_child = {c for h in hier for c in h["children"]}
    roots = [h["parent"] for h in hier if h["parent"] not in seen_as_child]

    groups, placed = [], set()
    for root in sorted(roots, key=lambda r: -counts.get(r, 0)):
        members = []

        def walk(node):
            if node in placed:
                return
            placed.add(node)
            if counts.get(node, 0):
                members.append(node)
            for ch in sorted(children.get(node, []),
                             key=lambda c: -counts.get(c, 0)):
                walk(ch)

        walk(root)
        if members:
            groups.append({"name": root, "members": members})

    rest = [c for c, n in counts.most_common()
            if c not in placed and n and c != "OntologyClass"]
    if rest:
        groups.append({"name": "other top-level", "members": rest})
    return groups


def _taxonomy_rows(hier: list[dict], counts: Counter) -> list[tuple]:
    """Flatten the class tree into drawable (depth, label, n, parent) rows."""
    children = {h["parent"]: list(h["children"]) for h in hier}
    seen_as_child = {c for h in hier for c in h["children"]}
    roots = [h["parent"] for h in hier if h["parent"] not in seen_as_child]
    rows: list[tuple] = []

    def walk(node: str, depth: int) -> None:
        rows.append((depth, node, counts.get(node, 0)))
        for ch in sorted(children.get(node, []),
                         key=lambda c: -counts.get(c, 0)):
            walk(ch, depth + 1)

    for r in sorted(roots, key=lambda r: -counts.get(r, 0)):
        walk(r, 0)
    # Classes with instances that the subclass_of spine never mentions.
    placed = {label for _, label, _ in rows}
    for cls, n in counts.most_common():
        if cls not in placed:
            rows.append((0, cls, n))
    return rows


def draw_structure(d: dict, out_stem: Path, formats=("png", "pdf")) -> list[Path]:
    fig = plt.figure(figsize=(13.4, 7.4), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.5], wspace=0.04,
                          left=0.012, right=0.988, top=0.885, bottom=0.035)

    # ---------------------------------------------------- (a) the taxonomy
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    rows = _taxonomy_rows(d["hierarchy"], d["counts"])
    n = len(rows)
    y_of = {}
    colour_of: dict[str, str] = {}
    root_i = -1
    for i, (depth, label, cnt) in enumerate(rows):
        if depth == 0:
            root_i += 1
        colour_of[label] = PALETTE[root_i % len(PALETTE)]

    step = 1.0 / (n + 1)
    for i, (depth, label, cnt) in enumerate(rows):
        y = 1 - (i + 1) * step
        y_of[label] = y
        x = 0.045 + depth * 0.115
        col = colour_of[label]
        weight = "bold" if depth == 0 else "normal"
        size = 8.6 if depth == 0 else 8.0
        ax.plot([x - 0.022], [y], marker="o", ms=5.0 if depth == 0 else 3.6,
                color=col, zorder=3, clip_on=False)
        ax.text(x, y, label, fontsize=size, color=INK, weight=weight,
                va="center", ha="left")
        ax.text(0.995, y, f"{cnt:,}" if cnt else "—", fontsize=7.6,
                color=MUTED if cnt else LINE, va="center", ha="right")

    # elbows from parent to child, drawn after positions are known
    for h in d["hierarchy"]:
        p = h["parent"]
        if p not in y_of:
            continue
        for ch in h["children"]:
            if ch not in y_of:
                continue
            depth_c = next(dep for dep, lab, _ in rows if lab == ch)
            xp = 0.045 + (depth_c - 1) * 0.115 - 0.022
            xc = 0.045 + depth_c * 0.115 - 0.022
            ax.plot([xp, xp, xc], [y_of[p], y_of[ch], y_of[ch]],
                    color=colour_of[p], lw=0.8, alpha=0.5, zorder=1,
                    solid_capstyle="round", clip_on=False)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"(a) Class taxonomy — {len(d['counts'])} classes, "
                 f"depth {d['hierarchy_depth']['max_depth']}",
                 fontsize=9.6, color=INK, loc="left", pad=8)

    # ------------------------------------------- (b) schema relation graph
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_axis_off()

    # A ring ordered by size scatters related classes and every chord crosses
    # the middle, which is unreadable at print size. Group the ring by taxonomy
    # root instead, so siblings are adjacent and most traffic becomes a short
    # arc, and draw only the class-pair links that carry real weight.
    import math

    groups = _taxonomy_groups(d["hierarchy"], d["counts"])
    classes = [c for g in groups for c in g["members"]]
    pos, ang = {}, {}
    R = 1.0
    gap = 0.085                     # radians of blank ring between groups
    span = 2 * math.pi - gap * len(groups)
    step = span / max(len(classes), 1)
    a = math.pi / 2 - 0.40      # keep the first group clear of the panel title
    for g in groups:
        g["a0"] = a
        for c in g["members"]:
            pos[c] = (R * math.cos(a), R * math.sin(a))
            ang[c] = a
            a -= step
        g["a1"] = a + step
        a -= gap

    top_preds = [p for p, _ in d["pred_total"].most_common(10)]
    pred_colour = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(top_preds)}

    agg = defaultdict(int)
    for (s, o, t), w in d["pair"].items():
        if s in pos and o in pos and t in top_preds:
            agg[(s, o, t)] += w
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])
    shown = ranked[:TOP_LINKS]
    drawn_weight = sum(w for _, w in shown)
    total_weight = sum(agg.values()) or 1
    biggest = max((w for _, w in shown), default=1)

    for (s, o, t), w in sorted(shown, key=lambda kv: kv[1]):
        x1, y1 = pos[s]
        x2, y2 = pos[o]
        # Bow each chord toward the centre in proportion to how far apart its
        # endpoints are, which is what stops long links from piling up on the
        # same straight lines through the middle.
        sep = abs(ang[s] - ang[o])
        sep = min(sep, 2 * math.pi - sep)
        rad = 0.10 + 0.26 * (sep / math.pi)
        lw = 0.4 + 2.9 * (w / biggest) ** 0.55
        ax2.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad:.3f}",
            arrowstyle="-|>", mutation_scale=7.5, lw=lw,
            color=pred_colour[t], alpha=0.52, zorder=1))

    # Group arcs just outside the nodes, so the taxonomy grouping is visible
    # rather than merely used.
    for gi, g in enumerate(groups):
        col = PALETTE[gi % len(PALETTE)]
        aa = [g["a0"] + 0.012, g["a1"] - 0.012]
        ts = [aa[0] + (aa[1] - aa[0]) * k / 40 for k in range(41)]
        ax2.plot([1.085 * math.cos(x) for x in ts],
                 [1.085 * math.sin(x) for x in ts],
                 color=col, lw=2.4, alpha=0.75, solid_capstyle="round",
                 zorder=2)
        g["colour"] = col

    biggest_n = max(d["counts"].values())
    for c in classes:
        x, y = pos[c]
        size = 60 + 560 * (d["counts"][c] / biggest_n) ** 0.5
        ax2.scatter([x], [y], s=size, color="white", edgecolors=NAVY,
                    linewidths=1.1, zorder=3)
        ax2.text(x, y, str(d["counts"][c]), fontsize=6.4, ha="center",
                 va="center", color=INK, zorder=4)
        a = math.atan2(y, x)
        lx, ly = (R + 0.135) * math.cos(a), (R + 0.135) * math.sin(a)
        rot = math.degrees(a)
        ha = "left"
        if 90 < rot <= 270 or -270 <= rot < -90:
            rot += 180
            ha = "right"
        ax2.text(lx, ly, c, fontsize=7.4, ha=ha, va="center",
                 rotation=rot, rotation_mode="anchor", color=INK)

    ax2.set_xlim(-1.58, 1.58)
    ax2.set_ylim(-1.50, 1.50)
    ax2.set_aspect("equal")
    ax2.set_title(f"(b) Schema-level relations — strongest {len(shown)} of "
                  f"{len(ranked)} class-pair links, "
                  f"{drawn_weight / total_weight * 100:.0f}% of all relation "
                  f"instances; arrow width ∝ count, classes grouped by "
                  f"taxonomy root",
                  fontsize=9.6, color=INK, loc="left", pad=14)
    # Two legends: the predicate colours on the right, and the taxonomy groups
    # that order the ring. Group names sat on the rim before and collided with
    # the longer class labels, which no amount of nudging fixes reliably.
    pred_leg = ax2.legend(
        handles=[Line2D([], [], color=pred_colour[p], lw=2.4,
                        label=f"{p.lower()}  ({d['pred_total'][p]:,})")
                 for p in top_preds],
        loc="center left", bbox_to_anchor=(0.985, 0.62), frameon=False,
        fontsize=7.0, labelspacing=0.5,
        title="predicate", title_fontsize=7.6)
    ax2.add_artist(pred_leg)
    ax2.legend(
        handles=[Line2D([], [], color=g["colour"], lw=3.2,
                        label=f"{g['name']}  ({len(g['members'])})")
                 for g in groups],
        loc="center left", bbox_to_anchor=(0.985, 0.17), frameon=False,
        fontsize=7.0, labelspacing=0.5,
        title="ring group (taxonomy root)", title_fontsize=7.6)

    fig.text(0.012, 0.963,
             "Figure 19 — Jeju tourism ontology: complete structure",
             fontsize=12.6, weight="bold", color=INK)
    fig.text(0.012, 0.925,
             f"v{d['version']} · {d['concepts']:,} concepts · "
             f"{d['relations']:,} relations · {len(d['counts'])} classes · "
             f"{len(d['relationship_types'])} predicates · "
             f"read from {d['engine'].replace('Neo4j Kernel ', 'Neo4j ')}",
             fontsize=8.6, color=MUTED)

    out = []
    for ext in formats:
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, facecolor="white")
        out.append(p)
    plt.close(fig)
    return out


# ------------------------------------------- figure 20: the qualitative cases


KEY_RE = re.compile(
    r"\b[A-Z][a-zA-Z]+(?:[-\s][A-Z][a-zA-Z]+)*\b")
STOP = {"The", "A", "An", "It", "Jeju", "Island", "Korea", "Near", "Day",
        "Morning", "Afternoon", "Evening", "Activities", "UNESCO", "Visit",
        "Explore", "Travel", "Stay", "South"}


LEADING = re.compile(r"^(The|A|An|Near|Visit|Explore|Arrive|South)\s+")


def key_entities(reference: str) -> list[str]:
    """Proper-noun phrases in the reference - the names that decide correctness.

    A sentence-initial article is part of the regex match but not part of the
    name, so "The UNESCO tuff cone" would otherwise be marked as the entity.
    """
    out, seen = [], set()
    for m in KEY_RE.findall(reference):
        m = LEADING.sub("", m.strip()).strip()
        if len(m) < 5 or m in STOP or m.lower() in seen:
            continue
        seen.add(m.lower())
        out.append(m)
    return out


def select_cases(arm: str = "onto_c", hi: float = 0.9, lo: float = 0.2,
                 want: int = 4) -> dict:
    """Cases where the ontology arm is right and RAG is wrong, plus base rates."""
    cands, pairs, onto_win, rag_win = [], 0, 0, 0
    for model in MODELS:
        paths = {k: EXP / sub / f"bench__{model}__{a}.jsonl"
                 for k, (sub, a) in {
                     "sr": ("scores", "rag"), "so": ("scores", arm),
                     "gr": ("generations", "rag"), "go": ("generations", arm),
                 }.items()}
        if not all(p.exists() for p in paths.values()):
            continue
        sr, so = jsonl(paths["sr"]), jsonl(paths["so"])
        gr, go = jsonl(paths["gr"]), jsonl(paths["go"])
        ids = set(sr) & set(so)
        pairs += len(ids)
        onto_win += sum(1 for i in ids
                        if so[i]["judge"] >= hi and sr[i]["judge"] <= lo)
        rag_win += sum(1 for i in ids
                       if sr[i]["judge"] >= hi and so[i]["judge"] <= lo)
        for i in ids:
            if so[i]["judge"] < hi or sr[i]["judge"] > lo:
                continue
            ref = go[i]["reference"]
            keys = key_entities(ref)
            if not keys:
                continue
            ho = sum(1 for k in keys if k.lower() in go[i]["answer"].lower())
            hr = sum(1 for k in keys if k.lower() in gr[i]["answer"].lower())
            cands.append({
                "model": model, "record_id": i, "question": go[i]["Input"],
                "reference": ref, "rag": gr[i]["answer"].strip(),
                "ontology": go[i]["answer"].strip(),
                "judge_rag": sr[i]["judge"], "judge_ontology": so[i]["judge"],
                "keytok_rag": sr[i]["keytok"], "keytok_ontology": so[i]["keytok"],
                "key_entities": keys,
                "entities_ontology": ho, "entities_rag": hr,
                "entity_gap": ho - hr,
            })

    # Prefer a wide entity gap and a short RAG answer, so the contrast is about
    # the fact and not about length; then one case per model for variety.
    cands.sort(key=lambda c: (-c["entity_gap"],
                              -(c["judge_ontology"] - c["judge_rag"]),
                              len(c["rag"])))
    chosen, used = [], set()
    for c in cands:
        if c["model"] in used:
            continue
        chosen.append(c)
        used.add(c["model"])
        if len(chosen) == want:
            break
    for c in cands:
        if len(chosen) == want:
            break
        if c not in chosen:
            chosen.append(c)

    return {"arm": arm, "threshold_ontology": hi, "threshold_rag": lo,
            "scored_pairs": pairs, "ontology_right_rag_wrong": onto_win,
            "rag_right_ontology_wrong": rag_win,
            "candidates": len(cands), "cases": chosen}


def _marked(ax, x, y, text, keys, width, colour, hit_colour, size=7.9,
            leading=0.0168):
    """Draw wrapped text, colouring the reference entities it does contain."""
    lines = textwrap.wrap(text, width=width) or [""]
    for li, line in enumerate(lines):
        yy = y - li * leading
        # split the line on the key entities so each can be coloured
        spans, idx = [], 0
        low = line.lower()
        while idx < len(line):
            nxt, nk = None, None
            for k in keys:
                j = low.find(k.lower(), idx)
                if j >= 0 and (nxt is None or j < nxt):
                    nxt, nk = j, k
            if nxt is None:
                spans.append((line[idx:], False))
                break
            if nxt > idx:
                spans.append((line[idx:nxt], False))
            spans.append((line[nxt:nxt + len(nk)], True))
            idx = nxt + len(nk)
        xx = x
        for s, hit in spans:
            t = ax.text(xx, yy, s, fontsize=size, va="top", ha="left",
                        color=hit_colour if hit else colour,
                        weight="bold" if hit else "normal",
                        family="DejaVu Sans")
            # advance by the rendered width, measured rather than estimated
            ax.figure.canvas.draw()
            bb = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
            xx += bb.width / ax.figure.bbox.width
    return len(lines)


WRAP = 132          # characters per wrapped line in the answer column
CLIP = 430          # characters of an answer shown before an ellipsis


def _case_height(c: dict) -> int:
    """Line-equivalents a case occupies, so the figure can be sized to fit.

    The two scored rows carry a sub-label a line below their title, so they
    never occupy fewer than two lines however short the answer is.
    """
    n = 0
    for text, scored in ((c["reference"], False), (c["rag"], True),
                         (c["ontology"], True)):
        shown = text[:CLIP] + ("…" if len(text) > CLIP else "")
        lines = max(1, len(textwrap.wrap(shown, width=WRAP)))
        n += max(lines, 2) if scored else lines
    return n


def draw_qualitative(sel: dict, out_stem: Path,
                     formats=("png", "pdf")) -> list[Path]:
    cases = sel["cases"]

    # Everything below is positioned in inches and converted at the end, so a
    # case that wraps to six lines takes six lines of space instead of leaving
    # the figure half empty or running off it.
    LH = 0.168          # inches per wrapped line
    ROW_PAD = 0.10      # inches below each of the three rows
    Q_H = 0.30          # inches for the question line
    CASE_PAD = 0.22     # inches between case boxes
    HEAD = 0.86         # inches of title and subtitle
    FOOT = 0.58         # inches of the base-rate footer

    body = sum(Q_H + _case_height(c) * LH + 3 * ROW_PAD + CASE_PAD
               for c in cases)
    H_IN = HEAD + body + FOOT
    W_IN = 13.4

    fig = plt.figure(figsize=(W_IN, H_IN), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def fy(inches_from_top: float) -> float:
        return 1.0 - inches_from_top / H_IN

    LHf, ROW_PADf, Q_Hf = LH / H_IN, ROW_PAD / H_IN, Q_H / H_IN
    CASE_PADf = CASE_PAD / H_IN

    fig.text(0.012, fy(0.20),
             "Figure 20 — Qualitative comparison: Simple RAG vs ontology-grounded "
             "retrieval", fontsize=12.6, weight="bold", color=INK, va="top")
    fig.text(0.012, fy(0.52),
             "Same prompt, same model, same character budget. Bold green marks a "
             "proper noun from the reference answer that the generated answer "
             "contains; bold red marks the entity the RAG answer named instead.",
             fontsize=8.4, color=MUTED, va="top")

    y = fy(HEAD)
    for ci, c in enumerate(cases):
        keys = c["key_entities"]
        wrong = [k for k in key_entities(c["rag"])
                 if k.lower() not in " ".join(keys).lower()
                 and k.lower() not in c["reference"].lower()][:3]

        box_top = y + 0.012
        rows_y = y

        ax.text(0.012, rows_y, f"Q{ci + 1}", fontsize=10.5, weight="bold",
                color=NAVY, va="top")
        ax.text(0.040, rows_y, c["question"], fontsize=8.6, va="top",
                color=INK, weight="bold")
        ax.text(0.988, rows_y, f"{c['model']} · record {c['record_id']}",
                fontsize=7.2, color=MUTED, va="top", ha="right")
        rows_y -= Q_Hf

        for label, text, colour, hit, score in (
                ("Reference", c["reference"], MUTED, GREEN, None),
                ("Simple RAG", c["rag"], INK, RED, c["judge_rag"]),
                ("Ontology", c["ontology"], INK, GREEN, c["judge_ontology"])):
            ax.text(0.040, rows_y, label, fontsize=7.8, weight="bold",
                    color={"Reference": MUTED, "Simple RAG": RED,
                           "Ontology": GREEN}[label], va="top")
            if score is not None:
                ax.text(0.040, rows_y - 0.92 * LHf, f"judge {score:.1f}",
                        fontsize=6.8, color=MUTED, va="top")
            marks = keys if label != "Simple RAG" else wrong
            nl = _marked(ax, 0.132, rows_y, text[:CLIP] +
                         ("…" if len(text) > CLIP else ""),
                         marks, width=WRAP, colour=colour, hit_colour=hit,
                         leading=LHf)
            # The "judge 0.2" sub-label sits a line below its row title, so a
            # one-line answer still needs two lines of room or the next row
            # lands on top of it.
            rows_y -= max(nl, 2 if score is not None else 1) * LHf + ROW_PADf

        ax.add_patch(FancyBboxPatch(
            (0.008, rows_y + 0.4 * ROW_PADf), 0.984,
            box_top - rows_y - 0.3 * ROW_PADf,
            boxstyle="round,pad=0.003", linewidth=0.7, edgecolor=LINE,
            facecolor="none", zorder=0, transform=ax.transData))
        y = rows_y - CASE_PADf

    foot = (
        f"Cases selected from the scored benchmark where the ontology arm's "
        f"judge score ≥ {sel['threshold_ontology']} and Simple RAG's ≤ "
        f"{sel['threshold_rag']} ({sel['candidates']} such cases). This is a "
        f"favourable slice, shown to illustrate how the two arms differ, not "
        f"how often: across {sel['scored_pairs']:,} scored (model, question) "
        f"pairs the ontology arm is right where RAG is wrong "
        f"{sel['ontology_right_rag_wrong']} times, and RAG is right where the "
        f"ontology arm is wrong {sel['rag_right_ontology_wrong']} times.")
    fig.text(0.012, y - 0.4 * LHf, "\n".join(textwrap.wrap(foot, width=168)),
             fontsize=7.4, color=MUTED, va="top", linespacing=1.5)

    out = []
    for ext in formats:
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, facecolor="white")
        out.append(p)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="onto_c",
                    help="which ontology arm to compare against rag")
    ap.add_argument("--cases", type=int, default=4)
    a = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)

    d = ontology_structure()
    wrote = draw_structure(d, FIG / "fig19_ontology_structure")
    for p in wrote:
        print(f"wrote {p.relative_to(ROOT)} "
              f"({p.stat().st_size / 1e6:.1f} MB)")

    sel = select_cases(arm=a.arm, want=a.cases)
    wrote = draw_qualitative(sel, FIG / "fig20_qualitative_rag_vs_ontology")
    for p in wrote:
        print(f"wrote {p.relative_to(ROOT)} "
              f"({p.stat().st_size / 1e6:.1f} MB)")

    out = STORE.parent / "task4_evaluation" / "qualitative_cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sel, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"\n{sel['scored_pairs']:,} scored pairs | ontology right / RAG wrong: "
          f"{sel['ontology_right_rag_wrong']} | RAG right / ontology wrong: "
          f"{sel['rag_right_ontology_wrong']}")
    for c in sel["cases"]:
        print(f"  {c['model']:<16} rec {c['record_id']}  judge "
              f"{c['judge_rag']:.1f} -> {c['judge_ontology']:.1f}  "
              f"entities {c['entities_rag']}/{len(c['key_entities'])} -> "
              f"{c['entities_ontology']}/{len(c['key_entities'])}")


if __name__ == "__main__":
    main()
