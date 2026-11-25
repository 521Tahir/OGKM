"""Layout primitives that reproduce the ETRI annual-report form (연차보고서) in .docx.

The source form is `(양식) 연차보고서_공동-위탁연구기관용(1001).hwp`. Its own
typographic rule, printed on its table-of-contents page, is: body text Dotum
12pt, tables Dotum 8pt. Section headings are a two-cell box - a dark numbered
cell followed by a wide title cell. Everything here exists to reproduce that
shape, so the English report sits in the same frame as the Korean form.
"""
from __future__ import annotations

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt, RGBColor

FONT = "Dotum"
BODY_PT = 12.0
TABLE_PT = 8.0
NAVY = "1F4E79"
HEAD_FILL = "D9E2F3"
GREY_FILL = "F2F2F2"


# ---------------------------------------------------------------- text/runs


def style_run(run, size=BODY_PT, bold=False, color=None, font=FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    # python-docx sets only the latin font; the east-asian slot needs the
    # raw attribute or Hangul and the Latin text disagree about typeface.
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), font)
    return run


def para(doc, text="", size=BODY_PT, bold=False, indent=0.0, space_after=2.0,
         align=None, color=None, container=None, space_before=0.0, line=1.25):
    host = container if container is not None else doc
    p = host.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(indent)
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing = line
    if align is not None:
        p.alignment = align
    if text:
        style_run(p.add_run(text), size=size, bold=bold, color=color)
    return p


def rich(doc, parts, size=BODY_PT, indent=0.0, space_after=2.0, container=None,
         line=1.25):
    """One paragraph from (text, bold) pairs, so a single line can carry emphasis."""
    host = container if container is not None else doc
    p = host.add_paragraph()
    p.paragraph_format.left_indent = Cm(indent)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line
    for text, bold in parts:
        style_run(p.add_run(text), size=size, bold=bold)
    return p


def bullet(doc, text, level=1, size=BODY_PT, bold=False, container=None):
    """The form's own bullet ladder: ◦ at the first level, - at the second."""
    marks = {1: ("◦ ", 0.5), 2: ("- ", 1.1), 3: ("· ", 1.7)}
    mark, ind = marks.get(level, marks[2])
    return para(doc, mark + text, size=size, bold=bold, indent=ind,
                container=container)


def page_break(doc):
    style_run(doc.add_paragraph().add_run()).add_break(WD_BREAK.PAGE)


# ---------------------------------------------------------------- shading/borders


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _mark_borders(element, sides, sz, color):
    borders = OxmlElement("w:tcBorders")
    for side in sides:
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:color"), color)
        borders.append(el)
    element.append(borders)


def outline(table, sz=12, color="000000"):
    """Thicken a table's outer edge, the way the form boxes a whole block."""
    rows = table.rows
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row.cells):
            sides = []
            if ri == 0:
                sides.append("top")
            if ri == len(rows) - 1:
                sides.append("bottom")
            if ci == 0:
                sides.append("left")
            if ci == len(row.cells) - 1:
                sides.append("right")
            if sides:
                _mark_borders(cell._tc.get_or_add_tcPr(), sides, sz, color)


def cell_text(cell, text, size=TABLE_PT, bold=False, align=None, fill=None,
              indent=0.0, color=None):
    cell.text = ""
    lines = str(text).split("\n")
    for i, line in enumerate(lines):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.left_indent = Cm(indent)
        if align is not None:
            p.alignment = align
        if line:
            style_run(p.add_run(line), size=size, bold=bold, color=color)
    if fill:
        shade(cell, fill)
    return cell


def grid(doc, rows, cols, widths=None, align_table=WD_TABLE_ALIGNMENT.CENTER):
    t = doc.add_table(rows=rows, cols=cols)
    t.style = "Table Grid"
    t.alignment = align_table
    t.autofit = False
    if widths:
        for row in t.rows:
            for cell, w in zip(row.cells, widths):
                cell.width = Cm(w)
    return t


def table_from(doc, header, body, widths, size=TABLE_PT, head_fill=HEAD_FILL,
               aligns=None, head_size=None):
    """A plain header-plus-rows table, 8pt, as the form specifies for tables."""
    t = grid(doc, 1 + len(body), len(header), widths)
    for ci, text in enumerate(header):
        cell_text(t.rows[0].cells[ci], text, size=head_size or size, bold=True,
                  align=WD_ALIGN_PARAGRAPH.CENTER, fill=head_fill)
    for ri, row in enumerate(body, start=1):
        for ci, text in enumerate(row):
            al = (aligns or [None] * len(header))[ci]
            cell_text(t.rows[ri].cells[ci], text, size=size, align=al)
    outline(t)
    return t


def label_table(doc, pairs, widths, size=BODY_PT, label_fill=HEAD_FILL):
    """Two-column label/value block - the form's commonest shape."""
    t = grid(doc, len(pairs), 2, widths)
    for ri, (label, value) in enumerate(pairs):
        cell_text(t.rows[ri].cells[0], label, size=size, bold=True,
                  align=WD_ALIGN_PARAGRAPH.CENTER, fill=label_fill)
        cell_text(t.rows[ri].cells[1], value, size=size, indent=0.1)
    outline(t)
    return t


# ---------------------------------------------------------------- headings


def section_header(doc, number, title, page_width_cm=17.0):
    """The form's numbered section box: dark number cell, then the title."""
    t = grid(doc, 1, 2, [1.6, page_width_cm - 1.6])
    cell_text(t.rows[0].cells[0], str(number), size=16, bold=True,
              align=WD_ALIGN_PARAGRAPH.CENTER, fill=NAVY, color="FFFFFF")
    cell_text(t.rows[0].cells[1], title, size=15, bold=True, indent=0.2,
              fill=GREY_FILL)
    outline(t, sz=16)
    para(doc, "", size=4)
    return t


CIRCLED = ["❶", "❷", "❸", "❹", "❺", "❻", "❼"]


def sub_header(doc, index, title):
    """❶ ❷ ❸ … sub-section heading, as the form marks them."""
    return para(doc, f"{CIRCLED[index - 1]} {title}", size=13.5, bold=True,
                space_before=8, space_after=4)


def box_header(doc, title, size=12.5):
    return para(doc, f"□ {title}", size=size, bold=True,
                space_before=6, space_after=3)


def figure(doc, path, number, title, width_cm=15.0, source=None):
    para(doc, f"[Figure {number}] {title}", size=10, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    if source:
        para(doc, f"※ Source: {source}", size=8,
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
    return p


def table_title(doc, number, title):
    return para(doc, f"<Table {number}> {title}", size=10, bold=True,
                space_before=6, space_after=2)


def table_source(doc, source):
    return para(doc, f"※ Source: {source}", size=8, space_after=8)


# ---------------------------------------------------------------- page setup


def new_document(margin_mm=20, top_mm=20):
    from docx import Document

    doc = Document()
    for s in doc.sections:
        s.page_width = Mm(210)
        s.page_height = Mm(297)
        s.left_margin = Mm(margin_mm)
        s.right_margin = Mm(margin_mm)
        s.top_margin = Mm(top_mm)
        s.bottom_margin = Mm(top_mm)
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY_PT)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    return doc


def page_width_cm(doc):
    s = doc.sections[0]
    return (s.page_width - s.left_margin - s.right_margin) / 360000.0


def add_page_numbers(doc):
    """PAGE field in the footer, so the printed report is paginated."""
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    style_run(run, size=9)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._element.addnext(fld)
    return p
