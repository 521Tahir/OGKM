"""Formatting primitives that reproduce the ETRI annual-report form in python-docx.

Form rules: body text Dotum 12 pt, tables Dotum 8 pt (narrative boxes Dotum 10 pt),
section markers ① □ ◦ -, table captions "<Table n>" above, figure captions
"[Figure n]" below, each followed by a "※ Source:" line in 8 pt.
"""
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Dotum"
FONT_EA = "돋움"
MONO = "Consolas"
NAVY = "1F3A5F"
HEAD_FILL = "DCE6F1"
LABEL_FILL = "F2F2F2"
TEXT_W = 17.0  # cm between margins on A4 with 2 cm side margins
CIRCLED = "①②③④⑤⑥⑦⑧⑨"


def set_font(run, size=None, bold=None, italic=None, color=None, font=FONT):
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(attr), font)
    rfonts.set(qn("w:eastAsia"), FONT_EA if font == FONT else font)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    return run


def add_rich(paragraph, text, size, bold=False, italic=False, color=None, font=FONT):
    """Add text where **x** is bold."""
    for i, part in enumerate(re.split(r"\*\*", text)):
        if part:
            set_font(paragraph.add_run(part), size, bold=bold or i % 2 == 1, italic=italic,
                     color=color, font=font)
    return paragraph


def fmt(p, align=None, before=0, after=4, left=0.0, hanging=0.0, line=1.15, keep=False):
    pf = p.paragraph_format
    if align:
        p.alignment = {"c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT,
                       "j": WD_ALIGN_PARAGRAPH.JUSTIFY, "l": WD_ALIGN_PARAGRAPH.LEFT}[align]
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if left or hanging:
        pf.left_indent = Cm(left + hanging)
        pf.first_line_indent = Cm(-hanging)
    if keep:
        pf.keep_with_next = True
    return p


def first_par(container):
    """Reuse the empty paragraph a fresh table cell starts with."""
    if hasattr(container, "_tc"):
        ps = container.paragraphs
        if len(ps) == 1 and not ps[0].text and not ps[0].runs:
            if not getattr(container, "_used_first", False):
                container._used_first = True
                return ps[0]
    return container.add_paragraph()


def para(container, text, size=12, bold=False, italic=False, align="j", before=0, after=4,
         left=0.0, hanging=0.0, color=None, keep=False, font=FONT, line=1.15):
    p = first_par(container)
    fmt(p, align, before, after, left, hanging, line=line, keep=keep)
    add_rich(p, text, size, bold, italic, color, font)
    return p


# Template marker hierarchy: ① section, □ heading, ◦ item, - sub-item, ※ note.
def box(container, text, size=12, after=3):
    return para(container, "□ " + text, size, bold=True, align="l", before=4, after=after,
                left=0.0, hanging=0.55, keep=True)


def item(container, text, size=12, after=2):
    return para(container, "◦ " + text, size, left=0.45, hanging=0.45, after=after)


def sub(container, text, size=12, after=2):
    return para(container, "- " + text, size, left=1.0, hanging=0.4, after=after)


def note(container, text, size=8, after=4):
    return para(container, "※ " + text, size, align="l", after=after, left=0.0, hanging=0.4,
                color="404040")


def bullets(container, items, size=12):
    """items: str for ◦, or (str, [subs]) for ◦ with - children."""
    for it in items:
        if isinstance(it, tuple):
            item(container, it[0], size)
            for s in it[1]:
                sub(container, s, size)
        else:
            item(container, it, size)


# ------------------------------------------------------------------ fields
def _fld(run, kind, instr=None):
    if kind == "instr":
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = instr
    else:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
    run._element.append(el)


def tc_entry(paragraph, text, level):
    """Hidden TC field so the TOC lists the form's own numbering (3-1., 5-4., ...)."""
    safe = text.replace('"', "'")
    for kind, instr in (("begin", None), ("instr", f' TC "{safe}" \\l {level} '), ("end", None)):
        r = paragraph.add_run()
        r.font.hidden = True
        _fld(r, kind, instr)


def page_field(paragraph, size=9):
    for kind, instr in (("begin", None), ("instr", " PAGE "), ("separate", None)):
        _fld(set_font(paragraph.add_run(), size), kind, instr)
    set_font(paragraph.add_run("1"), size)
    _fld(set_font(paragraph.add_run(), size), "end")


def toc(doc, entries, pages=None):
    """TOC built from TC fields; cached result pre-filled so it reads before Word refreshes it."""
    pages = pages or {}
    tab_pos = Cm(TEXT_W)
    for i, (level, text) in enumerate(entries):
        p = doc.add_paragraph()
        fmt(p, "l", after=3 if level == 1 else 1, left=0.0 if level == 1 else 0.8)
        p.paragraph_format.tab_stops.add_tab_stop(tab_pos, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        if i == 0:
            for kind, instr in (("begin", None), ("instr", ' TOC \\f \\l "1-2" \\h '), ("separate", None)):
                _fld(p.add_run(), kind, instr)
        set_font(p.add_run(text), 12 if level == 1 else 11, bold=(level == 1))
        set_font(p.add_run("\t" + str(pages.get(text, ""))), 11)
        if i == len(entries) - 1:
            _fld(p.add_run(), "end")


def update_fields_on_open(doc):
    settings = doc.settings.element
    el = OxmlElement("w:updateFields")
    el.set(qn("w:val"), "true")
    put(settings, el, ["updateFields"] + SETTINGS_AFTER_UPDATE)
    zoom = settings.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")


TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize",
               "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar",
               "tblLook", "tblCaption", "tblDescription"]
TCPR_ORDER = ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar",
              "textDirection", "tcFitText", "vAlign", "hideMark"]
SETTINGS_AFTER_UPDATE = ["hdrShapeDefaults", "footnotePr", "endnotePr", "compat", "docVars", "rsids", "mathPr",
                         "attachedSchema", "themeFontLang", "clrSchemeMapping", "doNotIncludeSubdocsInStats",
                         "doNotAutoCompressPictures", "forceUpgrade", "captions", "readModeInkLockDown", "smartTagType",
                         "schemaLibrary", "shapeDefaults", "doNotEmbedSmartTags", "decimalSymbol", "listSeparator"]


def put(parent, el, order):
    """Insert el into parent respecting the schema's child order, replacing any element of the same tag."""
    tag = el.tag.split("}")[1]
    for old in parent.findall(qn("w:" + tag)):
        parent.remove(old)
    later = order[order.index(tag) + 1:]
    for i, child in enumerate(parent):
        if child.tag.split("}")[1] in later:
            parent.insert(i, el)
            return el
    parent.append(el)
    return el


# ------------------------------------------------------------------ tables
def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    put(tcpr, shd, TCPR_ORDER)


def cell_margins(table, top=40, bottom=40, left=80, right=80):
    tblpr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(v))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    put(tblpr, mar, TBLPR_ORDER)


def borders(table, size=4, color="808080", outer=8):
    tblpr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(outer if side in ("top", "bottom") else size))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000" if side in ("top", "bottom") else color)
        b.append(el)
    put(tblpr, b, TBLPR_ORDER)


def no_borders(table):
    tblpr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "nil")
        b.append(el)
    put(tblpr, b, TBLPR_ORDER)


def repeat_header(row):
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trpr.append(el)


def no_split(row):
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit")
    el.set(qn("w:val"), "true")
    trpr.append(el)


def set_widths(table, widths):
    table.autofit = False
    tblpr = table._tbl.tblPr
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    put(tblpr, lay, TBLPR_ORDER)
    grid = table._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(Cm(w).twips)))
    for row in table.rows:
        for c, w in zip(row.cells, widths):
            c.width = Cm(w)


def write_cell(cell, text, size=8, bold=False, align="l", fill=None, italic=False, color=None):
    if fill:
        shade(cell, fill)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    lines = text.split("\n") if isinstance(text, str) else [str(text)]
    for ln in lines:
        p = first_par(cell)
        fmt(p, align, 0, 0, line=1.0)
        add_rich(p, ln, size, bold=bold, italic=italic, color=color)


def data_table(container, header, rows, widths, size=8, align=None, header_fill=HEAD_FILL,
               first_col_fill=None, merges=(), bold_rows=(), header_rows=1, col_align_header="c"):
    """header: list (or list of lists for multi-row headers); rows: list of lists."""
    heads = header if header and isinstance(header[0], list) else ([header] if header else [])
    n_cols = len(widths)
    t = container.add_table(rows=len(heads) + len(rows), cols=n_cols)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t)
    cell_margins(t)
    set_widths(t, widths)
    align = align or ["l"] + ["l" if w >= 4.0 else "c" for w in widths[1:]]
    for r, h in enumerate(heads):
        repeat_header(t.rows[r])
        for c, txt in enumerate(h):
            if txt is not None:
                write_cell(t.cell(r, c), txt, size, bold=True, align=col_align_header, fill=header_fill)
            else:
                shade(t.cell(r, c), header_fill)
    for i, row in enumerate(rows):
        r = i + len(heads)
        no_split(t.rows[r])
        for c, txt in enumerate(row):
            if txt is None:
                continue
            fill = first_col_fill if (c == 0 and first_col_fill) else None
            write_cell(t.cell(r, c), txt, size, bold=(i in bold_rows), align=align[c], fill=fill)
    for (r1, c1), (r2, c2) in merges:
        a, b = t.cell(r1, c1), t.cell(r2, c2)
        keep = a.text
        m = a.merge(b)
        for p in m.paragraphs[1:]:
            if not p.text.strip():
                p._element.getparent().remove(p._element)
    return t


def form_table(container, rows, widths, size=10, label_fill=LABEL_FILL, label_cols=(0,)):
    """Label/content box table used by the form (e.g. 'Overview | ◦ ...').

    rows: list of [label, content] where content is str, list of bullet items or a callable
    receiving the cell to fill it with arbitrary content.
    """
    t = container.add_table(rows=len(rows), cols=len(widths))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t)
    cell_margins(t, 60, 60, 100, 100)
    set_widths(t, widths)
    for r, row in enumerate(rows):
        for c, content in enumerate(row):
            cell = t.cell(r, c)
            if content is None:
                continue
            if c in label_cols:
                write_cell(cell, content, size, bold=True, align="c", fill=label_fill)
                continue
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            if callable(content):
                content(cell)
            elif isinstance(content, list):
                for it in content:
                    if isinstance(it, tuple):
                        para(cell, "○ " + it[0], size, left=0.0, hanging=0.4, after=1, align="l")
                        for s in it[1]:
                            para(cell, "- " + s, size, left=0.5, hanging=0.3, after=1, align="l")
                    else:
                        para(cell, "○ " + it, size, left=0.0, hanging=0.4, after=1, align="l")
            else:
                para(cell, content, size, after=1, align="l")
    return t


def caption_table(container, num, title, size=10):
    return para(container, f"<Table {num}> {title}", size, bold=True, align="c", before=6, after=3,
                keep=True)


def caption_figure(container, num, title, size=10):
    return para(container, f"[Figure {num}] {title}", size, bold=True, align="c", before=2, after=1)


def source(container, text):
    return para(container, "※ Source: " + text, 8, align="l", after=6, color="404040")


def figure(container, path, num, title, src, width=16.0):
    p = first_par(container)
    fmt(p, "c", 4, 2, keep=True)
    p.add_run().add_picture(str(path), width=Cm(width))
    caption_figure(container, num, title)
    source(container, src)


def code_block(container, lines, width=TEXT_W, size=8):
    t = container.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t, outer=4)
    cell_margins(t, 80, 80, 140, 140)
    set_widths(t, [width])
    c = t.cell(0, 0)
    shade(c, "F7F7F7")
    for ln in lines:
        p = first_par(c)
        fmt(p, "l", 0, 0, line=1.0)
        set_font(p.add_run(ln if ln else " "), size, font=MONO)
    return t


def spacer(container, pts=4):
    p = container.add_paragraph()
    fmt(p, after=0)
    p.paragraph_format.space_before = Pt(pts)
    return p


def page_break(doc):
    p = doc.add_paragraph()
    fmt(p, after=0)
    p.add_run().add_break(WD_BREAK.PAGE)


def chapter_banner(doc, num, title):
    """The form opens each chapter with a three-cell banner: number | gap | title."""
    t = doc.add_table(rows=1, cols=3)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    no_borders(t)
    set_widths(t, [1.4, 0.25, TEXT_W - 1.65])
    a, _, c = t.rows[0].cells
    shade(a, NAVY)
    write_cell(a, str(num), 18, bold=True, align="c", color="FFFFFF")
    c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = first_par(c)
    p.style = doc.styles["Heading 1"]
    fmt(p, "l", 0, 0)
    set_font(p.add_run(title), 16, bold=True, color="000000")
    tc_entry(p, f"{num}. {title}", 1)
    tcpr = c._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    el = OxmlElement("w:bottom")
    for k, v in (("w:val", "single"), ("w:sz", "18"), ("w:space", "0"), ("w:color", NAVY)):
        el.set(qn(k), v)
    b.append(el)
    put(tcpr, b, TCPR_ORDER)
    spacer(doc, 6)
    return t


def section_heading(doc, idx, title, toc_text):
    """①-style section heading with a TC entry carrying the form's 3-1. numbering."""
    p = doc.add_paragraph()
    p.style = doc.styles["Heading 2"]
    fmt(p, "l", 10, 6, keep=True)
    set_font(p.add_run(f"{CIRCLED[idx - 1]} {title}"), 13, bold=True, color="000000")
    if toc_text:
        tc_entry(p, toc_text, 2)
    return p


def new_document():
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(1.8)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.header_distance = sec.footer_distance = Cm(1.0)
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(12)
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(attr), FONT)
    rf.set(qn("w:eastAsia"), FONT_EA)
    for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 12)):
        h = doc.styles[name]
        h.font.name = FONT
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = RGBColor(0, 0, 0)
        hr = h.element.get_or_add_rPr()
        hf = hr.find(qn("w:rFonts"))
        if hf is None:
            hf = OxmlElement("w:rFonts")
            hr.insert(0, hf)
        for attr in ("w:ascii", "w:hAnsi", "w:cs"):
            hf.set(qn(attr), FONT)
        hf.set(qn("w:eastAsia"), FONT_EA)
        for th in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if hf.get(qn(th)) is not None:
                del hf.attrib[qn(th)]
    return doc


def header_footer(doc, header_text):
    sec = doc.sections[0]
    sec.different_first_page_header_footer = True
    hp = sec.header.paragraphs[0]
    fmt(hp, "r", after=0)
    set_font(hp.add_run(header_text), 8, color="606060")
    fp = sec.footer.paragraphs[0]
    fmt(fp, "c", after=0)
    set_font(fp.add_run("- "), 9)
    page_field(fp)
    set_font(fp.add_run(" -"), 9)
