"""Two-pass build: render with LibreOffice, read heading pages from the PDF, rebuild with a filled TOC.

Word recomputes the TOC itself on open; this only makes the cached copy correct for other viewers.
"""
import json
import os
import subprocess
from pathlib import Path

import fitz

import build_annual_report as B

SOFFICE = Path.home() / "opt/lo/root/opt/libreoffice26.2/program/soffice"
LIBS = Path.home() / "miniconda3/envs/docrender/lib"
WORK = Path("/tmp/annual_report/render")

# Body text that marks where each TOC entry starts, in document order.
ANCHORS = {
    "1. Overview of the R&D Project": "Overview of the R&D Project",
    "2. R&D Process and Content": "R&D Process and Content",
    "3. R&D Results and Degree of Goal Achievement": "R&D Results and Degree of Goal Achievement",
    "3-1. Achievement Level of This Year's Objectives": "Achievement Level of This Year's Objectives",
    "3-2. Qualitative R&D Outcomes": "Qualitative R&D Outcomes",
    "3-3. Quantitative R&D Outcomes": "Quantitative R&D Outcomes",
    "3-4. Detailed Quantitative R&D Outcomes": "Detailed Quantitative R&D Outcomes",
    "3-5. Unplanned Outcomes and Contributions to Related Fields": "Unplanned Outcomes and Contributions",
    "4. R&D Expenditure": "R&D Expenditure",
    "4-1. R&D Expenditure Statement of the R&D Institution": "R&D Expenditure Statement of the R&D Institution",
    "4-2. R&D Expenditure Statement of the R&D Project": "R&D Expenditure Statement of the R&D Project",
    "5. R&D Plan for the Next Year": "R&D Plan for the Next Year",
    "5-1. R&D Objectives and Content": "R&D Objectives and Content",
    "5-2. Changes in the Domestic and International Environment": "Changes in the Domestic and International",
    "5-3. R&D Strategy": "R&D Strategy",
    "5-4. R&D Schedule and Expected Outcomes": "R&D Schedule and Expected Outcomes",
    "5-5. R&D Budget Plan": "R&D Budget Plan",
    "5-6. Commercialisation Plan": "Commercialisation Plan",
    "5-7. Utilisation of R&D Outcomes and Expected Effects": "Utilisation of R&D Outcomes and Expected",
    B.TOC_ENTRIES[-1][1]: "< Attachments >",
}


def render(docx):
    env = dict(os.environ, LD_LIBRARY_PATH=str(LIBS))
    WORK.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(SOFFICE), "--headless", "--convert-to", "pdf", "--outdir", str(WORK), str(docx)],
                   check=True, env=env, capture_output=True)
    return WORK / (Path(docx).stem + ".pdf")


def page_map(pdf):
    doc = fitz.open(pdf)
    texts = [" ".join(p.get_text().split()) for p in doc]
    toc_page = next(i for i, t in enumerate(texts) if "Table of Contents" in t)
    pages, start = {}, toc_page + 1
    for entry, _ in [(e, l) for l, e in B.TOC_ENTRIES]:
        needle = ANCHORS[entry]
        for i in range(start, len(texts)):
            if needle in texts[i]:
                pages[entry] = i + 1
                start = i
                break
    return pages, len(doc)


if __name__ == "__main__":
    B.build()
    pages, n = page_map(render(B.OUT))
    B.build(pages)
    pages2, n2 = page_map(render(B.OUT))
    json.dump(pages2, open(B.OUT_DIR / "toc_pages.json", "w"), indent=1)
    print(f"{n2} pages; TOC stable: {pages == pages2}")
    for k, v in pages2.items():
        print(f"{v:>3}  {k}")
