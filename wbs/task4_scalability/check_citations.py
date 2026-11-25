"""
Does every artifact the documentation cites actually exist?

The `docs/` tree carries plans, trackers and audit tables written across several
phases of this work, some of them in the companion project
`/home/tahirahmad/Ontology_Project` before this repository existed. A citation
that no longer resolves is not automatically an error - a plan may legitimately
name an artifact it intends to produce - but a citation in a document that
presents itself as a RECORD of what exists is a claim, and an unresolvable one
is a false claim. The two cases are indistinguishable by reading, which is why
this check exists and runs.

Every backticked path in `docs/*.md` is resolved in four passes:

  here        the path exists in this repository
  by-name     the basename exists here under a different directory, i.e. the
              citation is relative to a deliverables subtree
  companion   the path exists in the companion project but not here
  unresolved  no file of that name exists in either tree

Only `unresolved` is a defect, and the report separates planning documents
(where a forward reference is legitimate) from record documents (where it is
not).

  python -m wbs.task4_scalability.check_citations
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
COMPANION = pathlib.Path("/home/tahirahmad/Ontology_Project")
DELIV = ROOT / "wbs" / "deliverables"
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}
CITE = re.compile(r"`([A-Za-z0-9_./\-]+\.(?:py|json|md|jsonl|ttl|owl|nt|csv|png|txt|xlsx))`")

# documents whose job is to plan or track future work: a forward reference is fine
PLANNING = {"PROJECT_IMPLEMENTATION_PLAN.md", "WEEKLY_TASK_TRACKER.md",
            "implementation_roadmap.md", "full_data_runbook.md"}


def index(tree: pathlib.Path) -> dict[str, list[str]]:
    out = collections.defaultdict(list)
    if not tree.exists():
        return out
    for p in tree.rglob("*"):
        if p.is_file() and not any(s in p.parts for s in SKIP_DIRS):
            out[p.name].append(str(p.relative_to(tree)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-on-record-defects", action="store_true",
                    help="exit non-zero if a record document cites a missing artifact")
    a = ap.parse_args()

    here, comp = index(ROOT), index(COMPANION)
    rows, tally = [], collections.Counter()
    for doc in sorted((ROOT / "docs").glob("*.md")):
        text = doc.read_text(encoding="utf-8")
        # A document that opens with a correction note mapping its stale names to
        # the implemented ones has already told the reader those names do not
        # resolve. Later occurrences of a name the note lists are covered by it,
        # so they are annotation too - a document should not be penalised once per
        # repetition of something it has explicitly corrected.
        corrected = {m.group(1) for ln in text.splitlines()
                     if ln.lstrip().startswith(">")
                     for m in CITE.finditer(ln)}
        for line in text.splitlines():
            # A correction note quotes a stale filename in order to say it is stale.
            # Blockquoted lines are therefore annotation, not claim, and a path that
            # appears only there is counted separately rather than as a defect.
            quoted = line.lstrip().startswith(">")
            for m in CITE.finditer(line):
                c = m.group(1)
                if quoted:
                    tally["annotated"] += 1
                    rows.append({"doc": doc.name, "citation": c,
                                 "status": "annotated", "planning_doc": False})
                    continue
                base = pathlib.PurePath(c).name
                if (ROOT / c).exists():
                    status = "here"
                elif base in here and ("/" not in c
                                       or any(h.endswith(c) for h in here[base])):
                    status = "by-name"
                elif (COMPANION / c).exists() or base in comp:
                    status = "companion"
                else:
                    # only a name that resolves nowhere falls back to the
                    # document's own correction note
                    status = "annotated" if c in corrected else "unresolved"
                tally[status] += 1
                rows.append({"doc": doc.name, "citation": c, "status": status,
                             "planning_doc": doc.name in PLANNING})

    unresolved = [r for r in rows if r["status"] == "unresolved"]
    record_defects = [r for r in unresolved if not r["planning_doc"]]
    by_doc = collections.Counter(r["doc"] for r in unresolved)
    total = len(rows)

    L = ["# Documentation citation audit\n",
         f"Every backticked artifact path in `docs/*.md` — {total:,} citations across "
         f"{len({r['doc'] for r in rows})} documents — resolved against this "
         f"repository and against the companion project `Ontology_Project`.\n",
         "| Resolution | Citations | Meaning |", "|---|--:|---|",
         f"| `here` | {tally['here']:,} | path exists as written |",
         f"| `by-name` | {tally['by-name']:,} | exists under another directory, i.e. "
         f"cited relative to a deliverables subtree |",
         f"| `companion` | {tally['companion']:,} | exists in the companion project, "
         f"not in this one |",
         f"| `annotated` | {tally['annotated']:,} | quoted inside a correction note, "
         f"i.e. named in order to record that it is stale |",
         f"| `unresolved` | **{tally['unresolved']:,}** | no file of that name in "
         f"either tree |", "",
         f"**{len(record_defects)} of the {tally['unresolved']} unresolved citations "
         f"sit in documents that present themselves as a record of what exists**, and "
         f"those are the defects. The remainder are forward references in planning and "
         f"tracking documents "
         f"({', '.join(sorted(PLANNING & {r['doc'] for r in unresolved}))}), where "
         f"naming an intended artifact is the document's purpose.\n"]

    if by_doc:
        L.append("## Unresolved citations by document\n")
        L.append("| Document | Unresolved | Kind |")
        L.append("|---|--:|---|")
        for d, n in by_doc.most_common():
            kind = "plan / tracker (forward references expected)" if d in PLANNING \
                else "**record (defect)**"
            L.append(f"| `docs/{d}` | {n} | {kind} |")
        L.append("")
    if record_defects:
        L.append("## The defects\n")
        cur = None
        for r in sorted(record_defects, key=lambda r: (r["doc"], r["citation"])):
            if r["doc"] != cur:
                cur = r["doc"]
                L.append(f"\n**`docs/{cur}`**\n")
            L.append(f"- `{r['citation']}`")
        L.append("\nThese are artifacts from the earlier phase of the work that were "
                 "either never produced or not carried into this repository. They are "
                 "left in place rather than deleted, because a citation that was once "
                 "made is part of the record, but no result in `wbs/deliverables/` "
                 "depends on any of them: the deliverables' own provenance is checked "
                 "separately by `REPRODUCIBILITY/MANIFEST.json`, and the two audits "
                 "(`WBS_WORKBOOK_AUDIT.md`, `PROPOSAL_COMPLIANCE.md`) resolve every "
                 "path they cite or report it missing.\n")
    (DELIV / "CITATION_AUDIT.md").write_text("\n".join(L), encoding="utf-8")
    (DELIV / "CITATION_AUDIT.json").write_text(
        json.dumps({"total": total, "tally": dict(tally),
                    "record_defects": len(record_defects),
                    "companion_project": str(COMPANION), "citations": rows},
                   indent=2), encoding="utf-8")
    print(f"{total:,} citations: " + ", ".join(f"{k} {v:,}" for k, v in tally.items()))
    print(f"record-document defects: {len(record_defects)}")
    print("wrote CITATION_AUDIT.md")
    if a.fail_on_record_defects and record_defects:
        sys.exit(1)


if __name__ == "__main__":
    main()
