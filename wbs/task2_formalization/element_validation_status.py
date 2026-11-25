"""
Per-element validation status for the ontology.

Proposal section 9.4 asks that retrieved subgraphs be ranked by "authority and
validation status of ontology elements". The project could not do that, because
validation produced a REPORT - counts and findings - and never a per-element
verdict a retriever could rank on. This script produces one.

For every concept: the number of L1 structural and L2 logical findings naming it
as subject, split by severity, and a verdict:

  clean     no finding names it
  warned    only warnings or info
  errored   at least one error

The output is a flat map from normalized concept name to verdict and counts, so
the retriever can look a card up in constant time without re-running validation.

Findings whose subject is absent (layer-level findings) are counted separately
and attributed to no element, rather than spread over all of them.

  python -m wbs.task2_formalization.element_validation_status
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "wbs" / "deliverables" / "task2_formalization"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.4.0")
    a = ap.parse_args()

    from wbs.task2_formalization.validate_ontology import (key, l1_structural,
                                                           l2_logical, load)
    cs, rels, hier, _ = load(a.version)
    f1 = l1_structural(cs, rels)
    f2 = l2_logical(cs, rels, hier)

    per = collections.defaultdict(lambda: {"error": 0, "warning": 0, "info": 0,
                                           "codes": collections.Counter()})
    unattributed = 0
    for f in f1 + f2:
        subj = f.get("subject")
        if not subj:
            unattributed += 1
            continue
        rec = per[key(subj)]
        rec[f.get("severity", "info")] = rec.get(f.get("severity", "info"), 0) + 1
        rec["codes"][f.get("code", "?")] += 1

    status = {}
    for name, rec in per.items():
        verdict = ("errored" if rec["error"] else
                   "warned" if (rec["warning"] or rec["info"]) else "clean")
        status[name] = {"verdict": verdict, "error": rec["error"],
                        "warning": rec["warning"], "info": rec["info"],
                        "findings": rec["error"] + rec["warning"] + rec["info"],
                        "top_codes": rec["codes"].most_common(3)}

    names = [key(c["name"]) for c in cs if c.get("name")]
    tally = collections.Counter(status.get(n, {}).get("verdict", "clean") for n in names)
    out = {"ontology_version": a.version, "concepts": len(names),
           "l1_findings": len(f1), "l2_findings": len(f2),
           "findings_not_attributable_to_an_element": unattributed,
           "verdict_counts": dict(tally),
           "status": status}
    p = OUT / "element_validation_status.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"v{a.version}: {len(f1)} L1 + {len(f2)} L2 findings over {len(names):,} "
          f"concepts")
    print("  " + ", ".join(f"{k} {v:,}" for k, v in sorted(tally.items())))
    print(f"  {unattributed} findings name no element and are attributed to none")
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
