"""
exp041 - inline the data bundle into the single-page app.

The app opens straight from file:// with no server, so the bundle is embedded in
a <script type="application/json"> block rather than fetched (file:// fetch is
blocked by CORS). Nothing is templated except that one substitution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "Jeju_Tourism_Research_Demo"
tpl = (D / "application" / "frontend" / "index_template.html").read_text(encoding="utf-8")
bundle = (D / "demo" / "recorded_traces" / "bundle.json").read_text(encoding="utf-8")

# guard: </script> inside the JSON would terminate the block early
assert "</script" not in bundle.lower(), "bundle contains a literal </script>"
html = tpl.replace("__BUNDLE__", bundle)

for out in (D / "demo" / "interactive_application" / "index.html",
            D / "application" / "frontend" / "index.html"):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024/1024:.2f} MB)")

b = json.loads(bundle)
s = b["ontology"]["stats"]
print(f"embedded: {s['typed_concepts']:,} concepts, {s['relations']:,} relations, "
      f"{len(b['queries'])} queries, "
      f"{sum(len(v) for m in b['full_dataset'].values() for v in m.values())} benchmark cells")
