"""
Combine all JSON files from dataset/ into a single full_dataset.json.

Handles:
- normal JSON arrays
- files with invalid \\uXXXX unicode escapes (auto-repaired)
- assigns global record IDs across files
- reports per-file counts and any skipped records
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger, setup_logging
from src.utils.io import save_json, ensure_dir

setup_logging(log_level="INFO")
logger = get_logger("combine_datasets")


def fix_unicode_escapes(s: str) -> str:
    """
    Replace invalid \\uXXXX escape sequences (fewer than 4 hex digits)
    with a literal backslash-u so JSON parser doesn't choke.
    """
    result: list[str] = []
    i = 0
    while i < len(s):
        if s[i : i + 2] == "\\u":
            hex_part = s[i + 2 : i + 6]
            if re.match(r"^[0-9a-fA-F]{4}$", hex_part):
                result.append(s[i : i + 6])
                i += 6
            else:
                # Neutralize the bad escape
                result.append("\\\\u")
                i += 2
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


def load_file(path: Path) -> list[dict]:
    """Load a JSON file, applying unicode escape repair if needed."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"  JSON error in {path.name} — applying unicode escape repair ...")
            fixed = fix_unicode_escapes(content)
            return json.loads(fixed)
    except Exception as e:
        logger.error(f"  FAILED to load {path.name}: {e}")
        return []


def main() -> None:
    dataset_dir = PROJECT_ROOT / "dataset"
    output_dir = PROJECT_ROOT / "data" / "raw"
    ensure_dir(output_dir)

    json_files = sorted(dataset_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {dataset_dir}")

    combined: list[dict] = []
    file_report: list[dict] = []
    global_id = 0

    for path in json_files:
        records = load_file(path)
        count = len(records)
        logger.info(f"  {path.name}: {count:,} records")

        for rec in records:
            # Ensure required fields exist
            if not all(k in rec for k in ("Input", "CoT", "Output")):
                continue
            rec["_source_file"] = path.name
            rec["_global_id"] = global_id
            combined.append(rec)
            global_id += 1

        file_report.append({"file": path.name, "records": count})

    logger.info(f"\nTotal combined records: {len(combined):,}")

    output_path = output_dir / "full_dataset.json"
    save_json(combined, output_path)
    logger.info(f"Saved → {output_path}")

    report_path = output_dir / "combine_report.json"
    save_json({
        "total_records": len(combined),
        "source_files": file_report,
        "output": str(output_path),
    }, report_path)
    logger.info(f"Report → {report_path}")

    print(f"\n{'='*55}")
    print(f"DATASET COMBINE COMPLETE")
    print(f"{'='*55}")
    for fr in file_report:
        print(f"  {fr['file']:20s}: {fr['records']:>7,}")
    print(f"{'─'*55}")
    print(f"  {'TOTAL':20s}: {len(combined):>7,}")
    print(f"  Output: {output_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
