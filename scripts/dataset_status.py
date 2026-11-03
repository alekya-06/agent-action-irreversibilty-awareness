"""Status of every built dataset file: schema check, counts and minimal-pair share."""
import json
import sys
from collections import Counter
from pathlib import Path

for name in ("sandbox", "sandbox_v2"):
    common = Path("src") / name / "common"
    if (common / "schema.py").exists():
        sys.path.insert(0, str(common))
        break
from schema import check_dataset, summarise  # noqa: E402

SKIP = ("before_", "drafts", "reviews", "fixes", "_summary", "reversibility_v2.jsonl", "legacy", "_expanded")
files = [p for p in sorted(Path("data").rglob("*.jsonl")) if not any(s in str(p) for s in SKIP)]
total = 0
for path in files:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows or "environment" not in rows[0]:
        continue  # not a dataset file
    print(f"\n=== {path}  ({len(rows)} rows)")
    try:
        check_dataset(rows)
        print("schema: OK")
    except Exception as exc:  # report, don't stop
        print(f"schema: FAILED - {exc}")
    groups = Counter(r.get("group_id") for r in rows if r.get("group_id"))
    paired = sum(n for n in groups.values() if n > 1)
    print(f"rows in minimal pairs: {paired} ({100 * paired / len(rows):.0f}%)")
    try:
        print(json.dumps(summarise(rows), indent=1)[:1500])
    except KeyError as exc:
        print(f"summary: skipped (missing field {exc})")
    if "splits" not in str(path):
        total += len(rows)
print(f"\nTOTAL rows across files: {total}  (target: 2,000)")