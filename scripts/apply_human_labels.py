"""Write tool_table_reviewed.csv: tool_table.csv with the labels from human_labels.csv applied."""
import csv
import sys
from pathlib import Path

TABLE = Path("data/toolemu/tool_table.csv")
LABELS = Path("data/toolemu/human_labels.csv")
OUT = Path("data/toolemu/tool_table_reviewed.csv")
VALID = {"reversible", "partially_reversible", "irreversible"}

with TABLE.open(encoding="utf-8", newline="") as fh:
    reader = csv.DictReader(fh)
    fields, rows = reader.fieldnames, list(reader)
with LABELS.open(encoding="utf-8", newline="") as fh:
    labels = {r["full_name"]: r for r in csv.DictReader(fh)}

unknown = sorted(set(labels) - {r["full_name"] for r in rows})
if unknown:
    sys.exit(f"Not in the tool table: {unknown}")
bad = sorted(n for n, r in labels.items() if r["human_label"] not in VALID)
if bad:
    sys.exit(f"Invalid human_label for: {bad} (use one of {sorted(VALID)})")

for r in rows:
    if r["full_name"] in labels:
        r["human_label"] = labels[r["full_name"]]["human_label"]
        r["human_notes"] = labels[r["full_name"]]["human_notes"]

with OUT.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"Applied {len(labels)} human labels -> {OUT}")