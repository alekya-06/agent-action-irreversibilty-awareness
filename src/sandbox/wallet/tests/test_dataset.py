import json
from pathlib import Path
from collections import Counter

path = Path("data/wallet/wallet.jsonl")

with path.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

print("Labels by action type:")
counts = Counter(
    (row["action"]["type"], row["trinary_label"])
    for row in rows
)

for (action_type, label), count in sorted(counts.items()):
    print(f"{action_type:20} {label:22} {count}")

print("\nIrreversible examples:")
for row in rows:
    if row["trinary_label"] == "irreversible":
        print(
            row["scenario_id"],
            row["action"],
            "cost =", row["recovery_cost"],
        )