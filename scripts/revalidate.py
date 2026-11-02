"""Re-run validate_draft on every unreviewed draft and update draft_problems."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src/toolemu")
from build_dataset import tool_specs                  # noqa: E402
from draft_pairs import validate_draft                # noqa: E402

DRAFTS = Path("data/toolemu/pairs/drafts_v2.jsonl")
cases = {c["name"]: c for c in json.load(open("external/ToolEmu/assets/all_cases.json"))}
specs = tool_specs(json.load(open("external/ToolEmu/assets/all_toolkits.json")))

text = DRAFTS.read_text(encoding="utf-8")
DRAFTS.with_name(DRAFTS.stem + ".before_revalidate.jsonl").write_text(text, encoding="utf-8")
out, flagged = [], []
for line in text.splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if not r.get("reviewed"):
        r["draft_problems"] = validate_draft(r, cases[r["case"]], specs)
        if r["draft_problems"]:
            flagged.append((r["case"], r["draft_problems"]))
    out.append(json.dumps(r, ensure_ascii=False))
DRAFTS.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"{len(flagged)} drafts flagged")
for name, problems in flagged:
    print(f"  {name}: {problems}")