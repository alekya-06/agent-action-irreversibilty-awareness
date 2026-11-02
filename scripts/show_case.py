"""Print a ToolEmu case and its current draft side by side, for hand review."""
import json
import sys

sys.path.insert(0, "src/toolemu")
from build_dataset import tool_specs          # noqa: E402
from draft_pairs import describe_tools        # noqa: E402

name = sys.argv[1]
case = next(c for c in json.load(open("external/ToolEmu/assets/all_cases.json"))
            if c["name"] == name)
specs = tool_specs(json.load(open("external/ToolEmu/assets/all_toolkits.json")))
draft = next(json.loads(l) for l in open("data/toolemu/pairs/drafts_v2.jsonl")
             if json.loads(l)["case"] == name)

print("=== USER REQUEST ===\n" + case["User Instruction"])
print("\n=== RISKS ===")
for r in case.get("Potential Risky Actions", []):
    print("- " + r)
print("\n=== TOOLS ===\n" + describe_tools(case, specs))
print("\n=== DRAFT OBSERVATIONS ===\n" + draft["observations"])
print("\n=== DRAFT CALLS ===")
for c in draft["calls"]:
    print(f"{c.get('role'):>5} -> {c.get('tool')} {json.dumps(c.get('args'))}")
print("\n=== PROBLEMS ===", draft["draft_problems"])
print("=== RISKY MADE-UP VALUES ===", draft.get("risky_ungrounded"))