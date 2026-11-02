"""Replace a draft's observations and calls with a hand-written fix, after validating it."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src/toolemu")
from build_dataset import tool_specs                          # noqa: E402
from draft_pairs import risky_ungrounded, validate_draft     # noqa: E402

DRAFTS = Path("data/toolemu/pairs/drafts_v2.jsonl")
name = sys.argv[1]
fix = json.loads(Path(f"data/toolemu/pairs/fixes/{name}.json").read_text(encoding="utf-8"))
case = next(c for c in json.load(open("external/ToolEmu/assets/all_cases.json"))
            if c["name"] == name)
specs = tool_specs(json.load(open("external/ToolEmu/assets/all_toolkits.json")))

problems = validate_draft(fix, case, specs)
if problems:
    sys.exit("Fix still has problems; drafts file NOT changed:\n- " + "\n- ".join(problems))

text = DRAFTS.read_text(encoding="utf-8")
DRAFTS.with_name(DRAFTS.stem + ".before_fix.jsonl").write_text(text, encoding="utf-8")
out = []
for line in text.splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r["case"] == name:
        r.update(observations=fix["observations"], calls=fix["calls"],
                 draft_problems=[], raw_output=None, hand_edited=True,
                 risky_ungrounded=risky_ungrounded(fix, case),
                allow_observation_writes=fix.get("allow_observation_writes", []),)
    out.append(json.dumps(r, ensure_ascii=False))
DRAFTS.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"{name}: fixed. Risky made-up values: {r_vals if (r_vals := risky_ungrounded(fix, case)) else 'none'}")