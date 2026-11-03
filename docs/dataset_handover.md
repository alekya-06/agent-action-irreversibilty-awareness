# Dataset handover (3 November 2026)

The labelled data for all three testbeds is built, passes `check_dataset`, and has
scenario-disjoint train/validation/test splits for seeds 42, 43 and 44. You can start
hidden-state extraction and probing now.

## Files

| Testbed | Full file | Rows | Scenarios | Splits |
|---|---|---|---|---|
| Sandbox: email | `data/email_outbox/email_outbox_v4.jsonl` | 460 | 115 | `data/sandbox_splits/seed_{42,43,44}/` (all sandbox files together) |
| Sandbox: filesystem | `data/filesystem/filesystem_v4.jsonl` | 324 | 81 | as above |
| Sandbox: wallet | `data/wallet/wallet_v4.jsonl` | 890 | 91 | as above |
| ScienceWorld | `data/scienceworld/scienceworld_reversibility_v3_shared.jsonl` | 1,129 | 30 tasks (144 task variations) | `data/scienceworld/splits/seed_{42,43,44}/` |
| ToolEmu | `data/toolemu/toolemu_v1.jsonl` | 539 | 143 cases | `data/toolemu/splits/seed_{42,43,44}/` |
| **Total** | | **3,342** | | |

Each split folder has `train.jsonl`, `validation.jsonl` and `test.jsonl` (80/10/10 by
scenario) plus a `manifest.json` with label counts. Do not use files in `legacy/`
folders or `scienceworld_reversibility_v3_expanded.jsonl` (raw generator output, old
schema). `scripts/dataset_status.py` prints statistics for every current file.

## Before you use the data (read this)

1. **Drop the 42 `inconclusive` ScienceWorld rows.** Their search ran out of budget, so
   they have no real label. `binary_label` is `None` for them, which makes them look
   partially reversible if you don't filter them.
2. **For binary tasks, drop rows where `binary_label` is `None`.** That removes partially
   reversible rows (and the inconclusive ones). Use `label` for the three-class task.
3. **Use the split files, never your own random split.** Splits are grouped by
   `scenario_id`; a random split leaks scenarios between train and test and inflates scores.
4. **Run all three seeds (42, 43, 44) and report mean ± sd.** ScienceWorld has only
   3 test tasks per seed, so single-seed numbers are noisy.
5. **Report balanced accuracy and class counts.** Classes are not balanced (22% irreversible
   overall, 16-31% per testbed).
6. **To pool testbeds, concatenate the same seed's files** (e.g. all three
   `seed_42/train.jsonl`). Scenario IDs are prefixed by testbed, so pooled splits stay
   disjoint.
7. **Leave ToolEmu out of the horizon ablation.** Its labels come from the tool table and
   are identical at k = 1, 2, 3.
8. **Keys in `labels_by_horizon` are strings:** `labels_by_horizon["3"]`, not `[3]`.
9. **Ignore `legacy/` folders and raw generator files** (`scienceworld_reversibility_v2_expanded.jsonl`).
   Use only the files in the table above.
10. **When the data is updated, pull `main` and re-run on new IDs only.** Row IDs are stable
    across versions.

```python
import json

def load(path, binary=False):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["label"] != "inconclusive"]          # item 1
    if binary:
        rows = [r for r in rows if r["binary_label"] is not None]      # item 2
    return rows

seed = 42
train = [r for env in ("sandbox_splits", "scienceworld/splits", "toolemu/splits")
         for r in load(f"data/{env}/seed_{seed}/train.jsonl", binary=True)]  # item 6
```

## Label counts (horizon 3)

| Testbed | Reversible | Partially reversible