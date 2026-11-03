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
folders or `scienceworld_reversibility_v2_expanded.jsonl` (raw generator output, old
schema). `scripts/dataset_status.py` prints statistics for every current file.

## Label counts (horizon 3)

| Testbed | Reversible | Partially reversible