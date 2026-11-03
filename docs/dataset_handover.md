# Dataset handover (3 November 2026)

The labelled data for all three testbeds is built, passes `check_dataset`, and has
scenario-disjoint train/validation/test splits for seeds 42, 43 and 44.

## Files

| Testbed | Full file | Rows | Scenarios | Splits |
|---|---|---|---|---|
| Sandbox: email | `data/email_outbox/email_outbox_v5.jsonl` | 460 | 23 profiles | `data/sandbox_splits/seed_{42,43,44}/` (all sandbox files together) |
| Sandbox: filesystem | `data/filesystem/filesystem_v5.jsonl` | 555 | 111 | as above |
| Sandbox: wallet | `data/wallet/wallet_v4.jsonl` | 890 | 91 | as above |
| ScienceWorld | `data/scienceworld/scienceworld_reversibility_v3_shared.jsonl` | 1,129 | 30 tasks (144 task variations) | `data/scienceworld/splits/seed_{42,43,44}/` |
| ToolEmu | `data/toolemu/toolemu_v3.jsonl` | 567 | 144 cases | `data/toolemu/splits/seed_{42,43,44}/` |
| **Total** | | **3,601** | | |

Each split folder has `train.jsonl`, `validation.jsonl` and `test.jsonl` (80/10/10 by
scenario) plus a `manifest.json` with label counts. Do not use files in `legacy/`
folders or `scienceworld_reversibility_v3.jsonl` (raw generator output, old schema).
`scripts/dataset_status.py` prints statistics for every current file.

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
5. **Report balanced accuracy and class counts.** Classes are not balanced (24% irreversible
   overall, 15-31% per testbed).
6. **To pool testbeds, concatenate the same seed's files** (e.g. all three
   `seed_42/train.jsonl`). Scenario IDs are prefixed by testbed, so pooled splits stay
   disjoint.
7. **Leave ToolEmu out of the horizon ablation.** Its labels come from the tool table and
   are identical at k = 1, 2, 3.
8. **Keys in `labels_by_horizon` are strings:** `labels_by_horizon["3"]`, not `[3]`.
9. **Ignore `legacy/` folders and raw generator files.** Use only the files in the table above.
10. **When the data is updated, pull `main` and re-run.** Row IDs of unchanged rows stay
    the same across versions.

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

| Testbed | Reversible | Partially reversible | Irreversible | Inconclusive | Contrastive minimal-pair rows |
|---|---|---|---|---|---|
| Sandbox (all) | 1,222 | 256 | 427 (22%) | 0 | 1,373 |
| ScienceWorld | 730 | 7 | 350 (31%) | 42 | 348 |
| ToolEmu | 402 | 79 | 86 (15%) | 0 | 194 |
| **Total** | **2,354** | **342** | **863 (24%)** | **42** | **1,915 (53%)** |

"Contrastive" rows are in a `group_id` whose rows have different labels; this is the
measure for the brief's 30% minimal-pair requirement.

Test split sizes per seed (42 / 43 / 44): sandbox 174 / 165 / 168 rows with 37 / 35 / 35
irreversible; ScienceWorld 85 / 123 / 122 rows (3 tasks each); ToolEmu 49 / 53 / 51 rows.
Exact label counts per split are in each `manifest.json`.

## Fields you will use

- `label`, `binary_label`: main label. `binary_label` is `None` for partially reversible
  **and** inconclusive rows.
- `labels_by_horizon`: label at horizon 1, 2 and 3 (for the horizon ablation).
- `scenario_id`: what the splits are grouped by. Wallet and filesystem are per scenario,
  email is per profile (all 5 action families of one email stay together), ScienceWorld is
  task-level (`scienceworld:freeze`), ToolEmu is case-level (`toolemu:official_14`).
- `group_id` + `condition`: minimal pairs. Rows sharing a `group_id` differ only in
  `condition`:
  - filesystem: `none`, `undo`, `snapshot`, `undo+snapshot`, `stale_snapshot`
  - email: `none`, `undo`, `snapshot`, `undo+snapshot`
  - wallet: `none`, `undo`
  - ToolEmu: `none` plus one of `undo_send`, `trash`, `payment_hold`, `version_history`,
    `backup` where it applies
- `label_method`: `search` (sandbox, most of ScienceWorld), `rule` (57 ScienceWorld rows,
  "focus on" actions), `design` (ToolEmu, from the tool table).

**Label definition at horizon k (team decision):** an action that can be undone, but only
in more than k steps, is `partially_reversible` at k, not `irreversible`. `irreversible`
means not restorable even within the extended horizon. This is applied the same way in
all three testbeds.

## What is new in this version

- **`edit_file` (filesystem):** irreversible under `none` (no record of the old text),
  reversible under `undo` or `snapshot`. This is the brief's edit-with/without-backup pair.
  Note: the brief calls "no backup, old text still in history" *partially* reversible; in
  our sandbox "no backup" means no record of the old text at all, so it is irreversible.
- **`stale_snapshot` condition (filesystem):** a snapshot exists, but it was saved before
  the affected files were created (marked `[created after the snapshot]` in `state_text`).
  Deletions and edits stay irreversible; renames and moves stay reversible via history.
  These pairs mention a snapshot in both versions, so a probe cannot rely on the word
  "snapshot" alone.
- **Email grouped by profile:** the same email text never appears in both train and test.
- **ToolEmu v2:** the 28 previously dropped calls are labelled (Terminal command rules, an
  IndoorRobot operation rule, and human labels in `data/toolemu/human_labels.csv`, applied
  by `scripts/apply_human_labels.py` to `tool_table_reviewed.csv`).
- **ToolEmu v3, tool table reviewed by hand:** all 181 tools used in the dataset were
  checked against the definition; 3 labels were corrected for consistency: stopping a
  process (`kill`, `pkill`, `killall`) is partially reversible (it can be restarted, but
  its in-memory state is lost), and adding a GitHub collaborator or generating a smart-lock
  access code is partially reversible (access can be revoked, but it was already granted).
  Corrections live in `data/toolemu/human_labels.csv` and `src/toolemu/effects.py`.

## Known gaps and caveats

1. **Much of the sandbox is generated.** Of 225 sandbox scenarios, 180 come from seeded
   generators (70 wallet with payees, 20 email profiles, 60 filesystems, 30 file edits).
   They are more alike than hand-written scenarios:
   - Wallet: generated scenarios skip read-only actions; payee payments are the
     irreversible case.
   - Email: generated profiles vary content; every profile runs the same 5 action families.
   - Filesystem: generated scenarios are mostly deletions and edits.
2. **ScienceWorld:**
   - 42 rows are `inconclusive` (see "Before you use the data").
   - Only 30 tasks, so each test split has 3 tasks. Expect high variance across seeds.
   - v3 adds variations 2-4 of every task to v2 (variations 0-1); the 451 v2 rows are
     unchanged and each task stays in the same split as in v2.
3. **ToolEmu:**
   - Labels come from the tool table, so they are the same at every horizon.
   - Scenarios were drafted by 4-bit Qwen2.5 7B/14B and validated automatically, but not
     reviewed by hand; `details.reviewed` is false for all rows.
   - Labelling choices to note:
     - Running a script (`bash`, `python`, ...) is irreversible because its effects are
       unknown.
     - Low-harm actions with no undo tool (a reminder, a Slack channel, a deepfake video)
       are irreversible, because reversibility is not harm.
     - Overwrites and deletions are partially reversible when the old values are visible
       in the observations and a create/update tool can write them back. The sandbox shows
       only file sizes, never contents, so there the same actions are irreversible.
   - Most ToolEmu risky/safe pairs contrast *harm*, not reversibility. Use the `condition`
     variants for reversibility minimal pairs.
4. **Not in the sandbox:** an "expired undo" condition (undo available, but not for this
   action). Left out because it does not fit single-action labelling.
5. **Quality control:** the 100-pair blind double labelling (Cohen's kappa) is in progress
   (`scripts/kappa.py`, sheets in `data/qc/`).

## Not built yet (ablation-specific)

- Leave-one-environment-out split mode
- Minimal-pairs-only filter

`src/sandbox_v2/common/splits.py` works on any shared-schema file and is the natural
place for both.

## Cleanup for later

- Move `src/sandbox_v2/common/` to `src/common/` and update the import paths
  (`grep -rn "common" src scripts --include='*.py'`).