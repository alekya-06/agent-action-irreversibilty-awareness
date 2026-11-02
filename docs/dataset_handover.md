# Dataset handover (2 November 2026)

The labelled data for all three testbeds is built, passes `check_dataset`, and has
scenario-disjoint train/validation/test splits for seeds 42, 43 and 44. You can start
hidden-state extraction and probing now.

More ScienceWorld rows are generating overnight. When they are merged, the ScienceWorld
splits keep the same task assignment (splits group by task, same seeds), so new rows
join their task's existing side. Re-run extraction on the new rows only.

## Files

| Testbed | Full file | Rows | Scenarios | Splits |
|---|---|---|---|---|
| Sandbox: email | `data/email_outbox/email_outbox_v4.jsonl` | 460 | 115 | `data/sandbox_splits/seed_{42,43,44}/` (all sandbox files together) |
| Sandbox: filesystem | `data/filesystem/filesystem_v4.jsonl` | 324 | 81 | as above |
| Sandbox: wallet | `data/wallet/wallet_v4.jsonl` | 890 | 91 | as above |
| ScienceWorld | `data/scienceworld/scienceworld_reversibility_v2_shared.jsonl` | 451 | 30 tasks | `data/scienceworld/splits/seed_{42,43,44}/` |
| ToolEmu | `data/toolemu/toolemu_v1.jsonl` | 539 | 143 cases | `data/toolemu/splits/seed_{42,43,44}/` |
| **Total** | | **2,664** | | |

Each split folder has `train.jsonl`, `validation.jsonl` and `test.jsonl` (80/10/10 by
scenario) plus a `manifest.json` with label counts. Do not use files in `legacy/`
folders, `scienceworld_reversibility_v2.jsonl` (raw generator output) or
`scienceworld_reversibility_v2_expanded.jsonl` (overnight run, unfinished).
`scripts/dataset_status.py` prints statistics for every current file.

## Label counts (horizon 3)

| Testbed | Reversible | Partially reversible | Irreversible | Inconclusive | Contrastive minimal-pair rows |
|---|---|---|---|---|---|
| Sandbox (all) | 1,110 | 250 | 314 (19%) | 0 | 1,164 |
| ScienceWorld | 285 | 3 | 144 (32%) | 19 | 132 |
| ToolEmu | 395 | 59 | 85 (16%) | 0 | 194 |
| **Total** | **1,790** | **312** | **543 (20%)** | **19** | **1,490 (56%)** |

"Contrastive" rows are in a `group_id` whose rows have different labels; this is the
measure for the brief's 30% minimal-pair requirement.

Sandbox test splits have 30-35 irreversible rows per seed (196, 192, 188 test rows
for seeds 42, 43, 44).

## Fields you will use

- `label`, `binary_label`: main label. `binary_label` is `None` for partially reversible
  **and** inconclusive rows (see caveats).
- `labels_by_horizon`: label at horizon 1, 2 and 3 (for the horizon ablation).
- `scenario_id`: what the splits are grouped by. Wallet and filesystem are per scenario,
  email is per (profile, action family), ScienceWorld is task-level
  (`scienceworld:freeze`), ToolEmu is case-level (`toolemu:official_14`).
- `group_id` + `condition`: minimal pairs. Rows sharing a `group_id` differ only in
  `condition` (filesystem and email: `none`, `undo`, `snapshot`, `undo+snapshot`;
  wallet: `none`, `undo`; ToolEmu: `undo_send`, `trash`, `payment_hold`,
  `version_history`, `backup`).
- `label_method`: `search` (sandbox, most of ScienceWorld), `rule` (24 ScienceWorld rows),
  `design` (ToolEmu, from the tool table).

**Label definition at horizon k (team decision):** an action that can be undone, but only
in more than k steps, is `partially_reversible` at k, not `irreversible`. `irreversible`
means not restorable even within the extended horizon. This is applied the same way in
all three testbeds.

## Known gaps and caveats

1. **Sandbox v4 is largely generated.** Of the 287 sandbox scenarios, 150 come from
   seeded generators (70 wallet with payees, 20 email profiles, 60 filesystems). They are
   more alike than hand-written scenarios:
   - Wallet: generated scenarios skip read-only actions (`check_balance`,
     `view_transactions`); payee payments are the irreversible case.
   - Email: generated profiles vary content (topic, sender, recipients); every profile
     runs the same 5 action families.
   - Filesystem: generated scenarios are mostly deletions; each gives at most 1
     irreversible row in 4, because undo and snapshot restore deletions.
2. **Email scenario grouping (decision pending):** `scenario_id` is per (profile, action
   family), so the same email text can appear in train and test with different actions.
   Grouping by profile would be stricter.
3. **ScienceWorld:**
   - 19 rows have label `inconclusive` (`search.status == "budget_exhausted"`, the search
     did not finish). **Filter them out**; otherwise `binary_label = None` makes them look
     partially reversible.
   - Only 30 tasks, so each test split has 3 tasks. Expect high variance across seeds.
4. **ToolEmu:**
   - Labels come from the tool table, so they are the same at every horizon. Leave ToolEmu
     out of the horizon ablation.
   - Scenarios were drafted by 4-bit Qwen2.5 7B/14B and validated automatically. Human
     review is in progress; `details.reviewed` marks reviewed rows (currently all false).
   - 28 calls were dropped because the tool table has no rule for them yet.
   - Most ToolEmu risky/safe pairs contrast *harm*, not reversibility. Use the `condition`
     variants for reversibility minimal pairs.
5. **Not yet in the sandbox:** `edit_file` with and without a backup (a minimal pair named
   in the brief), and recovery conditions that exist but do not help (expired undo,
   stale snapshot).
6. **Quality control:** the 100-pair blind double labelling (Cohen's kappa) is not done yet.

## Not built yet (ablation-specific)

- Leave-one-environment-out split mode
- Minimal-pairs-only filter

`src/sandbox_v2/common/splits.py` works on any shared-schema file and is the natural
place for both.

## Cleanup for later

- Move `src/sandbox_v2/common/` to `src/common/` and update the import paths
  (`grep -rn "common" src scripts --include='*.py'`).