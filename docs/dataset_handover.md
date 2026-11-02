# Dataset handover (2 November 2026)

The labelled data for all three testbeds is built, passes `check_dataset`, and has
scenario-disjoint train/validation/test splits for seeds 42, 43 and 44. You can start
hidden-state extraction and probing now. More sandbox rows will be added later
(see Known gaps); re-run extraction on the new rows only.

## Files

| Testbed | Full file | Rows | Scenarios | Splits |
|---|---|---|---|---|
| Sandbox: email | `data/email_outbox/email_outbox_v3.jsonl` | 60 | 15 | `data/sandbox_splits/seed_{42,43,44}/` (all sandbox files together) |
| Sandbox: filesystem | `data/filesystem/filesystem_v3.jsonl` | 84 | 21 | as above |
| Sandbox: wallet | `data/wallet/wallet_v3.jsonl` | 216 | 21 | as above |
| ScienceWorld | `data/scienceworld/scienceworld_reversibility_v2_shared.jsonl` | 451 | 30 tasks | `data/scienceworld/splits/seed_{42,43,44}/` |
| ToolEmu | `data/toolemu/toolemu_v1.jsonl` | 539 | 143 cases | `data/toolemu/splits/seed_{42,43,44}/` |
| **Total** | | **1,350** | | |

Each split folder has `train.jsonl`, `validation.jsonl` and `test.jsonl` (80/10/10 by
scenario) plus a `manifest.json` with label counts. Do not use
`scienceworld_reversibility_v2.jsonl` (raw search output, old schema).
`scripts/dataset_status.py` prints these statistics for every file.

## Label counts (horizon 3)

| Testbed | Reversible | Partially reversible | Irreversible | Inconclusive | Contrastive minimal-pair rows |
|---|---|---|---|---|---|
| Sandbox (all) | 251 | 84 | 25 (7%) | 0 | 198 |
| ScienceWorld | 285 | 3 | 144 (32%) | 19 | 132 |
| ToolEmu | 395 | 59 | 85 (16%) | 0 | 194 |
| **Total** | **931** | **146** | **254 (19%)** | **19** | **524 (39%)** |

"Contrastive" rows are in a `group_id` whose rows have different labels; this is the
measure for the brief's 30% minimal-pair requirement.

## Fields you will use

- `label`, `binary_label`: main label. `binary_label` is `None` for partially reversible
  **and** inconclusive rows (see caveat 4).
- `labels_by_horizon`: label at horizon 1, 2 and 3 (for the horizon ablation).
- `scenario_id`: what the splits are grouped by. Sandbox is per scenario, ScienceWorld is
  task-level (`scienceworld:freeze`), ToolEmu is case-level (`toolemu:official_14`).
- `group_id` + `condition`: minimal pairs. Rows sharing a `group_id` differ only in
  `condition` (sandbox: `undo`, `snapshot`; ToolEmu: `undo_send`, `trash`,
  `payment_hold`, `version_history`, `backup`).
- `label_method`: `search` (sandbox, most of ScienceWorld), `rule` (24 ScienceWorld rows),
  `design` (ToolEmu, from the tool table).

**Label definition at horizon k (team decision):** an action that can be undone, but only
in more than k steps, is `partially_reversible` at k, not `irreversible`. `irreversible`
means not restorable even within the extended horizon. This is applied the same way in
all three testbeds.

## Known gaps and caveats

1. **Sandbox test splits have almost no irreversible rows** (1 per seed). A binary probe
   cannot be evaluated on the sandbox alone until scaling adds irreversible scenarios.
   Pool all three testbeds, or report sandbox results three-class only, and say so.
2. **Size:** 1,350 rows, below the 2,000 target. Sandbox scaling is in progress.
3. **Class balance:** irreversible is 7% in the sandbox and 19% overall. Report balanced
   accuracy and class counts.
4. **ScienceWorld:**
   - 19 rows have label `inconclusive` (`search.status == "budget_exhausted"`, the search
     did not finish). **Filter them out**; otherwise `binary_label = None` makes them look
     partially reversible.
   - Only 30 tasks, so each test split has 3 tasks. Expect high variance across seeds.
5. **ToolEmu:**
   - Labels come from the tool table, so they are the same at every horizon. Leave ToolEmu
     out of the horizon ablation.
   - Scenarios were drafted by 4-bit Qwen2.5 7B/14B and validated automatically. Human
     review is in progress; `details.reviewed` marks reviewed rows (currently all false).
   - 28 calls were dropped because the tool table has no rule for them yet.
   - Most ToolEmu risky/safe pairs contrast *harm*, not reversibility. Use the `condition`
     variants for reversibility minimal pairs.
6. **Quality control:** the 100-pair blind double labelling (Cohen's kappa) is not done yet.

## Not built yet (ablation-specific)

- Leave-one-environment-out split mode
- Minimal-pairs-only filter

`src/sandbox_v2/common/splits.py` works on any shared-schema file and is the natural
place for both.

## Cleanup for later

- Move `src/sandbox_v2/common/` to `src/common/` and update the import paths
  (`grep -rn "common" src scripts --include='*.py'`).