# Data card: agent action reversibility dataset

## Summary

3,601 (state, action) pairs from three testbeds, each labelled as reversible, partially
reversible or irreversible at horizons k = 1, 2 and 3. Built to test whether LLM agents
represent action reversibility (probes) and act on it (agent behaviour). Format: JSON
lines, one shared schema for all testbeds.

| Testbed | Rows | Scenarios | Labelling method |
|---|---|---|---|
| Tool sandbox (email, filesystem, wallet) | 1,905 | 225 | exhaustive breadth-first search over the simulator |
| ScienceWorld | 1,129 | 30 tasks, 144 variations | breadth-first search over the game, plus a rule for "focus on" |
| ToolEmu (144 curated cases) | 567 | 144 cases | design: a hand-reviewed tool table plus argument rules |

## Label definition

For horizon k (main label: k = 3):

- **reversible:** the exact previous state can be restored in at most k actions.
- **partially reversible:** not reversible at k, but a restoration exists within the
  extended horizon (an exact one needing more than k actions, or an approximate one).
- **irreversible:** no exact or approximate restoration within the extended horizon.
- **inconclusive:** the search stopped before deciding (42 ScienceWorld rows only).

Extended horizon: 10 in the sandbox, 3 in ScienceWorld. "Approximate" is defined per
environment (e.g. same balances but compensating ledger entries in the wallet; same
objects but different temperatures in ScienceWorld). "Reversible" means restorable with
the tools of that environment.

## Class counts (k = 3)

| Testbed | Reversible | Partial | Irreversible | Inconclusive |
|---|---|---|---|---|
| Sandbox | 1,222 | 256 | 427 (22%) | 0 |
| ScienceWorld | 730 | 7 | 350 (31%) | 42 |
| ToolEmu | 402 | 79 | 86 (15%) | 0 |
| **Total** | **2,354** | **342** | **863 (24%)** | **42** |

## Minimal pairs

Rows sharing a `group_id` describe the same action in the same state and differ only in
the recovery resources stated in `state_text` (`condition`). 1,915 rows (53%) are in
groups whose rows have different labels, so the same action appears with different
reversibility. Two designs make the pairs harder to solve from surface words:
- **stale snapshot:** a snapshot is mentioned in both versions, but in one it predates
  the affected files and cannot restore them;
- **edit with or without a backup** (`edit_file`).

## How it was built

- **Sandbox:** three small Python simulators (file system, email outbox, wallet) with
  exact transitions and save/restore. Each action is run under every recovery condition
  and labelled by exhaustive search for a recovery path. 45 scenarios are hand-written
  (21 wallet, 3 email profiles, 21 filesystem); 180 come from seeded generators (wallets
  with payees, email profiles, random filesystems with deletions and edits). Every
  environment has unit tests for transitions, recovery and labelling.
- **ScienceWorld:** gold and random valid actions at fixed positions along the gold path
  of the first 5 training variations of all 30 tasks, each labelled by breadth-first
  search (budget 400 nodes) for a path back to the pre-action world state. "Focus on"
  actions are labelled irreversible by rule (they commit the task's answer).
- **ToolEmu:** for each of the 144 curated cases, observations and 3-5 candidate tool
  calls (risky, safe, read-only check) were drafted by Qwen2.5-7B/14B-Instruct (4-bit,
  run locally) and validated automatically (schema, grounding of ids, read-only
  observations); flagged drafts were redrafted or fixed by hand. Labels come from a tool
  table covering all 181 tools used, reviewed by hand (3 labels corrected), plus rules for
  argument-dependent tools such as Terminal commands. ToolEmu's GPT-4 emulator was not
  used.

No paid APIs were used; all models are open-weight and ran locally.

## Splits

80/10/10 train/validation/test, grouped by scenario (sandbox scenario, email profile,
ScienceWorld task, ToolEmu case), for seeds 42, 43 and 44. No scenario appears in more
than one split.

## Quality control

- **Sandbox labels** come from exhaustive search over exact simulators and are covered by
  unit tests, including the edit-file, stale-snapshot and undo-send cases.
- **ToolEmu labels:** the full tool table was reviewed by hand for consistency with the
  definition.
- **ToolEmu drafts:** validated automatically; not reviewed row by row.
- **Human agreement (Cohen's kappa):** a 100-pair blind double-labelling study is
  prepared (sampling script and labelling sheets) but has not been completed.

## Known limitations

- Most sandbox scenarios are generated from templates, so they are more alike than
  hand-written ones.
- ScienceWorld has 30 tasks, so each test split covers only 3 tasks.
- ToolEmu scenario text is LLM-drafted; its labels come from the tool table and do not
  vary with the horizon.
- In the email sandbox, sending is irreversible under every dataset condition: there is
  no "send with an undo window" variant in this dataset (it exists only in the agent
  tasks).
- The filesystem labels were computed before a copy tool existed; where two identical
  files exist, deleting one may be labelled irreversible although it could be restored by
  copying the other.
- Low-harm actions without an undo tool are labelled irreversible: reversibility is not
  harm.
- Labels are bounded by a horizon (3 actions, extended 10 / 3); longer recoveries count
  as partial or irreversible.
- The human agreement study is not completed.

## Licences

Sandbox data and code: ours. ToolEmu cases: Apache-2.0 (source repository).
ScienceWorld: ScienceWorld: Apache-2.0 (per its repository's LICENSE file; confirm before releasing).
Models used: Qwen2.5 (Apache-2.0),
Llama 3.1 (Llama 3.1 Community License).