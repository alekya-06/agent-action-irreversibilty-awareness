# Data card: agent action reversibility dataset

## Summary

3,601 (state, action) pairs from three testbeds, each labelled as reversible,
partially reversible or irreversible at horizons k = 1, 2 and 3. Built to test whether
LLM agents represent action reversibility (probes) and act on it (agent behaviour).
Format: JSON lines, one shared schema for all testbeds.

| Testbed | Rows | Scenarios | Labelling method |
|---|---|---|---|
| Tool sandbox (email, filesystem, wallet) | 1,905 | 225 | exact breadth-first search over the simulator |
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
objects but different temperatures in ScienceWorld).

## Class counts (k = 3)

| Testbed | Reversible | Partial | Irreversible | Inconclusive |
|---|---|---|---|---|
| Sandbox | 1,222 | 256 | 427 (22%) | 0 |
| ScienceWorld | 730 | 7 | 350 (31%) | 42 |
| ToolEmu | 402 | 79 | 86 (15%) | 0 |
| **Total** | **2,354** | **342** | **863 (24%)** | **42** |

## Minimal pairs

Rows sharing a `group_id` describe the same action in the same state and differ only in
the recovery resources stated in `state_text` (`condition`: undo, snapshot, stale
snapshot, trash, undo window, payment hold, version history, backup). 1,915 rows (53%)
are in groups whose rows have different labels, so the same action verb appears with
different reversibility. The stale-snapshot condition mentions a snapshot in both
versions of a pair, so the word alone does not predict the label.

## How it was built

- **Sandbox:** three small Python simulators with save and restore. Each action is run
  under every recovery condition and labelled by exhaustive search for a recovery path.
  21 + 3 + 21 scenarios are hand-written; 180 come from seeded generators (wallets with
  payees, email profiles, random filesystems with deletions and edits). Generators are
  deterministic for a fixed seed.
- **ScienceWorld:** gold and random valid actions at fixed positions along the gold path
  of the first 5 training variations of all 30 tasks; each labelled by breadth-first
  search (budget 400 nodes) for a path back to the pre-action world state.
- **ToolEmu:** for each case, observations and 3-5 candidate tool calls (risky, safe,
  read-only check) were drafted by Qwen2.5-7B/14B-Instruct (4-bit, MLX) and validated
  automatically (schema, grounding of ids, read-only observations). Labels come from a
  tool table covering all 181 tools used, reviewed by hand, plus rules for
  argument-dependent tools (e.g. Terminal commands).

## Splits

80/10/10 train/validation/test, grouped by scenario (sandbox scenario, email profile,
ScienceWorld task, ToolEmu case), for seeds 42, 43 and 44.

## Quality control

- **Human agreement:** 100 pairs (50 sandbox, 30 ScienceWorld, 20 ToolEmu; balanced
  across labels where possible; at most one row per minimal-pair group) were labelled
  independently by two team members who did not see the automatic labels.
  - Cohen's kappa, labeller A vs B: **TBD** (95% CI TBD)
  - A vs automatic: **TBD**; B vs automatic: **TBD**
- **Sandbox:** every environment has unit tests for transitions, recovery and labelling
  (including the edit-file and stale-snapshot pairs).
- **ToolEmu tool table:** all 181 tools checked by hand; 3 labels corrected.

## Known limitations

- Most sandbox scenarios are generated from templates, so they are more alike than
  hand-written ones.
- ScienceWorld has 30 tasks, so each test split covers only 3 tasks.
- ToolEmu scenario text is LLM-drafted and was not reviewed row by row; labels come from
  the tool table, so they do not vary with horizon.
- "Reversible" means restorable with the tools of that environment. Low-harm actions
  without an undo tool are labelled irreversible; reversibility is not harm.
- Labels are bounded by a horizon (3 actions, extended 10 / 3); longer recoveries count
  as partial or irreversible.
- Two labellers, both team members, for the agreement study.

## Licences

Sandbox data: ours. ToolEmu cases: Apache-2.0 (source repository). ScienceWorld: see the
LICENSE file of the ScienceWorld repository; check it before releasing. Generated with
open-weight models only; no paid APIs.