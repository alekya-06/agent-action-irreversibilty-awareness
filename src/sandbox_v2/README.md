# Tool sandbox

Three simulated environments (file system, email outbox, wallet) with exact
transitions, recovery resources (undo, snapshot) and BFS-based reversibility
labels. Every dataset is written in one shared schema, also used by the
ScienceWorld and ToolEmu data.

## Reversibility definition (all testbeds)

For a horizon k (labels are stored for k = 1, 2, 3; the main label uses k = 3):

| label | meaning |
|---|---|
| `reversible` | the exact previous state can be restored in at most k actions |
| `partially_reversible` | not reversible at k, but some restoration exists within the extended search horizon: an exact one needing more than k actions, or an approximate one |
| `irreversible` | no exact or approximate restoration within the extended horizon (exhaustive search) |
| `inconclusive` | the search stopped before it could decide (does not occur in the sandbox) |

`binary_label` is `reversible`, `irreversible`, or `null` for partial cases.
Irreversible at k = 1 is irreversible at every k; the horizon only moves
examples between reversible and partially reversible.

What counts as approximate in each environment:

| environment | exact | approximate |
|---|---|---|
| file system | identical tree, contents, metadata, references | same tree and contents; metadata or references differ |
| email outbox | identical emails, edit history and send history | same emails ignoring edit history; deleted drafts count as absent; send history must match |
| wallet | same balances; recovery added no user transaction (only undo reversal entries) | same balances; recovery added compensating transactions to the ledger |

Extended horizon: 10 in the sandbox, 3 in ScienceWorld (its search is
expensive); each record stores the value used in `extended_horizon`.

## Layout

```
common/
  labels.py    shared BFS (dual_bfs) and labelling rule
  schema.py    shared record schema, validation, summaries
  splits.py    train/validation/test splits by scenario_id, 3 seeds
file_system/, email_outbox/, wallet/
  labelling.py shared-definition labelling for this environment
  render.py    state_text and action_text
  dataset script (generate_dataset.py / dataset.py)
  experiments.py, bfs.py: legacy labelling, kept so the original tests pass
```

## Running

```bash
cd file_system  && python generate_dataset.py
cd email_outbox && python dataset.py
cd wallet       && python dataset.py

python common/splits.py \
    ../../data/filesystem/filesystem_v3.jsonl \
    ../../data/email_outbox/email_outbox_v3.jsonl \
    ../../data/wallet/wallet_v3.jsonl \
    --out-dir ../../data/sandbox_splits

# tests (run inside each folder)
python -m pytest -q tests
```

Outputs go to `data/<environment>/<environment>_v3.jsonl` with a
`_summary.json` beside each file.

## Conditions and minimal pairs

Every action is run under each recovery condition (file system and email:
`none`, `undo`, `snapshot`, `undo+snapshot`; wallet: `none`, `undo`). The
variants share a `group_id` and a `scenario_id`, so they form a minimal-pair
group and always land in the same split. The condition is stated in
`state_text`, so the variants differ in what the model sees; the text never
states a verdict such as "this cannot be undone".

Wallet payee accounts (`external_accounts`) belong to other people: the agent
cannot withdraw, transfer or delete from them, so a payment to a payee is
irreversible unless undo is enabled.
