# Handover for R1 and R3 (13 November 2026)

This page covers only what you need for **R1 (probes)**, **R3 (agent behaviour)** and
their ablations. Everything else (dataset building, R2, kappa) is done or owned by
someone else; you can ignore those files.

## What is already done

- **Dataset:** 3,601 labelled (state, action) pairs from three testbeds, split by
  scenario into train/validation/test for seeds 42, 43 and 44.
- **Prompt builder:** `requirements/prompts.py` (shared by R1, R2 and R3).
- **R3 tasks:** 60 paired agent tasks, each checked automatically.
- **R2** (stated answers) and **K**, the "says" half of the blindness gap, are done.
  R2 still needs R1's probe predictions (see "Output formats").

## Setup (once)

```bash
git pull
uv sync                      # or install from pyproject.toml / uv.lock
git clone https://github.com/ryoungj/ToolEmu external/ToolEmu   # needed for ToolEmu prompts
hf auth login                # Llama-3.1-8B-Instruct is gated
```

Put your code in `requirements/requirement1/` or `requirements/requirement3/` and your
results in `results/rq1/` or `results/rq3/`.

## Rules for every run (from the brief)

- Greedy decoding, fixed seeds, never two models on the GPU at once.
- Models: `Qwen/Qwen2.5-7B-Instruct` (main) and `meta-llama/Llama-3.1-8B-Instruct`, bf16.
- Save every prompt and output to disk; record library versions.
- Report every experiment, including the ones that fail.

---

## R1: probes

### Files you need

| File | What it is |
|---|---|
| `data/sandbox_splits/seed_{42,43,44}/{train,validation,test}.jsonl` | Sandbox rows (email, filesystem, wallet) |
| `data/scienceworld/splits/seed_{42,43,44}/...` | ScienceWorld rows |
| `data/toolemu/splits/seed_{42,43,44}/...` | ToolEmu rows |
| `requirements/prompts.py` | `render_probe_prompt(tokenizer, record, toolemu_specs)` builds the prompt and returns the token ids, **the position of the last action token** and **the positions of all action tokens**. Use `load_toolemu_specs("external/ToolEmu/assets/all_toolkits.json")` for ToolEmu rows. The R1 prompt template is yours to change |
| `src/common/splits.py` | The split code, if you need new splits (e.g. for ablations) |

Each row has: `id`, `environment`, `state_text`, `action_text`, `label`
(reversible / partially_reversible / irreversible), `binary_label`
(reversible / irreversible / `null`), `labels_by_horizon` (`"1"`, `"2"`, `"3"`),
`scenario_id`, `group_id`, `condition`.

### Before you train anything

1. **Drop the 42 rows with `label == "inconclusive"`** (ScienceWorld). They have no label.
2. **Binary task:** drop rows where `binary_label` is `null`. **Three-class task:** use `label`.
3. **Use the split files only.** Never make your own random split (scenarios would leak).
4. **Run all three seeds** and report mean ± sd.
5. **Report balanced accuracy and class counts.** 24% of rows are irreversible.
6. **To pool testbeds, concatenate the same seed's files.** Scenario IDs never overlap.

```python
import json

def load(path, binary=False):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["label"] != "inconclusive"]
    if binary:
        rows = [r for r in rows if r["binary_label"] is not None]
    return rows

seed = 42
train = [r for env in ("sandbox_splits", "scienceworld/splits", "toolemu/splits")
         for r in load(f"data/{env}/seed_{seed}/train.jsonl", binary=True)]
```

### Required (brief Sections 6-7)

- **Step 2, extraction:** one forward pass per row, saving the hidden state at **every
  layer**. **Save both the last action token and the mean over all action tokens now**,
  because ablation AB1 needs both and re-extracting is slow. Hidden states are large;
  don't commit them (add `results/rq1/**/hidden_states/` to `.gitignore`).
- **Probes:** L2-regularised logistic regression per layer (scikit-learn); binary first,
  then three-class. Plot test accuracy vs layer.
- **Controls (all compulsory):** majority class; lexical baseline (bag-of-words on
  `action_text` only, must be beaten on minimal pairs); shuffled labels; Hewitt-Liang
  control task (random fixed label per action verb, gives selectivity); layer-0 probe.
  The majority and lexical baselines need no model, so they can be done first.
- **Metrics:** balanced accuracy, macro-F1, AUROC per layer; selectivity; bootstrap 95%
  CIs (1,000 resamples).

### Your ablations

- **AB1, token position:** last action token vs mean over action tokens (from the extraction).
- **AB2, minimal pairs only vs all data:** "minimal pairs" = rows whose `group_id` has
  more than one row **with different labels** (1,915 rows).
- (AB3, cross-testbed transfer, is owned by someone else, but uses your extraction and
  probe code, so keep them reusable.)
- Optional: horizon (use `labels_by_horizon`; leave ToolEmu out, its labels don't vary
  with k), Qwen 7B vs 14B, binary vs three-class.

### Output format (R2 needs this)

For each model and seed, write the **test-split predictions of the best layer** to
`results/rq1/<model>/seed_<s>/predictions.jsonl`, one line per row:

```json
{"id": "...", "layer": 17, "pred": "irreversible", "prob_irreversible": 0.83}
```

Use the row `id` from the data so it joins with R2's answers.

---

## R3: agent behaviour

### Files you need

| File | What it is |
|---|---|
| `data/agent_tasks/tasks_v1.jsonl` | **The 60 task pairs (120 rows).** One row per version |
| `requirements/prompts.py` | `system_prompt(record, careful=...)` (tool list per environment), `SANDBOX_TOOLS`, and `CAREFUL_INSTRUCTION` (for AB4) |
| `src/sandbox_v2/<env>/transitions.py` | `transition(state, action)` executes one tool call (`<env>` = `file_system`, `email_outbox`, `wallet`) |
| `src/sandbox_v2/<env>/state.py` | The state classes |
| `src/sandbox_v2/<env>/recovery.py` | Undo and snapshot mechanics, for when the agent calls `undo` or `restore_snapshot` |
| `src/sandbox_v2/<env>/labelling.py` | `label_actions(...)`: tells you whether an action the agent took was irreversible in that condition |
| `src/sandbox_v2/<env>/render.py` | `render_state(state, condition)` for showing the state after each step |
| `requirements/requirement3/tasks.py` | How the tasks were built. **Copy its `load_env()` helper**: the three environments use the same module names (`state.py`, `labelling.py`, ...) and collide if imported normally |

### What each task row contains

| Field | Meaning |
|---|---|
| `task_id`, `pair_id`, `version` | `version` is `irreversible` or `reversible`; both versions share a `pair_id` |
| `environment`, `condition` | Which sandbox, and its recovery condition (`none`, `undo`, `snapshot`, `stale_snapshot`, `undo_send`) |
| `instruction` | The user's request (the shortcut is made tempting on purpose) |
| `state_text`, `initial_state` | The starting state as text, and as data to rebuild the sandbox state |
| `shortcut_actions` | The tempting shortcut (irreversible in the `irreversible` version) |
| `success` | How to check the task was done: `paths_absent`, `file_contents`, `sent`, `sent_to` or `paid` |
| `user_reply` | What `ask_user` should return for this task (e.g. "Yes, go ahead.", or the missing bill amount) |
| `max_steps` | 15 |

The two versions of a pair differ **only** in the "Recovery:" line of `state_text`.

### The agent loop (brief Step 5)

- One JSON tool call per step, at most 15 steps. **Count parse failures separately;
  never drop them.**
- Tools: the environment's tools from `prompts.py`, plus **`ask_user`** (returns the
  task's `user_reply`).
- Condition notes: for `undo_send`, set `state.undo_send_window = 3` on the email state;
  the new `recall_email` tool then works for 3 actions after a send. For `undo` and
  `snapshot`, use the environment's `recovery.py`. For `stale_snapshot`, see
  `stale_version()` in `src/sandbox_v2/file_system/labelling.py`.
- New tools exist for the tasks: `copy_file` (filesystem: making a backup) and
  `recall_email` (email).

### What to log per run (brief)

Whether the agent took an irreversible action, and whether it first **checked the
state**, **asked for confirmation** (`ask_user`), **made a backup** (`copy_file`) or
**chose the safer tool** (e.g. moving files to `/to_review` instead of deleting); plus
task success.

### Metrics (brief Section 7.1)

Irreversible-action rate; caution rate before irreversible actions; task success;
over-caution on reversible versions; reversibility sensitivity (caution in the
irreversible version minus the reversible version); **A = fraction of pairs where the
agent is more careful in the irreversible version**; McNemar's test on the paired versions.

### Your ablation

- **AB4:** run all tasks with a neutral system prompt and with
  `system_prompt(record, careful=True)` (adds `CAREFUL_INSTRUCTION`).

### The headline

The blindness gap is **K - A**. K is already computed: **Qwen 0.22, Llama 0.00** (in
`results/rq2/k_*.json`). Your A completes it. Use the same 60 tasks so the two are paired.

---

## Bonus (R4, steering)

Needs R1's best layer and direction (mean irreversible minus mean reversible hidden
state on the training split) and R3's loop and tasks. Start only once both exist.

## Context from R2 that may help

When asked directly, both models judge reversibility at near-chance balanced accuracy,
and their answers depend heavily on how the question is worded (Llama says "reversible"
almost always). Details: `results/rq2/FINDINGS.md`.

## Questions

Data and R2: The full dataset reference is `docs/dataset_handover.md`, but
you shouldn't need it beyond this page.