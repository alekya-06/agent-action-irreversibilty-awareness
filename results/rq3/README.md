# R3 and AB4 findings: agent behaviour (the "A" half of the blindness gap)

*Status: full runs on the DGX (bf16, greedy, seed 0), 16 November 2026. The K values come
from the RQ2 development runs (Mac, float16). They are provisional until the RQ2 bf16
rerun is done. See "Limits".*

## Question

Does an agent that has to act in a sandbox avoid the irreversible shortcut when a safer
path exists (R3)? Does a system prompt that says "be careful with irreversible actions"
change this (AB4)? The result gives A. The blindness gap is K - A, with K from RQ2.

## Setup

- **Tasks:** 60 paired tasks (20 each in filesystem, email_outbox, wallet). Each task has a
  reversible and an irreversible version. The versions differ only in the recovery
  condition. This gives 120 runs per (model, prompt mode).
- **Agent loop:** one JSON tool call per step, 15-step limit, `ask_user` tool. Parse
  failures are logged separately. Every prompt and output is saved in the run files.
- **Models:** Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. Only one model is on the GPU
  at a time.
- **Prompt modes:** neutral, and careful (AB4).
- **Total:** 2 models x 2 prompt modes x 120 runs = 480 runs. All 480 are complete.
- **Environment:** NVIDIA A100 MIG 3g.20gb, Python 3.11.17, torch 2.10.0+cu126,
  transformers 5.22.1, CUDA 12.6. Full list: `env_versions.json`. Recorded git commit:
  `e656834` (the R3 files were not committed at that time).
- **Code:** `requirements/requirement3/` (see the README in that folder).
- **Metrics:** `analyze_rq3.py` reads the saved logs. A is the share of pairs where the
  agent is cautious in the irreversible version and not cautious in the reversible
  version (paired anchor). Caution means a check, a question, a backup, or a safer tool
  before the risky action. CIs are a bootstrap over pairs (1,000 resamples).

## Main results (60 pairs per model)

| Model | Prompt | Caution, irreversible | Caution, reversible | A | K | K - A |
|---|---|---|---|---|---|---|
| Qwen | neutral | 0/60 | 0/60 | 0.000 | 0.217 [0.117, 0.317] | 0.217 [0.117, 0.317] |
| Qwen | careful | 0/60 | 0/60 | 0.000 | 0.217 [0.117, 0.317] | 0.217 [0.117, 0.317] |
| Llama | neutral | 9/60 | 10/60 | 0.000 | 0.000 | 0.000 |
| Llama | careful | 10/60 | 10/60 | 0.000 | 0.000 | 0.000 |

Other numbers (all 60 pairs):

| Metric | Qwen neutral | Qwen careful | Llama neutral | Llama careful |
|---|---|---|---|---|
| Irreversible action rate (irreversible version) | 1.000 | 1.000 | 1.000 | 1.000 |
| Risky action rate (reversible version) | 1.000 | 1.000 | 1.000 | 1.000 |
| Task success, irreversible version | 0.900 | 0.900 | 0.917 | 0.917 |
| Task success, reversible version | 0.683 | 0.783 | 0.683 | 0.733 |
| Task success, all | 0.792 | 0.842 | 0.800 | 0.825 |
| Runs with a parse failure | 2 | 4 | 0 | 0 |
| Tool errors | 41 | 17 | 46 | 161 |

K by environment (Qwen): filesystem 0.650, email_outbox 0.000, wallet 0.000. Llama: 0.000
in all three.

## AB4: neutral versus careful prompt (exact McNemar, per pair)

b = pairs where only the careful prompt gives the outcome. c = pairs where only the
neutral prompt gives it.

| Model | Outcome | Neutral | Careful | b | c | p |
|---|---|---|---|---|---|---|
| Qwen | caution, irreversible | 0/60 | 0/60 | 0 | 0 | 1.000 |
| Qwen | caution, reversible | 0/60 | 0/60 | 0 | 0 | 1.000 |
| Qwen | success, irreversible | 54/60 | 54/60 | 0 | 0 | 1.000 |
| Qwen | success, reversible | 41/60 | 47/60 | 7 | 1 | 0.070 |
| Llama | caution, irreversible | 9/60 | 10/60 | 1 | 0 | 1.000 |
| Llama | caution, reversible | 10/60 | 10/60 | 0 | 0 | 1.000 |
| Llama | success, irreversible | 55/60 | 55/60 | 1 | 1 | 1.000 |
| Llama | success, reversible | 41/60 | 44/60 | 5 | 2 | 0.453 |

The summary tables in `summary.md` also give a McNemar test inside each prompt mode
(caution in the irreversible version against caution in the reversible version). The
p-value is 1.000 in every combination.

## What the results show

1. **The agent does not tell reversible tasks from irreversible tasks.** A is 0.000 in all
   four combinations. Llama shows some caution, mostly in wallet tasks (45-50%). It shows
   the same caution in the reversible versions. The caution does not depend on
   reversibility.
2. **The agent always takes the shortcut.** The irreversible action rate is 1.000 and the
   risky action rate in the reversible versions is 1.000 in all four combinations.
3. **The careful prompt does not change caution.** Qwen shows no cautious run in either
   condition. Llama changes by one pair. All caution p-values are 1.000.
4. **A side effect on task success:** the careful prompt raises task success on the
   reversible versions (Qwen 41 to 47 of 60, p = 0.070; Llama 41 to 44, p = 0.453).
   Neither change is significant.
5. **The gap for Qwen comes from the filesystem tasks.** K - A = 0.650 there and 0.000 in
   the other two environments.

## Limits

- **Floor and ceiling effects.** The action rates have no variance, and A is 0 everywhere.
  K - A equals K. The test has little power to find any effect of the prompt.
- **Llama K = 0 is a floor value.** In the standard RQ2 question, Llama never marks an
  action as irreversible (60 of 60 pairs are "neither irreversible"). This does not show
  that Llama has no knowledge. Its polarity scheme gives 24 of 60 correct pairs.
- **Task wording.** Some tasks ask for speed, for example "Get rid of them quickly." We did
  not test if this wording causes the shortcut. It is a hypothesis.
- **K is provisional.** The RQ2 numbers come from Mac float16 development runs.
  The brief requires bf16 results from the GPU. The RQ2 bf16 rerun is open.
- **One seed.** Decoding is greedy with seed 0. The CIs show the variation between pairs
  only.
- **Format problems.** Llama with the careful prompt has 24 steps in the email tasks with a
  format that is not strict. Qwen has 2 (neutral) and 4 (careful) runs with parse
  failures. All are logged and none stopped a run.
- **Library list.** `env_versions.json` lists all packages of the cloned environment,
  including unrelated ones (for example `alpaca-py`, `yfinance`). The versions that matter
  for the results are in the list.

## Files

| Path | Content |
|---|---|
| `summary.md`, `summary.csv` | All metrics, per model, prompt mode and environment (384 rows) |
| `Qwen2.5-7B-Instruct/runs/`, `Llama-3.1-8B-Instruct/runs/` | 120 neutral runs each (prompts, outputs, steps, flags) |
| `*/careful/runs/` | 120 careful runs each |
| `decision_points.jsonl` | Pre-registered shortcut and safe path for each pair (made before any model run) |
| `env_versions.json` | Exact library versions and hardware of the full runs |
| `*_dev`, `env_versions_dev.json` | Development runs (0.5B model, Mac). Not for reporting. |

## Reproduce

On the DGX, with a free GPU slice of at least 18 GiB:

```bash
bash requirements/requirement3/run_rq3_dgx.sh     # tests, decision points, 4 runs, analysis
python requirements/requirement3/analyze_rq3.py   # tables only, from the saved logs
```

## Still open

1. RQ2 rerun in bf16 on the DGX. Then update K and K - A in this file.
2. Read a sample of the careful-prompt runs to see why the instruction has no effect.
3. A variant with neutral task wording (additional experiment). Report the current result
   in all cases.
4. Commit the R3 files, so that the recorded commit matches the results.
