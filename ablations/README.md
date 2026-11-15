# Ablations

| Ablation | Code | Results |
|---|---|---|
| AB1: last action token vs mean over action tokens | `requirements/requirement1/run_rq1_probes.py` (see note); summarised by `analyze_ab1.py` | `results/rq1/consolidated/r1_ab1_ab2_*` |
| AB2: minimal pairs only vs all data | `requirements/requirement1/run_rq1_probes.py` (see note); summarised by `analyze_ab2.py` | `results/rq1/consolidated/r1_ab1_ab2_*` |
| AB3: cross-testbed transfer (sandbox ↔ ScienceWorld) | `ab3_cross_environment.py` (needs the saved hidden states) | `results/rq1/consolidated/ab3_cross_environment_*` |
| AB4: neutral vs "be careful" system prompt | not written yet (needs the R3 agent loop) | `results/rq3/` |

`consolidate_r1_ab1_ab2.py` builds the three-seed R1/AB1/AB2 tables.

## Note: AB1 and AB2 live in the R1 code

AB1 and AB2 are not separate scripts, because they are not separate experiments: they
are extra settings of the same R1 probe run.

- **AB1** runs the R1 probe on both saved representations (`last_action_token` and
  `mean_action_tokens`). Hidden-state extraction saves both in one pass, and
  `run_rq1_probes.py` loops over them.
- **AB2** runs the same probe on two subsets of the data (`all` and `minimal_pairs`)
  inside the same loop, with the same layer selection and controls.

Splitting them out would mean duplicating the probe, layer-selection and evaluation
code, and the copies could drift apart. Keeping them in one place guarantees that the
R1 result and both ablations use exactly the same procedure. This folder holds only
the scripts that summarise AB1 and AB2 (`analyze_ab1.py`, `analyze_ab2.py`); to
re-run them, run `requirements/requirement1/run_rq1_probes.py`.

## Running

Run from the project root as modules, so imports of `requirements.*` resolve:

    python -m ablations.ab3_cross_environment
    python -m ablations.consolidate_r1_ab1_ab2
    python -m ablations.analyze_ab1
    python -m ablations.analyze_ab2