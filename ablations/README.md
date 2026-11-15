# Ablations

| Ablation | Code | Results |
|---|---|---|
| AB1: last action token vs mean over action tokens | runs inside `requirements/requirement1/run_rq1_probes.py` (both representations); summarised by `analyze_ab1.py` | `results/rq1/consolidated/r1_ab1_ab2_*` |
| AB2: minimal pairs only vs all data | runs inside `requirements/requirement1/run_rq1_probes.py` (`all` and `minimal_pairs` subsets); summarised by `analyze_ab2.py` | `results/rq1/consolidated/r1_ab1_ab2_*` |
| AB3: cross-testbed transfer (sandbox ↔ ScienceWorld) | `ab3_cross_environment.py` (needs the saved hidden states) | `results/rq1/consolidated/ab3_cross_environment_*` |
| AB4: neutral vs "be careful" system prompt | not written yet (needs the R3 agent loop) | `results/rq3/` |

`consolidate_r1_ab1_ab2.py` builds the three-seed R1/AB1/AB2 tables.

Run from the project root as modules, so imports of `requirements.*` resolve:

    python -m ablations.ab3_cross_environment
    python -m ablations.consolidate_r1_ab1_ab2