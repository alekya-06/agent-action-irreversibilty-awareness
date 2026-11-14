# R1 / AB1 / AB2 Handover

## Scope completed

This handoff covers the assigned work:

1. Hidden-state extraction
2. R1 layer-wise reversibility probes and controls
3. AB1: last action token vs mean action tokens
4. AB2: all data vs minimal pairs

No new model inference is required for the consolidated analyses below.

---

## 1. Hidden-state extraction

Models:

- Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct

Seeds:

- 42
- 43
- 44

Environments:

- filesystem
- email_outbox
- wallet
- scienceworld
- toolemu

For every `(state, action)` pair, hidden states were extracted for every model layer.

Both representations were retained:

- `last_action_token` — primary representation
- `mean_action_tokens` — AB1 representation

Extraction QC:

- 3,601 records per run
- 3,601 unique IDs
- expected hidden-state shapes
- all hidden-state values finite
- 7,202 `.npy` files per run
- all six model/seed extraction runs completed successfully

Hidden states are stored under:

`results/rq1/<model>/seed_<seed>/hidden_states/`

The hidden-state directories are intentionally excluded from Git because of their size.

---

## 2. R1: Layer-wise probes

The R1 probe analysis was run for:

- Qwen and Llama
- all five environments
- all layers
- binary and three-class tasks
- three scenario-disjoint seeds/splits
- last-action-token and mean-action-token representations
- all-data and minimal-pair subsets

Metrics:

- balanced accuracy
- macro-F1
- AUROC

Best-layer selection was based on validation balanced accuracy, with validation macro-F1 and then smallest layer as tie-breakers.

Required controls were also evaluated:

- majority baseline
- lexical BOW
- shuffled labels
- random fixed label per action verb
- layer 0 / embedding-layer probe

R1 structural and metric QC passed for all six model/seed runs.

### Important AUROC note

Some three-class test subsets contain only two classes. Their three-class AUROC is therefore undefined and is reported as NaN/NA rather than being treated as zero or 0.5.

This is an evaluation-set property, not a failed experiment.

---

## 3. AB1: representation ablation

AB1 compares:

- last action token
- mean over action tokens

Using the primary binary task and all data:

### Qwen

| Representation | BA | Macro-F1 | AUROC |
|---|---:|---:|---:|
| Last action token | 0.9595 | 0.9619 | 0.9943 |
| Mean action tokens | 0.9500 | 0.9603 | 0.9928 |

### Llama

| Representation | BA | Macro-F1 | AUROC |
|---|---:|---:|---:|
| Last action token | 0.9656 | 0.9733 | 0.9924 |
| Mean action tokens | 0.9381 | 0.9462 | 0.9881 |

Interpretation:

The predefined last-action-token representation provides equal or better overall decoding than mean pooling across the evaluated environments. The advantage is particularly pronounced for Llama. However, the preferred representation varies by environment, so mean pooling is competitive in some settings.

---

## 4. AB2: minimal-pair ablation

AB2 compares:

- all data
- minimal-pair subset

The primary representation (last action token) is used.

### Qwen

| Data condition | BA | Macro-F1 | AUROC |
|---|---:|---:|---:|
| All data | 0.9595 | 0.9619 | 0.9943 |
| Minimal pairs | 0.9132 | 0.9059 | 0.9926 |

### Llama

| Data condition | BA | Macro-F1 | AUROC |
|---|---:|---:|---:|
| All data | 0.9656 | 0.9733 | 0.9924 |
| Minimal pairs | 0.9386 | 0.9384 | 0.9954 |

Interpretation:

Restricting evaluation to minimal pairs generally reduces balanced accuracy and macro-F1 relative to the full dataset, particularly for Qwen, although the effect varies across environments and models. AUROC remains high in both conditions.

Minimal-pair test sets are smaller, so their point estimates should be interpreted with greater caution.

---

## 5. Consolidated result files

All consolidated results are in:

`results/rq1/consolidated/`

Files:

- `r1_ab1_ab2_per_seed.csv`
- `r1_ab1_ab2_three_seed_summary.csv`
- `r1_ab1_ab2_consolidated.json`

QC:

- Raw rows: 240 / 240
- Aggregate rows: 80 / 80
- Consolidation QC: PASSED

The 240 raw rows correspond to:

2 models × 3 seeds × 5 environments × 2 tasks × 2 representations × 2 subsets

The 80 aggregate rows correspond to:

2 models × 5 environments × 2 tasks × 2 representations × 2 subsets

---

## 6. Scope boundary

Completed here:

- hidden-state extraction
- R1
- R1 controls
- AB1
- AB2

Not part of this handoff:

- AB3 cross-environment analysis
- R2 direct stated-answer analysis
- R3 paired agent behavior
- blindness-gap analysis

Those analyses can use the outputs produced here where applicable.

---

## 7. Reproducibility

The extraction and probe scripts are located under:

`requirements/requirement1/`

The consolidated analysis scripts are under:

`scripts/`

No model inference needs to be rerun to reproduce the AB1/AB2 consolidated tables from the saved R1 summaries.
