
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

# This script should be run from the root of your Git repository.
REPO_ROOT = Path.cwd()

INPUT_CSV = (
    REPO_ROOT
    / "results"
    / "rq1"
    / "consolidated"
    / "r1_ab1_ab2_three_seed_summary.csv"
)

OUTPUT_DIR = REPO_ROOT / "ab7_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("AB7: BINARY VS THREE-CLASS LABEL COMPARISON")
print("=" * 65)


# ============================================================
# 2. LOAD DATA
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"\nCould not find the input CSV:\n{INPUT_CSV}\n"
        "Make sure you run this script from the repository root."
    )

df = pd.read_csv(INPUT_CSV)

print("\nInput file loaded successfully.")
print("Number of rows:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "model",
    "environment",
    "task",
    "representation",
    "subset",
    "balanced_accuracy_mean",
    "balanced_accuracy_sd",
    "macro_f1_mean",
    "macro_f1_sd",
    "auroc_mean",
    "auroc_sd",
    "auroc_n",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nThe CSV is missing these columns: {missing_columns}"
    )


# ============================================================
# 4. FILTER THE AB7 EXPERIMENT
# ============================================================

# AB7 compares:
#   Binary classification
#   Three-class classification
#
# We use:
#   Representation = last_action_token
#   Subset = all

df = df[
    (df["representation"] == "last_action_token")
    & (df["subset"] == "all")
    & (df["task"].isin(["binary", "three_class"]))
].copy()

print("\nFiltered AB7 dataset:")
print("Number of rows:", len(df))
print("\nTasks:")
print(df["task"].value_counts())

print("\nModels:")
print(df["model"].value_counts())

print("\nEnvironments:")
print(df["environment"].value_counts())


# ============================================================
# 5. CHECK THAT EACH COMPARISON EXISTS ONLY ONCE
# ============================================================

comparison_keys = ["model", "environment", "task"]

duplicates = df.duplicated(
    subset=comparison_keys,
    keep=False
)

if duplicates.any():
    print("\nDuplicate comparison rows found:")
    print(
        df.loc[duplicates, comparison_keys]
        .sort_values(comparison_keys)
        .to_string(index=False)
    )

    raise ValueError(
        "\nThere are duplicate rows for a model/environment/task. "
        "Please inspect the CSV before continuing."
    )

# Check that every model/environment has both tasks.
task_counts = df.groupby(
    ["model", "environment"]
)["task"].nunique()

if not (task_counts == 2).all():
    print("\nWarning: Some model/environment combinations are missing a task.")
    print(task_counts)


# ============================================================
# 6. CREATE BINARY VS THREE-CLASS COMPARISON TABLE
# ============================================================

keys = ["model", "environment"]

metrics = [
    "balanced_accuracy_mean",
    "macro_f1_mean",
    "auroc_mean",
]

wide = df[keys].drop_duplicates().copy()

for metric in metrics:

    metric_table = df.pivot(
        index=keys,
        columns="task",
        values=metric
    ).reset_index()

    metric_table = metric_table.rename(
        columns={
            "binary": f"{metric}_binary",
            "three_class": f"{metric}_three_class",
        }
    )

    wide = wide.merge(
        metric_table,
        on=keys,
        how="left"
    )

    # Difference = Three-class score - Binary score
    wide[f"{metric}_difference"] = (
        wide[f"{metric}_three_class"]
        - wide[f"{metric}_binary"]
    )


# ============================================================
# 7. ADD STANDARD DEVIATIONS
# ============================================================

sd_metrics = [
    "balanced_accuracy_sd",
    "macro_f1_sd",
    "auroc_sd",
]

for metric in sd_metrics:

    metric_table = df.pivot(
        index=keys,
        columns="task",
        values=metric
    ).reset_index()

    metric_table = metric_table.rename(
        columns={
            "binary": f"{metric}_binary",
            "three_class": f"{metric}_three_class",
        }
    )

    wide = wide.merge(
        metric_table,
        on=keys,
        how="left"
    )


# ============================================================
# 8. ADD AUROC SEED COUNTS
# ============================================================

auroc_count_table = df.pivot(
    index=keys,
    columns="task",
    values="auroc_n"
).reset_index()

auroc_count_table = auroc_count_table.rename(
    columns={
        "binary": "auroc_n_binary",
        "three_class": "auroc_n_three_class",
    }
)

wide = wide.merge(
    auroc_count_table,
    on=keys,
    how="left"
)

wide = wide.sort_values(
    ["model", "environment"]
).reset_index(drop=True)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("AB7 COMPARISON RESULTS")
print("=" * 65)

display_columns = [
    "model",
    "environment",
    "balanced_accuracy_mean_binary",
    "balanced_accuracy_mean_three_class",
    "balanced_accuracy_mean_difference",
    "macro_f1_mean_binary",
    "macro_f1_mean_three_class",
    "macro_f1_mean_difference",
    "auroc_mean_binary",
    "auroc_mean_three_class",
    "auroc_mean_difference",
]

print(
    wide[display_columns].round(4).to_string(index=False)
)


# ============================================================
# 10. SAVE COMPARISON TABLES
# ============================================================

all_results_path = OUTPUT_DIR / "ab7_comparison_all.csv"

wide.to_csv(
    all_results_path,
    index=False
)

print("\nSaved:", all_results_path)


# Save separate tables for each model.

for model in wide["model"].unique():

    model_results = wide[
        wide["model"] == model
    ].copy()

    model_path = OUTPUT_DIR / f"ab7_{model}_comparison.csv"

    model_results.to_csv(
        model_path,
        index=False
    )

    print("Saved:", model_path)


# Save the filtered source rows as well.

filtered_path = OUTPUT_DIR / "ab7_filtered_source_rows.csv"

df.to_csv(
    filtered_path,
    index=False
)

print("Saved:", filtered_path)


# ============================================================
# 11. GENERATE COMPARISON PLOTS
# ============================================================

plot_metrics = {
    "balanced_accuracy": {
        "title": "Balanced Accuracy",
        "mean_column": "balanced_accuracy_mean",
        "sd_column": "balanced_accuracy_sd",
    },
    "macro_f1": {
        "title": "Macro F1",
        "mean_column": "macro_f1_mean",
        "sd_column": "macro_f1_sd",
    },
    "auroc": {
        "title": "AUROC",
        "mean_column": "auroc_mean",
        "sd_column": "auroc_sd",
    },
}

tasks = ["binary", "three_class"]

for model in sorted(df["model"].unique()):

    model_df = df[
        df["model"] == model
    ].copy()

    environments = sorted(
        model_df["environment"].unique()
    )

    x = np.arange(len(environments))
    width = 0.35

    for metric_name, config in plot_metrics.items():

        fig, ax = plt.subplots(figsize=(11, 6))

        for i, task in enumerate(tasks):

            task_df = (
                model_df[model_df["task"] == task]
                .set_index("environment")
                .reindex(environments)
            )

            means = task_df[
                config["mean_column"]
            ].to_numpy(dtype=float)

            errors = task_df[
                config["sd_column"]
            ].to_numpy(dtype=float)

            offset = (i - 0.5) * width

            ax.bar(
                x + offset,
                means,
                width,
                yerr=errors,
                capsize=4,
                label=task.replace("_", " ").title(),
            )

        ax.set_title(
            f"{config['title']}: Binary vs Three-Class ({model.upper()})"
        )

        ax.set_xlabel("Environment")
        ax.set_ylabel(config["title"])

        ax.set_xticks(x)
        ax.set_xticklabels(
            environments,
            rotation=25,
            ha="right"
        )

        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

        plt.tight_layout()

        plot_path = (
            OUTPUT_DIR
            / f"ab7_{model}_{metric_name}.png"
        )

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("Saved plot:", plot_path)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("AB7 PROCESSING COMPLETE")
print("=" * 65)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nFiles generated:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)

print("\nNext step: inspect the comparison table and plots.")