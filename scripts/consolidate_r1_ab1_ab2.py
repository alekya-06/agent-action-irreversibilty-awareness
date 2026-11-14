import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("results/rq1")
OUT = ROOT / "consolidated"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ("qwen", "llama")
SEEDS = (42, 43, 44)
ENVIRONMENTS = (
    "filesystem",
    "email_outbox",
    "wallet",
    "scienceworld",
    "toolemu",
)
REPRESENTATIONS = ("last_action_token", "mean_action_tokens")
SUBSETS = ("all", "minimal_pairs")
TASKS = ("binary", "three_class")


def load_summary(model, seed):
    path = ROOT / model / f"seed_{seed}" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    with path.open() as f:
        return json.load(f)


summaries = {
    (model, seed): load_summary(model, seed)
    for model in MODELS
    for seed in SEEDS
}


# ---------------------------------------------------------------------
# Raw per-seed results
# ---------------------------------------------------------------------

raw_rows = []

for model in MODELS:
    for seed in SEEDS:
        summary = summaries[(model, seed)]

        for env in ENVIRONMENTS:
            for task in TASKS:
                for representation in REPRESENTATIONS:
                    task_result = (
                        summary["environments"][env]
                        [representation]["tasks"][task]
                    )

                    for subset in SUBSETS:
                        result = task_result["subsets"].get(subset)

                        if result is None:
                            raise ValueError(
                                f"Missing subset={subset} for "
                                f"{model} seed={seed} "
                                f"{env} {representation} {task}"
                            )

                        test = result["final_test"]

                        raw_rows.append({
                            "model": model,
                            "seed": seed,
                            "environment": env,
                            "task": task,
                            "representation": representation,
                            "subset": subset,
                            "best_layer": result["best_layer"],
                            "balanced_accuracy": test["balanced_accuracy"],
                            "macro_f1": test["macro_f1"],
                            "auroc": test["auroc"],
                        })


# ---------------------------------------------------------------------
# Three-seed aggregate
# ---------------------------------------------------------------------

def aggregate(values):
    finite = [
        float(x)
        for x in values
        if x is not None and math.isfinite(float(x))
    ]

    if not finite:
        return {
            "mean": None,
            "sd": None,
            "n": 0,
        }

    return {
        "mean": mean(finite),
        "sd": stdev(finite) if len(finite) >= 2 else 0.0,
        "n": len(finite),
    }


aggregate_rows = []

for model in MODELS:
    for env in ENVIRONMENTS:
        for task in TASKS:
            for representation in REPRESENTATIONS:
                for subset in SUBSETS:

                    matching = [
                        row
                        for row in raw_rows
                        if (
                            row["model"] == model
                            and row["environment"] == env
                            and row["task"] == task
                            and row["representation"] == representation
                            and row["subset"] == subset
                        )
                    ]

                    if len(matching) != 3:
                        raise ValueError(
                            f"Expected 3 seeds, found {len(matching)} "
                            f"for {model} {env} {task} "
                            f"{representation} {subset}"
                        )

                    ba = aggregate(
                        row["balanced_accuracy"]
                        for row in matching
                    )
                    f1 = aggregate(
                        row["macro_f1"]
                        for row in matching
                    )
                    auc = aggregate(
                        row["auroc"]
                        for row in matching
                    )

                    aggregate_rows.append({
                        "model": model,
                        "environment": env,
                        "task": task,
                        "representation": representation,
                        "subset": subset,

                        "best_layers_seed42": matching[0]["best_layer"],
                        "best_layers_seed43": matching[1]["best_layer"],
                        "best_layers_seed44": matching[2]["best_layer"],

                        "balanced_accuracy_mean": ba["mean"],
                        "balanced_accuracy_sd": ba["sd"],

                        "macro_f1_mean": f1["mean"],
                        "macro_f1_sd": f1["sd"],

                        "auroc_mean": auc["mean"],
                        "auroc_sd": auc["sd"],
                        "auroc_n": auc["n"],
                    })


# ---------------------------------------------------------------------
# Save raw CSV
# ---------------------------------------------------------------------

raw_path = OUT / "r1_ab1_ab2_per_seed.csv"

with raw_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
    writer.writeheader()
    writer.writerows(raw_rows)


# ---------------------------------------------------------------------
# Save aggregate CSV
# ---------------------------------------------------------------------

aggregate_path = OUT / "r1_ab1_ab2_three_seed_summary.csv"

with aggregate_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=aggregate_rows[0].keys())
    writer.writeheader()
    writer.writerows(aggregate_rows)


# ---------------------------------------------------------------------
# Save JSON containing both raw and aggregate results
# ---------------------------------------------------------------------

json_path = OUT / "r1_ab1_ab2_consolidated.json"

def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


payload = {
    "description": (
        "Consolidated R1, AB1 and AB2 results from the six "
        "precomputed model/seed runs."
    ),
    "models": list(MODELS),
    "seeds": list(SEEDS),
    "environments": list(ENVIRONMENTS),
    "tasks": list(TASKS),
    "representations": list(REPRESENTATIONS),
    "subsets": list(SUBSETS),

    "raw_per_seed": raw_rows,
    "three_seed_aggregate": aggregate_rows,
}

payload = json_safe(payload)

with json_path.open("w") as f:
    json.dump(payload, f, indent=2, allow_nan=False)


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------

expected_raw = (
    len(MODELS)
    * len(SEEDS)
    * len(ENVIRONMENTS)
    * len(TASKS)
    * len(REPRESENTATIONS)
    * len(SUBSETS)
)

expected_aggregate = (
    len(MODELS)
    * len(ENVIRONMENTS)
    * len(TASKS)
    * len(REPRESENTATIONS)
    * len(SUBSETS)
)

assert len(raw_rows) == expected_raw
assert len(aggregate_rows) == expected_aggregate

print()
print("=" * 70)
print("R1 / AB1 / AB2 CONSOLIDATION")
print("=" * 70)
print(f"Raw rows:       {len(raw_rows)} / {expected_raw}")
print(f"Aggregate rows: {len(aggregate_rows)} / {expected_aggregate}")
print()
print(f"Saved: {raw_path}")
print(f"Saved: {aggregate_path}")
print(f"Saved: {json_path}")
print()
print("QC: PASSED")
