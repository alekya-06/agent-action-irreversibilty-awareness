import csv
from pathlib import Path
from statistics import mean, stdev

path = Path("results/rq1/consolidated/r1_ab1_ab2_three_seed_summary.csv")

with path.open() as f:
    rows = list(csv.DictReader(f))

def finite_values(rows, key):
    values = []
    for row in rows:
        value = row[key]
        if value == "" or value.lower() == "nan":
            continue
        values.append(float(value))
    return values

print()
print("=" * 80)
print("AB1: LAST ACTION TOKEN vs MEAN ACTION TOKENS")
print("=" * 80)

# Primary AB1 comparison:
# binary task, all data, across all environments and both models.
for model in ("qwen", "llama"):
    print(f"\nMODEL: {model.upper()}")
    print("-" * 80)

    for env in ("filesystem", "email_outbox", "wallet", "scienceworld", "toolemu"):
        comparison = {}

        for representation in ("last_action_token", "mean_action_tokens"):
            matching = [
                r for r in rows
                if r["model"] == model
                and r["environment"] == env
                and r["task"] == "binary"
                and r["subset"] == "all"
                and r["representation"] == representation
            ]

            if len(matching) != 1:
                raise ValueError(
                    f"Expected exactly one aggregate row for "
                    f"{model} {env} {representation}, found {len(matching)}"
                )

            r = matching[0]

            comparison[representation] = {
                "ba": float(r["balanced_accuracy_mean"]),
                "f1": float(r["macro_f1_mean"]),
                "auroc": (
                    None
                    if r["auroc_mean"] in ("", "nan", "NaN")
                    else float(r["auroc_mean"])
                ),
            }

        last = comparison["last_action_token"]
        mean_rep = comparison["mean_action_tokens"]

        print(f"\n{env}")
        print(
            f"  Last token : "
            f"BA={last['ba']:.4f}, "
            f"F1={last['f1']:.4f}, "
            f"AUROC={last['auroc']:.4f}"
            if last["auroc"] is not None
            else
            f"  Last token : "
            f"BA={last['ba']:.4f}, "
            f"F1={last['f1']:.4f}, "
            f"AUROC=NA"
        )

        print(
            f"  Mean tokens: "
            f"BA={mean_rep['ba']:.4f}, "
            f"F1={mean_rep['f1']:.4f}, "
            f"AUROC={mean_rep['auroc']:.4f}"
            if mean_rep["auroc"] is not None
            else
            f"  Mean tokens: "
            f"BA={mean_rep['ba']:.4f}, "
            f"F1={mean_rep['f1']:.4f}, "
            f"AUROC=NA"
        )

        print(
            f"  Difference : "
            f"BA={last['ba'] - mean_rep['ba']:+.4f}, "
            f"F1={last['f1'] - mean_rep['f1']:+.4f}, "
            f"AUROC="
            f"{last['auroc'] - mean_rep['auroc']:+.4f}"
            if last["auroc"] is not None and mean_rep["auroc"] is not None
            else
            f"  Difference : "
            f"BA={last['ba'] - mean_rep['ba']:+.4f}, "
            f"F1={last['f1'] - mean_rep['f1']:+.4f}, "
            f"AUROC=NA"
        )

# Overall comparison across the 5 environments for each model.
print()
print("=" * 80)
print("AB1 OVERALL MEAN ACROSS ENVIRONMENTS")
print("=" * 80)

for model in ("qwen", "llama"):
    print(f"\n{model.upper()}")

    for representation in ("last_action_token", "mean_action_tokens"):
        matching = [
            r for r in rows
            if r["model"] == model
            and r["task"] == "binary"
            and r["subset"] == "all"
            and r["representation"] == representation
        ]

        ba = [float(r["balanced_accuracy_mean"]) for r in matching]
        f1 = [float(r["macro_f1_mean"]) for r in matching]
        auc = finite_values(matching, "auroc_mean")

        print(
            f"  {representation}: "
            f"BA={mean(ba):.4f}, "
            f"F1={mean(f1):.4f}, "
            f"AUROC={mean(auc):.4f}"
            if auc
            else
            f"  {representation}: "
            f"BA={mean(ba):.4f}, "
            f"F1={mean(f1):.4f}, "
            f"AUROC=NA"
        )

print()
print("AB1 analysis completed from precomputed results only.")
print("No model inference was performed.")
