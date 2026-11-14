import csv
from pathlib import Path
from statistics import mean

path = Path(
    "results/rq1/consolidated/r1_ab1_ab2_three_seed_summary.csv"
)

with path.open() as f:
    rows = list(csv.DictReader(f))


def get_value(row, key):
    value = row[key]
    if value in ("", "nan", "NaN", "None"):
        return None
    return float(value)


print()
print("=" * 80)
print("AB2: ALL DATA vs MINIMAL PAIRS")
print("=" * 80)

# AB2 uses the PRIMARY representation:
# last action token.
for model in ("qwen", "llama"):

    print(f"\nMODEL: {model.upper()}")
    print("-" * 80)

    for env in (
        "filesystem",
        "email_outbox",
        "wallet",
        "scienceworld",
        "toolemu",
    ):

        comparison = {}

        for subset in ("all", "minimal_pairs"):

            matching = [
                r
                for r in rows
                if r["model"] == model
                and r["environment"] == env
                and r["task"] == "binary"
                and r["representation"] == "last_action_token"
                and r["subset"] == subset
            ]

            if len(matching) != 1:
                raise ValueError(
                    f"Expected exactly one row for "
                    f"{model} {env} {subset}, "
                    f"found {len(matching)}"
                )

            r = matching[0]

            comparison[subset] = {
                "ba": get_value(r, "balanced_accuracy_mean"),
                "f1": get_value(r, "macro_f1_mean"),
                "auroc": get_value(r, "auroc_mean"),
            }

        all_data = comparison["all"]
        minimal = comparison["minimal_pairs"]

        print(f"\n{env}")

        print(
            f"  All data     : "
            f"BA={all_data['ba']:.4f}, "
            f"F1={all_data['f1']:.4f}, "
            f"AUROC={all_data['auroc']:.4f}"
            if all_data["auroc"] is not None
            else
            f"  All data     : "
            f"BA={all_data['ba']:.4f}, "
            f"F1={all_data['f1']:.4f}, "
            f"AUROC=NA"
        )

        print(
            f"  Minimal pairs: "
            f"BA={minimal['ba']:.4f}, "
            f"F1={minimal['f1']:.4f}, "
            f"AUROC={minimal['auroc']:.4f}"
            if minimal["auroc"] is not None
            else
            f"  Minimal pairs: "
            f"BA={minimal['ba']:.4f}, "
            f"F1={minimal['f1']:.4f}, "
            f"AUROC=NA"
        )

        print(
            f"  Difference (all - pairs): "
            f"BA={all_data['ba'] - minimal['ba']:+.4f}, "
            f"F1={all_data['f1'] - minimal['f1']:+.4f}, "
            f"AUROC="
            f"{all_data['auroc'] - minimal['auroc']:+.4f}"
            if (
                all_data["auroc"] is not None
                and minimal["auroc"] is not None
            )
            else
            f"  Difference (all - pairs): "
            f"BA={all_data['ba'] - minimal['ba']:+.4f}, "
            f"F1={all_data['f1'] - minimal['f1']:+.4f}, "
            f"AUROC=NA"
        )


print()
print("=" * 80)
print("AB2 OVERALL MEAN ACROSS ENVIRONMENTS")
print("=" * 80)

for model in ("qwen", "llama"):

    print(f"\n{model.upper()}")

    for subset in ("all", "minimal_pairs"):

        matching = [
            r
            for r in rows
            if r["model"] == model
            and r["task"] == "binary"
            and r["representation"] == "last_action_token"
            and r["subset"] == subset
        ]

        ba = [
            get_value(r, "balanced_accuracy_mean")
            for r in matching
        ]

        f1 = [
            get_value(r, "macro_f1_mean")
            for r in matching
        ]

        auc = [
            get_value(r, "auroc_mean")
            for r in matching
        ]

        auc = [x for x in auc if x is not None]

        print(
            f"  {subset:14s}: "
            f"BA={mean(ba):.4f}, "
            f"F1={mean(f1):.4f}, "
            f"AUROC={mean(auc):.4f}"
        )


print()
print("AB2 analysis completed from precomputed results only.")
print("No model inference was performed.")
