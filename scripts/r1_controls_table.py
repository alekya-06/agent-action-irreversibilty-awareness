"""Collect R1 probe and control results from every results/rq1/<model>/seed_<s>/summary.json
into one table (mean and sd over seeds). Binary task, last action token by default."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ["majority", "lexical_bow", "lexical_bow_minimal_pairs", "shuffled_labels", "layer_0_probe"]


def ba(block):
    return (block or {}).get("test", {}).get("balanced_accuracy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="binary")
    p.add_argument("--representation", default="last_action_token")
    args = p.parse_args()

    values = defaultdict(lambda: defaultdict(list))
    for path in sorted((ROOT / "results" / "rq1").glob("*/seed_*/summary.json")):
        d = json.loads(path.read_text())
        for env, reps in d["environments"].items():
            t = reps[args.representation]["tasks"][args.task]
            if t.get("status") != "ok":
                continue
            key = (d["model"], env)
            subsets, c = t["subsets"], t["controls"]
            row = {
                "probe_all": subsets["all"]["final_test"]["balanced_accuracy"],
                "probe_minimal_pairs": subsets.get("minimal_pairs", {}).get("final_test", {}).get("balanced_accuracy"),
                **{name: ba(c.get(name)) for name in CONTROLS},
                "control_task": c.get("action_identity_hewitt_liang", {}).get("control_target_test", {}).get("balanced_accuracy"),
            }
            row["selectivity"] = (row["probe_all"] - row["control_task"]) if row["control_task"] is not None else None
            for k, v in row.items():
                if v is not None:
                    values[key][k].append(v)

    cols = ["probe_all", "lexical_bow", "probe_minimal_pairs", "lexical_bow_minimal_pairs",
            "majority", "shuffled_labels", "layer_0_probe", "control_task", "selectivity"]
    out = ROOT / "results" / "rq1" / "consolidated" / f"r1_controls_{args.task}_{args.representation}.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "environment", "seeds"] + [f"{c}_{s}" for c in cols for s in ("mean", "sd")])
        print(f"{'model':<6} {'env':<13} " + " ".join(f"{c[:11]:>11}" for c in cols))
        for (model, env), vals in sorted(values.items()):
            stats = []
            for c in cols:
                xs = vals.get(c, [])
                stats += [statistics.mean(xs) if xs else None, statistics.stdev(xs) if len(xs) > 1 else None]
            w.writerow([model, env, len(vals["probe_all"])] + stats)
            print(f"{model:<6} {env:<13} " + " ".join(
                f"{stats[2 * i]:>11.3f}" if stats[2 * i] is not None else f"{'-':>11}" for i in range(len(cols))))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()