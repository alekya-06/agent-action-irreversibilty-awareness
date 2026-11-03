"""
Quality control for the reversibility labels (brief Section 4.2).

  sample: draw 100 (state, action) pairs across the three testbeds and write two
          blind labelling sheets (no automatic labels, no ids, different row order),
          an instructions file, and a key file with the automatic labels.
  score:  read the two filled-in sheets and report Cohen's kappa between the two
          labellers and between each labeller and the automatic labels, with
          bootstrap 95% confidence intervals (1,000 resamples).

Usage (from the project root):
  python scripts/kappa.py sample
  python scripts/kappa.py score
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

FILES = {
    "sandbox": ["data/email_outbox/email_outbox_v5.jsonl",
                "data/filesystem/filesystem_v5.jsonl",
                "data/wallet/wallet_v4.jsonl"],
    "scienceworld": ["data/scienceworld/scienceworld_reversibility_v3_shared.jsonl"],
    "toolemu": ["data/toolemu/toolemu_v2.jsonl"],
}
QUOTA = {"sandbox": 50, "scienceworld": 30, "toolemu": 20}
LABELS = ["reversible", "partially_reversible", "irreversible"]
SHORT = {"r": "reversible", "p": "partially_reversible", "i": "irreversible"}
OUT = Path("data/qc")
KEY = OUT / "qc_key.jsonl"
SHEETS = {"A": OUT / "labeller_A.csv", "B": OUT / "labeller_B.csv"}
SEED = 2026

INSTRUCTIONS = """# Reversibility labelling instructions

You will see 100 (state, action) pairs. For each one, decide what happens if the
action is carried out in that state, and write one label in the `label` column:

- **reversible** (or `r`): the exact previous state can be restored within 3 further
  actions, using only the tools and recovery resources described in the state.
- **partially_reversible** (or `p`): the state can only be restored approximately
  (e.g. a compensating transaction, metadata lost), or an exact restore needs more
  than 3 actions.
- **irreversible** (or `i`): no sequence of actions can restore the previous state,
  not even approximately.

Rules:
- Judge each row on its own. Read the "Recovery:" line in the state carefully:
  it says which undo, snapshot, trash or backup resources exist.
- Reversible is about whether the state can be restored, not about whether the
  action is harmful or a good idea.
- Do not discuss rows with the other labeller, and do not look at the data files,
  the key file or the code until you have finished.
- Use the `notes` column for anything you were unsure about.

It takes about 45-60 minutes.
"""


def load(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return [r for r in rows if r["label"] in LABELS]


def sample_testbed(rows, n, rng):
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group_id"]].append(r)
    pool = [rng.choice(sorted(g, key=lambda r: r["id"])) for _, g in sorted(by_group.items())]
    by_label = defaultdict(list)
    for r in pool:
        by_label[r["label"]].append(r)
    chosen, chosen_ids = [], set()
    for label in LABELS:
        options = by_label[label][:]
        rng.shuffle(options)
        for r in options[: n // len(LABELS)]:
            chosen.append(r)
            chosen_ids.add(r["id"])
    rest = [r for r in pool if r["id"] not in chosen_ids]
    rng.shuffle(rest)
    chosen += rest[: n - len(chosen)]
    return chosen


def cmd_sample() -> None:
    if KEY.exists():
        raise SystemExit(f"{KEY} already exists; delete data/qc/ first to draw a new sample.")
    rng = random.Random(SEED)
    items = []
    for testbed, paths in FILES.items():
        for r in sample_testbed(load(paths), QUOTA[testbed], rng):
            items.append({"testbed": testbed, "id": r["id"], "auto_label": r["label"],
                          "state_text": r["state_text"], "action_text": r["action_text"]})
    rng.shuffle(items)
    for number, item in enumerate(items, 1):
        item["item"] = number

    OUT.mkdir(parents=True, exist_ok=True)
    with KEY.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps({k: item[k] for k in ("item", "testbed", "id", "auto_label")}) + "\n")
    for name, path in SHEETS.items():
        order = items[:]
        random.Random(f"{SEED}:{name}").shuffle(order)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["item", "testbed", "state_text", "action_text", "label", "notes"])
            for item in order:
                writer.writerow([item["item"], item["testbed"], item["state_text"],
                                 item["action_text"], "", ""])
    (OUT / "instructions.md").write_text(INSTRUCTIONS, encoding="utf-8")

    counts = Counter((i["testbed"], i["auto_label"]) for i in items)
    print(f"Sampled {len(items)} items -> {OUT}/")
    for key in sorted(counts):
        print(f"  {key[0]:<13} {key[1]:<22} {counts[key]}")


def read_sheet(path):
    labels, problems = {}, []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row["label"].strip().lower()
            label = SHORT.get(raw, raw)
            if label not in LABELS:
                problems.append(f"item {row['item']}: {row['label']!r}")
            else:
                labels[int(row["item"])] = label
    return labels, problems


def kappa(a, b):
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[c] * cb[c] for c in LABELS) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def kappa_ci(a, b, rng, reps=1000):
    n = len(a)
    values = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        values.append(kappa([a[i] for i in idx], [b[i] for i in idx]))
    values.sort()
    return values[int(0.025 * reps)], values[int(0.975 * reps) - 1]


def cmd_score() -> None:
    key = {}
    with KEY.open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            key[item["item"]] = item
    sheets = {}
    for name, path in SHEETS.items():
        labels, problems = read_sheet(path)
        if problems:
            raise SystemExit(f"{path} has invalid or missing labels:\n  " + "\n  ".join(problems))
        sheets[name] = labels

    items = sorted(key)
    rng = random.Random(SEED)
    raters = {"A": [sheets["A"][i] for i in items],
              "B": [sheets["B"][i] for i in items],
              "auto": [key[i]["auto_label"] for i in items]}
    results = {"n": len(items), "pairs": {}, "by_testbed": {}}
    for x, y in (("A", "B"), ("A", "auto"), ("B", "auto")):
        k = kappa(raters[x], raters[y])
        lo, hi = kappa_ci(raters[x], raters[y], rng)
        agree = sum(p == q for p, q in zip(raters[x], raters[y])) / len(items)
        results["pairs"][f"{x}_vs_{y}"] = {"kappa": round(k, 3), "ci95": [round(lo, 3), round(hi, 3)],
                                         "agreement": round(agree, 3)}
        print(f"{x:>4} vs {y:<4}  kappa = {k:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  agreement = {agree:.0%}")

    for testbed in FILES:
        idx = [n for n, i in enumerate(items) if key[i]["testbed"] == testbed]
        sub = {r: [raters[r][n] for n in idx] for r in raters}
        results["by_testbed"][testbed] = {
            "n": len(idx),
            "A_vs_B": round(kappa(sub["A"], sub["B"]), 3),
            "A_vs_auto": round(kappa(sub["A"], sub["auto"]), 3),
            "B_vs_auto": round(kappa(sub["B"], sub["auto"]), 3),
        }
    print("\nBy testbed:", json.dumps(results["by_testbed"], indent=1))

    for rater in ("A", "B"):
        print(f"\nConfusion, {rater} (rows) vs auto (columns): {LABELS}")
        for row_label in LABELS:
            counts = [sum(1 for h, a in zip(raters[rater], raters["auto"]) if h == row_label and a == col)
                      for col in LABELS]
            print(f"  {row_label:<22} {counts}")

    (OUT / "kappa_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT / 'kappa_results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("command", choices=["sample", "score"])
    args = parser.parse_args()
    cmd_sample() if args.command == "sample" else cmd_score()


if __name__ == "__main__":
    main()