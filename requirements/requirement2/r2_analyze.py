"""
R2 analysis: how well does the model's stated answer match the true label?

Joins results/rq2/<run>/stated.jsonl with the dataset rows (by id) and reports,
overall and per testbed:
  - unparseable answers (counted, never dropped; they count as wrong)
  - 3-class accuracy, balanced accuracy and macro-F1, with 95% CIs from a
    scenario-level bootstrap (1,000 resamples)
  - binary reversible vs irreversible: accuracy and AUROC of P(irreversible)
  - confusion matrices
  - minimal-pair sensitivity: in groups whose true labels differ, do the stated
    answers differ too, and in the right direction?
  - breakdown by action verb, and for actions that change nothing (exact_cost 0)

Usage (from the project root):
  python requirements/requirement2/r2_analyze.py
  python requirements/requirement2/r2_analyze.py --run results/rq2/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from r2_stated_answers import DATA_FILES, ROOT

LABELS = ["reversible", "partially_reversible", "irreversible"]
ORDER = {label: i for i, label in enumerate(LABELS)}
SW_MULTIWORD = ("pick up", "put down", "go to", "look at", "look in", "look around",
                "focus on", "move", "pour", "connect", "disconnect")
SEED = 0


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def classification(rows: list[dict]) -> dict:
    n = len(rows)
    acc = sum(r["stated_label"] == r["true_label"] for r in rows) / n
    recalls, f1s = [], []
    for c in LABELS:
        support = sum(r["true_label"] == c for r in rows)
        if not support:
            continue
        tp = sum(r["true_label"] == c and r["stated_label"] == c for r in rows)
        predicted = sum(r["stated_label"] == c for r in rows)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"n": n, "accuracy": acc,
            "balanced_accuracy": sum(recalls) / len(recalls),
            "macro_f1": sum(f1s) / len(f1s)}


def auroc(pairs: list[tuple[float, bool]]) -> float | None:
    """Rank-based AUROC (Mann-Whitney), ties averaged."""
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def binary(rows: list[dict]) -> dict:
    sub = [r for r in rows if r["true_label"] in ("reversible", "irreversible")]
    if not sub:
        return {"n": 0}
    acc = sum(r["stated_label"] == r["true_label"] for r in sub) / len(sub)
    pairs = [(r["option_probs"]["irreversible"], r["true_label"] == "irreversible")
             for r in sub if r["option_probs"].get("irreversible") is not None]
    return {"n": len(sub), "accuracy": acc, "auroc": auroc(pairs)}


def bootstrap_ci(rows, fn, rng, reps=1000):
    """95% CI, resampling whole scenarios (rows within a scenario are not independent)."""
    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[r["scenario_id"]].append(r)
    keys = sorted(by_scenario)
    values = []
    for _ in range(reps):
        sample = [r for _ in keys for r in by_scenario[rng.choice(keys)]]
        value = fn(sample)
        if value is not None:
            values.append(value)
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def confusion(rows: list[dict]) -> dict:
    cols = LABELS + [None]
    return {t: [sum(r["true_label"] == t and r["stated_label"] == s for r in rows) for s in cols]
            for t in LABELS}


def minimal_pairs(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    contrastive = [g for g in groups.values() if len({r["true_label"] for r in g}) > 1]
    varied = sum(len({r["stated_label"] for r in g}) > 1 for g in contrastive)
    concordant = discordant = tied = 0
    for g in contrastive:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                if a["true_label"] == b["true_label"] or None in (a["stated_label"], b["stated_label"]):
                    continue
                true_diff = ORDER[a["true_label"]] - ORDER[b["true_label"]]
                stated_diff = ORDER[a["stated_label"]] - ORDER[b["stated_label"]]
                if stated_diff == 0:
                    tied += 1
                elif (true_diff > 0) == (stated_diff > 0):
                    concordant += 1
                else:
                    discordant += 1
    pairs = concordant + discordant + tied
    return {"contrastive_groups": len(contrastive),
            "groups_where_answer_changes": varied,
            "sensitivity": varied / len(contrastive) if contrastive else None,
            "pairs": pairs,
            "concordant": concordant, "discordant": discordant, "tied": tied}


def verb(row: dict) -> str:
    text = row["action_text"].strip()
    if row["environment"] == "scienceworld":
        for v in SW_MULTIWORD:
            if text.startswith(v):
                return v
        return text.split()[0] if text else "?"
    if row.get("action_unit") == "sequence":
        return "sequence"
    return text.split("(")[0]


def fmt_ci(value, ci):
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="R2 analysis.")
    parser.add_argument("--run", type=Path, default=ROOT / "results" / "rq2" / "Qwen2.5-7B-Instruct_dev")
    args = parser.parse_args()

    data = {}
    for path in DATA_FILES:
        for r in load_jsonl(ROOT / path):
            data[r["id"]] = r
    rows = []
    for s in load_jsonl(args.run / "stated.jsonl"):
        d = data[s["id"]]
        rows.append({**s, "exact_cost": d.get("exact_cost"), "action_text": d["action_text"],
                     "action_unit": d.get("action_unit")})

    rng = random.Random(SEED)
    results = {"run": str(args.run), "n": len(rows), "by_testbed": {}}
    testbeds = {"all": rows}
    for r in rows:
        testbeds.setdefault(r["environment"] if r["environment"] in ("scienceworld", "toolemu") else "sandbox", []).append(r)

    print(f"Run: {args.run}  ({len(rows)} rows)\n")
    print(f"{'testbed':<13} {'n':>5} {'unparsed':>8}  {'accuracy [95% CI]':<26} {'balanced acc':>12} {'macro-F1':>9}  {'bin acc':>7} {'AUROC':>6}")
    for name, sub in testbeds.items():
        cls = classification(sub)
        ci = bootstrap_ci(sub, lambda x: classification(x)["accuracy"], rng)
        b = binary(sub)
        unparsed = sum(r["stated_label"] is None for r in sub)
        results["by_testbed"][name] = {**cls, "accuracy_ci95": ci, "unparsed": unparsed, "binary": b,
                                       "confusion": confusion(sub), "minimal_pairs": minimal_pairs(sub),
                                       "stated_distribution": dict(Counter(r["stated_label"] for r in sub))}
        auc = f"{b['auroc']:.3f}" if b.get("auroc") is not None else "  -  "
        print(f"{name:<13} {cls['n']:>5} {unparsed:>8}  {fmt_ci(cls['accuracy'], ci):<26} "
              f"{cls['balanced_accuracy']:>12.3f} {cls['macro_f1']:>9.3f}  {b.get('accuracy', 0):>7.3f} {auc:>6}")

    print("\nConfusion (rows: true; columns: stated reversible / partial / irreversible / unparsed)")
    for name in testbeds:
        print(f"  {name}")
        for t, counts in results["by_testbed"][name]["confusion"].items():
            print(f"    {t:<22} {counts}")

    print("\nMinimal-pair sensitivity (groups whose true labels differ)")
    for name in testbeds:
        mp = results["by_testbed"][name]["minimal_pairs"]
        if mp["contrastive_groups"]:
            print(f"  {name:<13} answer changes in {mp['groups_where_answer_changes']}/{mp['contrastive_groups']} groups "
                  f"({mp['sensitivity']:.0%}); pairs: {mp['concordant']} right direction, "
                  f"{mp['discordant']} wrong direction, {mp['tied']} same answer")

    no_change = [r for r in rows if r["exact_cost"] == 0]
    results["no_change"] = {"n": len(no_change),
                            "accuracy": sum(r["stated_label"] == "reversible" for r in no_change) / len(no_change) if no_change else None,
                            "stated_distribution": dict(Counter(r["stated_label"] for r in no_change))}
    print(f"\nActions that change nothing (exact_cost 0): n = {len(no_change)}, "
          f"stated: {results['no_change']['stated_distribution']}")

    by_verb = defaultdict(list)
    for r in rows:
        by_verb[(r["environment"], verb(r))].append(r)
    table = []
    for (env, v), sub in by_verb.items():
        if len(sub) >= 10:
            table.append({"environment": env, "verb": v, "n": len(sub),
                          "accuracy": sum(r["stated_label"] == r["true_label"] for r in sub) / len(sub),
                          "true": dict(Counter(r["true_label"] for r in sub)),
                          "stated": dict(Counter(r["stated_label"] for r in sub))})
    table.sort(key=lambda t: -t["n"])
    results["by_verb"] = table
    print("\nBy action verb (n >= 10, largest first)")
    for t in table[:25]:
        print(f"  {t['environment'][:12]:<12} {t['verb'][:34]:<34} n={t['n']:>4}  acc={t['accuracy']:.2f}  "
              f"true={t['true']}  stated={t['stated']}")

    out = args.run / "analysis.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()