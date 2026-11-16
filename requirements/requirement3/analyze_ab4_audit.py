import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("results/rq3")
OUT_CSV = ROOT / "ab4_audit.csv"
OUT_MD = ROOT / "ab4_audit.md"

MODELS = [
    "Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct",
]

PROMPTS = ["neutral", "careful"]
VERSIONS = ["irreversible", "reversible"]


def load_runs():
    rows = []

    for model in MODELS:
        for prompt in PROMPTS:
            run_dir = ROOT / model / ("runs" if prompt == "neutral" else "careful/runs")

            for path in sorted(run_dir.glob("*.json")):
                data = json.loads(path.read_text())

                steps = data.get("steps", [])
                first_risky = data.get("first_risky_step")

                parse_failures = 0
                non_strict = 0
                tool_errors = 0
                errors_before_risky = 0
                errors_after_risky = 0
                parse_before_risky = 0
                parse_after_risky = 0
                non_strict_before_risky = 0
                non_strict_after_risky = 0

                kind_counts = Counter()
                action_counts = Counter()

                for i, step in enumerate(steps):
                    kind = step.get("kind")
                    kind_counts[str(kind)] += 1

                    action = step.get("action")
                    if action:
                        action_type = action.get("type")
                        if action_type:
                            action_counts[action_type] += 1

                    is_parse = bool(step.get("parse_failure", False))
                    is_non_strict = not bool(step.get("strict_format", True))

                    # A failed non-parse step is treated as a tool/execution error.
                    # This matches the run-level distinction between parse failures
                    # and tool errors while avoiding double-counting parse failures.
                    is_tool_error = (
                        not is_parse
                        and step.get("ok") is False
                    )

                    if is_parse:
                        parse_failures += 1
                    if is_non_strict:
                        non_strict += 1
                    if is_tool_error:
                        tool_errors += 1

                    if first_risky is not None:
                        if i < first_risky:
                            if is_parse:
                                parse_before_risky += 1
                            if is_non_strict:
                                non_strict_before_risky += 1
                            if is_tool_error:
                                errors_before_risky += 1
                        elif i > first_risky:
                            if is_parse:
                                parse_after_risky += 1
                            if is_non_strict:
                                non_strict_after_risky += 1
                            if is_tool_error:
                                errors_after_risky += 1

                rows.append({
                    "model": model,
                    "prompt_mode": prompt,
                    "environment": data.get("environment"),
                    "version": data.get("version"),
                    "pair_id": data.get("pair_id"),
                    "task_id": data.get("task_id"),
                    "caution": bool(data.get("caution", False)),
                    "careful_literal": bool(data.get("careful_literal", False)),
                    "first_risky_step": first_risky,
                    "first_irreversible_step": data.get("first_irreversible_step"),
                    "steps_used": data.get("steps_used"),
                    "success": bool(data.get("success", False)),
                    "took_risky_action": bool(data.get("took_risky_action", False)),
                    "took_irreversible_action": bool(
                        data.get("took_irreversible_action", False)
                    ),
                    "parse_failures": parse_failures,
                    "non_strict_format": non_strict,
                    "tool_errors": tool_errors,
                    "top_level_tool_errors": data.get("tool_errors", 0),
                    "errors_before_risky": errors_before_risky,
                    "errors_after_risky": errors_after_risky,
                    "parse_before_risky": parse_before_risky,
                    "parse_after_risky": parse_after_risky,
                    "non_strict_before_risky": non_strict_before_risky,
                    "non_strict_after_risky": non_strict_after_risky,
                    "kind_counts": dict(kind_counts),
                    "action_counts": dict(action_counts),
                })

    return rows


def pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d


def aggregate(rows):
    groups = defaultdict(list)

    for r in rows:
        key = (
            r["model"],
            r["prompt_mode"],
            r["environment"],
            r["version"],
        )
        groups[key].append(r)

    output = []

    for key, rs in sorted(groups.items()):
        model, prompt, env, version = key
        n = len(rs)

        output.append({
            "model": model,
            "prompt_mode": prompt,
            "environment": env,
            "version": version,
            "n": n,
            "caution_n": sum(r["caution"] for r in rs),
            "caution_rate": sum(r["caution"] for r in rs) / n,
            "risky_n": sum(r["took_risky_action"] for r in rs),
            "risky_rate": sum(r["took_risky_action"] for r in rs) / n,
            "irreversible_n": sum(r["took_irreversible_action"] for r in rs),
            "irreversible_rate": sum(
                r["took_irreversible_action"] for r in rs
            ) / n,
            "success_n": sum(r["success"] for r in rs),
            "success_rate": sum(r["success"] for r in rs) / n,
            "runs_with_parse_failure": sum(
                r["parse_failures"] > 0 for r in rs
            ),
            "runs_with_non_strict": sum(
                r["non_strict_format"] > 0 for r in rs
            ),
            "runs_with_tool_error": sum(
                r["tool_errors"] > 0 for r in rs
            ),
            "total_parse_failures": sum(
                r["parse_failures"] for r in rs
            ),
            "total_non_strict_steps": sum(
                r["non_strict_format"] for r in rs
            ),
            "total_tool_errors": sum(
                r["tool_errors"] for r in rs
            ),
            "total_errors_before_risky": sum(
                r["errors_before_risky"] for r in rs
            ),
            "total_errors_after_risky": sum(
                r["errors_after_risky"] for r in rs
            ),
            "total_parse_before_risky": sum(
                r["parse_before_risky"] for r in rs
            ),
            "total_parse_after_risky": sum(
                r["parse_after_risky"] for r in rs
            ),
            "total_non_strict_before_risky": sum(
                r["non_strict_before_risky"] for r in rs
            ),
            "total_non_strict_after_risky": sum(
                r["non_strict_after_risky"] for r in rs
            ),
        })

    return output


def pair_audit(rows):
    """
    Compare irreversible/reversible versions within each pair.

    This is specifically to determine whether execution errors are
    systematically concentrated in one member of the paired comparison.
    """
    by_pair = defaultdict(dict)

    for r in rows:
        key = (
            r["model"],
            r["prompt_mode"],
            r["environment"],
            r["pair_id"],
        )
        by_pair[key][r["version"]] = r

    results = []

    for key, pair in sorted(by_pair.items()):
        if "irreversible" not in pair or "reversible" not in pair:
            continue

        irr = pair["irreversible"]
        rev = pair["reversible"]

        results.append({
            "model": key[0],
            "prompt_mode": key[1],
            "environment": key[2],
            "pair_id": key[3],
            "irr_caution": int(irr["caution"]),
            "rev_caution": int(rev["caution"]),
            "irr_success": int(irr["success"]),
            "rev_success": int(rev["success"]),
            "irr_tool_errors": irr["tool_errors"],
            "rev_tool_errors": rev["tool_errors"],
            "irr_errors_before_risky": irr["errors_before_risky"],
            "rev_errors_before_risky": rev["errors_before_risky"],
            "irr_errors_after_risky": irr["errors_after_risky"],
            "rev_errors_after_risky": rev["errors_after_risky"],
            "irr_parse_failures": irr["parse_failures"],
            "rev_parse_failures": rev["parse_failures"],
            "irr_non_strict": irr["non_strict_format"],
            "rev_non_strict": rev["non_strict_format"],
        })

    return results


def write_csv(rows):
    fields = [
        "model",
        "prompt_mode",
        "environment",
        "version",
        "n",
        "caution_n",
        "caution_rate",
        "risky_n",
        "risky_rate",
        "irreversible_n",
        "irreversible_rate",
        "success_n",
        "success_rate",
        "runs_with_parse_failure",
        "runs_with_non_strict",
        "runs_with_tool_error",
        "total_parse_failures",
        "total_non_strict_steps",
        "total_tool_errors",
        "total_errors_before_risky",
        "total_errors_after_risky",
        "total_parse_before_risky",
        "total_parse_after_risky",
        "total_non_strict_before_risky",
        "total_non_strict_after_risky",
    ]

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(agg, pairs):
    lines = []

    lines.append("# AB4 trajectory audit")
    lines.append("")
    lines.append(
        "Audit of the saved AB4 trajectories. No model inference or experiment "
        "rerun is performed."
    )
    lines.append("")

    lines.append("## Aggregate execution diagnostics")
    lines.append("")
    lines.append(
        "| Model | Prompt | Environment | Version | n | "
        "Caution | Success | Tool-error runs | Tool errors | "
        "Errors before risky | Errors after risky | "
        "Parse failures | Non-strict steps |"
    )
    lines.append(
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for r in agg:
        lines.append(
            f"| {r['model']} | {r['prompt_mode']} | {r['environment']} | "
            f"{r['version']} | {r['n']} | "
            f"{r['caution_n']}/{r['n']} | "
            f"{r['success_n']}/{r['n']} | "
            f"{r['runs_with_tool_error']} | "
            f"{r['total_tool_errors']} | "
            f"{r['total_errors_before_risky']} | "
            f"{r['total_errors_after_risky']} | "
            f"{r['total_parse_failures']} | "
            f"{r['total_non_strict_steps']} |"
        )

    lines.append("")
    lines.append("## Paired execution comparison")
    lines.append("")
    lines.append(
        "For each pair, irreversible and reversible versions are compared "
        "directly. This checks whether execution problems are concentrated "
        "in one member of the pair."
    )
    lines.append("")
    lines.append(
        "| Model | Prompt | Environment | Pairs | "
        "Irr caution | Rev caution | "
        "Irr success | Rev success | "
        "Irr errors before risky | Rev errors before risky | "
        "Irr errors after risky | Rev errors after risky |"
    )
    lines.append(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    grouped = defaultdict(list)
    for r in pairs:
        grouped[
            (r["model"], r["prompt_mode"], r["environment"])
        ].append(r)

    for key, rs in sorted(grouped.items()):
        n = len(rs)

        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {n} | "
            f"{sum(r['irr_caution'] for r in rs)} | "
            f"{sum(r['rev_caution'] for r in rs)} | "
            f"{sum(r['irr_success'] for r in rs)} | "
            f"{sum(r['rev_success'] for r in rs)} | "
            f"{sum(r['irr_errors_before_risky'] for r in rs)} | "
            f"{sum(r['rev_errors_before_risky'] for r in rs)} | "
            f"{sum(r['irr_errors_after_risky'] for r in rs)} | "
            f"{sum(r['rev_errors_after_risky'] for r in rs)} |"
        )

    lines.append("")
    lines.append("## Interpretation guidance")
    lines.append("")
    lines.append(
        "- Errors occurring after the first risky action do not change the "
        "pre-risk caution classification."
    )
    lines.append(
        "- Execution errors are reported as diagnostics and are not silently "
        "reclassified as caution."
    )
    lines.append(
        "- The paired comparison is used to assess whether execution problems "
        "are disproportionately associated with either version."
    )

    OUT_MD.write_text("\n".join(lines) + "\n")


def main():
    rows = load_runs()

    print("Total run files:", len(rows))

    expected = 2 * 2 * 3 * 2 * 20
    print("Expected run files:", expected)

    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} runs but found {len(rows)}"
        )

    print("\nRuns by model/prompt:")
    counts = Counter(
        (r["model"], r["prompt_mode"])
        for r in rows
    )
    for key, n in sorted(counts.items()):
        print(f"  {key}: {n}")

    print("\nStep kind counts:")
    kinds = Counter()
    for r in rows:
        kinds.update(r["kind_counts"])
    for kind, n in sorted(kinds.items()):
        print(f"  {kind}: {n}")

    agg = aggregate(rows)
    pairs = pair_audit(rows)

    write_csv(agg)
    write_markdown(agg, pairs)

    print("\nWrote:")
    print(" ", OUT_CSV)
    print(" ", OUT_MD)

    print("\nAggregate diagnostics:")
    for r in agg:
        print(
            f"{r['model']} | {r['prompt_mode']} | "
            f"{r['environment']} | {r['version']} | "
            f"n={r['n']} | caution={r['caution_n']}/{r['n']} | "
            f"success={r['success_n']}/{r['n']} | "
            f"tool_errors={r['total_tool_errors']} | "
            f"before_risky={r['total_errors_before_risky']} | "
            f"after_risky={r['total_errors_after_risky']} | "
            f"parse={r['total_parse_failures']} | "
            f"non_strict={r['total_non_strict_steps']}"
        )


if __name__ == "__main__":
    main()
