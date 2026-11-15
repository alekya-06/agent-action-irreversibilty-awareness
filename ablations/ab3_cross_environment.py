#!/usr/bin/env python3
"""
AB3: Cross-environment transfer for the primary binary reversibility probe.

Protocol:
- Source environments are the three Python sandbox environments:
    filesystem, email_outbox, wallet
  pooled into one "sandbox" source environment.
- Target environment is ScienceWorld.
- Run both directions:
    sandbox -> scienceworld
    scienceworld -> sandbox
- Primary representation: last_action_token.
- Primary task: binary reversibility.
- For each model and seed:
    1. Select the best layer using SOURCE TRAIN -> SOURCE VALIDATION only.
    2. Refit the probe on SOURCE TRAIN + SOURCE VALIDATION.
    3. Evaluate once on TARGET TEST only.
- No target training or validation rows are used.
- Uses the existing saved hidden states; no model inference is performed.
- Seeds 42/43/44 are kept separate, matching the existing R1 methodology.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

# Reuse the exact R1 implementation for:
#   - model names
#   - label definitions
#   - hidden-state loading
#   - metadata loading
#   - binary cleaning
#   - logistic regression
#   - metric calculation
#   - best-layer selection
#   - aligned probabilities
from requirements.requirement1.run_rq1_probes import (
    BINARY_LABELS,
    MODELS,
    SEEDS,
    aligned_predict_proba,
    clean_binary,
    load_hidden_states,
    load_metadata,
    make_model,
    metrics,
    select_best_layer,
)


SANDBOX_ENVIRONMENTS = (
    "filesystem",
    "email_outbox",
    "wallet",
)

TARGET_ENVIRONMENT = "scienceworld"
REPRESENTATION = "last_action_token"
TASK = "binary"


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def validate_metadata_and_hidden(
    metadata: list[dict],
    hidden: np.ndarray,
    *,
    model_name: str,
    seed: int,
) -> None:
    if len(metadata) != 3601:
        raise ValueError(
            f"{model_name} seed {seed}: expected 3601 metadata rows, "
            f"got {len(metadata)}."
        )

    if hidden.ndim != 3:
        raise ValueError(
            f"Expected hidden states with shape [layers, rows, dim], "
            f"got {hidden.shape}."
        )

    if hidden.shape[1] != len(metadata):
        raise ValueError(
            f"{model_name} seed {seed}: hidden-state row count "
            f"{hidden.shape[1]} != metadata rows {len(metadata)}."
        )

    ids = [row["id"] for row in metadata]
    if len(ids) != len(set(ids)):
        raise ValueError(
            f"{model_name} seed {seed}: metadata IDs are not unique."
        )

    if not np.isfinite(hidden).all():
        raise ValueError(
            f"{model_name} seed {seed}: hidden states contain non-finite values."
        )


def source_target_rows(
    metadata: list[dict],
    source_name: str,
    target_name: str,
) -> tuple[list[dict], list[dict]]:
    binary_rows = clean_binary(metadata)

    source = [
        row
        for row in binary_rows
        if row["environment"] in SANDBOX_ENVIRONMENTS
        if source_name == "sandbox"
    ]

    if source_name == "scienceworld":
        source = [
            row
            for row in binary_rows
            if row["environment"] == TARGET_ENVIRONMENT
        ]

    target = [
        row
        for row in binary_rows
        if row["environment"] in SANDBOX_ENVIRONMENTS
        if target_name == "sandbox"
    ]

    if target_name == "scienceworld":
        target = [
            row
            for row in binary_rows
            if row["environment"] == TARGET_ENVIRONMENT
        ]

    return source, target


def row_indices(
    metadata: list[dict],
    rows: list[dict],
) -> np.ndarray:
    """Map rows back to canonical metadata indices using unique IDs."""
    index_by_id = {row["id"]: i for i, row in enumerate(metadata)}
    indices = np.array(
        [index_by_id[row["id"]] for row in rows],
        dtype=int,
    )

    if len(indices) != len(set(indices.tolist())):
        raise ValueError("Duplicate metadata index detected.")

    return indices


def split_indices(rows: list[dict], split: str) -> np.ndarray:
    return np.array(
        [i for i, row in enumerate(rows) if row["_split"] == split],
        dtype=int,
    )


def select_source_layer(
    source_hidden: np.ndarray,
    source_rows: list[dict],
) -> tuple[int, list[dict]]:
    """
    Select the transfer layer using source training data only for fitting
    and source validation data only for selection.
    """
    train_idx = split_indices(source_rows, "train")
    val_idx = split_indices(source_rows, "validation")

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Source train/validation split is empty.")

    y = np.array([row["binary_label"] for row in source_rows])

    layer_results = []

    for layer in range(source_hidden.shape[0]):
        model = make_model(multiclass=False)
        model.fit(
            source_hidden[layer, train_idx, :],
            y[train_idx],
        )

        val_pred = model.predict(
            source_hidden[layer, val_idx, :]
        )

        val_prob = aligned_predict_proba(
            model,
            source_hidden[layer, val_idx, :],
            BINARY_LABELS,
        )

        val_metrics = metrics(
            y[val_idx],
            val_pred,
            val_prob,
            BINARY_LABELS,
        )

        layer_results.append(
            {
                "subset": "source",
                "layer": layer,
                "validation": val_metrics,
            }
        )

    best_layer = select_best_layer(layer_results)
    return best_layer, layer_results


def fit_source_and_test_target(
    source_hidden: np.ndarray,
    source_rows: list[dict],
    target_hidden: np.ndarray,
    target_rows: list[dict],
    best_layer: int,
) -> dict[str, Any]:
    source_train_val = np.array(
        [
            i
            for i, row in enumerate(source_rows)
            if row["_split"] in ("train", "validation")
        ],
        dtype=int,
    )

    target_test = np.array(
        [
            i
            for i, row in enumerate(target_rows)
            if row["_split"] == "test"
        ],
        dtype=int,
    )

    if len(source_train_val) == 0:
        raise ValueError("Source train+validation is empty.")

    if len(target_test) == 0:
        raise ValueError("Target test is empty.")

    y_source = np.array(
        [row["binary_label"] for row in source_rows]
    )

    y_target = np.array(
        [row["binary_label"] for row in target_rows]
    )

    model = make_model(multiclass=False)

    model.fit(
        source_hidden[best_layer, source_train_val, :],
        y_source[source_train_val],
    )

    target_pred = model.predict(
        target_hidden[best_layer, target_test, :]
    )

    target_prob = aligned_predict_proba(
        model,
        target_hidden[best_layer, target_test, :],
        BINARY_LABELS,
    )

    target_metrics = metrics(
        y_target[target_test],
        target_pred,
        target_prob,
        BINARY_LABELS,
    )

    return {
        "target_test": target_metrics,
        "target_test_n": int(len(target_test)),
        "target_test_class_counts": {
            label: int(np.sum(y_target[target_test] == label))
            for label in BINARY_LABELS
        },
    }


def bootstrap_target(
    source_hidden: np.ndarray,
    source_rows: list[dict],
    target_hidden: np.ndarray,
    target_rows: list[dict],
    best_layer: int,
    seed: int,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """
    Bootstrap the fixed transfer predictions on the untouched target test set.
    The source model/layer are not refit inside bootstrap samples.
    """
    source_train_val = np.array(
        [
            i
            for i, row in enumerate(source_rows)
            if row["_split"] in ("train", "validation")
        ],
        dtype=int,
    )

    target_test = np.array(
        [
            i
            for i, row in enumerate(target_rows)
            if row["_split"] == "test"
        ],
        dtype=int,
    )

    y_source = np.array(
        [row["binary_label"] for row in source_rows]
    )

    y_target = np.array(
        [row["binary_label"] for row in target_rows]
    )

    model = make_model(multiclass=False)

    model.fit(
        source_hidden[best_layer, source_train_val, :],
        y_source[source_train_val],
    )

    X_target = target_hidden[best_layer, target_test, :]
    y_true = y_target[target_test]

    y_pred = model.predict(X_target)

    y_prob = aligned_predict_proba(
        model,
        X_target,
        BINARY_LABELS,
    )

    rng = np.random.default_rng(seed)

    metric_samples = {
        "balanced_accuracy": [],
        "macro_f1": [],
        "auroc": [],
    }

    for _ in range(n_bootstrap):
        indices = rng.integers(
            0,
            len(y_true),
            size=len(y_true),
        )

        sample_true = y_true[indices]

        # AUROC requires both classes in a bootstrap sample.
        if len(np.unique(sample_true)) < 2:
            continue

        sample_metrics = metrics(
            sample_true,
            y_pred[indices],
            y_prob[indices],
            BINARY_LABELS,
        )

        for key in metric_samples:
            metric_samples[key].append(sample_metrics[key])

    ci = {}

    for key, values in metric_samples.items():
        if not values:
            ci[f"{key}_95ci"] = None
        else:
            ci[f"{key}_95ci"] = [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]

    ci["n_bootstrap"] = n_bootstrap
    ci["valid_resamples"] = len(
        metric_samples["balanced_accuracy"]
    )

    return ci


def run_one(
    *,
    model_name: str,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    model_root = output_root / model_name / f"seed_{seed}"

    metadata = load_metadata(model_root)

    # Load the complete saved representation once. No model inference.
    hidden = load_hidden_states(
        model_root,
        REPRESENTATION,
    )

    validate_metadata_and_hidden(
        metadata,
        hidden,
        model_name=model_name,
        seed=seed,
    )

    # Binary cleaning exactly matches the existing R1 implementation.
    binary_rows = clean_binary(metadata)

    results = {
        "model": model_name,
        "model_id": MODELS[model_name],
        "seed": seed,
        "representation": REPRESENTATION,
        "task": TASK,
        "protocol": {
            "source_layer_selection": "source_train_to_source_validation",
            "final_fit": "source_train_plus_source_validation",
            "final_evaluation": "target_test_only",
            "target_train_used": False,
            "target_validation_used": False,
            "sandbox_definition": list(SANDBOX_ENVIRONMENTS),
            "target_environment": TARGET_ENVIRONMENT,
            "best_layer_criterion": "validation_balanced_accuracy",
            "best_layer_tie_break": (
                "validation_macro_f1_then_smallest_layer"
            ),
        },
        "directions": {},
    }

    for source_name, target_name in (
        ("sandbox", "scienceworld"),
        ("scienceworld", "sandbox"),
    ):
        source_rows, target_rows = source_target_rows(
            metadata,
            source_name,
            target_name,
        )

        if not source_rows or not target_rows:
            raise ValueError(
                f"Empty source/target rows for "
                f"{source_name} -> {target_name}"
            )

        source_indices = row_indices(
            metadata,
            source_rows,
        )

        target_indices = row_indices(
            metadata,
            target_rows,
        )

        source_hidden = hidden[:, source_indices, :]
        target_hidden = hidden[:, target_indices, :]

        # Safety: target rows are never included in source fitting/selection.
        if set(source_indices.tolist()) & set(target_indices.tolist()):
            raise AssertionError(
                f"Source/target overlap detected for "
                f"{source_name} -> {target_name}."
            )

        best_layer, layer_results = select_source_layer(
            source_hidden,
            source_rows,
        )

        transfer = fit_source_and_test_target(
            source_hidden,
            source_rows,
            target_hidden,
            target_rows,
            best_layer,
        )

        ci = bootstrap_target(
            source_hidden,
            source_rows,
            target_hidden,
            target_rows,
            best_layer,
            seed=seed + 50000 + best_layer,
        )

        results["directions"][
            f"{source_name}_to_{target_name}"
        ] = {
            "source": source_name,
            "target": target_name,
            "source_num_rows": len(source_rows),
            "source_split_counts": {
                split: int(
                    sum(
                        row["_split"] == split
                        for row in source_rows
                    )
                )
                for split in (
                    "train",
                    "validation",
                    "test",
                )
            },
            "target_num_rows": len(target_rows),
            "target_split_counts": {
                split: int(
                    sum(
                        row["_split"] == split
                        for row in target_rows
                    )
                )
                for split in (
                    "train",
                    "validation",
                    "test",
                )
            },
            "best_layer": best_layer,
            "best_layer_selection": {
                "criterion": "source_validation_balanced_accuracy",
                "tie_break": "source_validation_macro_f1",
                "final_tie_break": "smallest_layer",
            },
            "source_validation_at_best_layer": next(
                r["validation"]
                for r in layer_results
                if r["layer"] == best_layer
            ),
            "source_layers": layer_results,
            "final_target_test": transfer["target_test"],
            "target_test_n": transfer["target_test_n"],
            "target_test_class_counts": (
                transfer["target_test_class_counts"]
            ),
            "target_test_bootstrap_95ci": ci,
        }

        print(
            f"{model_name} seed={seed} "
            f"{source_name}->{target_name}: "
            f"best_layer={best_layer}, "
            f"target_BA="
            f"{transfer['target_test']['balanced_accuracy']:.4f}, "
            f"target_F1="
            f"{transfer['target_test']['macro_f1']:.4f}, "
            f"target_AUROC="
            f"{transfer['target_test']['auroc']}",
            flush=True,
        )

    results["environment"] = environment_info()

    return results


def write_summary(
    results: list[dict[str, Any]],
    output_root: Path,
) -> None:
    consolidated_dir = output_root / "consolidated"
    consolidated_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_seed_json = (
        consolidated_dir
        / "ab3_cross_environment_per_seed.json"
    )

    # Preserve previously completed seeds and add/replace only the
    # model/seed pairs produced by this invocation.
    existing_results = []

    if per_seed_json.exists():
        with per_seed_json.open(
            "r",
            encoding="utf-8",
        ) as f:
            existing_results = json.load(f)

    merged = {}

    for result in existing_results:
        key = (
            result["model"],
            result["seed"],
        )
        merged[key] = result

    for result in results:
        key = (
            result["model"],
            result["seed"],
        )
        merged[key] = result

    results = sorted(
        merged.values(),
        key=lambda result: (
            result["model"],
            result["seed"],
        ),
    )

    json_dump(
        per_seed_json,
        results,
    )

    rows = []

    for result in results:
        for direction, data in result["directions"].items():
            metrics_dict = data["final_target_test"]

            rows.append(
                {
                    "model": result["model"],
                    "seed": result["seed"],
                    "direction": direction,
                    "source": data["source"],
                    "target": data["target"],
                    "representation": result["representation"],
                    "best_layer": data["best_layer"],
                    "target_test_n": data["target_test_n"],
                    "balanced_accuracy": (
                        metrics_dict["balanced_accuracy"]
                    ),
                    "macro_f1": metrics_dict["macro_f1"],
                    "auroc": metrics_dict["auroc"],
                }
            )

    csv_path = (
        consolidated_dir
        / "ab3_cross_environment_per_seed.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []

    for direction in (
        "sandbox_to_scienceworld",
        "scienceworld_to_sandbox",
    ):
        for model_name in MODELS:
            matching = [
                row
                for row in rows
                if row["model"] == model_name
                and row["direction"] == direction
            ]

            if len(matching) != len(SEEDS):
                raise ValueError(
                    f"Expected {len(SEEDS)} rows for "
                    f"{model_name}/{direction}, "
                    f"got {len(matching)}."
                )

            aggregate = {
                "model": model_name,
                "direction": direction,
            }

            for metric_name in (
                "balanced_accuracy",
                "macro_f1",
                "auroc",
            ):
                values = np.array(
                    [
                        float(row[metric_name])
                        for row in matching
                    ],
                    dtype=float,
                )

                aggregate[
                    f"{metric_name}_mean"
                ] = float(np.mean(values))

                aggregate[
                    f"{metric_name}_std"
                ] = float(np.std(values, ddof=1))

            aggregate_rows.append(aggregate)

    aggregate_csv = (
        consolidated_dir
        / "ab3_cross_environment_three_seed_summary.csv"
    )

    with aggregate_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                aggregate_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(aggregate_rows)

    json_dump(
        consolidated_dir
        / "ab3_cross_environment_three_seed_summary.json",
        aggregate_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/rq1"),
        help=(
            "Existing R1 result root containing "
            "qwen/ and llama/."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODELS),
        default=list(MODELS),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=SEEDS,
        default=list(SEEDS),
    )

    args = parser.parse_args()

    all_results = []

    for model_name in args.models:
        for seed in args.seeds:
            all_results.append(
                run_one(
                    model_name=model_name,
                    seed=seed,
                    output_root=args.output_root,
                )
            )

    write_summary(
        all_results,
        args.output_root,
    )

    print(
        "\nAB3 completed successfully.",
        flush=True,
    )

    print(
        "No model inference was performed; "
        "saved hidden states were reused.",
        flush=True,
    )


if __name__ == "__main__":
    main()