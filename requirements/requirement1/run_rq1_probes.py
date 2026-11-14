from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


# ============================================================================
# Configuration
# ============================================================================

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

SEEDS = (42, 43, 44)

ENVIRONMENTS = (
    "filesystem",
    "email_outbox",
    "wallet",
    "scienceworld",
    "toolemu",
)

BINARY_LABELS = (
    "reversible",
    "irreversible",
)

THREE_CLASS_LABELS = (
    "reversible",
    "partially_reversible",
    "irreversible",
)

REPRESENTATIONS = (
    "last_action_token",
    "mean_action_tokens",
)

N_BOOTSTRAP = 1000
CONTROL_SEED = 12345

# Logistic-regression settings are fixed across all experiments.
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 2000


# ============================================================================
# General utilities
# ============================================================================


def json_dump(path: Path, obj: Any) -> None:
    """Write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def jsonl_write(path: Path, rows: list[dict]) -> None:
    """Write JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

    return rows


def environment_info() -> dict:
    """Record the software/runtime environment used for the probe."""
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }

    try:
        import scipy

        info["scipy"] = scipy.__version__
    except Exception:
        info["scipy"] = None

    return info


# ============================================================================
# Metadata and hidden-state loading
# ============================================================================


def load_metadata(model_dir: Path) -> list[dict]:
    """
    Load extraction metadata.

    The extraction script writes one metadata row per original dataset row.
    The metadata order is therefore the canonical hidden-state row order.
    """
    path = model_dir / "metadata.jsonl"
    rows = load_jsonl(path)

    if not rows:
        raise RuntimeError(f"metadata.jsonl is empty: {path}")

    required = {
        "id",
        "environment",
        "scenario_id",
        "group_id",
        "label",
        "binary_label",
        "action_text",
        "action_token_count",
        "split",
        "batch",
        "index",
    }

    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            "metadata.jsonl is missing required fields: "
            + ", ".join(sorted(missing))
        )

    for global_index, row in enumerate(rows):
        row["_metadata_index"] = global_index

        # The extraction script uses "_split" in its internal representation
        # but saves "split" to disk. Normalize to "_split".
        row["_split"] = row.get("_split", row["split"])

        if row["_split"] not in {"train", "validation", "test"}:
            raise ValueError(
                f"Invalid split {row['_split']!r} for row {row['id']}"
            )

        if row["environment"] not in ENVIRONMENTS:
            raise ValueError(
                f"Unknown environment {row['environment']!r} "
                f"for row {row['id']}"
            )

        if row["label"] not in (
            "reversible",
            "partially_reversible",
            "irreversible",
            "inconclusive",
        ):
            raise ValueError(
                f"Unknown label {row['label']!r} for row {row['id']}"
            )

    return rows


def _batch_number(path: Path) -> int:
    """
    Extract the numeric batch number from a hidden-state filename.

    This avoids the classic lexicographic ordering problem:
        batch_10.npy appearing before batch_2.npy.
    """
    match = re.search(r"(\d+)", path.stem)

    if match is None:
        raise ValueError(
            f"Could not determine batch number from filename: {path.name}"
        )

    return int(match.group(1))


def load_hidden_states(
    model_dir: Path,
    representation: str,
) -> np.ndarray:
    """
    Load saved hidden states.

    Returns:
        ndarray with shape [num_layers, num_rows, hidden_dimension].
    """
    hidden_dir = model_dir / "hidden_states"

    if not hidden_dir.exists():
        raise FileNotFoundError(
            f"Hidden-state directory does not exist: {hidden_dir}"
        )

    if representation == "last_action_token":
        pattern = "*_last.npy"
    elif representation == "mean_action_tokens":
        pattern = "*_mean.npy"
    else:
        raise ValueError(f"Unknown representation: {representation}")

    files = list(hidden_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No hidden-state files found for representation "
            f"{representation!r} in {hidden_dir}"
        )

    files.sort(key=_batch_number)

    arrays = []

    for path in files:
        array = np.load(path, mmap_mode="r")

        if array.ndim != 3:
            raise ValueError(
                f"Expected 3-D hidden state array in {path}, "
                f"got shape {array.shape}"
            )

        arrays.append(array)

    hidden = np.concatenate(arrays, axis=1)

    return hidden


def verify_hidden_state_alignment(
    hidden: np.ndarray,
    metadata: list[dict],
) -> None:
    """Verify that saved hidden states align with metadata rows."""
    if hidden.shape[1] != len(metadata):
        raise ValueError(
            "Hidden-state/metadata row mismatch: "
            f"hidden states contain {hidden.shape[1]} rows but "
            f"metadata contains {len(metadata)} rows."
        )

    if hidden.shape[0] < 2:
        raise ValueError(
            f"Expected layer 0 plus at least one transformer layer; "
            f"got {hidden.shape[0]} layers."
        )

    if hidden.shape[2] <= 0:
        raise ValueError(
            f"Invalid hidden dimension: {hidden.shape[2]}"
        )


# ============================================================================
# Dataset cleaning
# ============================================================================


def clean_binary(rows: list[dict]) -> list[dict]:
    """
    Primary binary task.

    Keep only:
        reversible
        irreversible

    Therefore:
        - inconclusive rows are removed
        - partially_reversible rows are removed

    This follows the predefined binary_label field rather than silently
    merging partial reversibility into either binary class.
    """
    return [
        row
        for row in rows
        if row["label"] != "inconclusive"
        and row["binary_label"] in BINARY_LABELS
    ]


def clean_three_class(rows: list[dict]) -> list[dict]:
    """
    Three-class task.

    Keep:
        reversible
        partially_reversible
        irreversible

    Remove:
        inconclusive
    """
    return [
        row
        for row in rows
        if row["label"] in THREE_CLASS_LABELS
    ]


def add_minimal_pair_flag(rows: list[dict]) -> None:
    """
    Mark rows belonging to minimal-pair groups.

    A minimal-pair group is a group_id containing more than one
    reversibility label.

    The grouping is computed from the full metadata for the relevant
    environment, before binary/three-class filtering.
    """
    labels_by_group: dict[str, set[str]] = {}

    for row in rows:
        group_id = row["group_id"]
        label = row["label"]

        labels_by_group.setdefault(group_id, set()).add(label)

    minimal_groups = {
        group_id
        for group_id, labels in labels_by_group.items()
        if len(labels) > 1
    }

    for row in rows:
        row["_minimal_pair"] = row["group_id"] in minimal_groups


def action_verb(action_text: str) -> str:
    """
    Extract the action/tool name.

    Examples:
        delete_file(path="/a.txt") -> delete_file
        send_email(to="x")         -> send_email
        move_file(src="/a")        -> move_file
    """
    match = re.match(
        r"\s*([A-Za-z_][A-Za-z0-9_]*)",
        action_text,
    )

    if match:
        return match.group(1)

    stripped = action_text.strip()

    if not stripped:
        return ""

    return stripped.split()[0]


# ============================================================================
# Dataset / split validation
# ============================================================================


def validate_split_structure(
    rows: list[dict],
    *,
    context: str,
) -> None:
    """Validate that the rows contain the expected scenario-disjoint split labels."""

    counts = Counter(row["_split"] for row in rows)

    for split in ("train", "validation", "test"):
        if counts[split] == 0:
            raise ValueError(
                f"{context}: split {split!r} contains zero rows."
            )

    scenario_to_splits: dict[str, set[str]] = {}

    for row in rows:
        scenario_to_splits.setdefault(
            row["scenario_id"],
            set(),
        ).add(row["_split"])

    leakage = {
        scenario: sorted(splits)
        for scenario, splits in scenario_to_splits.items()
        if len(splits) > 1
    }

    if leakage:
        preview = dict(list(leakage.items())[:10])

        raise ValueError(
            f"{context}: scenario leakage detected. "
            f"Some scenario_ids occur in multiple splits: {preview}"
        )


def validate_class_availability(
    rows: list[dict],
    label_key: str,
    labels: tuple[str, ...],
    *,
    context: str,
) -> dict:
    """
    Verify class availability.

    Logistic regression requires at least two training classes.
    For three-class analysis, every class must occur in training.
    """
    split_counts = {}

    for split in ("train", "validation", "test"):
        split_rows = [
            row
            for row in rows
            if row["_split"] == split
        ]

        counts = class_counts(split_rows, label_key)
        split_counts[split] = counts

    train_labels = set(split_counts["train"])

    if len(train_labels) < 2:
        raise ValueError(
            f"{context}: training split has fewer than two classes: "
            f"{split_counts['train']}"
        )

    missing_train = [
        label
        for label in labels
        if label not in train_labels
    ]

    if missing_train:
        raise ValueError(
            f"{context}: training split is missing required class(es): "
            f"{missing_train}. "
            f"Training counts: {split_counts['train']}"
        )

    return split_counts


# ============================================================================
# Logistic regression
# ============================================================================


def make_model(multiclass: bool = False) -> LogisticRegression:
    """
    Fixed L2-regularized logistic regression.

    C=1.0 is held fixed across layers and environments so that the
    representation comparison is not confounded by per-layer tuning.
    """
    if multiclass:
        # Do not pass the deprecated multi_class parameter.
        # lbfgs automatically handles multinomial logistic regression
        # when more than two classes are present.
        return LogisticRegression(
            penalty="l2",
            C=LOGISTIC_C,
            solver="lbfgs",
            max_iter=LOGISTIC_MAX_ITER,
        )

    return LogisticRegression(
        penalty="l2",
        C=LOGISTIC_C,
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
    )


def aligned_predict_proba(
    model: Any,
    X: np.ndarray,
    labels: tuple[str, ...],
) -> np.ndarray:
    """Return class probabilities in the project's fixed label order.

    scikit-learn returns predict_proba columns in ``model.classes_`` order,
    not necessarily in the order used by the project constants. Keeping the
    columns aligned is essential for AUROC, especially because
    ``irreversible`` is not the first label in BINARY_LABELS.
    """
    raw_prob = model.predict_proba(X)
    classes = [str(value) for value in model.classes_]
    expected = list(labels)

    if set(classes) != set(expected):
        raise ValueError(
            "Classifier probability classes do not match the requested "
            f"labels. classes_={classes}, labels={expected}"
        )

    column_indices = [
        classes.index(label)
        for label in expected
    ]

    return raw_prob[:, column_indices]


def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    multiclass: bool,
    labels: tuple[str, ...],
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """Fit the probe and return model, predictions and aligned probabilities."""
    model = make_model(multiclass=multiclass)

    model.fit(X_train, y_train)

    pred = model.predict(X_eval)
    prob = aligned_predict_proba(
        model,
        X_eval,
        labels,
    )

    return model, pred, prob


# ============================================================================
# Metrics
# ============================================================================


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    labels: tuple[str, ...],
) -> dict:
    """Compute balanced accuracy, macro-F1 and AUROC."""
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set.")

    result = {
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
    }

    if y_prob is None:
        result["auroc"] = None
        return result

    try:
        if len(labels) == 2:
            positive_index = list(labels).index("irreversible")

            # sklearn treats the larger string label as the positive class
            # when y_true contains strings. That would make "reversible"
            # positive because it sorts after "irreversible". The project
            # defines irreversible as the positive class, so encode it
            # explicitly as 1 and reversible as 0.
            y_true_binary = np.asarray(
                [1 if value == "irreversible" else 0 for value in y_true],
                dtype=int,
            )

            result["auroc"] = float(
                roc_auc_score(
                    y_true_binary,
                    y_prob[:, positive_index],
                )
            )
        else:
            # scikit-learn requires multiclass ``labels`` to be ordered.
            # The project label order is fixed as:
            # reversible, partially_reversible, irreversible.
            # Encode the labels to 0, 1, 2 in that project order while
            # keeping the probability columns in the same aligned order.
            label_to_index = {label: index for index, label in enumerate(labels)}
            y_true_multiclass = np.asarray(
                [label_to_index[value] for value in y_true],
                dtype=int,
            )

            result["auroc"] = float(
                roc_auc_score(
                    y_true_multiclass,
                    y_prob,
                    labels=list(range(len(labels))),
                    multi_class="ovr",
                    average="macro",
                )
            )

    except ValueError:
        result["auroc"] = None

    return result


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    labels: tuple[str, ...],
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """
    Non-parametric bootstrap 95% confidence intervals.

    The test predictions are fixed; bootstrap resampling is performed
    only over test examples.
    """
    n = len(y_true)

    if n == 0:
        return {
            "balanced_accuracy_95ci": None,
            "macro_f1_95ci": None,
            "auroc_95ci": None,
        }

    rng = np.random.default_rng(seed)

    ba_values: list[float] = []
    f1_values: list[float] = []
    auc_values: list[float] = []

    for _ in range(n_bootstrap):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample_true = y_true[indices]
        sample_pred = y_pred[indices]

        ba_values.append(
            float(
                balanced_accuracy_score(
                    sample_true,
                    sample_pred,
                )
            )
        )

        f1_values.append(
            float(
                f1_score(
                    sample_true,
                    sample_pred,
                    labels=list(labels),
                    average="macro",
                    zero_division=0,
                )
            )
        )

        if y_prob is not None:
            try:
                if len(labels) == 2:
                    positive_index = list(labels).index(
                        "irreversible"
                    )

                    # Keep the bootstrap AUROC convention identical to
                    # the main metric: irreversible = 1, reversible = 0.
                    sample_true_binary = np.asarray(
                        [
                            1 if value == "irreversible"
                            else 0
                            for value in sample_true
                        ],
                        dtype=int,
                    )

                    auc = roc_auc_score(
                        sample_true_binary,
                        y_prob[
                            indices,
                            positive_index,
                        ],
                    )
                else:
                    # Use the same project-defined numeric class order as
                    # the main multiclass AUROC calculation.
                    label_to_index = {
                        label: index
                        for index, label in enumerate(labels)
                    }
                    sample_true_multiclass = np.asarray(
                        [
                            label_to_index[value]
                            for value in sample_true
                        ],
                        dtype=int,
                    )

                    auc = roc_auc_score(
                        sample_true_multiclass,
                        y_prob[indices],
                        labels=list(range(len(labels))),
                        multi_class="ovr",
                        average="macro",
                    )

                auc_values.append(float(auc))

            except ValueError:
                pass

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None

        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]

    return {
        "balanced_accuracy_95ci": interval(ba_values),
        "macro_f1_95ci": interval(f1_values),
        "auroc_95ci": interval(auc_values),
        "n_bootstrap": n_bootstrap,
    }


# ============================================================================
# Counts / layer evaluation
# ============================================================================


def class_counts(
    rows: list[dict],
    label_key: str,
) -> dict[str, int]:
    """Return deterministic class counts."""
    return dict(
        sorted(
            Counter(
                row[label_key]
                for row in rows
            ).items()
        )
    )


def split_indices(
    rows: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, validation and test indices."""
    train = np.array(
        [
            i
            for i, row in enumerate(rows)
            if row["_split"] == "train"
        ],
        dtype=int,
    )

    validation = np.array(
        [
            i
            for i, row in enumerate(rows)
            if row["_split"] == "validation"
        ],
        dtype=int,
    )

    test = np.array(
        [
            i
            for i, row in enumerate(rows)
            if row["_split"] == "test"
        ],
        dtype=int,
    )

    return train, validation, test


def evaluate_validation_layer(
    X: np.ndarray,
    rows: list[dict],
    layer: int,
    label_key: str,
    labels: tuple[str, ...],
    subset_name: str,
) -> dict:
    """
    Evaluate one layer on validation only.

    IMPORTANT:
    The test set is intentionally not touched here.
    This function is used for best-layer selection.
    """
    train_indices, validation_indices, _ = split_indices(rows)

    if len(train_indices) == 0:
        raise ValueError(
            f"{subset_name}: no training examples."
        )

    if len(validation_indices) == 0:
        raise ValueError(
            f"{subset_name}: no validation examples."
        )

    y = np.array(
        [row[label_key] for row in rows]
    )

    multiclass = len(labels) == 3

    _, val_pred, val_prob = fit_predict(
        X[train_indices],
        y[train_indices],
        X[validation_indices],
        multiclass=multiclass,
        labels=labels,
    )

    val_metrics = metrics(
        y[validation_indices],
        val_pred,
        val_prob,
        labels,
    )

    return {
        "subset": subset_name,
        "layer": layer,
        "validation": val_metrics,
    }


def select_best_layer(
    results: list[dict],
) -> int:
    """
    Select the best layer using validation only.

    Primary criterion:
        validation balanced accuracy

    Tie-break:
        validation macro-F1

    Final tie-break:
        smallest layer
    """
    if not results:
        raise ValueError("Cannot select a layer from empty results.")

    best = max(
        results,
        key=lambda r: (
            r["validation"]["balanced_accuracy"],
            r["validation"]["macro_f1"],
            -r["layer"],
        ),
    )

    return int(best["layer"])


def train_test_best_layer(
    X: np.ndarray,
    rows: list[dict],
    layer: int,
    label_key: str,
    labels: tuple[str, ...],
) -> tuple[
    dict,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Final evaluation for a selected layer.

    The model is refit on:
        train + validation

    and evaluated once on:
        untouched test

    This is the only test evaluation used for the selected R1 layer.
    """
    train_val_indices = np.array(
        [
            i
            for i, row in enumerate(rows)
            if row["_split"] in ("train", "validation")
        ],
        dtype=int,
    )

    test_indices = np.array(
        [
            i
            for i, row in enumerate(rows)
            if row["_split"] == "test"
        ],
        dtype=int,
    )

    if len(train_val_indices) == 0:
        raise ValueError(
            f"Layer {layer}: no train+validation examples."
        )

    if len(test_indices) == 0:
        raise ValueError(
            f"Layer {layer}: no test examples."
        )

    y = np.array(
        [row[label_key] for row in rows]
    )

    model = make_model(
        multiclass=len(labels) == 3
    )

    model.fit(
        X[train_val_indices],
        y[train_val_indices],
    )

    test_pred = model.predict(
        X[test_indices]
    )

    test_prob = aligned_predict_proba(
        model,
        X[test_indices],
        labels,
    )

    test_metrics = metrics(
        y[test_indices],
        test_pred,
        test_prob,
        labels,
    )

    return (
        test_metrics,
        test_pred,
        test_prob,
        test_indices,
    )


# ============================================================================
# Required controls
# ============================================================================


def run_majority(
    rows: list[dict],
    label_key: str,
    labels: tuple[str, ...],
) -> dict:
    """Majority-class baseline."""
    train_rows = [
        row
        for row in rows
        if row["_split"] == "train"
    ]

    validation_rows = [
        row
        for row in rows
        if row["_split"] == "validation"
    ]

    test_rows = [
        row
        for row in rows
        if row["_split"] == "test"
    ]

    counts = Counter(
        row[label_key]
        for row in train_rows
    )

    majority = counts.most_common(1)[0][0]

    def evaluate(eval_rows: list[dict]) -> dict:
        y_true = np.array(
            [row[label_key] for row in eval_rows]
        )

        y_pred = np.array(
            [majority] * len(eval_rows)
        )

        return metrics(
            y_true,
            y_pred,
            None,
            labels,
        )

    return {
        "control": "majority",
        "majority_class": majority,
        "validation": evaluate(validation_rows),
        "test": evaluate(test_rows),
        "class_counts_train": class_counts(
            train_rows,
            label_key,
        ),
        "class_counts_validation": class_counts(
            validation_rows,
            label_key,
        ),
        "class_counts_test": class_counts(
            test_rows,
            label_key,
        ),
    }


def run_bow(
    rows: list[dict],
    label_key: str,
    labels: tuple[str, ...],
) -> dict:
    """
    Lexical bag-of-words baseline using action_text only.

    No hidden-state information is used.
    """
    train_rows = [
        row
        for row in rows
        if row["_split"] == "train"
    ]

    validation_rows = [
        row
        for row in rows
        if row["_split"] == "validation"
    ]

    test_rows = [
        row
        for row in rows
        if row["_split"] == "test"
    ]

    vectorizer = CountVectorizer(
        lowercase=True,
        binary=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    classifier = LogisticRegression(
        penalty="l2",
        C=LOGISTIC_C,
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITER,
    )

    pipeline = Pipeline(
        [
            ("bow", vectorizer),
            ("classifier", classifier),
        ]
    )

    X_train_text = [
        row["action_text"]
        for row in train_rows
    ]

    y_train = np.array(
        [
            row[label_key]
            for row in train_rows
        ]
    )

    pipeline.fit(
        X_train_text,
        y_train,
    )

    def evaluate(eval_rows: list[dict]) -> dict:
        texts = [
            row["action_text"]
            for row in eval_rows
        ]

        y_true = np.array(
            [
                row[label_key]
                for row in eval_rows
            ]
        )

        y_pred = pipeline.predict(texts)
        y_prob = aligned_predict_proba(
            pipeline,
            texts,
            labels,
        )

        return metrics(
            y_true,
            y_pred,
            y_prob,
            labels,
        )

    return {
        "control": "lexical_bow",
        "input": "action_text_only",
        "validation": evaluate(validation_rows),
        "test": evaluate(test_rows),
    }


def run_shuffled_label_control(
    X: np.ndarray,
    rows: list[dict],
    layer: int,
    label_key: str,
    labels: tuple[str, ...],
    seed: int,
) -> dict:
    """
    Shuffled-label probe.

    The representation remains unchanged, but training labels are randomly
    permuted. Test labels remain the true labels and are used only for
    evaluating how much performance survives label randomization.

    This control is evaluated at the same selected layer as the primary
    representation probe.
    """
    train_indices, _, test_indices = split_indices(rows)

    y = np.array(
        [row[label_key] for row in rows]
    )

    rng = np.random.default_rng(seed)

    shuffled_train_labels = y[train_indices].copy()
    rng.shuffle(shuffled_train_labels)

    model = make_model(
        multiclass=len(labels) == 3
    )

    model.fit(
        X[train_indices],
        shuffled_train_labels,
    )

    test_pred = model.predict(
        X[test_indices]
    )

    test_prob = aligned_predict_proba(
        model,
        X[test_indices],
        labels,
    )

    test_metrics = metrics(
        y[test_indices],
        test_pred,
        test_prob,
        labels,
    )

    return {
        "control": "shuffled_labels",
        "layer": layer,
        "shuffle_seed": seed,
        "test": test_metrics,
    }


def run_action_identity_control(
    X: np.ndarray,
    rows: list[dict],
    layer: int,
    labels: tuple[str, ...],
    seed: int,
) -> dict:
    """
    Hewitt-Liang-style selectivity control.

    Each action verb is assigned one random fixed label.
    Every occurrence of that verb receives the same random target.

    The classifier is trained to predict this RANDOM ACTION-IDENTITY TARGET,
    not reversibility.

    Therefore:
        control_target_test
    is the meaningful metric. The random mapping is constrained only enough
    to ensure every required control class occurs in the training split.

    We do NOT report the classifier's predictions against the true
    reversibility labels as if that were a reversibility probe.
    """
    rng = np.random.default_rng(seed)

    verbs = sorted(
        {
            action_verb(row["action_text"])
            for row in rows
        }
    )

    if not verbs:
        raise ValueError(
            "Action-identity control found no action verbs."
        )

    train_indices, _, test_indices = split_indices(rows)

    train_verbs = sorted(
        {
            action_verb(rows[i]["action_text"])
            for i in train_indices
        }
    )

    if len(train_verbs) < len(labels):
        raise ValueError(
            "Action-identity control cannot assign all required control "
            f"classes: only {len(train_verbs)} distinct action verbs occur "
            f"in training, but {len(labels)} labels are required."
        )

    # Random fixed labels per verb, with rejection sampling only to ensure
    # every classifier class is represented in training. This preserves the
    # intended random action-identity control while preventing an invalid
    # one-class/two-class training target by chance.
    for _ in range(10000):
        verb_to_label = {
            verb: labels[
                int(
                    rng.integers(
                        0,
                        len(labels),
                    )
                )
            ]
            for verb in verbs
        }

        trial_train_y = np.array(
            [
                verb_to_label[
                    action_verb(
                        rows[i]["action_text"]
                    )
                ]
                for i in train_indices
            ]
        )

        if set(trial_train_y) == set(labels):
            break
    else:
        raise RuntimeError(
            "Could not generate a random fixed-label action-identity "
            "mapping containing every required class in training."
        )

    control_train_y = trial_train_y

    control_test_y = np.array(
        [
            verb_to_label[
                action_verb(
                    rows[i]["action_text"]
                )
            ]
            for i in test_indices
        ]
    )

    model = make_model(
        multiclass=len(labels) == 3
    )

    model.fit(
        X[train_indices],
        control_train_y,
    )

    predicted_control = model.predict(
        X[test_indices]
    )

    control_prob = aligned_predict_proba(
        model,
        X[test_indices],
        labels,
    )

    return {
        "control": "action_identity_hewitt_liang",
        "layer": layer,
        "random_seed": seed,
        "num_action_verbs": len(verbs),
        "control_target_test": metrics(
            control_test_y,
            predicted_control,
            control_prob,
            labels,
        ),
    }


def run_layer0_control(
    X: np.ndarray,
    rows: list[dict],
    label_key: str,
    labels: tuple[str, ...],
) -> dict:
    """
    Explicit layer-0 probe.

    Layer 0 is the embedding-layer control required by the project.
    It is evaluated using the same train+validation -> test protocol as
    any other selected layer, but it is reported separately as a control.
    """
    test_metrics, _, _, _ = train_test_best_layer(
        X=X,
        rows=rows,
        layer=0,
        label_key=label_key,
        labels=labels,
    )

    return {
        "control": "layer_0_probe",
        "layer": 0,
        "test": test_metrics,
    }


# ============================================================================
# R1 experiment for one environment / task / subset
# ============================================================================


def run_probe_subset(
    *,
    hidden: np.ndarray,
    rows: list[dict],
    label_key: str,
    labels: tuple[str, ...],
    representation: str,
    task_name: str,
    subset_name: str,
    model_name: str,
    seed: int,
    environment: str,
    output_dir: Path,
) -> dict:
    """
    Run the complete R1 layer-wise probe for one:

        model
        seed
        environment
        representation
        task
        subset

    Layer selection:
        train -> validation

    Final evaluation:
        train + validation -> untouched test
    """
    if not rows:
        raise ValueError(
            f"{environment}/{task_name}/{subset_name}: no rows."
        )

    split_counts = Counter(
        row["_split"]
        for row in rows
    )

    print(
        f"      subset={subset_name} "
        f"rows={len(rows)} "
        f"splits={dict(split_counts)}",
        flush=True,
    )

    validate_split_structure(
        rows,
        context=(
            f"{environment}/{task_name}/{subset_name}"
        ),
    )

    split_class_counts = validate_class_availability(
        rows,
        label_key,
        labels,
        context=(
            f"{environment}/{task_name}/{subset_name}"
        ),
    )

    layer_results: list[dict] = []

    num_layers = hidden.shape[0]

    for layer in range(num_layers):
        result = evaluate_validation_layer(
            X=hidden[layer],
            rows=rows,
            layer=layer,
            label_key=label_key,
            labels=labels,
            subset_name=subset_name,
        )

        layer_results.append(result)

    best_layer = select_best_layer(
        layer_results
    )

    final_test_metrics, final_test_pred, final_test_prob, test_indices = (
        train_test_best_layer(
            X=hidden[best_layer],
            rows=rows,
            layer=best_layer,
            label_key=label_key,
            labels=labels,
        )
    )

    y = np.array(
        [row[label_key] for row in rows]
    )

    y_test = y[test_indices]

    ci = bootstrap_ci(
        y_true=y_test,
        y_pred=final_test_pred,
        y_prob=final_test_prob,
        labels=labels,
        seed=(
            seed
            + 1000 * best_layer
            + len(environment)
            + len(task_name)
            + len(subset_name)
        ),
    )

    # Locate the validation result for the selected layer.
    selected_validation = next(
        result["validation"]
        for result in layer_results
        if result["layer"] == best_layer
    )

    result = {
        "model": model_name,
        "seed": seed,
        "environment": environment,
        "representation": representation,
        "task": task_name,
        "subset": subset_name,
        "label_key": label_key,
        "labels": list(labels),
        "num_rows": len(rows),
        "split_counts": dict(split_counts),
        "class_counts": split_class_counts,
        "best_layer": best_layer,
        "best_layer_selection": {
            "criterion": "validation_balanced_accuracy",
            "tie_break": "validation_macro_f1",
            "final_tie_break": "smallest_layer",
        },
        "validation_at_best_layer": selected_validation,
        "final_test": final_test_metrics,
        "bootstrap_95ci": ci,
        "layers": layer_results,
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_dump(
        output_dir / "layers.json",
        result,
    )

    print(
        f"        best_layer={best_layer} "
        f"val_BA={selected_validation['balanced_accuracy']:.4f} "
        f"test_BA={final_test_metrics['balanced_accuracy']:.4f} "
        f"test_F1={final_test_metrics['macro_f1']:.4f} "
        f"test_AUROC={final_test_metrics['auroc']}",
        flush=True,
    )

    return result


# ============================================================================
# Environment-specific R1 analysis
# ============================================================================


def run_environment_r1(
    *,
    hidden: np.ndarray,
    metadata: list[dict],
    environment: str,
    representation: str,
    model_name: str,
    seed: int,
    output_dir: Path,
) -> dict:
    """
    Run all required R1 analyses for one environment.

    Structure:

        binary / three_class
            all / minimal_pairs
                every layer
                best layer
                final test
                bootstrap CI

        controls
            majority
            lexical BOW
            shuffled labels
            action identity
            layer 0
    """
    environment_metadata = [
        row
        for row in metadata
        if row["environment"] == environment
    ]

    if not environment_metadata:
        raise ValueError(
            f"No metadata rows found for environment {environment!r}"
        )

    # Convert global metadata indices to hidden-state indices.
    global_indices = np.array(
        [
            row["_metadata_index"]
            for row in environment_metadata
        ],
        dtype=int,
    )

    environment_hidden = hidden[
        :,
        global_indices,
        :,
    ]

    if environment_hidden.shape[1] != len(environment_metadata):
        raise ValueError(
            f"{environment}: hidden-state/metadata mismatch."
        )

    environment_output = (
        output_dir / environment / representation
    )

    environment_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    environment_summary: dict[str, Any] = {
        "model": model_name,
        "seed": seed,
        "environment": environment,
        "representation": representation,
        "num_rows": len(environment_metadata),
        "class_counts": class_counts(
            environment_metadata,
            "label",
        ),
        "binary_class_counts": class_counts(
            [
                row
                for row in environment_metadata
                if row["binary_label"] in BINARY_LABELS
            ],
            "binary_label",
        ),
        "tasks": {},
    }

    # ------------------------------------------------------------------
    # R1 binary + three-class
    # ------------------------------------------------------------------

    task_specs = (
        (
            "binary",
            "binary_label",
            BINARY_LABELS,
            clean_binary,
        ),
        (
            "three_class",
            "label",
            THREE_CLASS_LABELS,
            clean_three_class,
        ),
    )

    for (
        task_name,
        label_key,
        labels,
        cleaner,
    ) in task_specs:

        task_rows = cleaner(
            environment_metadata
        )

        task_output = (
            environment_output / task_name
        )

        if not task_rows:
            environment_summary["tasks"][task_name] = {
                "status": "skipped_empty"
            }
            continue

        # Original indices within environment_metadata.
        task_row_ids = {
            row["id"]
            for row in task_rows
        }

        task_local_indices = np.array(
            [
                i
                for i, row in enumerate(
                    environment_metadata
                )
                if row["id"] in task_row_ids
            ],
            dtype=int,
        )

        if len(task_local_indices) != len(task_rows):
            raise ValueError(
                f"{environment}/{task_name}: task-row alignment "
                "failed between metadata and hidden states."
            )

        task_hidden = environment_hidden[
            :,
            task_local_indices,
            :,
        ]

        task_summary: dict[str, Any] = {
            "status": "ok",
            "num_rows": len(task_rows),
            "class_counts": class_counts(
                task_rows,
                label_key,
            ),
            "subsets": {},
        }

        # AB2:
        #   all data
        #   minimal pairs only
        subset_specs = {
            "all": np.arange(
                len(task_rows),
                dtype=int,
            ),
            "minimal_pairs": np.array(
                [
                    i
                    for i, row in enumerate(task_rows)
                    if row.get("_minimal_pair", False)
                ],
                dtype=int,
            ),
        }

        for subset_name, subset_indices in subset_specs.items():

            if len(subset_indices) == 0:
                task_summary["subsets"][subset_name] = {
                    "status": "skipped_empty"
                }
                continue

            subset_rows = [
                task_rows[i]
                for i in subset_indices
            ]

            subset_hidden = task_hidden[
                :,
                subset_indices,
                :,
            ]

            subset_output = (
                task_output / subset_name
            )

            try:
                subset_result = run_probe_subset(
                    hidden=subset_hidden,
                    rows=subset_rows,
                    label_key=label_key,
                    labels=labels,
                    representation=representation,
                    task_name=task_name,
                    subset_name=subset_name,
                    model_name=model_name,
                    seed=seed,
                    environment=environment,
                    output_dir=subset_output,
                )

                task_summary["subsets"][subset_name] = (
                    subset_result
                )

            except ValueError as exc:
                # A rare class may be absent from a scenario-disjoint
                # training split, making a three-class logistic probe
                # mathematically undefined. Record this explicitly instead
                # of silently changing the experiment.
                task_summary["subsets"][subset_name] = {
                    "status": "skipped_invalid_training_setup",
                    "error": str(exc),
                    "num_rows": len(subset_rows),
                    "class_counts": class_counts(
                        subset_rows,
                        label_key,
                    ),
                }

                print(
                    f"        SKIPPED "
                    f"{environment}/{task_name}/{subset_name}: "
                    f"{exc}",
                    flush=True,
                )

        # --------------------------------------------------------------
        # Controls for this exact environment/task.
        # Controls use the ALL subset so they remain directly comparable
        # with the main all-data probe. AB2 has its own lexical results
        # below as well.
        # --------------------------------------------------------------

        all_indices = subset_specs["all"]
        all_rows = [
            task_rows[i]
            for i in all_indices
        ]
        all_hidden = task_hidden[
            :,
            all_indices,
            :,
        ]

        controls_output = (
            task_output / "controls"
        )

        controls_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        controls: dict[str, Any] = {}

        try:
            controls["majority"] = run_majority(
                all_rows,
                label_key,
                labels,
            )

            controls["lexical_bow"] = run_bow(
                all_rows,
                label_key,
                labels,
            )

            all_probe_result = (
                task_summary["subsets"]
                .get("all")
            )

            if (
                all_probe_result is not None
                and all_probe_result.get("status", "ok") == "ok"
            ):
                best_layer = all_probe_result["best_layer"]

                controls["shuffled_labels"] = (
                    run_shuffled_label_control(
                        X=all_hidden[best_layer],
                        rows=all_rows,
                        layer=best_layer,
                        label_key=label_key,
                        labels=labels,
                        seed=CONTROL_SEED,
                    )
                )

                controls["action_identity_hewitt_liang"] = (
                    run_action_identity_control(
                        X=all_hidden[best_layer],
                        rows=all_rows,
                        layer=best_layer,
                        labels=labels,
                        seed=CONTROL_SEED,
                    )
                )

            else:
                controls["shuffled_labels"] = {
                    "status": "skipped_no_valid_primary_probe"
                }

                controls["action_identity_hewitt_liang"] = {
                    "status": "skipped_no_valid_primary_probe"
                }

            # Layer 0 is always evaluated independently as an explicit
            # embedding-layer control.
            controls["layer_0_probe"] = run_layer0_control(
                X=all_hidden[0],
                rows=all_rows,
                label_key=label_key,
                labels=labels,
            )

            # ----------------------------------------------------------
            # AB2 lexical BOW on minimal pairs only.
            # This is important because the handover specifically says
            # the lexical baseline should be beaten on minimal pairs.
            # ----------------------------------------------------------

            minimal_indices = subset_specs[
                "minimal_pairs"
            ]

            if len(minimal_indices) > 0:
                minimal_rows = [
                    task_rows[i]
                    for i in minimal_indices
                ]

                controls["lexical_bow_minimal_pairs"] = (
                    run_bow(
                        minimal_rows,
                        label_key,
                        labels,
                    )
                )
            else:
                controls["lexical_bow_minimal_pairs"] = {
                    "status": "skipped_empty"
                }

        except ValueError as exc:
            controls["status"] = "failed"
            controls["error"] = str(exc)

        json_dump(
            controls_output / "controls.json",
            {
                "model": model_name,
                "seed": seed,
                "environment": environment,
                "representation": representation,
                "task": task_name,
                "controls": controls,
            },
        )

        task_summary["controls"] = controls

        json_dump(
            task_output / "summary.json",
            task_summary,
        )

        environment_summary["tasks"][
            task_name
        ] = task_summary

    json_dump(
        environment_output / "summary.json",
        environment_summary,
    )

    return environment_summary


# ============================================================================
# Pooled binary analysis for R2 predictions
# ============================================================================


def run_pooled_r2_predictions(
    *,
    hidden: np.ndarray,
    metadata: list[dict],
    model_name: str,
    seed: int,
    output_dir: Path,
) -> dict:
    """
    Produce the pooled binary predictions needed by R2.

    This is deliberately kept separate from primary R1.

    R1:
        model × environment × layer

    R2:
        pooled test-set prediction file

    The R2 best layer is selected from pooled validation data only.
    """
    pooled_rows = clean_binary(
        metadata
    )

    pooled_indices = np.array(
        [
            row["_metadata_index"]
            for row in pooled_rows
        ],
        dtype=int,
    )

    pooled_hidden = hidden[
        :,
        pooled_indices,
        :,
    ]

    validate_split_structure(
        pooled_rows,
        context="pooled_binary_R2",
    )

    validate_class_availability(
        pooled_rows,
        "binary_label",
        BINARY_LABELS,
        context="pooled_binary_R2",
    )

    layer_results = []

    for layer in range(
        pooled_hidden.shape[0]
    ):
        result = evaluate_validation_layer(
            X=pooled_hidden[layer],
            rows=pooled_rows,
            layer=layer,
            label_key="binary_label",
            labels=BINARY_LABELS,
            subset_name="pooled_all",
        )

        layer_results.append(result)

    best_layer = select_best_layer(
        layer_results
    )

    (
        test_metrics,
        test_pred,
        test_prob,
        test_indices,
    ) = train_test_best_layer(
        X=pooled_hidden[best_layer],
        rows=pooled_rows,
        layer=best_layer,
        label_key="binary_label",
        labels=BINARY_LABELS,
    )

    y = np.array(
        [
            row["binary_label"]
            for row in pooled_rows
        ]
    )

    y_test = y[test_indices]

    ci = bootstrap_ci(
        y_true=y_test,
        y_pred=test_pred,
        y_prob=test_prob,
        labels=BINARY_LABELS,
        seed=seed + 10000,
    )

    predictions_path = (
        output_dir / "predictions.jsonl"
    )

    irreversible_index = BINARY_LABELS.index(
        "irreversible"
    )

    prediction_rows = []

    for row, pred, prob in zip(
        [pooled_rows[i] for i in test_indices],
        test_pred,
        test_prob,
    ):
        prediction_rows.append(
            {
                "id": row["id"],
                "environment": row["environment"],
                "layer": best_layer,
                "pred": str(pred),
                "prob_irreversible": float(
                    prob[irreversible_index]
                ),
            }
        )

    jsonl_write(
        predictions_path,
        prediction_rows,
    )

    result = {
        "model": model_name,
        "seed": seed,
        "analysis": "pooled_binary_R2_predictions",
        "representation": "last_action_token",
        "best_layer": best_layer,
        "best_layer_selection": {
            "criterion": "validation_balanced_accuracy",
            "tie_break": "validation_macro_f1",
            "final_tie_break": "smallest_layer",
        },
        "layers": layer_results,
        "final_test": test_metrics,
        "bootstrap_95ci": ci,
        "num_test_rows": len(prediction_rows),
        "predictions_file": str(
            predictions_path
        ),
    }

    json_dump(
        output_dir / "pooled_binary_R2.json",
        result,
    )

    return result


# ============================================================================
# Main experiment
# ============================================================================


def run_experiment(
    model_name: str,
    seed: int,
    model_dir: Path,
    output_dir: Path,
) -> None:
    """
    Complete R1 execution for one model and one scenario split seed.
    """
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model {model_name!r}"
        )

    if seed not in SEEDS:
        raise ValueError(
            f"Seed must be one of {SEEDS}; got {seed}"
        )

    print(
        "\n"
        + "=" * 90
        + "\n"
        + f"R1 PROBES: MODEL={model_name} SEED={seed}"
        + "\n"
        + "=" * 90,
        flush=True,
    )

    print(
        f"Model: {MODELS[model_name]}",
        flush=True,
    )

    print(
        f"Model directory: {model_dir}",
        flush=True,
    )

    print(
        f"Output directory: {output_dir}",
        flush=True,
    )

    metadata = load_metadata(
        model_dir
    )

    print(
        f"Loaded metadata rows: {len(metadata)}",
        flush=True,
    )

    if len(metadata) != 3601:
        print(
            "WARNING: expected the current 3,601-row dataset, "
            f"but metadata contains {len(metadata)} rows.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Add minimal-pair information.
    #
    # This is computed from the complete metadata so partial/reversible/
    # irreversible differences are not lost before the binary filtering.
    # ------------------------------------------------------------------

    for row in metadata:
        row["_minimal_pair"] = False

    add_minimal_pair_flag(
        metadata
    )

    minimal_pair_count = sum(
        row["_minimal_pair"]
        for row in metadata
    )

    print(
        f"Minimal-pair rows: {minimal_pair_count} "
        f"({minimal_pair_count / len(metadata):.2%})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Global dataset QC.
    # ------------------------------------------------------------------

    global_counts = {
        "label": class_counts(
            metadata,
            "label",
        ),
        "binary_label": class_counts(
            [
                row
                for row in metadata
                if row["binary_label"] in BINARY_LABELS
            ],
            "binary_label",
        ),
    }

    environment_counts = {
        environment: sum(
            row["environment"] == environment
            for row in metadata
        )
        for environment in ENVIRONMENTS
    }

    split_counts = Counter(
        row["_split"]
        for row in metadata
    )

    # Verify each environment has scenario-disjoint splits.
    for environment in ENVIRONMENTS:
        environment_rows = [
            row
            for row in metadata
            if row["environment"] == environment
        ]

        if not environment_rows:
            raise ValueError(
                f"No rows for environment {environment}"
            )

        validate_split_structure(
            environment_rows,
            context=environment,
        )

    # ------------------------------------------------------------------
    # Create output directory.
    # ------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Run both representations.
    #
    # AB1:
    #   last_action_token
    #   mean_action_tokens
    #
    # For each representation:
    #   environment
    #       binary / three_class
    #           all / minimal_pairs
    #               layer-wise probe
    #               controls
    # ------------------------------------------------------------------

    r1_summary: dict[str, Any] = {
        "model": model_name,
        "model_id": MODELS[model_name],
        "seed": seed,
        "representations": list(REPRESENTATIONS),
        "environments": {},
        "global_dataset": {
            "num_rows": len(metadata),
            "label_counts": global_counts,
            "environment_counts": environment_counts,
            "split_counts": dict(split_counts),
            "minimal_pair_rows": minimal_pair_count,
        },
        "runtime": environment_info(),
        "probe_configuration": {
            "classifier": "logistic_regression",
            "penalty": "l2",
            "C": LOGISTIC_C,
            "max_iter": LOGISTIC_MAX_ITER,
            "bootstrap_resamples": N_BOOTSTRAP,
            "control_seed": CONTROL_SEED,
            "layer_selection": {
                "primary": "validation_balanced_accuracy",
                "tie_break": "validation_macro_f1",
                "final_tie_break": "smallest_layer",
            },
            "final_test_protocol": (
                "fit on train+validation; evaluate once on test"
            ),
        },
    }

    # We load one representation at a time to keep memory usage reasonable.
    for representation in REPRESENTATIONS:
        print(
            "\n"
            + "-" * 90
            + "\n"
            + f"REPRESENTATION: {representation}"
            + "\n"
            + "-" * 90,
            flush=True,
        )

        hidden = load_hidden_states(
            model_dir,
            representation,
        )

        verify_hidden_state_alignment(
            hidden,
            metadata,
        )

        print(
            f"Hidden-state shape: {hidden.shape}",
            flush=True,
        )

        representation_summary: dict[str, Any] = {}

        for environment in ENVIRONMENTS:
            print(
                "\n"
                + "=" * 80
                + "\n"
                + f"ENVIRONMENT: {environment}"
                + "\n"
                + "=" * 80,
                flush=True,
            )

            environment_summary = run_environment_r1(
                hidden=hidden,
                metadata=metadata,
                environment=environment,
                representation=representation,
                model_name=model_name,
                seed=seed,
                output_dir=output_dir,
            )

            representation_summary[
                environment
            ] = environment_summary

            r1_summary["environments"].setdefault(
                environment,
                {},
            )[representation] = environment_summary

        # Explicitly delete the representation before loading the next one.
        del hidden

    # ------------------------------------------------------------------
    # R2 pooled predictions.
    #
    # Use last-action-token representation as the primary R2 representation,
    # matching the handover and predictions.jsonl contract.
    # ------------------------------------------------------------------

    print(
        "\n"
        + "=" * 90
        + "\n"
        + "POOLED BINARY R2 PREDICTIONS"
        + "\n"
        + "=" * 90,
        flush=True,
    )

    hidden = load_hidden_states(
        model_dir,
        "last_action_token",
    )

    verify_hidden_state_alignment(
        hidden,
        metadata,
    )

    r2_result = run_pooled_r2_predictions(
        hidden=hidden,
        metadata=metadata,
        model_name=model_name,
        seed=seed,
        output_dir=output_dir,
    )

    del hidden

    r1_summary["r2_pooled_binary"] = r2_result

    # ------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------

    summary_path = (
        output_dir / "summary.json"
    )

    json_dump(
        summary_path,
        r1_summary,
    )

    print(
        "\n"
        + "=" * 90
        + "\n"
        + f"FINISHED R1: MODEL={model_name} SEED={seed}"
        + "\n"
        + "=" * 90,
        flush=True,
    )

    print(
        f"Summary: {summary_path}",
        flush=True,
    )

    print(
        f"R2 predictions: "
        f"{output_dir / 'predictions.jsonl'}",
        flush=True,
    )

    print(
        f"R2 pooled best layer: "
        f"{r2_result['best_layer']}",
        flush=True,
    )

    print(
        "R2 pooled test BA: "
        f"{r2_result['final_test']['balanced_accuracy']:.4f}",
        flush=True,
    )


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run R1 hidden-state probes, controls, AB1/AB2 ablations, "
            "and pooled R2 predictions."
        )
    )

    parser.add_argument(
        "--model",
        choices=tuple(MODELS.keys()),
        required=True,
        help="Model extraction directory to analyze.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        required=True,
        help="Scenario-disjoint split seed.",
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/rq1"),
        help="Root directory containing extraction results.",
    )

    args = parser.parse_args()

    model_dir = (
        args.results_root
        / args.model
        / f"seed_{args.seed}"
    )

    output_dir = model_dir

    run_experiment(
        model_name=args.model,
        seed=args.seed,
        model_dir=model_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()