from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from requirements.prompts import load_toolemu_specs, render_probe_prompt


MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

SEEDS = (42, 43, 44)

SPLIT_PATHS = {
    "sandbox": Path("data/sandbox_splits"),
    "scienceworld": Path("data/scienceworld/splits"),
    "toolemu": Path("data/toolemu/splits"),
}

TOOLEMU_SPECS = Path("external/ToolEmu/assets/all_toolkits.json")


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(seed: int) -> list[dict]:
    """
    Load the existing scenario-disjoint train/validation/test splits.

    The split files are treated as authoritative. No new random split
    is created here.
    """
    records = []

    for environment, base_dir in SPLIT_PATHS.items():
        for split in ("train", "validation", "test"):
            path = base_dir / f"seed_{seed}" / f"{split}.jsonl"

            if not path.exists():
                raise FileNotFoundError(f"Missing split file: {path}")

            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        record["_split"] = split
                        records.append(record)

    return records


def get_environment_info() -> dict:
    """
    Capture the software and hardware environment used for extraction.

    This is stored in extraction_manifest.json so the exact extraction
    environment can be reported later.
    """
    cuda_available = torch.cuda.is_available()

    gpu_info = []

    if cuda_available:
        for index in range(torch.cuda.device_count()):
            gpu_info.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(
                        torch.cuda.get_device_capability(index)
                    ),
                    "total_memory_bytes": torch.cuda.get_device_properties(
                        index
                    ).total_memory,
                }
            )

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpus": gpu_info,
    }


def extract(
    model_name: str,
    seed: int,
    output_root: Path,
    batch_size: int,
    overwrite: bool = False,
) -> None:
    """
    Extract hidden states for every row in one model/seed combination.

    For every (state, action) row:
      - one forward pass is performed as part of a batch
      - hidden states from layer 0 through the final transformer layer
        are retained
      - the last action-token representation is saved
      - the mean action-token representation is saved
      - the exact rendered prompt and action-token positions are recorded

    Output:
      results/rq1/<model>/seed_<seed>/
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for R1 extraction. "
            "Do not run the full extraction on the Mac."
        )

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODELS)}"
        )

    if seed not in SEEDS:
        raise ValueError(
            f"Unknown seed '{seed}'. Choose from: {SEEDS}"
        )

    set_seed(seed)

    model_id = MODELS[model_name]

    model_output_dir = output_root / model_name / f"seed_{seed}"

    # Never accidentally mix a new extraction with files from an older run.
    # Existing output is removed only when --overwrite is explicitly supplied.
    if model_output_dir.exists() and any(model_output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output already exists at {model_output_dir}. "
                "Use --overwrite only if you intentionally want to replace it."
            )

        print(
            f"Overwriting existing output directory: {model_output_dir}",
            flush=True,
        )
        shutil.rmtree(model_output_dir)

    hidden_dir = model_output_dir / "hidden_states"
    hidden_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = model_output_dir / "metadata.jsonl"

    print(f"Loading tokenizer: {model_id}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Right padding is important because render_probe_prompt() returns
    # token positions relative to each unpadded prompt.
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    model.eval()

    records = load_records(seed)

    print(
        f"Loaded {len(records)} rows for seed {seed}",
        flush=True,
    )

    if len(records) == 0:
        raise RuntimeError(
            f"No records were loaded for seed {seed}."
        )

    toolemu_specs = None

    if any(r["environment"] == "toolemu" for r in records):
        if not TOOLEMU_SPECS.exists():
            raise FileNotFoundError(
                f"Missing ToolEmu toolkit specification file: "
                f"{TOOLEMU_SPECS}"
            )

        toolemu_specs = load_toolemu_specs(TOOLEMU_SPECS)

    # Capture the environment at the beginning of the actual extraction.
    environment_info = get_environment_info()

    metadata_file = metadata_path.open("w", encoding="utf-8")

    batch_last = []
    batch_mean = []
    batch_number = 0

    num_layers = None
    hidden_dimension = None

    try:
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]

            prompts = []
            action_positions = []

            for record in batch_records:
                specs = (
                    toolemu_specs
                    if record["environment"] == "toolemu"
                    else None
                )

                rendered = render_probe_prompt(
                    tokenizer,
                    record,
                    toolemu_specs=specs,
                    careful=False,
                )

                prompts.append(rendered["text"])
                action_positions.append(
                    rendered["action_token_indices"]
                )

                if not action_positions[-1]:
                    raise RuntimeError(
                        "No action tokens were identified for record "
                        f"{record['id']}"
                    )

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )

            input_ids = encoded["input_ids"].to("cuda")
            attention_mask = encoded["attention_mask"].to("cuda")

            # Right padding means the original action-token positions
            # returned by render_probe_prompt() remain unchanged.
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

            hidden_states = outputs.hidden_states

            if hidden_states is None:
                raise RuntimeError(
                    "Model did not return hidden states."
                )

            # hidden_states:
            #   layer 0 = embedding output
            #   layer 1...N = transformer layers
            num_layers = len(hidden_states)
            hidden_dimension = hidden_states[0].shape[-1]

            last_batch = []
            mean_batch = []

            for layer_hidden in hidden_states:
                # layer_hidden shape:
                # [batch, sequence_length, hidden_size]
                layer_last = []
                layer_mean = []

                for row_idx, positions in enumerate(action_positions):
                    pos = torch.tensor(
                        positions,
                        device=layer_hidden.device,
                        dtype=torch.long,
                    )

                    action_hidden = layer_hidden[row_idx, pos, :]

                    if action_hidden.shape[0] == 0:
                        raise RuntimeError(
                            "Empty action-token representation for "
                            f"row {start + row_idx}"
                        )

                    # AB1 representation 1:
                    # hidden state of the last action token.
                    layer_last.append(action_hidden[-1])

                    # AB1 representation 2:
                    # mean hidden state across all action tokens.
                    layer_mean.append(action_hidden.mean(dim=0))

                last_batch.append(
                    torch.stack(layer_last)
                    .float()
                    .cpu()
                    .numpy()
                )

                mean_batch.append(
                    torch.stack(layer_mean)
                    .float()
                    .cpu()
                    .numpy()
                )

            # Array shape:
            # [layers, row_within_batch, hidden_dimension]
            last_array = np.stack(last_batch, axis=0)
            mean_array = np.stack(mean_batch, axis=0)

            batch_id = f"batch_{batch_number:04d}"

            last_path = hidden_dir / f"{batch_id}_last.npy"
            mean_path = hidden_dir / f"{batch_id}_mean.npy"

            np.save(last_path, last_array)
            np.save(mean_path, mean_array)

            # Metadata tells the probe code exactly which row corresponds
            # to each batch position.
            #
            # We also save the complete rendered prompt and exact action
            # token positions so the extraction can be independently audited.
            for local_idx, record in enumerate(batch_records):
                metadata = {
                    "batch": batch_id,
                    "index": local_idx,
                    "id": record["id"],
                    "environment": record["environment"],
                    "scenario_id": record["scenario_id"],
                    "group_id": record["group_id"],
                    "condition": record["condition"],
                    "split": record["_split"],
                    "label": record["label"],
                    "binary_label": record["binary_label"],
                    "action_text": record["action_text"],
                    "prompt_text": prompts[local_idx],
                    "action_token_indices": action_positions[local_idx],
                    "action_token_count": len(
                        action_positions[local_idx]
                    ),
                }

                metadata_file.write(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # Flush metadata after every batch so completed metadata
            # remains available if the process is interrupted.
            metadata_file.flush()

            batch_last.append(str(last_path))
            batch_mean.append(str(mean_path))

            processed = min(
                start + batch_size,
                len(records),
            )

            print(
                f"[{model_name} seed={seed}] "
                f"processed {processed}/{len(records)} rows; "
                f"layers={num_layers}; "
                f"hidden_dim={hidden_dimension}",
                flush=True,
            )

            batch_number += 1

            # Explicitly release GPU tensors before the next batch.
            del outputs
            del hidden_states
            del input_ids
            del attention_mask
            del encoded

            torch.cuda.empty_cache()

    finally:
        metadata_file.close()

    if num_layers is None or hidden_dimension is None:
        raise RuntimeError(
            "Extraction completed without processing any batches."
        )

    manifest = {
        "model_name": model_name,
        "model_id": model_id,
        "seed": seed,

        "environment": environment_info,

        "tokenizer": {
            "name": model_id,
            "padding_side": tokenizer.padding_side,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
        },

        "num_records": len(records),
        "num_batches": batch_number,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "hidden_dimension": hidden_dimension,

        "prompt_configuration": {
            "careful": False,
            "prompt_renderer": "requirements.prompts.render_probe_prompt",
            "one_forward_pass_per_row": True,
            "truncation": False,
            "padding": "right",
        },

        "representations": [
            "last_action_token",
            "mean_action_tokens",
        ],

        "hidden_state_files": {
            "last_action_token": batch_last,
            "mean_action_tokens": batch_mean,
        },

        "representation_axis_order": [
            "layer",
            "row_within_batch",
            "hidden_dimension",
        ],

        "metadata_file": str(metadata_path),
        "metadata_fields": [
            "batch",
            "index",
            "id",
            "environment",
            "scenario_id",
            "group_id",
            "condition",
            "split",
            "label",
            "binary_label",
            "action_text",
            "prompt_text",
            "action_token_indices",
            "action_token_count",
        ],
    }

    manifest_path = model_output_dir / "extraction_manifest.json"

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nFinished {model_name}, seed {seed}. "
        f"Results saved to {model_output_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract R1 hidden states from probe prompts."
    )

    parser.add_argument(
        "--model",
        choices=MODELS,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/rq1"),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacing an existing model/seed output directory."
        ),
    )

    args = parser.parse_args()

    extract(
        model_name=args.model,
        seed=args.seed,
        output_root=args.output_root,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()