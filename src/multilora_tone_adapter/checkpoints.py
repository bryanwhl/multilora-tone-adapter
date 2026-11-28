"""Create immutable private candidates from recorded training checkpoints."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path

from .common import private_dir, write_json
from .runtime import (
    file_sha256,
    require_training_objective,
    verify_adapter_contract,
    verify_dataset_contract,
)


def _private_location(path: Path) -> None:
    for ancestor in (path, *path.parents):
        if (ancestor / ".git").exists():
            raise ValueError("Checkpoint candidates must be outside a source checkout.")


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError("Checkpoint source contains an invalid file.")
    shutil.copyfile(source, destination)
    destination.chmod(0o600)


def fork_checkpoint(source_run: Path, destination_run: Path, step: int) -> dict:
    """Fork one recorded checkpoint; never copy qualification or choose a winner.

    The selected candidate shares the parent's dataset and evaluation family,
    while its exact weights, configuration and training record receive new
    hashes. Existing source runs are never modified.
    """
    if type(step) is not int or step < 1:
        raise ValueError("Checkpoint step must be a positive integer.")
    source_run = Path(source_run).expanduser().resolve()
    destination_run = Path(destination_run).expanduser().resolve()
    _private_location(source_run)
    _private_location(destination_run)
    if destination_run.exists():
        raise ValueError("Use a fresh destination run for every checkpoint candidate.")
    if source_run == destination_run or source_run in destination_run.parents:
        raise ValueError("Checkpoint destination must be outside its source run.")
    training = verify_adapter_contract(source_run / "adapter")
    require_training_objective(training)
    if training.get("status") != "succeeded":
        raise ValueError("Only successful training runs can supply checkpoints.")
    verify_dataset_contract(source_run / "dataset")
    for name, expected in training.get("dataset_files", {}).items():
        if Path(name).name != name or file_sha256(source_run / "dataset" / name) != expected:
            raise ValueError("Training dataset changed after training.")
    if not training.get("dataset_files"):
        raise ValueError("Training dataset integrity record is missing.")
    for filename, key in (
        ("adapters.safetensors", "adapter_sha256"),
        ("adapter_config.json", "adapter_config_sha256"),
    ):
        if file_sha256(source_run / "adapter" / filename) != training.get(key):
            raise ValueError("Parent adapter changed after training.")
    matches = [row for row in training.get("checkpoints", []) if row.get("step") == step]
    if len(matches) != 1:
        raise ValueError("Requested checkpoint is absent or ambiguous in the inventory.")
    checkpoint = matches[0]
    filename = checkpoint.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Invalid checkpoint filename.")
    weights = source_run / "adapter" / filename
    if file_sha256(weights) != checkpoint.get("sha256"):
        raise ValueError("Checkpoint weights changed after training.")
    model_record = source_run / "model.json"
    if not model_record.is_file() or not isinstance(json.loads(model_record.read_text()).get("path"), str):
        raise ValueError("Source run requires a model record.")

    private_dir(destination_run.parent)
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=destination_run.parent))
    try:
        dataset_dir = private_dir(staging / "dataset")
        for path in (source_run / "dataset").iterdir():
            _copy_file(path, dataset_dir / path.name)
        _copy_file(model_record, staging / "model.json")
        adapter_dir = private_dir(staging / "adapter")
        _copy_file(weights, adapter_dir / "adapters.safetensors")
        config = json.loads((source_run / "adapter/adapter_config.json").read_text())
        config["iters"] = step
        config["adapter_path"] = str(destination_run / "adapter")
        write_json(adapter_dir / "adapter_config.json", config)
        selected = copy.deepcopy(training)
        selected["checkpoint_parent"] = {
            "run": str(source_run),
            "training_sha256": file_sha256(source_run / "adapter/training.json"),
            "adapter_sha256": training["adapter_sha256"],
            "original_iters": training["args"]["iters"],
            "step": step,
        }
        selected["args"]["iters"] = step
        selected["selected_checkpoint_step"] = step
        selected["evaluation_family_root"] = training.get("evaluation_family_root", str(source_run))
        selected["qualified"] = False
        selected["adapter_sha256"] = file_sha256(adapter_dir / "adapters.safetensors")
        selected["adapter_config_sha256"] = file_sha256(adapter_dir / "adapter_config.json")
        selected["checkpoints"] = [
            {"step": step, "file": "adapters.safetensors", "sha256": selected["adapter_sha256"]}
        ]
        # These describe the parent execution, not a second training process.
        for key in ("command", "execution", "started_at", "finished_at"):
            if key in selected:
                selected["checkpoint_parent"][key] = selected.pop(key)
        write_json(adapter_dir / "training.json", selected)
        if destination_run.exists():
            raise ValueError("Checkpoint destination appeared during creation.")
        staging.rename(destination_run)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "run": str(destination_run),
        "source_run": str(source_run),
        "step": step,
        "adapter_sha256": selected["adapter_sha256"],
        "qualified": False,
    }
