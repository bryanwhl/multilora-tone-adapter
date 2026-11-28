"""Synthetic checkpoint candidates preserve provenance without inheriting approval."""

import json

import pytest
from checkpoint_fixtures import curated_dataset

from multilora_tone_adapter.checkpoints import fork_checkpoint
from multilora_tone_adapter.common import input_contract, write_json
from multilora_tone_adapter.runtime import file_sha256, training_objective_contract


@pytest.fixture
def source_run(tmp_path):
    run = tmp_path / "parent"
    data, adapter = run / "dataset", run / "adapter"
    data.mkdir(parents=True)
    adapter.mkdir()
    curated_dataset(data)
    write_json(run / "model.json", {"path": str(tmp_path / "public-model-cache")})
    (adapter / "adapters.safetensors").write_bytes(b"final-synthetic-weights")
    (adapter / "0000002_adapters.safetensors").write_bytes(b"early-synthetic-weights")
    write_json(adapter / "adapter_config.json", {"iters": 8, "num_layers": 4})
    write_json(adapter / "training.json", {
        "input_contract": input_contract(),
        "training_objective": training_objective_contract(),
        "status": "succeeded", "qualified": False, "args": {"iters": 8},
        "evaluation_family_root": str(run.resolve()),
        "dataset_files": {p.name: file_sha256(p) for p in data.iterdir() if p.suffix in {".json", ".jsonl"}},
        "adapter_sha256": file_sha256(adapter / "adapters.safetensors"),
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
        "checkpoints": [
            {"step": 2, "file": "0000002_adapters.safetensors", "sha256": file_sha256(adapter / "0000002_adapters.safetensors")},
            {"step": 8, "file": "adapters.safetensors", "sha256": file_sha256(adapter / "adapters.safetensors")},
        ],
        "command": ["synthetic-training-command"], "execution": {"exit_code": 0},
    })
    write_json(run / "evaluation.json", {"qualified": True})
    return run


def test_fork_selects_recorded_weights_and_preserves_parent_immutably(source_run, tmp_path):
    parent_files = {str(p.relative_to(source_run)): p.read_bytes() for p in source_run.rglob("*") if p.is_file()}
    destination = tmp_path / "selected"
    result = fork_checkpoint(source_run, destination, 2)
    assert result["step"] == 2 and result["qualified"] is False
    assert (destination / "adapter/adapters.safetensors").read_bytes() == b"early-synthetic-weights"
    assert not (destination / "evaluation.json").exists()
    selected = json.loads((destination / "adapter/training.json").read_text())
    config = json.loads((destination / "adapter/adapter_config.json").read_text())
    assert selected["args"]["iters"] == config["iters"] == 2
    assert selected["adapter_sha256"] == file_sha256(destination / "adapter/adapters.safetensors")
    assert selected["adapter_config_sha256"] == file_sha256(destination / "adapter/adapter_config.json")
    assert selected["checkpoint_parent"]["training_sha256"] == file_sha256(source_run / "adapter/training.json")
    assert selected["checkpoint_parent"]["original_iters"] == 8
    assert selected["evaluation_family_root"] == str(source_run.resolve())
    assert "command" not in selected
    for relative, contents in parent_files.items():
        assert (source_run / relative).read_bytes() == contents
    for path in destination.rglob("*"):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)


@pytest.mark.parametrize("change", ["weights", "dataset", "configuration", "objective"])
def test_fork_refuses_changed_or_legacy_source(source_run, tmp_path, change):
    if change == "weights":
        (source_run / "adapter/0000002_adapters.safetensors").write_bytes(b"changed")
    elif change == "dataset":
        (source_run / "dataset/train.jsonl").write_text("changed")
    elif change == "configuration":
        (source_run / "adapter/adapter_config.json").write_text("{}")
    else:
        path = source_run / "adapter/training.json"
        record = json.loads(path.read_text())
        record.pop("training_objective")
        write_json(path, record)
    with pytest.raises(ValueError):
        fork_checkpoint(source_run, tmp_path / "candidate", 2)
    assert not (tmp_path / "candidate").exists()


def test_fork_refuses_existing_checkout_nested_and_unknown_destinations(source_run, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        fork_checkpoint(source_run, existing, 2)
    with pytest.raises(ValueError, match="outside its source"):
        fork_checkpoint(source_run, source_run / "child", 2)
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="source checkout"):
        fork_checkpoint(source_run, repository / "candidate", 2)
    with pytest.raises(ValueError, match="absent"):
        fork_checkpoint(source_run, tmp_path / "missing", 3)


def test_nested_fork_keeps_original_evaluation_family(source_run, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    fork_checkpoint(source_run, first, 2)
    fork_checkpoint(first, second, 2)
    record = json.loads((second / "adapter/training.json").read_text())
    assert record["evaluation_family_root"] == str(source_run.resolve())
