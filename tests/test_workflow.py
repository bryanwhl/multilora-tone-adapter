"""Synthetic end-to-end contracts for private curation and qualification boundaries."""

import json
import subprocess

import pytest
from test_dataset import synthetic_reviews_for
from test_evaluation_quality import covered_cases, import_judgments, run_comparison

from multilora_tone_adapter.cli import main, run_dir
from multilora_tone_adapter.common import (
    digest,
    input_contract,
    read_jsonl,
    write_json,
    write_jsonl,
)
from multilora_tone_adapter.dataset import assign_splits, build_dataset
from multilora_tone_adapter.evaluation import activate, verify_active
from multilora_tone_adapter.labeling import import_labels, make_request, validate_meaning
from multilora_tone_adapter.publication import audit
from multilora_tone_adapter.runtime import (
    file_sha256,
    model_fingerprint,
    training_objective_contract,
)


def write_episodes(directory):
    directory.mkdir(parents=True, exist_ok=True)
    episodes = [
        {
            "id": "first",
            "context": [{"speaker": "other", "text": "when?"}],
            "target": "monday at 14 works!",
            "source_ids": ["s1", "s2"],
            "group_id": "g1",
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "second",
            "context": [],
            "target": "no thanks, can't make it",
            "source_ids": ["s3"],
            "group_id": "g2",
            "timestamp": "2026-01-02T00:00:00+00:00",
        },
        {
            "id": "third",
            "context": [],
            "target": "sounds good",
            "source_ids": ["s4"],
            "group_id": "g3",
            "timestamp": "2026-01-03T00:00:00+00:00",
        },
    ]
    write_jsonl(directory / "episodes.jsonl", episodes)
    return episodes


def label_response(path, rows):
    write_json(path, {"labels": rows})
    return path


def test_label_import_preserves_authentic_target_and_binds_batch(tmp_path):
    episodes = write_episodes(tmp_path)
    before = (tmp_path / "episodes.jsonl").read_bytes()
    result = make_request(tmp_path, 1)
    request = json.loads((tmp_path / "label-request.json").read_text())
    assert result["request_hash"] == digest(request["episodes"])
    assert request["episodes"][0]["target"] == episodes[0]["target"]
    import_labels(
        tmp_path,
        label_response(
            tmp_path / "response.json",
            [{"id": "first", "draft": "Monday at 14 is suitable.", "meaning_preserved": True}],
        ),
    )
    label = read_jsonl(tmp_path / "labels.jsonl")[0]
    assert label["target_hash"] == digest(episodes[0]["target"])
    assert label["request_hash"] == result["request_hash"]
    assert (tmp_path / "episodes.jsonl").read_bytes() == before
    assert make_request(tmp_path, 1)["count"] == 1
    assert (
        json.loads((tmp_path / "label-request.json").read_text())["episodes"][0]["id"] == "second"
    )


@pytest.mark.parametrize("ids", [["unknown"], ["first", "first"], []])
def test_label_response_requires_exact_pending_ids(tmp_path, ids):
    write_episodes(tmp_path)
    make_request(tmp_path, 1)
    response = label_response(
        tmp_path / "response.json",
        [
            {"id": key, "draft": "Monday at 14 is suitable.", "meaning_preserved": True}
            for key in ids
        ],
    )
    with pytest.raises(ValueError, match="IDs"):
        import_labels(tmp_path, response)
    assert not (tmp_path / "labels.jsonl").exists()


def test_modified_pending_request_cannot_rebind_old_labels(tmp_path):
    write_episodes(tmp_path)
    make_request(tmp_path, 1)
    request = json.loads((tmp_path / "label-request.json").read_text())
    request["episodes"][0]["target"] = "Tuesday at 15 is suitable."
    write_json(tmp_path / "label-request.json", request)
    with pytest.raises(ValueError):
        import_labels(
            tmp_path,
            label_response(
                tmp_path / "response.json",
                [{"id": "first", "draft": "Tuesday at 15 works.", "meaning_preserved": True}],
            ),
        )


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/b", "a\\b", ".", "..", ""])
def test_run_path_traversal_rejected(tmp_path, monkeypatch, name):
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(tmp_path / "private"))
    with pytest.raises(ValueError):
        run_dir(name)


def test_private_state_cannot_be_inside_git_or_symlink_into_git(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(checkout / "private"))
    with pytest.raises(ValueError, match="Git"):
        run_dir("candidate")
    private = tmp_path / "private"
    (private / "runs").mkdir(parents=True)
    (private / "runs" / "candidate").symlink_to(checkout, target_is_directory=True)
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(private))
    with pytest.raises(ValueError, match="Git"):
        run_dir("candidate")


@pytest.mark.parametrize(
    "draft,changed",
    [
        ("Send 12 units", "Send 13 units"),
        ("See https://example.invalid/a", "See https://example.invalid/b"),
        ("Contact a@example.invalid", "Contact b@example.invalid"),
        ("Monday is fine", "Tuesday is fine"),
        ("I cannot attend", "I can attend"),
    ],
)
def test_obvious_protected_value_changes_are_not_qualified(draft, changed):
    assert validate_meaning(draft, changed)


def trained_fixture(directory, accepted=500):
    model = directory / "fake-model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"synthetic-model-not-real-weights")
    adapter = directory / "adapter"
    adapter.mkdir()
    write_json(adapter / "adapter_config.json", {"lora_parameters": {"rank": 8}})
    (adapter / "adapters.safetensors").write_bytes(b"synthetic-adapter-not-real-weights")
    dataset = directory / "dataset"
    episodes = [
        {"id": str(index), "context": [], "target": f"item {index} works!",
         "group_id": f"synthetic-session-{index}", "source_ids": [f"synthetic-source-{index}"],
         "timestamp": f"{index:06d}"}
        for index in range(accepted)
    ]
    labels = [{"id": str(index), "draft": f"Item {index} is suitable.", "meaning_preserved": True}
              for index in range(accepted)]
    build_dataset(episodes, labels, dataset, min_episodes=1, identity_fraction=0,
                  reviews=synthetic_reviews_for(episodes, labels))
    write_json(directory / "model.json", {"path": str(model)})
    write_json(
        adapter / "training.json",
        {
            "status": "succeeded",
            "input_contract": input_contract(),
            "training_objective": training_objective_contract(),
            "evaluation_family_root": str(directory.resolve()),
            "adapter_sha256": file_sha256(adapter / "adapters.safetensors"),
            "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
            "dataset_files": {path.name: file_sha256(path) for path in dataset.iterdir()
                              if path.suffix in {".json", ".jsonl"}},
            "model": model_fingerprint(model),
            "model_path": str(model),
        },
    )
    return model


def qualify_synthetic(directory, monkeypatch):
    from multilora_tone_adapter import evaluation

    monkeypatch.setattr(evaluation, "_run_variant",
                        lambda directory, model, mode, cases, references, memory_gb:
                        [case["draft"] for case in cases])
    monkeypatch.setattr(evaluation, "_fit_style_examples",
                        lambda model, cases, references: (references, {}))
    comparison, judgments = run_comparison(directory, covered_cases())
    import_judgments(directory, comparison, judgments)
    evaluation.select_candidate(directory)
    comparison, judgments = run_comparison(directory, covered_cases("Final"), purpose="qualification")
    report = import_judgments(directory, comparison, judgments)
    assert report["qualified"]
    return report


def test_failed_and_unqualified_training_cannot_activate(tmp_path):
    trained_fixture(tmp_path)
    write_json(tmp_path / "evaluation.json", {"qualified": False})
    with pytest.raises(ValueError, match="qualification"):
        activate(tmp_path)
    training = json.loads((tmp_path / "adapter/training.json").read_text())
    training["status"] = "failed"
    write_json(tmp_path / "adapter/training.json", training)
    write_json(tmp_path / "evaluation.json", {"qualified": True})
    with pytest.raises(ValueError, match="succeeded"):
        activate(tmp_path)
    assert not (tmp_path / "active.json").exists()


def test_altered_base_cannot_retain_qualification(tmp_path, monkeypatch):
    model = trained_fixture(tmp_path)
    qualify_synthetic(tmp_path, monkeypatch)
    activate(tmp_path)
    verify_active(tmp_path)
    (model / "model.safetensors").write_bytes(b"changed-base-model")
    with pytest.raises(ValueError):
        verify_active(tmp_path)


def test_overlapping_context_sources_never_cross_splits():
    episodes = [
        {
            "id": str(i),
            "target": f"word{i}",
            "group_id": f"session{i}",
            "source_ids": [f"s{i}"],
            "timestamp": str(i),
        }
        for i in range(10)
    ]
    episodes[8]["source_ids"].append("s0")
    splits = assign_splits(episodes)
    assert splits["8"] == splits["0"]
    assert set(splits.values()) == {"train", "valid", "test"}


def test_publication_audit_inspects_staged_bytes_not_only_worktree(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "example.txt"
    # Construct a fake credential without committing or contacting a service.
    path.write_text("gh" + "p_" + "A" * 30)
    subprocess.run(["git", "add", "example.txt"], cwd=tmp_path, check=True)
    path.write_text("clean working tree")
    assert any(item["rule"] == "credential" for item in audit(tmp_path)["findings"])


def test_cli_errors_do_not_echo_message_payloads(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(tmp_path / "private"))
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "id": 1,
                "messages": [
                    {
                        "id": 1,
                        "type": "message",
                        "from_id": "user1",
                        "text": "PRIVATE_SYNTHETIC_MESSAGE",
                        "date": "ambiguous",
                    }
                ],
            }
        )
    )
    assert main(["inspect", str(path), "--author-id", "user1"]) == 2
    captured = capsys.readouterr()
    assert "PRIVATE_SYNTHETIC_MESSAGE" not in captured.out + captured.err


@pytest.mark.parametrize(
    "count,accepted,change_number,qualified",
    [
        (49, 500, False, False),
        (50, 499, False, False),
        (50, 500, True, False),
        (50, 500, False, True),
    ],
)
def test_real_qualification_gates(tmp_path, monkeypatch, count, accepted, change_number, qualified):
    from multilora_tone_adapter.evaluation import select_candidate

    trained_fixture(tmp_path, accepted=accepted)

    def synthetic_variant(directory, model, mode, cases, references, memory_gb):
        return [
            case["draft"].replace("12", "99")
            if mode == "adapter" and change_number
            else case["draft"]
            for case in cases
        ]

    monkeypatch.setattr("multilora_tone_adapter.evaluation._run_variant", synthetic_variant)
    monkeypatch.setattr("multilora_tone_adapter.evaluation._fit_style_examples",
                        lambda model, cases, references: (references, {}))
    comparison, judgments = run_comparison(tmp_path, covered_cases())
    import_judgments(tmp_path, comparison, judgments)
    select_candidate(tmp_path)
    cases = covered_cases("Final")[:count]
    if count < 50:
        with pytest.raises(ValueError, match="at least 50"):
            run_comparison(tmp_path, cases, purpose="qualification")
        with pytest.raises(ValueError, match="qualification"):
            activate(tmp_path)
        return
    comparison, judgments = run_comparison(tmp_path, cases, purpose="qualification")
    report = import_judgments(tmp_path, comparison, judgments)
    assert report["qualified"] is qualified
    assert report["scoring"] == "automated_proxy_not_human_validation"
    if qualified:
        assert activate(tmp_path)["active"] is True
        verify_active(tmp_path)
    else:
        with pytest.raises(ValueError):
            activate(tmp_path)


def test_committed_secret_is_caught_even_after_removal(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "example.txt"
    path.write_text("gh" + "p_" + "B" * 30)
    subprocess.run(["git", "add", "example.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Synthetic Tester",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "synthetic fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    path.write_text("clean replacement")
    subprocess.run(["git", "add", "example.txt"], cwd=tmp_path, check=True)
    assert any(
        item["rule"] == "credential" and item["source"] == "history"
        for item in audit(tmp_path)["findings"]
    )


def test_altered_adapter_config_invalidates_qualified_activation(tmp_path, monkeypatch):
    trained_fixture(tmp_path)
    qualify_synthetic(tmp_path, monkeypatch)
    activate(tmp_path)
    verify_active(tmp_path)
    write_json(tmp_path / "adapter/adapter_config.json", {"lora_parameters": {"rank": 16}})
    with pytest.raises(ValueError, match="configuration changed"):
        verify_active(tmp_path)
    with pytest.raises(ValueError, match="configuration changed"):
        activate(tmp_path)
