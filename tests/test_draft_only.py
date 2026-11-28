"""History is curation data, never a local rewrite-model input."""

import copy
import json

import pytest
from test_dataset import independent_episodes, labels_for, synthetic_reviews_for
from test_runtime import Tokenizer, fake_generation, fake_mlx_dataset  # noqa: F401
from test_workflow import trained_fixture

from multilora_tone_adapter import cli, eval_worker, evaluation, runtime
from multilora_tone_adapter.common import (
    input_contract,
    messages_for,
    read_jsonl,
    write_json,
    write_jsonl,
)
from multilora_tone_adapter.dataset import build_dataset
from multilora_tone_adapter.labeling import make_request


def test_history_remains_for_curation_but_never_enters_training_messages(tmp_path):
    episodes = independent_episodes()
    for episode in episodes:
        episode["context"] = [{"speaker": "other", "text": "CURATION_ONLY_SENTINEL"}]
    original = copy.deepcopy(episodes)
    labels = labels_for(episodes)
    write_jsonl(tmp_path / "episodes.jsonl", episodes)
    make_request(tmp_path)
    assert "CURATION_ONLY_SENTINEL" in (tmp_path / "label-request.json").read_text()
    output = tmp_path / "dataset"
    manifest = build_dataset(episodes, labels, output, min_episodes=10,
                             reviews=synthetic_reviews_for(episodes, labels), identity_fraction=0)
    assert manifest["schema_version"] == 3
    assert manifest["input_contract"] == input_contract()
    assert episodes == original
    by_id = {episode["id"]: episode for episode in episodes}
    label_by_id = {label["id"]: label for label in labels}
    for provenance in read_jsonl(output / "provenance.jsonl"):
        row = read_jsonl(output / f"{provenance['split']}.jsonl")[provenance["row"]]
        assert row["messages"] == messages_for(
            label_by_id[provenance["id"]]["draft"], target=by_id[provenance["id"]]["target"]
        )
        assert "CURATION_ONLY_SENTINEL" not in json.dumps(row)
        assert json.loads(row["messages"][1]["content"]).keys() == {"draft"}


@pytest.mark.usefixtures("fake_mlx_dataset")
def test_runtime_prompt_is_exact_training_prefix(monkeypatch, tmp_path):
    draft, target = "I will update you tomorrow.", "i’ll update you tomorrow"
    training_messages = messages_for(draft, target=target)
    write_jsonl(tmp_path / "train.jsonl", [{"messages": training_messages}])
    checked = runtime.verify_training_rows(tmp_path, Tokenizer(), Tokenizer(), 4096)
    assert checked["input_contract"] == input_contract()
    engine, _ = fake_generation(monkeypatch)
    seen = []

    class RecordingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            seen.append(copy.deepcopy(messages))
            return super().apply_chat_template(messages, **kwargs)

    engine._tokenizer = RecordingTokenizer()
    engine.rewrite(draft)
    assert seen == [training_messages[:-1]]
    with pytest.raises(TypeError):
        engine.rewrite(draft, [])
    with pytest.raises(TypeError):
        messages_for(draft, context=[])


@pytest.mark.parametrize("context", [[], None, [{"speaker": "other", "text": "incoming"}]])
@pytest.mark.usefixtures("fake_mlx_dataset")
def test_legacy_training_row_rejected_even_with_empty_context(tmp_path, context):
    messages = messages_for("Draft", target="Target")
    messages[1]["content"] = json.dumps({"draft": "Draft", "context": context})
    write_jsonl(tmp_path / "train.jsonl", [{"messages": messages}])
    with pytest.raises(ValueError, match="draft-only serialization"):
        runtime.verify_training_rows(tmp_path, Tokenizer(), Tokenizer(), 4096)


@pytest.mark.parametrize("context", [[], None, [{"speaker": "other", "text": "incoming"}]])
def test_evaluation_rejects_context_before_binding_or_loading(monkeypatch, tmp_path, context):
    case = {"draft": "Draft", "context": context, "independently_authored": True}
    write_json(tmp_path / "cases.json", [case])

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid cases must fail before model/artifact work")

    monkeypatch.setattr(evaluation, "_artifact_binding", forbidden)
    monkeypatch.setattr(eval_worker, "MLXEngine", forbidden)
    with pytest.raises(ValueError, match="remove context"):
        evaluation.generate_comparison(tmp_path, tmp_path / "cases.json")
    with pytest.raises(ValueError, match="remove context"):
        eval_worker.generate({"mode": "base", "cases": [case]})


@pytest.mark.parametrize("contract", [None, {"name": "draft-only-v1", "sha256": "wrong"}])
def test_old_dataset_rejected_before_preflight_or_model_work(tmp_path, monkeypatch, contract):
    write_json(tmp_path / "dataset/manifest.json", {"input_contract": contract})

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy data must fail before model work")

    monkeypatch.setattr(runtime, "preflight", forbidden)
    with pytest.raises(ValueError, match="rebuild.*retrain"):
        runtime.train(tmp_path / "dataset", tmp_path / "model", tmp_path / "adapter")
    metadata = json.loads((tmp_path / "adapter/training.json").read_text())
    assert metadata["status"] == "failed"
    assert not (tmp_path / "adapter/adapters.safetensors").exists()


@pytest.mark.parametrize("candidate", [False, True])
def test_cli_rejects_legacy_adapter_including_candidate_before_server(monkeypatch, tmp_path, capsys, candidate):
    state = tmp_path / "state"
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(state))
    write_json(state / "runs/old/adapter/training.json", {"status": "succeeded"})

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy adapter must not reach model resolution")

    monkeypatch.setattr(cli, "resolve_model", forbidden)
    args = ["serve", "--run", "old"] + (["--candidate"] if candidate else [])
    assert cli.main(args) == 2
    assert "rebuild a draft-only dataset" in capsys.readouterr().err


def test_engine_and_evaluation_binding_reject_legacy_adapter(tmp_path, monkeypatch):
    model = trained_fixture(tmp_path)
    path = tmp_path / "adapter/training.json"
    metadata = json.loads(path.read_text())
    metadata.pop("input_contract")
    write_json(path, metadata)

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy adapter must not reach preflight")

    monkeypatch.setattr(runtime, "preflight", forbidden)
    with pytest.raises(ValueError, match="rebuild.*retrain"):
        runtime.MLXEngine(model, tmp_path / "adapter").load()
    with pytest.raises(ValueError, match="rebuild.*retrain"):
        evaluation._artifact_binding(tmp_path)


def test_prompt_baseline_references_cannot_be_combined_with_adapter():
    with pytest.raises(ValueError, match="cannot be combined"):
        runtime.MLXEngine("model", "adapter", evaluation_style_references=["style"])


def test_prompt_baseline_keeps_draft_payload_and_uses_fixed_system_references(monkeypatch):
    engine, _ = fake_generation(monkeypatch)
    references = [{"draft": "Hello.", "target": "hello!"},
                  {"draft": "Thank you.", "target": "thanks"}]
    engine._evaluation_style_references = references
    seen = []

    class RecordingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            seen.append(copy.deepcopy(messages))
            return super().apply_chat_template(messages, **kwargs)

    engine._tokenizer = RecordingTokenizer()
    engine.rewrite("Draft")
    assert seen[0][1:] == messages_for("Draft")[1:]
    assert json.dumps(references) in seen[0][0]["content"]


def test_current_adapter_cannot_bind_a_legacy_dataset(tmp_path):
    trained_fixture(tmp_path)
    path = tmp_path / "dataset/manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("input_contract")
    write_json(path, manifest)
    with pytest.raises(ValueError, match="Dataset.*rebuild.*retrain"):
        evaluation._artifact_binding(tmp_path)


@pytest.mark.parametrize("stage", ["comparison", "evaluation", "active"])
def test_current_artifacts_cannot_reuse_legacy_evaluation_bindings(tmp_path, stage):
    from multilora_tone_adapter.common import digest

    trained_fixture(tmp_path)
    binding = evaluation._artifact_binding(tmp_path)
    old_binding = {key: value for key, value in binding.items() if key != "input_contract"}
    if stage == "comparison":
        write_json(tmp_path / "comparison-private.json",
                   {"protocol": evaluation.EVALUATION_PROTOCOL, "binding": old_binding})
        with pytest.raises(ValueError, match="Compared artifacts have changed"):
            evaluation.import_evaluation(tmp_path, tmp_path / "not-read.json")
    elif stage == "evaluation":
        write_json(tmp_path / "evaluation.json", {"qualified": True, "binding": old_binding})
        with pytest.raises(ValueError, match="qualification"):
            evaluation.activate(tmp_path)
        assert not (tmp_path / "active.json").exists()
    else:
        report = {"qualified": True, "binding": binding}
        write_json(tmp_path / "evaluation.json", report)
        write_json(tmp_path / "active.json", {
            "binding": old_binding, "evaluation_hash": digest(report),
        })
        with pytest.raises(ValueError, match="stale or unqualified"):
            evaluation.verify_active(tmp_path)
