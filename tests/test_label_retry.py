"""Synthetic label responses: retry model mistakes, never source corruption."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from multilora_tone_adapter import labeling
from multilora_tone_adapter.common import digest, read_jsonl, write_json, write_jsonl


@pytest.fixture
def episodes(tmp_path):
    rows = [
        {
            "id": digest("synthetic-first"),
            "target": "bring a notebook pls",
            "context": [{"speaker": "other", "text": "what should I bring?"}],
        },
        {"id": digest("synthetic-second"), "target": "thanks will do", "context": []},
    ]
    write_jsonl(tmp_path / "episodes.jsonl", rows)
    return rows


def response_for(prompt):
    request = json.loads(prompt)
    return {
        "labels": [
            {
                "id": row["id"],
                "draft": "Please bring a notebook.",
                "meaning_preserved": True,
                "reason": "Synthetic equivalent draft.",
            }
            for row in request["episodes"]
        ]
    }


def output_path(args):
    from pathlib import Path

    return Path(args[args.index("--output-last-message") + 1])


def test_wrong_id_then_fresh_valid_response_imports_exact_pending_batch(
    tmp_path, episodes, monkeypatch
):
    prompts = []
    before = (tmp_path / "episodes.jsonl").read_bytes()

    def run(args, **kwargs):
        prompts.append(kwargs["input"])
        response = response_for(kwargs["input"])
        if len(prompts) == 1:
            response["labels"][0]["id"] = "f" * 64
        write_json(output_path(args), response)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(labeling.subprocess, "run", run)
    result = labeling.codex_label(tmp_path, batch_size=1, max_batches=1)
    assert result["labeled"] == 1
    assert result["response_retries_used"] == 1
    assert len(prompts) == 2 and prompts[0] == prompts[1]
    labels = read_jsonl(tmp_path / "labels.jsonl")
    assert [row["id"] for row in labels] == [episodes[0]["id"]]
    assert labels[0]["target_hash"] == digest(episodes[0]["target"])
    assert (tmp_path / "episodes.jsonl").read_bytes() == before


def test_exhausting_three_attempts_preserves_previously_imported_labels(
    tmp_path, episodes, monkeypatch
):
    calls = []
    saved_first_batch = []

    def run(args, **kwargs):
        calls.append(kwargs["input"])
        response = response_for(kwargs["input"])
        if len(calls) > 1:
            if not saved_first_batch:
                saved_first_batch.append((tmp_path / "labels.jsonl").read_bytes())
            response["labels"][0]["id"] = "f" * 64
        write_json(output_path(args), response)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(labeling.subprocess, "run", run)
    with pytest.raises(labeling.LabelResponseError, match="IDs"):
        labeling.codex_label(tmp_path, batch_size=1, max_batches=2)
    assert len(calls) == 4  # One successful batch, then three failed fresh attempts.
    assert len(set(calls[1:])) == 1
    assert (tmp_path / "labels.jsonl").read_bytes() == saved_first_batch[0]
    assert [row["id"] for row in read_jsonl(tmp_path / "labels.jsonl")] == [episodes[0]["id"]]


@pytest.mark.parametrize("tamper", ["request_hash", "request_context", "source_context"])
def test_pending_request_or_source_tamper_is_fatal_without_retry(
    tmp_path, episodes, monkeypatch, tamper
):
    calls = []

    def run(args, **kwargs):
        calls.append(kwargs["input"])
        write_json(output_path(args), response_for(kwargs["input"]))
        request_path = tmp_path / "label-request.json"
        request = json.loads(request_path.read_text())
        if tamper == "request_hash":
            request["request_hash"] = "tampered"
            write_json(request_path, request)
        elif tamper == "request_context":
            request["episodes"][0]["context"] = []
            request["request_hash"] = digest(request["episodes"])
            write_json(request_path, request)
        else:
            changed = [dict(row) for row in episodes]
            changed[0]["context"] = []
            write_jsonl(tmp_path / "episodes.jsonl", changed)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(labeling.subprocess, "run", run)
    with pytest.raises(ValueError) as error:
        labeling.codex_label(tmp_path, batch_size=1, max_batches=1)
    assert not isinstance(error.value, labeling.LabelResponseError)
    assert len(calls) == 1
    assert not (tmp_path / "labels.jsonl").exists()


@pytest.mark.parametrize("failure", ["nonzero", "missing", "timeout"])
def test_process_or_authentication_failures_do_not_retry(tmp_path, episodes, monkeypatch, failure):
    calls = []
    (tmp_path / "label-response.json").write_text("stale successful-looking response")

    def run(args, **kwargs):
        calls.append(kwargs["input"])
        assert not output_path(args).exists()
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 1)
        return SimpleNamespace(returncode=1 if failure == "nonzero" else 0)

    monkeypatch.setattr(labeling.subprocess, "run", run)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        labeling.codex_label(tmp_path, batch_size=1, max_batches=1)
    assert len(calls) == 1
    assert not (tmp_path / "labels.jsonl").exists()


@pytest.mark.parametrize(
    "bad_response", ["not-json", "[]", "{}", '{"labels":{}}', '{"labels":[{"id":[]}]}']
)
def test_malformed_model_response_has_distinct_retryable_error(tmp_path, episodes, bad_response):
    labeling.make_request(tmp_path, 1)
    path = tmp_path / "response.json"
    path.write_text(bad_response)
    with pytest.raises(labeling.LabelResponseError):
        labeling.import_labels(tmp_path, path)
    assert not (tmp_path / "labels.jsonl").exists()


@pytest.mark.parametrize(
    "field,value", [("draft", None), ("meaning_preserved", "true"), ("reason", 42)]
)
def test_schema_type_failures_are_retryable(tmp_path, episodes, field, value):
    labeling.make_request(tmp_path, 1)
    response = response_for((tmp_path / "label-request.json").read_text())
    response["labels"][0][field] = value
    path = tmp_path / "response.json"
    write_json(path, response)
    with pytest.raises(labeling.LabelResponseError):
        labeling.import_labels(tmp_path, path)
    assert not (tmp_path / "labels.jsonl").exists()
