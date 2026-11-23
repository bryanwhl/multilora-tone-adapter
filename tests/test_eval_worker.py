"""Variant isolation and worker lifecycle contracts without loading MLX."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multilora_tone_adapter import eval_worker, evaluation
from multilora_tone_adapter.common import write_json, write_jsonl


@pytest.mark.parametrize("mode", ["base", "prompt", "adapter"])
def test_worker_inputs_match_rewrite_contract_without_mutating_request(monkeypatch, mode):
    calls = []
    closed = []

    class RecordingEngine:
        def __init__(self, model, adapter, **kwargs):
            calls.append(("init", model, adapter, kwargs))

        def rewrite(self, draft, context, **kwargs):
            calls.append(("rewrite", draft, copy.deepcopy(context), kwargs))
            return "styled " + draft

        def close(self):
            closed.append(True)

    monkeypatch.setattr(eval_worker, "MLXEngine", RecordingEngine)
    request = {
        "mode": mode,
        "model_path": "synthetic-model",
        "adapter_path": "synthetic-adapter",
        "cases": [
            {"draft": "First draft", "context": [{"speaker": "other", "text": "Hi"}]},
            {"draft": "Second draft"},
        ],
        "references": ["authentic synthetic style"],
        "memory_gb": 6,
    }
    original = copy.deepcopy(request)
    assert eval_worker.generate(request) == ["styled First draft", "styled Second draft"]
    assert request == original
    assert closed == [True]
    assert calls[0] == (
        "init",
        Path("synthetic-model"),
        Path("synthetic-adapter") if mode == "adapter" else None,
        {"memory_gb": 6},
    )
    for index, call in enumerate(calls[1:]):
        expected_context = original["cases"][index].get("context", [])
        if mode == "prompt":
            assert call[2][0]["speaker"] == "style_reference"
            assert json.dumps(original["references"], ensure_ascii=False) in call[2][0]["text"]
            assert call[2][1:] == expected_context
        else:
            assert call[2] == expected_context
        assert call[1] == original["cases"][index]["draft"]
        assert call[3] == {"max_output_tokens": 128}


def test_worker_closes_engine_when_generation_fails(monkeypatch):
    closed = []

    class FailingEngine:
        def __init__(self, *args, **kwargs):
            pass

        def rewrite(self, *args, **kwargs):
            raise RuntimeError("synthetic generation failure")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(eval_worker, "MLXEngine", FailingEngine)
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        eval_worker.generate(
            {
                "mode": "adapter",
                "model_path": "model",
                "adapter_path": "adapter",
                "cases": [{"draft": "draft"}],
                "references": [],
            }
        )
    assert closed == [True]


def test_unsupported_mode_does_not_create_engine(monkeypatch):
    def should_not_load(*args, **kwargs):
        raise AssertionError("Unsupported mode must not load a model")

    monkeypatch.setattr(eval_worker, "MLXEngine", should_not_load)
    with pytest.raises(ValueError, match="Unsupported"):
        eval_worker.generate({"mode": "unsupported"})


@pytest.mark.parametrize(
    "returncode,worker_output",
    [
        (2, None),
        (0, None),
        (0, ["one", "extra"]),
        (0, [42]),
        (0, {"wrong": "shape"}),
    ],
)
def test_parent_worker_failure_cannot_publish_comparison_or_qualification(
    tmp_path,
    monkeypatch,
    returncode,
    worker_output,
):
    monkeypatch.setattr(evaluation, "_artifact_binding", lambda _: {"synthetic": True})
    write_json(tmp_path / "model.json", {"path": "synthetic-model"})
    write_jsonl(
        tmp_path / "dataset/train.jsonl",
        [{"messages": [{"role": "assistant", "content": "synthetic style"}]}],
    )
    write_json(
        tmp_path / "cases.json",
        [{"draft": "PRIVATE_SYNTHETIC_DRAFT", "independently_authored": True}],
    )
    # A stale successful worker response must not be reused after a new failure.
    write_json(tmp_path / "evaluation-base-output.json", ["stale output"])

    def child_run(args, **kwargs):
        if worker_output is not None:
            write_json(Path(args[-1]), worker_output)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(evaluation.subprocess, "run", child_run)
    with pytest.raises((RuntimeError, ValueError)) as error:
        evaluation.generate_comparison(tmp_path, tmp_path / "cases.json")
    assert "PRIVATE_SYNTHETIC_DRAFT" not in str(error.value)
    for filename in (
        "comparison-private.json",
        "evaluation-request.json",
        "evaluation.json",
        "active.json",
    ):
        assert not (tmp_path / filename).exists()


@pytest.fixture
def headroom_clock(monkeypatch):
    state = {"elapsed": 0.0, "sleeps": []}
    monkeypatch.setattr(eval_worker.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(eval_worker.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(eval_worker.time, "monotonic", lambda: state["elapsed"])

    def sleep(seconds):
        state["sleeps"].append(seconds)
        state["elapsed"] += seconds

    monkeypatch.setattr(eval_worker.time, "sleep", sleep)
    return state


def test_headroom_retry_recovers_after_transient_gpu_release_delay(monkeypatch, headroom_clock):
    calls = []

    def preflight(*, memory_gb):
        calls.append(memory_gb)
        if len(calls) < 3:
            raise RuntimeError("application headroom is still settling")
        return {"budget_bytes": 7 * 1024**3}

    monkeypatch.setattr(eval_worker, "preflight", preflight)
    eval_worker.wait_for_headroom(7, seconds=3)
    assert calls == [7, 7, 7]
    assert headroom_clock == {"elapsed": 2.0, "sleeps": [1, 1]}


def test_headroom_retry_is_bounded_when_pressure_never_recovers(monkeypatch, headroom_clock):
    calls = []

    def preflight(*, memory_gb):
        calls.append(memory_gb)
        raise RuntimeError("memory pressure remains elevated")

    monkeypatch.setattr(eval_worker, "preflight", preflight)
    with pytest.raises(RuntimeError, match="pressure remains elevated"):
        eval_worker.wait_for_headroom(7, seconds=3)
    assert calls == [7, 7, 7, 7]
    assert headroom_clock == {"elapsed": 3.0, "sleeps": [1, 1, 1]}


def test_invalid_memory_configuration_is_not_retried(monkeypatch, headroom_clock):
    def preflight(*, memory_gb):
        raise ValueError("invalid budget")

    monkeypatch.setattr(eval_worker, "preflight", preflight)
    with pytest.raises(ValueError, match="invalid budget"):
        eval_worker.wait_for_headroom(float("nan"), seconds=30)
    assert headroom_clock == {"elapsed": 0.0, "sleeps": []}
