import json
import subprocess

import pytest
from fastapi.testclient import TestClient

from multilora_tone_adapter.api import create_app
from multilora_tone_adapter.common import write_json
from multilora_tone_adapter.rewrite_review import (
    CHECKS,
    codex_review,
    finalize,
    import_review,
    make_request,
    store_request,
)


def approval(request, **changes):
    return {
        "request_hash": request["request_hash"],
        "checks": {**dict.fromkeys(CHECKS, "pass"), **changes},
        "reason": "Every proposition and its participants are retained.",
    }


class Engine:
    loaded = False

    def __init__(self, output="i might bring the sketchbook"):
        self.output = output
        self.calls = 0

    def rewrite(self, draft, *, max_output_tokens=128):
        self.loaded = True
        self.calls += 1
        return self.output


def test_unconfigured_final_api_returns_original_without_loading_model():
    engine = Engine()
    draft = "I might bring the sketchbook."
    response = TestClient(create_app(engine)).post("/v1/rewrite", json={"draft": draft})
    assert response.status_code == 200
    assert response.json()["output"] == draft
    assert response.json()["status"] == "fallback_original"
    assert response.json()["reason"] == "review_required"
    assert engine.calls == 0


def test_candidate_has_no_final_output_and_is_bound_to_actual_draft():
    engine = Engine()
    draft = "I might bring the sketchbook."
    response = TestClient(create_app(engine, "test-adapter")).post(
        "/v1/rewrite/candidate", json={"draft": draft}
    ).json()
    assert "output" not in response
    assert response["status"] == "requires_semantic_review"
    assert response["candidate"] == engine.output
    assert response["request_hash"] == make_request(draft, engine.output, "test-adapter")["request_hash"]


@pytest.mark.parametrize(
    "candidate,check,result",
    [
        ("you might bring the sketchbook", "actors", "fail"),
        ("i will bring the sketchbook", "uncertainty", "fail"),
        ("bring the sketchbook", "actors", "uncertain"),
    ],
)
def test_regex_invisible_semantic_errors_return_original(candidate, check, result):
    draft = "I might bring the sketchbook."
    def reviewer(request):
        return approval(request, **{check: result})
    response = TestClient(create_app(Engine(candidate), semantic_reviewer=reviewer)).post(
        "/v1/rewrite", json={"draft": draft}
    ).json()
    assert response["output"] == draft
    assert response["status"] == "fallback_original"
    assert candidate not in response.values()


def test_reviewed_equivalent_rewrite_is_released():
    engine = Engine()
    response = TestClient(create_app(engine, semantic_reviewer=approval)).post(
        "/v1/rewrite", json={"draft": "I might bring the sketchbook."}
    ).json()
    assert response["output"] == engine.output
    assert response["status"] == "rewritten"


def test_reviewer_errors_and_malformed_responses_fail_closed():
    def broken(request):
        raise RuntimeError("private reviewer information")

    for reviewer in (broken, lambda request: {}, lambda request: approval(make_request("x", "y", "z"))):
        response = TestClient(create_app(Engine(), semantic_reviewer=reviewer)).post(
            "/v1/rewrite", json={"draft": "I might bring the sketchbook."}
        )
        assert response.json()["reason"] == "review_unavailable"
        assert "private reviewer" not in response.text


def test_reviewer_cannot_change_candidate_under_review():
    from multilora_tone_adapter.common import digest

    def mutating_reviewer(request):
        request["binding"]["candidate"] = "i will bring a camera"
        request["request_hash"] = digest(request["binding"])
        return approval(request)

    draft = "I might bring the sketchbook."
    response = TestClient(create_app(Engine(), semantic_reviewer=mutating_reviewer)).post(
        "/v1/rewrite", json={"draft": draft}
    ).json()
    assert response["output"] == draft
    assert response["reason"] == "review_unavailable"


def test_english_qualification_does_not_release_uncovered_script():
    draft, target = "明天带画本。", "明天带画本"
    request = make_request(draft, target, "test-adapter")
    assert finalize(request, approval(request))["reason"] == "outside_qualified_scope"
    multilingual = make_request(draft, target, "test-adapter", language_scope="multilingual")
    assert finalize(multilingual, approval(multilingual))["status"] == "rewritten"
    assert multilingual["request_hash"] != request["request_hash"]


def test_semantic_pass_cannot_override_changed_protected_number():
    request = make_request("Bring 6 markers.", "bring 8 markers", "test-adapter")
    assert finalize(request, approval(request))["reason"] == "protected_check_failed"


def test_pronominal_one_does_not_introduce_a_spurious_number():
    request = make_request("That is the one I meant.", "that's what i meant", "test-adapter")
    assert finalize(request, approval(request))["status"] == "rewritten"


def test_review_is_bound_to_candidate_draft_and_version():
    request = make_request("Bring the sketchbook.", "bring the sketchbook", "test-adapter")
    for field in ("draft", "candidate", "adapter_version"):
        changed = json.loads(json.dumps(request))
        changed["binding"][field] += " changed"
        with pytest.raises(ValueError, match="changed"):
            finalize(changed, approval(request))
    for check in CHECKS:
        response = approval(request)
        del response["checks"][check]
        with pytest.raises(ValueError):
            finalize(request, response)


def test_private_request_import_is_idempotent_and_rejects_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(tmp_path))
    draft, candidate = "Bring the sketchbook.", "bring the sketchbook"
    request = make_request(draft, candidate, "test-adapter")
    result = store_request(draft, {
        "status": "requires_semantic_review", "candidate": candidate,
        "adapter_version": "test-adapter", "request_hash": request["request_hash"],
    })
    from pathlib import Path

    request_path = Path(result["request_file"])
    assert request_path.stat().st_mode & 0o777 == 0o600
    assert request_path.parent.stat().st_mode & 0o777 == 0o700
    response = request_path.parent / "response.json"
    write_json(response, approval(request))
    first = import_review(request_path, response)
    assert import_review(request_path, response) == first
    write_json(response, approval(request, actors="uncertain"))
    with pytest.raises(ValueError, match="already reviewed"):
        import_review(request_path, response)


def test_private_requests_reject_git_state_override(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(tmp_path / "state"))
    request = make_request("Bring it.", "bring it", "test-adapter")
    with pytest.raises(ValueError, match="Git"):
        store_request("Bring it.", {
            "status": "requires_semantic_review", "candidate": "bring it",
            "adapter_version": "test-adapter", "request_hash": request["request_hash"],
        })


def test_auto_review_process_failure_returns_original(tmp_path, monkeypatch):
    request = make_request("Bring the sketchbook.", "bring the sketchbook", "test-adapter")
    request_path = tmp_path / "request.json"
    write_json(request_path, request)

    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired("codex", 1)

    monkeypatch.setattr(subprocess, "run", unavailable)
    result = codex_review(request_path)
    assert result["status"] == "fallback_original"
    assert result["output"] == request["binding"]["draft"]
    assert result["reason"] == "review_unavailable"


def test_auto_review_uses_existing_config_and_only_bound_request(tmp_path, monkeypatch):
    request = make_request("Bring the sketchbook.", "bring the sketchbook", "test-adapter")
    request_path = tmp_path / "request.json"
    write_json(request_path, request)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert args[:3] == ["codex", "exec", "--ephemeral"]
        assert "--sandbox" in args and "read-only" in args
        assert "--model" not in args
        assert json.loads(kwargs["input"]) == request
        from pathlib import Path

        write_json(Path(args[args.index("--output-last-message") + 1]), approval(request))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", run)
    assert codex_review(request_path)["status"] == "rewritten"
    assert codex_review(request_path)["status"] == "rewritten"
    assert len(calls) == 1
