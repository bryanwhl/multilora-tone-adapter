"""CLI integration with synthetic data and an in-process fake local model."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from multilora_tone_adapter import cli, runtime
from multilora_tone_adapter.api import create_app
from multilora_tone_adapter.common import write_json, write_jsonl
from multilora_tone_adapter.rewrite_review import CHECKS


class Engine:
    loaded = False

    def rewrite(self, draft, *, max_output_tokens=128):
        return "i might bring the sketchbook"


@pytest.mark.parametrize("certainty,result", [("pass", "rewritten"), ("uncertain", "fallback_original")])
def test_cli_candidate_review_import_round_trip(tmp_path, monkeypatch, capsys, certainty, result):
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(tmp_path / "private"))
    client = TestClient(create_app(Engine(), "synthetic-adapter"))

    def local_post(url, **kwargs):
        assert url == "http://127.0.0.1:8765/v1/rewrite/candidate"
        return client.post("/v1/rewrite/candidate", json=kwargs["json"])

    monkeypatch.setattr(httpx, "post", local_post)
    draft = tmp_path / "draft.txt"
    draft.write_text("I might bring the sketchbook.")
    assert cli.main(["rewrite", "--draft-file", str(draft)]) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert "output" not in prepared
    request_path = Path(prepared["request_file"])
    request = json.loads(request_path.read_text())
    response = request_path.parent / "response.json"
    write_json(response, {
        "request_hash": request["request_hash"],
        "checks": {**dict.fromkeys(CHECKS, "pass"), "uncertainty": certainty},
        "reason": "Synthetic review of equivalent certainty.",
    })
    assert cli.main(["rewrite-import", "--request", str(request_path), "--input", str(response)]) == 0
    final = json.loads(capsys.readouterr().out)
    assert final["status"] == result
    assert final["output"] == ("i might bring the sketchbook" if certainty == "pass" else draft.read_text())


def test_cli_model_override_requires_explicit_auto_review(tmp_path, capsys):
    assert cli.main(["rewrite", "--draft-file", str(tmp_path / "not-read"), "--model", "override"]) == 2
    assert "--auto-review" in capsys.readouterr().err


def test_cli_checkpoint_fork_does_not_precreate_destination(tmp_path, monkeypatch, capsys):
    from multilora_tone_adapter import checkpoints

    monkeypatch.setenv("MULTILORA_STATE_DIR", str(tmp_path))

    def fork(source, destination, step):
        assert not destination.exists()
        assert source.name == "source" and destination.name == "candidate" and step == 8
        return {"forked": True}

    monkeypatch.setattr(checkpoints, "fork_checkpoint", fork)
    assert cli.main(["checkpoint-fork", "--run", "source", "--new-run", "candidate", "--step", "8"]) == 0
    assert json.loads(capsys.readouterr().out)["forked"] is True


@pytest.mark.parametrize("extra,expected", [([], 6), (["--epochs", "0.5"], 2), (["--iters", "4"], 4)])
def test_cli_training_duration_uses_actual_rows(tmp_path, monkeypatch, capsys, extra, expected):
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(tmp_path))
    directory = cli.run_dir("synthetic")
    write_jsonl(directory / "dataset/train.jsonl", [{"synthetic": True}] * 3)
    monkeypatch.setattr(cli, "resolve_model", lambda _: tmp_path / "fake-model")

    def train(*args, **kwargs):
        assert kwargs["iters"] == expected
        assert kwargs["save_every"] == 1
        return {"synthetic": True}

    monkeypatch.setattr(runtime, "train", train)
    assert cli.main(["train", "--run", "synthetic", "--save-every", "1", *extra]) == 0
    assert json.loads(capsys.readouterr().out)["synthetic"] is True


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_invalid_epoch_duration_fails_before_model_work(tmp_path, monkeypatch, capsys, value):
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "resolve_model", lambda _: pytest.fail("Must not resolve model"))
    assert cli.main(["train", "--run", "synthetic", "--epochs", value]) == 2
    assert "epochs" in capsys.readouterr().err
