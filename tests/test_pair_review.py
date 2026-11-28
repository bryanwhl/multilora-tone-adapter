import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_dataset import independent_episodes, labels_for, synthetic_reviews_for

from multilora_tone_adapter import pair_review
from multilora_tone_adapter.common import digest, read_jsonl, write_json, write_jsonl
from multilora_tone_adapter.dataset import build_dataset, verify_curation_files
from multilora_tone_adapter.labeling import has_correction_chain, make_request, validate_pair


@pytest.fixture
def run(tmp_path):
    episodes = independent_episodes()
    episodes[0]["context"] = [{"speaker": "other", "text": "PRIVATE HISTORY SENTINEL"}]
    labels = labels_for(episodes)
    for label in labels:
        label["reason"] = "GENERATOR VERDICT SENTINEL"
    write_jsonl(tmp_path / "episodes.jsonl", episodes)
    write_jsonl(tmp_path / "labels.jsonl", labels)
    return tmp_path


def response_for(request, verdict="pass"):
    rows = []
    for pair in request["pairs"]:
        checks = {name: "pass" for name in pair_review.REVIEW_CHECKS}
        checks["actors"] = verdict
        rows.append({**{k: pair[k] for k in ("id", "source_hash", "draft_hash", "target_hash")},
                     "checks": checks, "verdict": verdict,
                     "reason": "Synthetic independent semantic assessment."})
    return {"request_hash": request["request_hash"], "reviews": rows}


def prepare_response(run, verdict="pass", limit=2):
    result = pair_review.make_review_request(run, limit)
    request = json.loads(Path(result["request_file"]).read_text())
    response = response_for(request, verdict)
    path = run / "response.json"
    write_json(path, response)
    return request, response, path


def test_request_is_blind_and_binds_exact_source_draft_and_target(run):
    result = pair_review.make_review_request(run, 2)
    text = Path(result["request_file"]).read_text()
    request = json.loads(text)
    assert "PRIVATE HISTORY SENTINEL" not in text
    assert "GENERATOR VERDICT SENTINEL" not in text
    assert '"meaning_preserved"' not in text
    assert set(request["pairs"][0]) == {"id", "draft", "target", "source_hash",
                                        "draft_hash", "target_hash"}
    episode = read_jsonl(run / "episodes.jsonl")[0]
    label = read_jsonl(run / "labels.jsonl")[0]
    assert request["pairs"][0]["source_hash"] == digest(episode)
    assert request["pairs"][0]["draft_hash"] == digest(label["draft"])
    assert request["pairs"][0]["target_hash"] == digest(episode["target"])
    assert (Path(result["request_file"]).stat().st_mode & 0o777) == 0o600


@pytest.mark.parametrize("verdict", ["pass", "fail", "uncertain"])
def test_import_retains_checks_and_does_not_re_request_completed_pair(run, verdict):
    request, _, path = prepare_response(run, verdict)
    original_sources = (run / "episodes.jsonl").read_bytes()
    result = pair_review.import_reviews(run, path)
    assert result["accepted"] == (2 if verdict == "pass" else 0)
    reviews = read_jsonl(run / "pair-reviews.jsonl")
    assert len(reviews) == 2 and reviews[0]["verdict"] == verdict
    assert reviews[0]["review_contract"] == pair_review.review_contract()
    assert (run / "episodes.jsonl").read_bytes() == original_sources
    pair_review.import_reviews(run, path)
    assert read_jsonl(run / "pair-reviews.jsonl") == reviews
    assert (run / "pair-review-requests" / f"{request['request_hash']}.json").exists()
    assert (run / "pair-review-responses" / f"{request['request_hash']}.json").exists()
    next_request = pair_review.make_review_request(run, 100)
    assert next_request["count"] == 8


@pytest.mark.parametrize("tamper", ["request_hash", "unknown_id", "duplicate_id", "draft_hash",
                                    "source_hash", "missing_check", "unknown_check", "bad_check",
                                    "inconsistent_verdict", "reason_type", "empty_reason"])
def test_invalid_review_is_rejected_without_writes(run, tamper):
    _, response, path = prepare_response(run)
    row = response["reviews"][0]
    if tamper == "request_hash":
        response["request_hash"] = "wrong"
    elif tamper == "unknown_id":
        row["id"] = "unknown"
    elif tamper == "duplicate_id":
        response["reviews"][1] = copy.deepcopy(row)
    elif tamper in ("draft_hash", "source_hash"):
        row[tamper] = "wrong"
    elif tamper == "missing_check":
        del row["checks"]["actors"]
    elif tamper == "unknown_check":
        row["checks"]["new"] = "pass"
    elif tamper == "bad_check":
        row["checks"]["actors"] = True
    elif tamper == "inconsistent_verdict":
        row["checks"]["actors"] = "fail"
    elif tamper == "reason_type":
        row["reason"] = []
    else:
        row["reason"] = " "
    write_json(path, response)
    with pytest.raises(pair_review.ReviewResponseError):
        pair_review.import_reviews(run, path)
    assert not (run / "pair-reviews.jsonl").exists()


@pytest.mark.parametrize("field", ["target", "context", "source_ids", "draft"])
def test_stale_source_or_label_fails_closed(run, field):
    _, _, path = prepare_response(run)
    source = run / ("labels.jsonl" if field == "draft" else "episodes.jsonl")
    rows = read_jsonl(source)
    rows[0][field] = "changed" if field in ("draft", "target") else []
    write_jsonl(source, rows)
    with pytest.raises(ValueError, match="source or draft"):
        pair_review.import_reviews(run, path)
    assert not (run / "pair-reviews.jsonl").exists()


def test_cross_batch_response_rejected_and_completed_batches_are_immutable(run):
    request, response, path = prepare_response(run)
    pair_review.import_reviews(run, path)
    old = (run / "pair-reviews.jsonl").read_bytes()
    response["reviews"][0]["checks"]["actors"] = "fail"
    response["reviews"][0]["verdict"] = "fail"
    write_json(path, response)
    with pytest.raises(ValueError, match="immutable"):
        pair_review.import_reviews(run, path)
    assert (run / "pair-reviews.jsonl").read_bytes() == old
    pair_review.make_review_request(run, 2)
    write_json(path, response_for(request))
    with pytest.raises(pair_review.ReviewResponseError, match="request hash"):
        pair_review.import_reviews(run, path)
    assert (run / "pair-reviews.jsonl").read_bytes() == old


def test_regenerated_draft_gets_new_review_and_preserves_old_review(run):
    _, _, path = prepare_response(run, "fail", 1)
    pair_review.import_reviews(run, path)
    old = read_jsonl(run / "pair-reviews.jsonl")
    labels = read_jsonl(run / "labels.jsonl")
    labels[0]["draft"] = "An apple, please."
    write_jsonl(run / "labels.jsonl", labels)
    result = pair_review.make_review_request(run, 1)
    request = json.loads(Path(result["request_file"]).read_text())
    assert request["pairs"][0]["id"] == old[0]["id"]
    assert request["pairs"][0]["draft_hash"] != old[0]["draft_hash"]
    write_json(path, response_for(request))
    pair_review.import_reviews(run, path)
    assert read_jsonl(run / "pair-reviews.jsonl")[0] == old[0]
    assert len(read_jsonl(run / "pair-reviews.jsonl")) == 2


def test_automatic_review_uses_fresh_read_only_codex_and_retries_only_response_errors(run, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        request = json.loads(kwargs["input"])
        response = response_for(request)
        if len(calls) == 1:
            response["reviews"][0]["checks"]["actors"] = "fail"
        write_json(Path(args[args.index("--output-last-message") + 1]), response)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pair_review.subprocess, "run", fake_run)
    result = pair_review.codex_review(run, batch_size=1, max_batches=1, model="chosen-model")
    assert result["reviewed"] == 1 and result["response_retries_used"] == 1
    assert calls[0][1]["input"] == calls[1][1]["input"]
    assert "--ephemeral" in calls[0][0]
    assert calls[0][0][calls[0][0].index("--sandbox") + 1] == "read-only"
    assert calls[0][0][calls[0][0].index("--model") + 1] == "chosen-model"
    assert calls[0][1]["cwd"] != run
    assert len(list((run / "pair-review-attempts").glob("*.json"))) == 2


@pytest.mark.parametrize("failure", ["process", "missing_output", "timeout", "changed_source"])
def test_automatic_review_does_not_retry_process_or_source_failures(run, monkeypatch, failure):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 1)
        if failure == "changed_source":
            source = read_jsonl(run / "episodes.jsonl")
            source[0]["context"] = []
            write_jsonl(run / "episodes.jsonl", source)
            write_json(Path(args[args.index("--output-last-message") + 1]),
                       response_for(json.loads(kwargs["input"])))
        return SimpleNamespace(returncode=1 if failure == "process" else 0)

    monkeypatch.setattr(pair_review.subprocess, "run", fake_run)
    with pytest.raises((ValueError, RuntimeError, subprocess.TimeoutExpired)):
        pair_review.codex_review(run, batch_size=1, max_batches=1)
    assert len(calls) == 1
    assert not (run / "pair-reviews.jsonl").exists()


def test_unreviewed_or_stale_pairs_cannot_enter_build(tmp_path):
    episodes = independent_episodes()
    labels = labels_for(episodes)
    with pytest.raises(ValueError, match="Only 0 independently reviewed"):
        build_dataset(episodes, labels, tmp_path / "unreviewed", 1)
    reviews = synthetic_reviews_for(episodes, labels)
    labels[0]["draft"] = "An apple, please."
    with pytest.raises(ValueError, match="Only 9 independently reviewed"):
        build_dataset(episodes, labels, tmp_path / "stale", 10, reviews=reviews)
    assert not (tmp_path / "stale").exists()


@pytest.mark.parametrize("verdict", ["fail", "uncertain"])
def test_rejected_review_cannot_enter_build(tmp_path, verdict):
    episodes = independent_episodes()
    labels = labels_for(episodes)
    reviews = synthetic_reviews_for(episodes, labels)
    reviews[0]["checks"]["actors"] = verdict
    reviews[0]["verdict"] = verdict
    with pytest.raises(ValueError, match="Only 9 independently reviewed"):
        build_dataset(episodes, labels, tmp_path / "dataset", 10, reviews=reviews)


def test_identity_examples_are_bounded_deduplicated_train_only_and_not_counted(tmp_path):
    episodes = []
    for i in range(30):
        episodes.append({"id": f"{i:03}", "target": f"item{i} please!", "context": [],
                         "group_id": f"session:{i}", "source_ids": [f"source:{i}"],
                         "timestamp": f"2026-01-{i+1:02}T00:00:00+00:00"})
    episodes[1]["target"] = episodes[0]["target"]
    labels = labels_for(episodes)
    labels[0]["draft"] = episodes[0]["target"]
    reviews = synthetic_reviews_for(episodes, labels)
    output = tmp_path / "dataset"
    manifest = build_dataset(episodes, labels, output, 29, reviews=reviews)
    verify_curation_files(output, manifest)
    assert manifest["accepted_pairs"] == 29
    assert manifest["identity_pairs"] == 2
    assert sum(manifest["counts"].values()) == 31
    assert manifest["base_counts"]["train"] == 23
    provenance = read_jsonl(output / "provenance.jsonl")
    identities = [row for row in provenance if row["kind"] == "identity"]
    assert len(identities) == 2 and all(row["split"] == "train" for row in identities)
    training = read_jsonl(output / "train.jsonl")
    targets = []
    for row in identities:
        sample = training[row["row"]]["messages"]
        assert json.loads(sample[1]["content"])["draft"] == sample[2]["content"]
        assert row["source_ids"] == next(e for e in episodes if e["id"] == row["id"])["source_ids"]
        targets.append(sample[2]["content"])
    assert len(set(targets)) == len(targets)
    assert build_dataset(list(reversed(episodes)), list(reversed(labels)), output, 29,
                         reviews=list(reversed(reviews))) == manifest


@pytest.mark.parametrize("target", ["bring the cove\ncover*", "pick the tray\n*train"])
def test_explicit_correction_chains_are_quarantined_without_editing_targets(tmp_path, target):
    assert has_correction_chain(target)
    assert "correction_chain" in validate_pair("Please bring the cover.", target)
    episodes = independent_episodes()
    episodes[0]["target"] = target
    write_jsonl(tmp_path / "episodes.jsonl", episodes)
    before = (tmp_path / "episodes.jsonl").read_bytes()
    request = make_request(tmp_path)
    assert request["count"] == 9
    assert target not in (tmp_path / "label-request.json").read_text()
    assert (tmp_path / "episodes.jsonl").read_bytes() == before
    labels = labels_for(episodes)
    manifest = build_dataset(episodes, labels, tmp_path / "dataset", 9,
                             reviews=synthetic_reviews_for(episodes, labels))
    assert manifest["rejected"] == {"correction_chain": 1}


@pytest.mark.parametrize("target", ["cover*", "bring the cover*", "shopping list\n* milk",
                                    "thank you\n*thanks*", "read the files\n*.md",
                                    "a note*\nHere is its explanation", "A * B = C"])
def test_correction_screen_preserves_markdown_wildcards_and_ordinary_text(target):
    assert not has_correction_chain(target)


def test_private_review_output_refused_inside_checkout(run):
    (run / ".git").mkdir()
    with pytest.raises(ValueError, match="outside"):
        pair_review.make_review_request(run)


def test_frozen_review_file_and_exact_row_binding_verified(tmp_path):
    episodes = independent_episodes()
    labels = labels_for(episodes)
    reviews = synthetic_reviews_for(episodes, labels)
    output = tmp_path / "dataset"
    manifest = build_dataset(episodes, labels, output, 10, reviews=reviews)
    verify_curation_files(output, manifest)
    assert read_jsonl(output / "pair-reviews.jsonl") == reviews
    rows = read_jsonl(output / "train.jsonl")
    rows[0]["messages"][1]["content"] = json.dumps({"draft": "A different request."})
    write_jsonl(output / "train.jsonl", rows)
    manifest["files"]["train.jsonl"] = hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest()
    manifest["dataset_hash"] = digest({k: v for k, v in manifest.items() if k != "dataset_hash"})
    (output / "READY").write_text(manifest["dataset_hash"] + "\n")
    with pytest.raises(ValueError, match="matching accepted independent review"):
        verify_curation_files(output, manifest)


def test_legacy_dataset_and_forged_identity_counts_rejected(tmp_path):
    episodes = independent_episodes()
    labels = labels_for(episodes)
    output = tmp_path / "dataset"
    manifest = build_dataset(episodes, labels, output, 10,
                             reviews=synthetic_reviews_for(episodes, labels))
    legacy = {**manifest, "schema_version": 2}
    with pytest.raises(ValueError, match="independent pair review"):
        verify_curation_files(output, legacy)
    manifest["identity_policy"]["limit"] = 100
    with pytest.raises(ValueError, match="identity limits"):
        verify_curation_files(output, manifest)


def refresh_manifest_files(output, manifest):
    """Simulate rehashed artifact mutation, not merely a corrupt file digest."""
    manifest["provenance_hash"] = digest(read_jsonl(output / "provenance.jsonl"))
    for name in manifest["files"]:
        manifest["files"][name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    manifest["dataset_hash"] = digest({k: v for k, v in manifest.items() if k != "dataset_hash"})
    write_json(output / "manifest.json", manifest)
    (output / "READY").write_text(manifest["dataset_hash"] + "\n")


@pytest.mark.parametrize("binding", ["frozen_split", "group", "source"])
def test_rehashed_train_test_swaps_cannot_break_frozen_source_groups(tmp_path, binding):
    episodes = independent_episodes()
    if binding == "source":
        episodes[1]["source_ids"].append(episodes[0]["source_ids"][0])
    else:
        episodes[1]["group_id"] = episodes[0]["group_id"]
    labels = labels_for(episodes)
    output = tmp_path / "dataset"
    manifest = build_dataset(episodes, labels, output, 10,
                             reviews=synthetic_reviews_for(episodes, labels))
    verify_curation_files(output, manifest)
    training, held_out = read_jsonl(output / "train.jsonl"), read_jsonl(output / "test.jsonl")
    training[0], held_out[0] = held_out[0], training[0]
    write_jsonl(output / "train.jsonl", training)
    write_jsonl(output / "test.jsonl", held_out)
    provenance = read_jsonl(output / "provenance.jsonl")
    moved_train = next(row for row in provenance if row["split"] == "train" and row["row"] == 0)
    moved_test = next(row for row in provenance if row["split"] == "test" and row["row"] == 0)
    moved_train["split"], moved_test["split"] = "test", "train"
    if binding != "frozen_split":
        # Even an edited split map cannot place shared groups/messages across splits.
        manifest["splits"][moved_train["id"]] = "test"
        manifest["splits"][moved_test["id"]] = "train"
        manifest["split_hash"] = digest(manifest["splits"])
    write_jsonl(output / "provenance.jsonl", provenance)
    refresh_manifest_files(output, manifest)
    reason = {"frozen_split": "frozen source split", "group": "source group crosses",
              "source": "source message crosses"}[binding]
    with pytest.raises(ValueError, match=reason):
        verify_curation_files(output, manifest)


def test_split_map_hash_and_membership_are_required(tmp_path):
    episodes = independent_episodes()
    labels = labels_for(episodes)
    output = tmp_path / "dataset"
    manifest = build_dataset(episodes, labels, output, 10,
                             reviews=synthetic_reviews_for(episodes, labels))
    del manifest["splits"][episodes[0]["id"]]
    refresh_manifest_files(output, manifest)
    with pytest.raises(ValueError, match="split assignment changed"):
        verify_curation_files(output, manifest)
    manifest["split_hash"] = digest(manifest["splits"])
    refresh_manifest_files(output, manifest)
    with pytest.raises(ValueError, match="frozen source split"):
        verify_curation_files(output, manifest)
