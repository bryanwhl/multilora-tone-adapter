import json

import pytest

from multilora_tone_adapter.common import read_jsonl
from multilora_tone_adapter.dataset import assign_splits, build_dataset
from multilora_tone_adapter.pair_review import REVIEW_CHECKS, review_binding, review_contract
from multilora_tone_adapter.telegram import import_telegram


def message(mid, sender, text, timestamp):
    return {
        "id": mid,
        "type": "message",
        "from_id": sender,
        "text": text,
        "date_unixtime": str(timestamp),
    }


def export(tmp_path, messages, account=False):
    chat = {"id": 77, "messages": messages}
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"chats": {"list": [chat]}} if account else chat))
    return path


def test_telegram_bursts_context_and_session_boundaries(tmp_path):
    path = export(
        tmp_path,
        [
            message(1, "user2", "question", 1000),
            message(2, "user1", ["yes ", {"text": "please"}], 1010),
            message(3, "user1", "thanks", 1020),
            message(4, "user2", "another question", 1030),
            message(5, "user1", "reply", 1040),
            message(6, "user1", "new session", 5000),
        ],
        account=True,
    )
    episodes, report = import_telegram([path, path], "1")
    assert [e["target"] for e in episodes] == ["yes please\nthanks", "reply", "new session"]
    assert episodes[0]["context"][0]["text"] == "question"
    assert "telegram:77:2" not in episodes[0]["source_ids"][:1]
    assert episodes[-1]["context"] == []
    assert episodes[0]["group_id"] == episodes[1]["group_id"]
    assert episodes[0]["group_id"] != episodes[2]["group_id"]
    assert report["selected_episodes"] == 3
    assert "question" not in json.dumps(report)


def test_ambiguous_dates_and_identity_fail_closed(tmp_path):
    path = export(
        tmp_path,
        [
            {
                "id": 1,
                "type": "message",
                "from_id": "user1",
                "date": "2026-01-01T10:00:00",
                "text": "hello",
            }
        ],
    )
    with pytest.raises(ValueError, match="ambiguous"):
        import_telegram([path], "1")
    with pytest.raises(ValueError, match="numeric"):
        import_telegram([path], "Example Author")


def test_forwarded_messages_excluded_and_break_bursts(tmp_path):
    forwarded = message(2, "user1", "copied", 1010)
    forwarded["forwarded_from"] = "Somebody"
    path = export(
        tmp_path,
        [message(1, "user1", "hello", 1000), forwarded, message(3, "user1", "world", 1020)],
    )
    episodes, _ = import_telegram([path], "user1")
    assert [e["target"] for e in episodes] == ["hello", "world"]
    assert all("copied" not in json.dumps(e) for e in episodes)


def episodes_fixture(count=10):
    return [
        {
            "id": str(i),
            "chat_id": "hashed",
            "context": [],
            "target": f"thanks, appointment {i} works for me!",
            "source_ids": [f"source:{i}"],
            "group_id": f"session:{i}",
            "timestamp": f"2026-01-{i + 1:02d}T00:00:00+00:00",
        }
        for i in range(count)
    ]


def independent_episodes():
    words = ["apple", "boat", "cat", "door", "eagle", "fox", "grape", "hat", "ice", "jet"]
    result = episodes_fixture()
    for episode, word in zip(result, words):
        episode["target"] = f"{word} please!"
    return result


def labels_for(episodes):
    return [
        {
            "id": e["id"],
            "draft": "I would like " + e["target"].replace(" please!", "."),
            "meaning_preserved": True,
        }
        for e in episodes
    ]


def synthetic_reviews_for(episodes, labels):
    """Explicit passing review fixtures for synthetic unit-test pairs only."""
    by_id = {e["id"]: e for e in episodes}
    return [
        {**review_binding(by_id[label["id"]], label),
         "review_contract": review_contract(), "request_hash": "synthetic-unit-test-request",
         "checks": {name: "pass" for name in REVIEW_CHECKS}, "verdict": "pass",
         "reason": "Synthetic equivalent pair used by this unit test."}
        for label in labels if label["id"] in by_id
    ]


def test_splits_shared_sources_and_templates_never_leak():
    episodes = independent_episodes()
    episodes[1]["source_ids"].append(episodes[0]["source_ids"][0])
    episodes[2]["target"] = "Please confirm the appointment for 14 December at the usual location."
    episodes[3]["target"] = "Please confirm the appointment for 15 December at the usual location."
    splits = assign_splits(episodes)
    assert splits["0"] == splits["1"]
    assert splits["2"] == splits["3"]
    assert set(splits.values()) == {"train", "valid", "test"}
    assert assign_splits(list(reversed(episodes))) == splits


def test_generic_acknowledgements_do_not_merge_all_sessions():
    episodes = independent_episodes()
    for episode in episodes:
        episode["target"] = "ok"
    assert set(assign_splits(episodes).values()) == {"train", "valid", "test"}


def test_style_pairs_preserve_case_and_punctuation_only_changes(tmp_path):
    episodes = independent_episodes()
    labels = [{"id": e["id"], "draft": e["target"].capitalize().replace("!", "."),
               "meaning_preserved": True} for e in episodes]
    out = tmp_path / "dataset"
    manifest = build_dataset(episodes, labels, out, min_episodes=10, reviews=synthetic_reviews_for(episodes, labels))
    assert manifest["accepted_pairs"] == 10
    row = read_jsonl(out / "train.jsonl")[0]
    assert json.loads(row["messages"][1]["content"])["draft"] == "Apple please."
    assert row["messages"][2]["content"] == "apple please!"


def test_exact_copies_do_not_inflate_independent_rewrite_count(tmp_path):
    episodes = independent_episodes()
    labels = labels_for(episodes)
    labels[0]["draft"] = episodes[0]["target"]
    manifest = build_dataset(episodes, labels, tmp_path / "dataset", min_episodes=9, reviews=synthetic_reviews_for(episodes, labels))
    assert manifest["accepted_pairs"] == 9
    assert manifest["identity_pairs"] == 0  # Floor of 10% of seven rewrite train rows.
    assert "copied_target" not in manifest["rejected"]


def test_build_freezes_pure_mlx_rows_and_provenance(tmp_path):
    episodes = independent_episodes()
    labels = labels_for(episodes)
    out = tmp_path / "dataset"
    manifest = build_dataset(episodes, labels, out, min_episodes=10, reviews=synthetic_reviews_for(episodes, labels))
    assert manifest["counts"] == {"train": 8, "valid": 1, "test": 1}
    assert (
        build_dataset(list(reversed(episodes)), list(reversed(labels)), out, min_episodes=10, reviews=synthetic_reviews_for(episodes, labels))
        == manifest
    )
    row = read_jsonl(out / "train.jsonl")[0]
    assert set(row) == {"messages"}
    assert row["messages"][-1]["role"] == "assistant"
    assert len(read_jsonl(out / "provenance.jsonl")) == 10
    (out / "train.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        build_dataset(episodes, labels, out, min_episodes=10, reviews=synthetic_reviews_for(episodes, labels))


def test_label_validation_and_floor(tmp_path):
    episodes = independent_episodes()
    labels = labels_for(episodes)
    labels[0]["draft"] = episodes[0]["target"]
    with pytest.raises(ValueError, match="Only 9"):
        build_dataset(episodes, labels, tmp_path / "out", min_episodes=10, reviews=synthetic_reviews_for(episodes, labels))
    with pytest.raises(ValueError, match="unknown"):
        build_dataset(episodes, labels + [{"id": "missing"}], tmp_path / "out", 1)
    labels[0]["draft"] = "Meet at 11"
    with pytest.raises(ValueError, match="Only 9"):
        build_dataset(episodes, labels, tmp_path / "out", min_episodes=10, reviews=synthetic_reviews_for(episodes, labels))


def test_private_outputs_rejected_inside_checkout(tmp_path):
    (tmp_path / ".git").mkdir()
    episodes = independent_episodes()
    with pytest.raises(ValueError, match="outside"):
        build_dataset(episodes, labels_for(episodes), tmp_path / "out", 1, reviews=synthetic_reviews_for(episodes, labels_for(episodes)))


def test_conflicting_exports_cannot_silently_replace_history(tmp_path):
    original = export(tmp_path, [message(1, "user1", "original", 1000)])
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps({"id": 77, "messages": [message(1, "user1", "edited", 1000)]}))
    with pytest.raises(ValueError, match="Conflicting"):
        import_telegram([original, changed], "1")


def test_group_collapse_requires_more_independent_data():
    episodes = independent_episodes()
    for episode in episodes:
        episode["group_id"] = "one-session"
    with pytest.raises(ValueError, match="three independent"):
        assign_splits(episodes)


def test_interrupted_dataset_write_resumes_identically(tmp_path):
    episodes = independent_episodes()
    labels = labels_for(episodes)
    out = tmp_path / "dataset"
    manifest = build_dataset(episodes, labels, out, 10, reviews=synthetic_reviews_for(episodes, labels))
    (out / "manifest.json").unlink()
    (out / "READY").unlink()
    (out / "test.jsonl").unlink()
    assert build_dataset(episodes, labels, out, 10, reviews=synthetic_reviews_for(episodes, labels)) == manifest
    assert (out / "READY").read_text().strip() == manifest["dataset_hash"]


def test_session_reset_and_explicit_timezone_timestamp(tmp_path):
    path = export(
        tmp_path,
        [
            {
                "id": 1,
                "from_id": "user2",
                "type": "message",
                "text": "earlier",
                "date": "2026-01-01T10:00:00+02:00",
            },
            {
                "id": 2,
                "from_id": "user1",
                "type": "message",
                "text": "later",
                "date": "2026-01-01T11:00:00+02:00",
            },
        ],
    )
    episodes, _ = import_telegram([path], "1")
    assert episodes[0]["timestamp"] == "2026-01-01T09:00:00+00:00"
    assert episodes[0]["context"] == []


def test_allowlist_and_utc_half_open_dates_filter_both_target_and_context(tmp_path):
    path = tmp_path / "account.json"
    path.write_text(
        json.dumps(
            {
                "chats": {
                    "list": [
                        {
                            "id": 77,
                            "messages": [
                                message(1, "user2", "outside prior", 1000),
                                message(2, "user1", "included", 1010),
                                message(3, "user1", "exclusive boundary", 1020),
                            ],
                        },
                        {"id": 88, "messages": [message(1, "user1", "other chat", 1010)]},
                    ]
                }
            }
        )
    )
    episodes, report = import_telegram(
        [path],
        "1",
        allowed_chat_ids=["77"],
        since="1970-01-01T00:16:50Z",
        until="1970-01-01T00:17:00+00:00",
    )
    assert len(episodes) == 1
    assert episodes[0]["target"] == "included"
    assert episodes[0]["context"] == []
    assert report["outside_date_messages"] == 2
    assert report["excluded_chats"] == 1
    assert "outside prior" not in json.dumps(episodes)
    assert "other chat" not in json.dumps(episodes)


@pytest.mark.parametrize(
    "since,until",
    [
        ("2026-01-01T00:00:00", None),
        ("2026-01-03", "2026-01-02"),
        (None, "not-a-date"),
    ],
)
def test_ambiguous_or_inverted_filter_dates_rejected(tmp_path, since, until):
    path = export(tmp_path, [message(1, "user1", "hello", 1000)])
    with pytest.raises(ValueError):
        import_telegram([path], "1", since=since, until=until)


@pytest.mark.parametrize(
    "field,value",
    [
        ("photo", "photos/example.jpg"),
        ("file", "files/example.pdf"),
        ("media_type", "voice_message"),
        ("sticker_emoji", "😀"),
        ("poll", {"question": "synthetic poll"}),
    ],
)
def test_media_captions_are_not_targets_or_context(tmp_path, field, value):
    media = message(2, "user1", "caption depends on attachment", 1010)
    media[field] = value
    path = export(
        tmp_path, [message(1, "user1", "before", 1000), media, message(3, "user1", "after", 1020)]
    )
    episodes, report = import_telegram([path], "1")
    assert [e["target"] for e in episodes] == ["before", "after"]
    assert "caption depends" not in json.dumps(episodes)
    assert report["media_messages"] == 1


@pytest.mark.parametrize(
    "secret",
    [
        "gh" + "o_" + "A" * 30,
        "sk" + "-proj-" + "B" * 30,
        "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
        "api_key=" + "C" * 30,
    ],
)
def test_secrets_quarantined_from_own_targets_and_incoming_context(tmp_path, secret):
    from multilora_tone_adapter.common import write_jsonl
    from multilora_tone_adapter.labeling import make_request

    path = export(
        tmp_path,
        [
            message(1, "user1", "before", 1000),
            message(2, "user1", secret, 1010),
            message(3, "user2", secret, 1020),
            message(4, "user1", "after", 1030),
        ],
    )
    episodes, report = import_telegram([path], "1")
    assert [e["target"] for e in episodes] == ["before", "after"]
    assert secret not in json.dumps(episodes)
    assert report["quarantined_secret_messages"] == 2
    assert secret not in json.dumps(report)
    run = tmp_path / "run"
    write_jsonl(run / "episodes.jsonl", episodes)
    make_request(run)
    assert secret not in (run / "label-request.json").read_text()


def test_bounded_collection_balances_chats_before_recency(tmp_path):
    path = tmp_path / "account.json"
    path.write_text(
        json.dumps(
            {
                "chats": {
                    "list": [
                        {
                            "id": 77,
                            "messages": [
                                message(i, "user1", f"busy-{i}", i * 4000) for i in range(1, 10)
                            ],
                        },
                        {"id": 88, "messages": [message(1, "user1", "quiet", 1000)]},
                    ]
                }
            }
        )
    )
    episodes, report = import_telegram([path], "1", max_episodes=4)
    assert len(episodes) == 4
    assert "quiet" in [e["target"] for e in episodes]
    assert [e["target"] for e in episodes if e["target"].startswith("busy")] == [
        "busy-7",
        "busy-8",
        "busy-9",
    ]
    assert len({e["chat_id"] for e in episodes}) == 2
    assert report["selection_policy"] == "recent_round_robin_by_chat"
    assert import_telegram([path], "1", max_episodes=4) == (episodes, report)
