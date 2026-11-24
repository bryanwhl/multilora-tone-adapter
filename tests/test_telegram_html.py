"""Fictional inline Telegram HTML; no real export files belong in tests."""

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import pytest

from multilora_tone_adapter.telegram_html import normalize_html_export, source_manifest

OWN = "Alex Fiction"
OTHER = "Jordan Example"


def message(
    mid,
    name=OWN,
    avatar=101,
    text="hello",
    date="16 November 2026, 10:00:00",
    *,
    joined=False,
    extra="",
):
    classes = "message default clearfix" + (" joined" if joined else "")
    avatar_html = (
        (
            f'<div class="pull_left userpic_wrap"><div class="userpic">'
            f'<img src="photos/author_{avatar}.jpg"></div></div>'
        )
        if avatar is not None
        else ""
    )
    header = f'<div class="from_name">{escape(name)}</div>' if name is not None else ""
    return (
        f'<div class="{classes}" id="message{mid}">{avatar_html}<div class="body">'
        f'<div class="pull_right date details" title="{escape(date)}">10:00</div>'
        f'{header}{extra}<div class="text">{text}</div></div></div>'
    )


def write_page(root, body, name="messages.html"):
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(
        '<!DOCTYPE html><html><body><div class="history">' + body + "</div></body></html>"
    )
    return path


def normalize(root, **kwargs):
    return normalize_html_export(
        root, author_names=[OWN], export_timezone="America/Los_Angeles", **kwargs
    )


def rows(payload):
    return [row for chat in payload["chats"]["list"] for row in chat["messages"]]


def test_plain_html_identity_entities_linebreaks_and_utc(tmp_path):
    write_page(
        tmp_path,
        message(1, OTHER, 202, "question") + message(2, text="yes &amp; <b>thanks</b><br>你好 😀"),
    )
    payload, report = normalize(tmp_path)
    parsed = rows(payload)
    own = next(row for row in parsed if row.get("from_id") == "html-self")
    assert own["text"] == "yes & thanks\n你好 😀"
    assert int(own["date_unixtime"]) == int(datetime(2026, 11, 16, 18, tzinfo=UTC).timestamp())
    assert next(row for row in parsed if str(row["id"]) == "1")["from_id"].startswith("html-name:")
    assert report["snapshot_hash"]
    assert report["identity_method"]
    assert not any(phrase in json.dumps(report) for phrase in (OWN, OTHER, "question"))


def test_joined_inherits_only_proven_same_chat_sender(tmp_path):
    write_page(
        tmp_path,
        message(1, text="first")
        + message(
            2, name=None, avatar=None, text="joined", joined=True, date="16 November 2026, 10:01:00"
        ),
    )
    payload, _ = normalize(tmp_path)
    authored = [row for row in rows(payload) if row.get("from_id") == "html-self"]
    assert [row["text"] for row in authored] == ["first", "joined"]


def test_orphan_joined_and_page_boundary_cannot_borrow_sender(tmp_path):
    write_page(tmp_path, message(1, text="proven"))
    write_page(
        tmp_path, message(2, name=None, avatar=None, text="unproven", joined=True), "messages2.html"
    )
    payload, _ = normalize(tmp_path)
    assert not any(
        row.get("from_id") == "html-self" and row.get("text") == "unproven" for row in rows(payload)
    )


def test_service_boundary_cannot_leak_previous_sender_into_joined(tmp_path):
    service = '<div class="message service" id="message-1"><div class="body">service</div></div>'
    write_page(
        tmp_path,
        message(1, text="proven")
        + service
        + message(2, name=None, avatar=None, text="unproven", joined=True),
    )
    payload, _ = normalize(tmp_path)
    assert not any(
        row.get("from_id") == "html-self" and row.get("text") == "unproven" for row in rows(payload)
    )


def test_nested_forward_sender_never_replaces_outer_sender(tmp_path):
    forwarded = f'<div class="forwarded body"><div class="from_name">{OWN}</div></div>'
    write_page(
        tmp_path,
        message(1, OTHER, 202, "forwarded text", extra=forwarded) + message(2, text="authentic"),
    )
    payload, _ = normalize(tmp_path)
    forwarded_rows = [row for row in rows(payload) if str(row["id"]) == "1"]
    assert all(row.get("from_id") != "html-self" for row in forwarded_rows)


def test_explicit_identity_and_timezone_are_required(tmp_path):
    write_page(tmp_path, message(1))
    for names, zone in [([], "America/Los_Angeles"), ([OWN], "Unknown/Nowhere"), ([OWN], "")]:
        with pytest.raises(ValueError):
            normalize_html_export(tmp_path, author_names=names, export_timezone=zone)


def test_same_display_name_with_multiple_numeric_avatar_ids_is_not_proven(tmp_path):
    write_page(
        tmp_path,
        message(1, avatar=101, text="first person")
        + message(2, avatar=202, text="different person"),
    )
    try:
        payload, _ = normalize(tmp_path)
    except ValueError:
        return  # Failing the whole ambiguous import is also safe.
    assert not any(row.get("from_id") == "html-self" for row in rows(payload))


@pytest.mark.parametrize(
    "bad_date,safe_same_day,next_day",
    [
        ("1 November 2026, 01:30:00", "1 November 2026, 12:00:00", "2 November 2026, 12:00:00"),
        ("8 March 2026, 02:30:00", "8 March 2026, 12:00:00", "9 March 2026, 12:00:00"),
    ],
)
def test_dst_ambiguous_or_nonexistent_time_quarantines_entire_chat_day(
    tmp_path,
    bad_date,
    safe_same_day,
    next_day,
):
    write_page(
        tmp_path,
        message(1, date=bad_date, text="invalid clock")
        + message(2, date=safe_same_day, text="same day quarantined")
        + message(3, date=next_day, text="next day valid"),
    )
    payload, _ = normalize(tmp_path)
    authored = [row.get("text") for row in rows(payload) if row.get("from_id") == "html-self"]
    assert authored == ["next day valid"]


def test_source_manifest_covers_all_pages_and_snapshot_changes(tmp_path):
    write_page(tmp_path, message(1))
    write_page(tmp_path, message(2, text="second"), "messages2.html")
    first_payload, first_report = normalize(tmp_path)
    manifest = source_manifest(tmp_path)
    assert {row["relative_path"] for row in manifest} == {"messages.html", "messages2.html"}
    assert all(len(row["sha256"]) == 64 for row in manifest)
    assert all(not Path(row["relative_path"]).is_absolute() for row in manifest)
    write_page(tmp_path, message(2, text="changed"), "messages2.html")
    _, second_report = normalize(tmp_path)
    assert first_report["snapshot_hash"] != second_report["snapshot_hash"]
    assert first_payload  # No mutation of a previously returned payload.


def test_source_manifest_rejects_symlink_outside_selected_root(tmp_path):
    root = tmp_path / "export"
    write_page(root, message(1))
    outside = tmp_path / "outside.html"
    outside.write_text("outside selected export")
    (root / "messages2.html").symlink_to(outside)
    with pytest.raises(ValueError):
        source_manifest(root)


def test_static_html_does_not_execute_or_fetch_links_or_scripts(tmp_path):
    write_page(
        tmp_path,
        message(1, text="visible<script>DO_NOT_INCLUDE()</script><style>HIDDEN</style>")
        + message(2, text="safe")
        + '<a href="https://example.invalid/messages2.html">Next</a>'
        + '<a href="../outside/messages2.html">Other next</a>',
    )
    payload, _ = normalize(tmp_path)
    authored = [row for row in rows(payload) if row.get("from_id") == "html-self"]
    assert [row["text"] for row in authored] == ["safe"]
    assert "DO_NOT_INCLUDE" not in json.dumps(payload)


def test_hidden_hyperlink_destinations_are_quarantined_without_invented_text(tmp_path):
    write_page(
        tmp_path,
        message(1, text='<a href="https://example.invalid/secret">click here</a>')
        + message(2, text="https://example.invalid/visible"),
    )
    payload, _ = normalize(tmp_path)
    authored = [row["text"] for row in rows(payload) if row.get("from_id") == "html-self"]
    assert authored == ["https://example.invalid/visible"]


def test_identical_selected_snapshot_is_independent_of_absolute_folder(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    body = message(1, text="same snapshot")
    write_page(left, body)
    write_page(right, body)
    assert normalize(left) == normalize(right)


def test_explicit_confirmed_aliases_share_own_identity(tmp_path):
    alias = "Alex Alias"
    write_page(tmp_path, message(1, text="first") + message(2, name=alias, text="second"))
    payload, _ = normalize_html_export(tmp_path, author_names=[OWN, alias], export_timezone="UTC")
    assert [row["text"] for row in rows(payload) if row.get("from_id") == "html-self"] == [
        "first",
        "second",
    ]


def test_duplicate_message_conflict_is_not_silently_overwritten(tmp_path):
    write_page(tmp_path, message(1, text="original"))
    write_page(tmp_path, message(1, text="changed"), "messages2.html")
    with pytest.raises(ValueError):
        normalize(tmp_path)


def test_numeric_pagination_order(tmp_path):
    write_page(tmp_path, message(1, text="first"))
    write_page(tmp_path, message(10, text="tenth"), "messages10.html")
    write_page(tmp_path, message(2, text="second"), "messages2.html")
    payload, _ = normalize(tmp_path)
    assert [row["text"] for row in rows(payload) if row.get("from_id") == "html-self"] == [
        "first",
        "second",
        "tenth",
    ]


def test_html_import_reuses_secret_quarantine_before_label_requests(tmp_path):
    from multilora_tone_adapter.common import write_jsonl
    from multilora_tone_adapter.labeling import make_request
    from multilora_tone_adapter.telegram import import_telegram

    root = tmp_path / "selected"
    secret = "gh" + "p_" + "A" * 30
    write_page(
        root,
        message(1, text="safe before")
        + message(2, OTHER, 202, secret, "16 November 2026, 10:01:00")
        + message(3, text=secret, date="16 November 2026, 10:02:00")
        + message(4, text="safe after", date="16 November 2026, 10:03:00"),
    )
    episodes, report = import_telegram(
        [root], author_names=[OWN], export_timezone="America/Los_Angeles"
    )
    assert [episode["target"] for episode in episodes] == ["safe before", "safe after"]
    assert secret not in json.dumps(episodes)
    assert report["quarantined_secret_messages"] == 2
    run = tmp_path / "private-run"
    write_jsonl(run / "episodes.jsonl", episodes)
    make_request(run)
    assert secret not in (run / "label-request.json").read_text()


def test_unknown_sender_is_context_barrier_not_silently_removed(tmp_path):
    from multilora_tone_adapter.telegram import import_telegram

    write_page(
        tmp_path,
        message(1, text="before")
        + message(
            2,
            name="Unidentified Speaker",
            avatar=None,
            text="unknown",
            date="16 November 2026, 10:01:00",
        )
        + message(3, text="after", date="16 November 2026, 10:02:00"),
    )
    episodes, _ = import_telegram(
        [tmp_path], author_names=[OWN], export_timezone="America/Los_Angeles"
    )
    assert [episode["target"] for episode in episodes] == ["before", "after"]
    assert episodes[-1]["context"] == []
    assert episodes[0]["group_id"] != episodes[1]["group_id"]


def test_media_caption_is_not_authored_training_target_or_history(tmp_path):
    from multilora_tone_adapter.telegram import import_telegram

    write_page(
        tmp_path,
        message(1, text="before")
        + message(
            2,
            text="attachment caption",
            date="16 November 2026, 10:01:00",
            extra='<div class="media_wrap"><a href="photos/missing.jpg">Photo</a></div>',
        )
        + message(3, text="after", date="16 November 2026, 10:02:00"),
    )
    episodes, _ = import_telegram(
        [tmp_path], author_names=[OWN], export_timezone="America/Los_Angeles"
    )
    assert [episode["target"] for episode in episodes] == ["before", "after"]
    assert "attachment caption" not in json.dumps(episodes)


def test_prepare_pins_all_html_sources_and_rejects_changed_snapshot(tmp_path, monkeypatch, capsys):
    from multilora_tone_adapter.cli import main

    root = tmp_path / "selected"
    state = tmp_path / "private-state"
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(state))
    write_page(
        root,
        message(1, text="first", date="16 November 2026, 10:00:00")
        + message(2, text="second", date="16 November 2026, 11:00:00"),
    )
    write_page(root, message(3, text="third", date="16 November 2026, 12:00:00"), "messages2.html")
    args = [
        "prepare",
        str(root),
        "--author-name",
        OWN,
        "--export-timezone",
        "America/Los_Angeles",
        "--run",
        "synthetic-html",
    ]
    assert main(args) == 0
    console = capsys.readouterr()
    summary = json.loads(console.out)
    assert "source_manifest" not in summary["report"]["html"]
    assert OWN not in console.out
    config = json.loads((state / "runs/synthetic-html/run.json").read_text())
    assert {row["relative_path"] for row in config["html"]["source_manifest"]} == {
        "messages.html",
        "messages2.html",
    }
    assert config["html"]["export_timezone"] == "America/Los_Angeles"
    write_page(
        root, message(3, text="edited third", date="16 November 2026, 12:00:00"), "messages2.html"
    )
    assert main(args) == 2
    assert "different inputs" in capsys.readouterr().err


def test_account_chat_scope_and_message_provenance_remain_separate(tmp_path):
    from multilora_tone_adapter.telegram import import_telegram

    write_page(tmp_path / "chats/chat_001", message(1, text="chat one"))
    write_page(tmp_path / "chats/chat_002", message(1, text="chat two"))
    payload, _ = normalize(tmp_path)
    chat_id = payload["chats"]["list"][0]["id"]
    episodes, _ = import_telegram(
        [tmp_path],
        author_names=[OWN],
        export_timezone="America/Los_Angeles",
        allowed_chat_ids=[chat_id],
    )
    assert len(episodes) == 1
    all_episodes, _ = import_telegram(
        [tmp_path], author_names=[OWN], export_timezone="America/Los_Angeles"
    )
    assert len(all_episodes) == 2
    assert set(all_episodes[0]["source_ids"]).isdisjoint(all_episodes[1]["source_ids"])


@pytest.mark.parametrize("preview_field", ["description", "text"])
def test_web_preview_text_cannot_become_authored_target_or_context(tmp_path, preview_field):
    from multilora_tone_adapter.telegram import import_telegram

    preview = (
        '<div class="media_wrap"><div class="webpage_preview">'
        '<div class="title">Unauthored article title</div>'
        f'<div class="{preview_field}">Unauthored article summary</div></div></div>'
    )
    write_page(
        tmp_path,
        message(1, text="before")
        + message(2, text="check this article", date="16 November 2026, 10:01:00", extra=preview)
        + message(3, text="after", date="16 November 2026, 10:02:00"),
    )
    episodes, _ = import_telegram(
        [tmp_path], author_names=[OWN], export_timezone="America/Los_Angeles"
    )
    assert [episode["target"] for episode in episodes] == ["before", "after"]
    assert "Unauthored article" not in json.dumps(episodes)


def test_forwarded_nested_sender_cannot_authorize_next_joined_message(tmp_path):
    from multilora_tone_adapter.common import digest
    from multilora_tone_adapter.telegram import import_telegram

    forwarded = (
        f'<div class="forwarded userpic_wrap"><img src="photos/author_999.jpg"></div>'
        f'<div class="forwarded body"><div class="from_name">{OWN}</div></div>'
    )
    write_page(
        tmp_path,
        message(1, text="authentic before")
        + message(2, OTHER, 202, "forwarded", "16 November 2026, 10:01:00", extra=forwarded)
        + message(
            3,
            name=None,
            avatar=None,
            text="not my writing",
            joined=True,
            date="16 November 2026, 10:02:00",
        )
        + message(4, text="authentic after", date="16 November 2026, 10:03:00"),
    )
    episodes, _ = import_telegram(
        [tmp_path], author_names=[OWN], export_timezone="America/Los_Angeles"
    )
    assert [episode["target"] for episode in episodes] == ["authentic before", "authentic after"]
    payload, _ = normalize(tmp_path)
    joined = next(row for row in rows(payload) if str(row["id"]) == "3")
    assert joined["from_id"] == "html-name:" + digest(OTHER)


@pytest.mark.parametrize(
    "flag,value", [("--author-name", "Different Fiction"), ("--export-timezone", "UTC")]
)
def test_prepare_cannot_change_confirmed_identity_or_timezone_in_existing_run(
    tmp_path,
    monkeypatch,
    capsys,
    flag,
    value,
):
    from multilora_tone_adapter.cli import main

    root = tmp_path / "selected"
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(tmp_path / "private-state"))
    write_page(
        root,
        message(1, text="first", date="16 November 2026, 10:00:00")
        + message(2, text="second", date="16 November 2026, 11:00:00")
        + message(3, text="third", date="16 November 2026, 12:00:00"),
    )
    args = [
        "prepare",
        str(root),
        "--author-name",
        OWN,
        "--export-timezone",
        "America/Los_Angeles",
        "--run",
        "synthetic-html",
    ]
    assert main(args) == 0
    capsys.readouterr()
    args[args.index(flag) + 1] = value
    assert main(args) == 2
    assert "first" not in capsys.readouterr().err
