"""Private HTML exports cannot bypass audits by changing their filename."""

import subprocess

import pytest

from multilora_tone_adapter.publication import _inspect, audit


@pytest.mark.parametrize(
    "html",
    [
        '<div class="message default clearfix" id="message123">private text</div>',
        "<div id='message123' class='message default clearfix joined'>private text</div>",
        '<DIV id="message-1" class = "message service">private text</DIV>',
    ],
)
def test_renamed_telegram_html_detected_by_content(html):
    findings = _inspect("innocent.html", html.encode(), "index")
    assert any(item["rule"] == "telegram_html_export" for item in findings)
    assert all("private text" not in str(item) for item in findings)


@pytest.mark.parametrize(
    "name",
    [
        "messages.html",
        "messages2.html",
        "export_results.html",
        "chats.html",
        "personal_information.html",
        "contacts.html",
    ],
)
def test_export_filenames_are_private_even_if_content_is_unrecognized(name):
    assert any(
        item["rule"] == "private_artifact" for item in _inspect(name, b"unrecognized", "index")
    )


def test_generic_html_and_inline_synthetic_tests_remain_publishable():
    assert (
        _inspect("docs/example.html", b'<div class="message">Example interface</div>', "index")
        == []
    )
    assert (
        _inspect(
            "tests/test_fixture.py",
            b'<div class="message default" id="message1">fictional</div>',
            "index",
        )
        == []
    )


def test_html_audit_inspects_staged_export_after_clean_worktree_replacement(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "renamed.txt"
    path.write_text('<div class="message default clearfix" id="message1">synthetic secret</div>')
    subprocess.run(["git", "add", "renamed.txt"], cwd=tmp_path, check=True)
    path.write_text("clean replacement")
    findings = audit(tmp_path)["findings"]
    assert any(
        item["rule"] == "telegram_html_export" and item["source"] == "index" for item in findings
    )
