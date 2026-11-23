"""Deterministic, read-only normalization of official Telegram JSON exports."""

# Export schema errors are invalid data values, consistently surfaced as ValueError.
# ruff: noqa: TRY004
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from .common import digest

# Conservative high-confidence credential screens. No message contents enter reports.
_SECRET = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}"
    r"|\bgithub_pat_[A-Za-z0-9_]{20,}"
    r"|\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"
    r"|\bxox[baprs]-[A-Za-z0-9-]{15,}"
    r"|\bAKIA[A-Z0-9]{16}\b"
    r"|(?i:api[_ -]?key|access[_ -]?token|password|client[_ -]?secret)"
    r"\s*[:=]\s*[\"']?[A-Za-z0-9_./+~=-]{12,}"
)
_MEDIA_FIELDS = {
    "photo",
    "file",
    "sticker",
    "sticker_emoji",
    "video_file",
    "audio_file",
    "voice_message",
    "video_message",
    "animation",
    "media_type",
    "mime_type",
    "contact_information",
    "location_information",
    "poll",
    "game_information",
    "invoice_information",
}


def _date_bound(value: str | None) -> datetime | None:
    """Accept UTC calendar dates or timezone-qualified ISO instants; never local time."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Date filter must be an ISO date or timezone-qualified timestamp")
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Date filter timestamps must include a timezone")
    return stamp.astimezone(UTC)


def flatten_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(flatten_text(part) for part in value)
    if isinstance(value, dict):
        return flatten_text(value.get("text", ""))
    return ""


def _author(value: object) -> str:
    text = str(value)
    if re.fullmatch(r"(?:user)?[0-9]+", text):
        return "user" + text.removeprefix("user")
    raise ValueError("Author identity must be an explicit Telegram numeric user ID")


def _timestamp(message: dict) -> datetime:
    raw = message.get("date_unixtime")
    if raw is not None:
        if isinstance(raw, bool) or not re.fullmatch(r"[0-9]+", str(raw)):
            raise ValueError("Invalid Telegram Unix timestamp")
        return datetime.fromtimestamp(int(raw), tz=UTC)
    raw = message.get("date")
    if not isinstance(raw, str):
        raise ValueError("Missing unambiguous Telegram timestamp")
    stamp = datetime.fromisoformat(raw)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Timezone-free export dates require date_unixtime")
    return stamp.astimezone(UTC)


def import_telegram(
    paths: list[Path],
    author_id: str,
    max_episodes: int = 2000,
    *,
    burst_gap_seconds: int = 120,
    session_gap_seconds: int = 1800,
    max_context_messages: int = 8,
    allowed_chat_ids: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
) -> tuple[list[dict], dict]:
    """Normalize individual/full-account exports; never infer identity from display names.

    Repeated exports merge by exact app/chat/message identity. Conflicting versions fail
    closed so edited history cannot silently change a frozen training run. Date filtering
    uses [since, until); a calendar date means UTC midnight. Media/secret-bearing
    messages are quarantined before episodes or context can reach a remote labeler.
    """
    own = _author(author_id)
    lower, upper = _date_bound(since), _date_bound(until)
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("Date range requires since before exclusive until")
    allowed = None if allowed_chat_ids is None else {str(value) for value in allowed_chat_ids}
    if max_episodes < 1 or burst_gap_seconds < 0 or session_gap_seconds <= 0:
        raise ValueError("Episode and timing limits must be positive")
    if max_context_messages < 0 or burst_gap_seconds > session_gap_seconds:
        raise ValueError("Invalid context or session limits")
    counts: Counter = Counter()
    chats: dict[str, dict[str, dict]] = {}
    for path in sorted({Path(p).resolve() for p in paths}):
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("Expected an official Telegram JSON object")
        if isinstance(payload.get("chats"), dict):
            exports = payload["chats"].get("list", [])
        elif isinstance(payload.get("messages"), list):
            exports = [payload]
        else:
            raise ValueError("Expected an official Telegram chat or account JSON export")
        if not isinstance(exports, list):
            raise ValueError("Malformed Telegram account chat list")
        for chat in exports:
            if not isinstance(chat, dict) or not isinstance(chat.get("messages"), list):
                raise ValueError("Malformed Telegram chat")
            chat_id = chat.get("id")
            if not isinstance(chat_id, (int, str)) or isinstance(chat_id, bool):
                raise ValueError("Telegram chat is missing its stable ID")
            if allowed is not None and str(chat_id) not in allowed:
                counts["excluded_chats"] += 1
                continue
            bucket = chats.setdefault(str(chat_id), {})
            for message in chat["messages"]:
                counts["input_messages"] += 1
                if not isinstance(message, dict):
                    raise ValueError("Malformed Telegram message")
                mid = message.get("id")
                if not isinstance(mid, (int, str)) or isinstance(mid, bool):
                    raise ValueError("Telegram message is missing its stable ID")
                key = f"telegram:{chat_id}:{mid}"
                if key in bucket:
                    if bucket[key] != message:
                        raise ValueError("Conflicting exported versions of the same message")
                    counts["duplicate_export_messages"] += 1
                else:
                    bucket[key] = message
    episodes: list[dict] = []
    own_seen = 0
    for chat_id, bucket in sorted(chats.items()):
        normalized = []
        for source, message in bucket.items():
            try:
                stamp = _timestamp(message)
            except (ValueError, OverflowError, OSError) as exc:
                raise ValueError("Export contains an invalid or ambiguous timestamp") from exc
            sender = message.get("from_id")
            is_own = sender == own or str(sender) == own.removeprefix("user")
            own_seen += int(is_own)
            if (lower is not None and stamp < lower) or (upper is not None and stamp >= upper):
                counts["outside_date_messages"] += 1
                continue
            text = flatten_text(message.get("text", ""))
            secret = bool(_SECRET.search(text))
            media = any(bool(message.get(field)) for field in _MEDIA_FIELDS)
            if secret:
                counts["quarantined_secret_messages"] += 1
            if media:
                counts["media_messages"] += 1
            eligible = (
                not secret
                and not media
                and message.get("type") == "message"
                and bool(text.strip())
                and not message.get("forwarded_from")
                and not message.get("saved_from")
            )
            if not eligible:
                counts["excluded_messages"] += 1
            normalized.append(
                {
                    "source": source,
                    "stamp": stamp,
                    "text": text,
                    "own": is_own,
                    "eligible": eligible,
                    "sender": str(sender),
                    "raw_id": str(message["id"]),
                }
            )
        normalized.sort(
            key=lambda m: (
                m["stamp"],
                int(m["raw_id"]) if m["raw_id"].isdigit() else 0,
                m["source"],
            )
        )
        session: list[dict] = []
        session_start = ""
        index = 0
        while index < len(normalized):
            current = normalized[index]
            if (
                not session
                or (current["stamp"] - session[-1]["stamp"]).total_seconds() > session_gap_seconds
            ):
                session = []
                session_start = current["source"]
            if not current["eligible"]:
                # Excluded service/forwarded/media messages break contiguous bursts.
                session.append(current)
                index += 1
                continue
            if not current["own"]:
                session.append(current)
                index += 1
                continue
            burst = [current]
            end = index + 1
            while end < len(normalized):
                candidate = normalized[end]
                if (
                    not candidate["eligible"]
                    or not candidate["own"]
                    or (candidate["stamp"] - burst[-1]["stamp"]).total_seconds() > burst_gap_seconds
                ):
                    break
                burst.append(candidate)
                end += 1
            prior = [m for m in session if m["eligible"]]
            prior = prior[-max_context_messages:] if max_context_messages else []
            context = [
                {
                    "speaker": "self" if m["own"] else "other-" + digest(m["sender"])[:12],
                    "text": m["text"],
                }
                for m in prior
            ]
            target_sources = [m["source"] for m in burst]
            episodes.append(
                {
                    "id": digest(target_sources),
                    "chat_id": digest(["telegram", chat_id]),
                    "context": context,
                    "target": "\n".join(m["text"] for m in burst),
                    "source_ids": [m["source"] for m in prior] + target_sources,
                    "target_source_ids": target_sources,
                    "timestamp": current["stamp"].isoformat(),
                    "group_id": digest([chat_id, session_start]),
                }
            )
            session.extend(burst)
            index = end
    if not own_seen:
        raise ValueError("No messages matched the explicit author ID")
    episodes.sort(key=lambda e: (e["timestamp"], e["id"]))
    total = len(episodes)
    # Round-robin recent episodes across chats so a busy chat cannot monopolize
    # the bounded collection. Final rows remain chronologically ordered.
    by_chat: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        by_chat[episode["chat_id"]].append(episode)
    selected = []
    while len(selected) < min(max_episodes, total):
        for chat_id in sorted(by_chat):
            if by_chat[chat_id] and len(selected) < max_episodes:
                selected.append(by_chat[chat_id].pop())
    episodes = sorted(selected, key=lambda e: (e["timestamp"], e["id"]))
    return episodes, {
        "schema_version": 1,
        **dict(counts),
        "chats": len(chats),
        "selected_chats": len({episode["chat_id"] for episode in episodes}),
        "selection_policy": "recent_round_robin_by_chat",
        "eligible_episodes": total,
        "selected_episodes": len(episodes),
        "truncated_episodes": total - len(episodes),
    }
