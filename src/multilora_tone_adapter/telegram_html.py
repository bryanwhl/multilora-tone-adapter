"""Read official Telegram HTML snapshots without executing their scripts.

HTML lacks stable chat IDs and timestamp offsets. Identity names and the export's
IANA timezone must therefore be confirmed explicitly. Numeric avatar filenames
are corroborating sender evidence, never a guessed account owner identity.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .common import digest

_PAGE = re.compile(r"messages(?P<number>[0-9]*)\.html")
_AVATAR = re.compile(r"photos/author_([0-9]+)\.jpg")
_DATE = re.compile(
    r"(?P<day>[0-9]{1,2}) (?P<month>[A-Za-z]+) (?P<year>[0-9]{4}),? "
    r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
)
_MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        1,
    )
}
_VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


def _pages(root: Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("HTML input must be one Telegram export directory.")
    candidates = list(root.glob("chats/*/messages*.html"))
    if (root / "messages.html").exists():
        if candidates:
            raise ValueError("HTML input mixes account and single-chat export layouts.")
        candidates = list(root.glob("messages*.html"))
    pages = []
    numbers = set()
    for path in candidates:
        match = _PAGE.fullmatch(path.name)
        if not match:
            continue
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("HTML message page escapes the supplied export directory.")
        number = int(match["number"]) if match["number"] else 1
        key = (str(path.parent.relative_to(root)), number)
        if number < 1 or key in numbers:
            raise ValueError("HTML export has ambiguous page numbering.")
        numbers.add(key)
        pages.append((key, path))
    if not pages:
        raise ValueError("No official Telegram messages.html pages were found.")
    return [path for _, path in sorted(pages)]


def source_manifest(root: Path) -> list[dict]:
    """Hash exactly the message pages used, with paths scoped to one snapshot."""
    root = Path(root).expanduser().resolve()
    result = []
    for path in _pages(root):
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        result.append({"relative_path": str(path.relative_to(root)), "sha256": hasher.hexdigest()})
    return result


class _MessageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.records = []
        self.current = None
        self.depth = 0
        self.name_depth = self.text_depth = self.body_depth = None
        self.stack = []
        self.link = None
        self.suppressed = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        if tag == "div" and "message" in classes:
            if self.current is not None:
                raise ValueError("HTML contains nested or unclosed message containers.")
            self.current = {
                "classes": classes,
                "id": attrs.get("id"),
                "name": [],
                "text": [],
                "photos": set(),
                "timestamp": None,
                "media": False,
                "media_wrap": False,
                "web_preview": False,
                "forwarded": False,
                "unsupported": False,
            }
            self.depth = 1
            self.name_depth = self.text_depth = self.body_depth = None
            self.stack = [(tag, classes)]
            return
        if self.current is None:
            return
        if tag == "div":
            self.depth += 1
        if tag not in _VOID:
            self.stack.append((tag, classes))
        if tag in {"script", "style"}:
            self.suppressed += 1
            if self.text_depth is not None:
                self.current["unsupported"] = True
        if tag == "div" and "body" in classes and self.depth == 2:
            self.body_depth = self.depth
        outer_field = tag == "div" and self.body_depth == 2 and self.depth == 3
        if "from_name" in classes and outer_field:
            self.name_depth = self.depth
        if "text" in classes and tag == "div":
            if self.text_depth is not None:
                self.current["unsupported"] = True
            elif outer_field:
                self.text_depth = self.depth
        if "date" in classes and "title" in attrs and outer_field:
            if self.current["timestamp"] is not None:
                raise ValueError("HTML message contains multiple timestamp fields.")
            self.current["timestamp"] = attrs["title"]
        if "forwarded_from" in classes or "forwarded" in classes:
            self.current["forwarded"] = True
        if "media_wrap" in classes:
            self.current["media_wrap"] = True
        if "webpage_preview" in classes:
            self.current["web_preview"] = True
            # Article metadata is unauthored; exclude previews like other media.
            self.current["media"] = True
        if any(value.startswith("media_") and value != "media_wrap" for value in classes):
            self.current["media"] = True
        if tag == "img":
            match = _AVATAR.fullmatch(attrs.get("src", ""))
            header = len(self.stack) > 1 and "userpic_wrap" in self.stack[1][1]
            if match and header:
                self.current["photos"].add(match[1])
            elif self.text_depth is not None:
                # Inline media/custom emoji may have meaning not represented as text.
                self.current["unsupported"] = True
        if tag == "br" and self.text_depth is not None:
            self.current["text"].append("\n")
            if self.link:
                self.link["text"].append("\n")
        if tag == "a" and self.text_depth is not None:
            if self.link is not None:
                self.current["unsupported"] = True
            self.link = {"href": attrs.get("href", ""), "text": []}

    def handle_data(self, data):
        if self.current is None or self.suppressed:
            return
        if self.name_depth is not None:
            self.current["name"].append(data)
        if self.text_depth is not None:
            self.current["text"].append(data)
            if self.link:
                self.link["text"].append(data)

    def handle_endtag(self, tag):
        if self.current is None:
            return
        if tag == "a" and self.link is not None:
            href, visible = self.link["href"], "".join(self.link["text"]).strip()
            expected = href.removeprefix("mailto:").removeprefix("tel:")
            if not href or visible != expected:
                self.current["unsupported"] = True
            self.link = None
        if tag in {"script", "style"} and self.suppressed:
            self.suppressed -= 1
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
        if tag != "div":
            return
        if self.body_depth == self.depth:
            self.body_depth = None
        if self.name_depth == self.depth:
            self.name_depth = None
        if self.text_depth == self.depth:
            self.text_depth = None
        self.depth -= 1
        if self.depth == 0:
            record = self.current
            record["name"] = "".join(record["name"]).strip()
            record["text"] = "".join(record["text"]).strip()
            record["media"] |= record["media_wrap"] and not record["web_preview"]
            self.records.append(record)
            self.current = None
            self.stack = []
            self.link = None
            self.suppressed = 0

    def finish(self):
        self.close()
        if self.current is not None:
            raise ValueError("HTML ends inside a message container.")
        return self.records


def _local_time(title: str | None) -> datetime:
    match = _DATE.fullmatch(title or "")
    if not match or match["month"] not in _MONTHS:
        raise ValueError("HTML message lacks a supported full English timestamp title.")
    try:
        return datetime(
            int(match["year"]),
            _MONTHS[match["month"]],
            int(match["day"]),
            int(match["hour"]),
            int(match["minute"]),
            int(match["second"]),
        )
    except ValueError as exc:
        raise ValueError("HTML contains an invalid calendar timestamp.") from exc


def _utc_time(local: datetime, zone: ZoneInfo) -> datetime | None:
    candidates = set()
    for fold in (0, 1):
        aware = local.replace(tzinfo=zone, fold=fold)
        utc = aware.astimezone(UTC)
        if utc.astimezone(zone).replace(tzinfo=None) == local:
            candidates.add(utc)
    return next(iter(candidates)) if len(candidates) == 1 else None


def normalize_html_export(
    root: Path, *, author_names: list[str], export_timezone: str
) -> tuple[dict, dict]:
    """Convert one immutable HTML snapshot into canonical importer input.

    Service markers and quarantined rows become a barrier on the next ordinary
    row; the shared importer must honor html_barrier_before in sessions/bursts.
    All messages on a chat/day containing ambiguous or nonexistent DST times are
    excluded together, so no UTC time or adjacent conversation is invented.
    """
    if (
        not isinstance(author_names, list)
        or not author_names
        or any(not isinstance(name, str) or not name.strip() for name in author_names)
    ):
        raise ValueError("HTML requires explicitly confirmed, nonempty author names.")
    own_names = {name.strip() for name in author_names}
    if not export_timezone or not isinstance(export_timezone, str):
        raise ValueError("HTML requires the explicitly confirmed export timezone.")
    try:
        zone = ZoneInfo(export_timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("HTML export timezone must be a valid IANA timezone.") from exc
    root = Path(root).expanduser().resolve()
    manifest = source_manifest(root)
    snapshot_hash = digest(manifest)
    page_groups = defaultdict(list)
    for item in manifest:
        path = root / item["relative_path"]
        page_groups[str(path.parent.relative_to(root))].append((path, item["sha256"]))
    counts = Counter()
    chats = []
    global_names = defaultdict(set)
    parsed_chats = []
    for relative_folder, pages in sorted(page_groups.items()):
        records = []
        local_names = defaultdict(set)
        for path, expected_hash in pages:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected_hash:
                raise ValueError("HTML source changed while its snapshot was being normalized.")
            parser = _MessageParser()
            try:
                parser.feed(data.decode("utf-8-sig"))
                page_records = parser.finish()
            except UnicodeDecodeError as exc:
                raise ValueError("HTML page must use UTF-8 text.") from exc
            if page_records:
                page_records[0]["page_start"] = True
            for record in page_records:
                if "default" in record["classes"]:
                    counts["ordinary_messages"] += 1
                    if not re.fullmatch(r"message-?[0-9]+", record["id"] or ""):
                        raise ValueError("HTML ordinary message is missing its numeric message ID.")
                    local = _local_time(record["timestamp"])
                    record["local"] = local
                    record["utc"] = _utc_time(local, zone)
                    if "joined" not in record["classes"] and record["name"]:
                        local_names[record["name"]].update(record["photos"])
                        global_names[record["name"]].update(record["photos"])
                records.append(record)
        counts["ambiguous_sender_names"] += sum(len(ids) > 1 for ids in local_names.values())
        parsed_chats.append((relative_folder, records, local_names))
    if any(len(global_names[name]) > 1 for name in own_names):
        raise ValueError("A confirmed author name maps to multiple numeric avatar identities.")
    if not any(global_names[name] for name in own_names):
        raise ValueError("Confirmed author names have no numeric avatar evidence in this snapshot.")
    for relative_folder, records, local_names in parsed_chats:
        bad_days = {
            record["local"].date()
            for record in records
            if "local" in record and record["utc"] is None
        }
        counts["excluded_dst_days"] += len(bad_days)
        messages = []
        previous_sender = None
        barrier = True
        seen = {}
        for record in records:
            if record.get("page_start"):
                previous_sender = None
                barrier = True
            if "default" not in record["classes"]:
                counts["service_barriers"] += 1
                previous_sender = None
                barrier = True
                continue
            if record["local"].date() in bad_days:
                counts["excluded_dst_messages"] += 1
                previous_sender = None
                barrier = True
                continue
            joined = "joined" in record["classes"]
            if joined:
                sender = previous_sender
                if sender is None:
                    counts["quarantined_orphan_joined"] += 1
                    barrier = True
                    continue
            else:
                name = record["name"]
                ids = local_names[name]
                if not name or len(ids) != 1 or len(record["photos"]) > 1:
                    counts["quarantined_unknown_sender"] += 1
                    previous_sender = None
                    barrier = True
                    continue
                sender = "html-self" if name in own_names else "html-name:" + digest(name)
                previous_sender = sender
            if record["unsupported"]:
                counts["quarantined_unsupported_markup"] += 1
                barrier = True
                continue
            message = {
                "id": record["id"].removeprefix("message"),
                "type": "message",
                "date_unixtime": str(int(record["utc"].timestamp())),
                "text": record["text"],
            }
            message["from_id"] = sender
            if record["media"]:
                message["media_type"] = "html_attachment"
                counts["media_messages"] += 1
            if record["forwarded"]:
                message["forwarded_from"] = "html-forwarded"
                counts["forwarded_messages"] += 1
            if barrier:
                message["html_barrier_before"] = True
            previous = seen.get(message["id"])
            if previous is not None:
                if previous != message:
                    raise ValueError(
                        "HTML snapshot contains conflicting versions of one message ID."
                    )
                counts["duplicate_messages"] += 1
                continue
            seen[message["id"]] = message
            messages.append(message)
            counts["own_messages" if sender == "html-self" else "other_messages"] += 1
            barrier = False
        chats.append({"id": f"html:{snapshot_hash}:{relative_folder}", "messages": messages})
    report = {
        "parser_version": 1,
        "source_manifest": manifest,
        "chat_manifest": [
            {"id": chat["id"], "relative_folder": relative_folder}
            for chat, (relative_folder, _, _) in zip(chats, parsed_chats, strict=True)
        ],
        "snapshot_hash": snapshot_hash,
        "identity_method": "explicit_name_mapping",
        "export_timezone": export_timezone,
        "timestamp_method": "confirmed_export_timezone_with_dst_day_quarantine",
        "counts": dict(counts),
        "chats": len(chats),
        "pages": len(manifest),
        "html_barrier_required": True,
    }
    return {"chats": {"list": chats}}, report
