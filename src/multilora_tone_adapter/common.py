"""Shared private storage and train/serve serialization contracts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

SYSTEM_PROMPT = (
    "Rewrite the supplied draft in the author's Telegram style. Preserve facts, intent, "
    "negation, uncertainty, commitments, names, numbers, URLs, and language. "
    "Do not add facts or translate. Return only the rewritten message."
)
DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"


def state_root() -> Path:
    """Keep private data outside checkouts, including when invoked inside a repo."""
    return Path(os.environ.get("MULTILORA_STATE_DIR", "~/.local/share/multilora-tone-adapter")).expanduser().resolve()


def private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def write_json(path: Path, value: object) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write(path: Path, text: str) -> None:
    private_dir(path.parent)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def messages_for(draft: str, context: list[dict] | str | None = None, target: str | None = None) -> list[dict]:
    """One identical serialization for training, evaluation and HTTP inference."""
    payload = {"draft": draft, "context": context or []}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages
