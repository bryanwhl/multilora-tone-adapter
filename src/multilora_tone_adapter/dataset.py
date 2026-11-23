"""Frozen, provenance-aware training datasets with leakage-safe grouped splits."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .common import atomic_write, digest, messages_for, private_dir, write_json
from .labeling import validate_pair


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _template(text: str) -> str:
    return re.sub(r"\b\d+(?:[.,:]\d+)*\b", "NUMBER", _normalized(text))


def assign_splits(episodes: list[dict]) -> dict[str, str]:
    """Freeze splits BEFORE labeling using sessions, shared sources and duplicate targets.

    Generic short acknowledgements remain separate. Longer normalized or near-template
    targets are connected transitively to prevent common cross-split leakage.
    """
    by_id = {e["id"]: e for e in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("Duplicate episode IDs")
    parent = {key: key for key in by_id}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        a, b = sorted((find(left), find(right)))
        parent[b] = a

    seen: dict[tuple, str] = {}
    shingle_index: dict[str, list[str]] = defaultdict(list)
    shingles: dict[str, set[str]] = {}
    for key, episode in sorted(by_id.items()):
        if not isinstance(episode.get("target"), str) or not episode["target"].strip():
            raise ValueError("Episode target must be nonempty text")
        identity = episode.get("group_id")
        if not identity or not episode.get("source_ids"):
            raise ValueError("Episode requires a session group and source provenance")
        tags = [("session", identity)] + [("source", s) for s in episode["source_ids"]]
        text = _template(episode["target"])
        if len(text) >= 32:
            tags.append(("template", text))
        for tag in tags:
            if tag in seen:
                union(key, seen[tag])
            else:
                seen[tag] = key
        # Character shingles handle spacing, emoji and multilingual templates.
        if len(text) >= 48:
            current = {text[i : i + 8] for i in range(len(text) - 7)}
            candidates: Counter = Counter()
            for part in current:
                candidates.update(shingle_index[part])
            for other, overlap in candidates.items():
                previous = shingles[other]
                if overlap / len(current | previous) >= 0.85:
                    union(key, other)
            shingles[key] = current
            for part in current:
                shingle_index[part].append(key)
    components: dict[str, list[str]] = defaultdict(list)
    for key in sorted(by_id):
        components[find(key)].append(key)
    groups = sorted(
        components.values(),
        key=lambda g: (min(by_id[key].get("timestamp", "") for key in g), digest(g)),
    )
    if len(groups) < 3:
        raise ValueError("At least three independent session/duplicate groups are required")
    # Chronological cumulative boundaries approximate 80/10/10 without breaking groups.
    train_end = max(1, min(len(groups) - 2, _boundary(groups, len(episodes) * 0.8)))
    valid_end = max(train_end + 1, min(len(groups) - 1, _boundary(groups, len(episodes) * 0.9)))
    result = {}
    for index, group in enumerate(groups):
        split = "train" if index < train_end else "valid" if index < valid_end else "test"
        result.update({key: split for key in group})
    return result


def _boundary(groups: list[list[str]], desired: float) -> int:
    total = 0
    best = (float("inf"), 1)
    for index, group in enumerate(groups, start=1):
        total += len(group)
        best = min(best, (abs(total - desired), index))
    return best[1]


def build_dataset(
    episodes: list[dict],
    labels: list[dict],
    output_dir: Path,
    min_episodes: int = 500,
) -> dict:
    """Write MLX chat JSONL and private provenance; never modify a frozen output.

    min_episodes is a collection floor, not a claim of training sufficiency. All split
    membership derives from the complete curated episode set before label rejection.
    """
    if min_episodes < 1:
        raise ValueError("min_episodes must be positive")
    splits = assign_splits(episodes)
    by_id = {e["id"]: e for e in episodes}
    by_label: dict[str, dict] = {}
    for label in labels:
        if label.get("id") not in by_id:
            raise ValueError("Label refers to an unknown episode ID")
        if label["id"] in by_label:
            raise ValueError("Duplicate label ID")
        by_label[label["id"]] = label
    rejected: Counter = Counter()
    rows: dict[str, list[dict]] = {name: [] for name in ("train", "valid", "test")}
    provenance = []
    for key, episode in sorted(by_id.items()):
        label = by_label.get(key)
        if label is None:
            rejected["missing_label"] += 1
            continue
        draft = label.get("draft")
        target = episode["target"]
        if label.get("meaning_preserved") is not True:
            rejected["meaning_not_confirmed"] += 1
        elif not isinstance(draft, str) or not draft.strip():
            rejected["empty_draft"] += 1
        elif _normalized(draft) == _normalized(target):
            rejected["copied_target"] += 1
        elif label.get("target_hash", digest(target)) != digest(target):
            rejected["target_hash_mismatch"] += 1
        elif validate_pair(draft, target):
            rejected["pair_screen_failed"] += 1
        else:
            split = splits[key]
            rows[split].append({"messages": messages_for(draft, episode.get("context"), target)})
            provenance.append(
                {
                    "id": key,
                    "split": split,
                    "group_id": episode["group_id"],
                    "source_ids": episode["source_ids"],
                    "row": len(rows[split]) - 1,
                }
            )
    accepted = len(provenance)
    if accepted < min_episodes:
        raise ValueError(f"Only {accepted} curated pairs meet checks; need {min_episodes}")
    if any(not value for value in rows.values()):
        raise ValueError("Label rejection left an empty split; collect independent examples")
    expected = {
        f"{name}.jsonl": "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in value
        )
        for name, value in rows.items()
    }
    expected["provenance.jsonl"] = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in provenance
    )
    manifest = {
        "schema_version": 1,
        "profile": "telegram",
        "episodes": len(episodes),
        "accepted_pairs": accepted,
        "rejected": dict(sorted(rejected.items())),
        "counts": {name: len(value) for name, value in rows.items()},
        "episode_hash": digest(sorted(episodes, key=lambda e: e["id"])),
        "label_hash": digest(sorted(labels, key=lambda e: e["id"])),
        "split_hash": digest(splits),
        "splits": splits,
        "files": {
            name: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for name, value in expected.items()
        },
        "provenance_hash": digest(provenance),
        "min_episodes": min_episodes,
    }
    manifest["dataset_hash"] = digest(manifest)
    output_dir = Path(output_dir).expanduser().resolve()
    # Refuse private content inside this checkout or any git checkout ancestor.
    for ancestor in (output_dir, *output_dir.parents):
        if (ancestor / ".git").exists():
            raise ValueError("Private dataset output must be outside a source checkout")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != manifest:
            raise ValueError("Frozen dataset exists; choose a new output directory")
        for name, value in expected.items():
            if not (output_dir / name).exists() or (output_dir / name).read_text() != value:
                raise ValueError("Frozen dataset file is missing or changed")
        ready = output_dir / "READY"
        if ready.exists() and ready.read_text(encoding="utf-8") != manifest["dataset_hash"] + "\n":
            raise ValueError("Frozen dataset READY marker changed")
        if not ready.exists():
            atomic_write(ready, manifest["dataset_hash"] + "\n")
        return existing
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.name not in expected or path.is_symlink() or not path.is_file():
                raise ValueError("Output contains unrecognized files without a frozen manifest")
            if path.read_text(encoding="utf-8") != expected[path.name]:
                raise ValueError("Partial dataset changed; choose a new output directory")
    private_dir(output_dir)
    for name, value in expected.items():
        atomic_write(output_dir / name, value)
    write_json(output_dir / "manifest.json", manifest)
    atomic_write(output_dir / "READY", manifest["dataset_hash"] + "\n")
    return manifest
