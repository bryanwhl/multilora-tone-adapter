"""Frozen, provenance-aware training datasets with leakage-safe grouped splits."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .common import (
    atomic_write,
    digest,
    input_contract,
    messages_for,
    private_dir,
    read_jsonl,
    write_json,
)
from .labeling import has_correction_chain, validate_pair
from .pair_review import accepted_review, review_contract, reviewed_pairs

CURATED_FILES = {"train.jsonl", "valid.jsonl", "test.jsonl", "provenance.jsonl", "pair-reviews.jsonl"}


def require_curated_dataset(manifest: dict) -> None:
    """Reject legacy or inconsistent curation claims before training or qualification."""
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 3
            or manifest.get("review_contract") != review_contract()):
        raise ValueError("Dataset requires independent pair review; rebuild in a new private run")
    base = manifest.get("base_counts")
    counts = manifest.get("counts")
    policy = manifest.get("identity_policy")
    accepted, identities = manifest.get("accepted_pairs"), manifest.get("identity_pairs")
    if (not isinstance(base, dict) or set(base) != {"train", "valid", "test"}
            or any(type(n) is not int or n < 1 for n in base.values())
            or not isinstance(counts, dict) or set(counts) != set(base)
            or any(type(n) is not int for n in counts.values())
            or type(accepted) is not int or accepted != sum(base.values())
            or type(identities) is not int or identities < 0
            or not isinstance(policy, dict)):
        raise ValueError("Invalid curated dataset counts or identity policy")
    fraction = policy.get("fraction")
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 <= fraction <= 0.25
            or policy.get("train_only") is not True
            or policy.get("excluded_from_accepted_pairs") is not True
            or type(policy.get("limit")) is not int
            or policy["limit"] != math.floor(base["train"] * fraction)
            or identities > policy["limit"]
            or counts != {**base, "train": base["train"] + identities}
            or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("accepted_reviews_hash", "")))):
        raise ValueError("Invalid curated identity limits or independent review binding")
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != CURATED_FILES:
        raise ValueError("Curated dataset requires the complete frozen file inventory")
    splits = manifest.get("splits")
    if (not isinstance(splits, dict) or not splits
            or any(not isinstance(key, str) or not key or split not in ("train", "valid", "test")
                   for key, split in splits.items())
            or manifest.get("split_hash") != digest(splits)):
        raise ValueError("Curated dataset frozen split assignment changed")


def verify_curation_files(directory: Path, manifest: dict) -> None:
    """Verify actual model rows, reviews and identity counts, not just manifest claims."""
    require_curated_dataset(manifest)
    directory = Path(directory)
    if manifest.get("dataset_hash") != digest({k: v for k, v in manifest.items()
                                              if k != "dataset_hash"}):
        raise ValueError("Curated dataset manifest hash changed")
    try:
        for name, expected in manifest["files"].items():
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Curated dataset file is missing or is a symlink")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Curated dataset file changed")
        ready = directory / "READY"
        if ready.is_symlink() or ready.read_text() != manifest["dataset_hash"] + "\n":
            raise ValueError("Curated dataset READY marker changed")
        reviews = read_jsonl(directory / "pair-reviews.jsonl")
        provenance = read_jsonl(directory / "provenance.jsonl")
        data = {split: read_jsonl(directory / f"{split}.jsonl")
                for split in ("train", "valid", "test")}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Curated dataset files are incomplete or invalid") from exc
    reviewed_pairs(reviews)
    if (any(row["verdict"] != "pass" for row in reviews)
            or len(reviews) != manifest["accepted_pairs"]
            or len({row["id"] for row in reviews}) != len(reviews)
            or digest(sorted(reviews, key=lambda r: r["id"])) != manifest["accepted_reviews_hash"]
            or digest(provenance) != manifest.get("provenance_hash")
            or {split: len(rows) for split, rows in data.items()} != manifest["counts"]):
        raise ValueError("Independent reviews or provenance do not match dataset counts")
    by_review = {row["id"]: row for row in reviews}
    seen_rows, rewrite_ids, identity_targets = set(), set(), set()
    base_counts: Counter = Counter()
    held_out_targets = set()
    group_splits: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    episode_rows: dict[str, tuple[tuple, set[str]]] = {}
    for item in provenance:
        if not isinstance(item, dict):
            raise ValueError("Invalid curation provenance")
        split, row_index = item.get("split"), item.get("row")
        if (not isinstance(split, str) or split not in data
                or type(row_index) is not int or row_index < 0
                or row_index >= len(data[split]) or (split, row_index) in seen_rows):
            raise ValueError("Invalid or duplicate curation row provenance")
        episode_id, group_id, source_ids = item.get("id"), item.get("group_id"), item.get("source_ids")
        if (not isinstance(episode_id, str)
                or manifest["splits"].get(episode_id) != split):
            raise ValueError("Curation row disagrees with its frozen source split")
        if (not isinstance(group_id, str) or not group_id
                or not isinstance(source_ids, list) or not source_ids
                or any(not isinstance(source, str) or not source for source in source_ids)
                or len(set(source_ids)) != len(source_ids)
                or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("source_hash", "")))):
            raise ValueError("Curation row requires valid group and source provenance")
        if group_splits.setdefault(group_id, split) != split:
            raise ValueError("A source group crosses curated dataset splits")
        if any(source_splits.setdefault(source, split) != split for source in source_ids):
            raise ValueError("A source message crosses curated dataset splits")
        seen_rows.add((split, row_index))
        row = data[split][row_index]
        try:
            messages = row["messages"]
            draft = json.loads(messages[1]["content"])["draft"]
            target = messages[2]["content"]
            if row != {"messages": messages_for(draft, target=target)}:
                raise ValueError("Curated model row does not match the input contract")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid curated model row") from exc
        if validate_pair(draft, target):
            raise ValueError("Curated model row no longer passes pair screens")
        kind = item.get("kind")
        if kind not in ("rewrite", "identity"):
            raise ValueError("Unknown curation provenance kind")
        source_record = (item["source_hash"], group_id, tuple(source_ids), target)
        previous = episode_rows.get(episode_id)
        if previous is not None:
            previous_record, kinds = previous
            if (split != "train" or previous_record != source_record
                    or kind in kinds or kinds | {kind} != {"rewrite", "identity"}):
                raise ValueError("Duplicate source rows may only be one train rewrite and its identity")
            kinds.add(kind)
        else:
            episode_rows[episode_id] = (source_record, {kind})
        if kind == "rewrite":
            review = by_review.get(item.get("id"))
            if (draft == target or item.get("id") in rewrite_ids or review is None
                    or review["draft_hash"] != digest(draft)
                    or review["target_hash"] != digest(target)
                    or review["source_hash"] != item.get("source_hash")):
                raise ValueError("Model row lacks a matching accepted independent review")
            rewrite_ids.add(item["id"])
            base_counts[split] += 1
            if split != "train":
                held_out_targets.add(target)
        elif kind == "identity":
            if split != "train" or draft != target or target in identity_targets:
                raise ValueError("Identity examples must be exact, unique and train-only")
            identity_targets.add(target)
        else:
            raise ValueError("Unknown curation provenance kind")
    if (len(seen_rows) != sum(manifest["counts"].values())
            or dict(base_counts) != manifest["base_counts"]
            or len(identity_targets) != manifest["identity_pairs"]
            or identity_targets & held_out_targets):
        raise ValueError("Actual curation rows do not satisfy the declared identity policy")


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
    *,
    reviews: list[dict] | None = None,
    identity_fraction: float = 0.1,
) -> dict:
    """Write MLX chat JSONL and private provenance; never modify a frozen output.

    min_episodes is a collection floor, not a claim of training sufficiency. All split
    membership derives from the complete curated episode set before label rejection.
    """
    if min_episodes < 1:
        raise ValueError("min_episodes must be positive")
    if (isinstance(identity_fraction, bool) or not isinstance(identity_fraction, (int, float))
            or not math.isfinite(identity_fraction) or not 0 <= identity_fraction <= 0.25):
        raise ValueError("identity_fraction must be between zero and 0.25")
    review_history = reviews or []
    approved = reviewed_pairs(review_history)
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
    accepted_reviews = []
    identity_candidates: list[dict] = []

    def append_row(episode: dict, draft: str, split: str, kind: str) -> None:
        rows[split].append({"messages": messages_for(draft, target=episode["target"])})
        provenance.append({"id": episode["id"], "split": split, "kind": kind,
                           "group_id": episode["group_id"], "source_ids": episode["source_ids"],
                           "source_hash": digest(episode),
                           "row": len(rows[split]) - 1})

    for key, episode in sorted(by_id.items()):
        if has_correction_chain(episode["target"]):
            rejected["correction_chain"] += 1
            continue
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
        elif label.get("target_hash", digest(target)) != digest(target):
            rejected["target_hash_mismatch"] += 1
        elif validate_pair(draft, target):
            rejected["pair_screen_failed"] += 1
        elif draft == target:
            # Exact identities require no semantic inference, but never count as
            # independently reviewed rewrite episodes or enter held-out splits.
            if splits[key] == "train":
                identity_candidates.append(episode)
            else:
                rejected["identity_nontrain_excluded"] += 1
        elif (review := accepted_review(episode, label, approved)) is None:
            rejected["independent_review_not_accepted"] += 1
        else:
            split = splits[key]
            append_row(episode, draft, split, "rewrite")
            accepted_reviews.append(review)
            if split == "train":
                identity_candidates.append(episode)
    accepted = len(provenance)
    if accepted < min_episodes:
        raise ValueError(f"Only {accepted} independently reviewed curated pairs meet checks; "
                         f"need {min_episodes}. Review unreviewed or changed pairs before build.")
    if any(not value for value in rows.values()):
        raise ValueError("Label rejection left an empty split; collect independent examples")
    base_counts = {name: len(value) for name, value in rows.items()}
    identity_limit = math.floor(base_counts["train"] * identity_fraction)
    held_out_targets = {e["target"] for e in episodes if splits[e["id"]] != "train"}
    unique_identity: dict[str, dict] = {}
    for episode in sorted(identity_candidates, key=lambda e: e["id"]):
        if episode["target"] not in held_out_targets:
            unique_identity.setdefault(episode["target"], episode)
    selected_identity = sorted(unique_identity.values(),
                               key=lambda e: (digest(e["target"]), e["id"]))[:identity_limit]
    for episode in selected_identity:
        append_row(episode, episode["target"], "train", "identity")
    expected = {
        f"{name}.jsonl": "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in value
        )
        for name, value in rows.items()
    }
    expected["provenance.jsonl"] = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in provenance
    )
    expected["pair-reviews.jsonl"] = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in sorted(accepted_reviews, key=lambda r: r["id"])
    )
    manifest = {
        "schema_version": 3,
        "input_contract": input_contract(),
        "profile": "telegram",
        "episodes": len(episodes),
        "accepted_pairs": accepted,
        "base_counts": base_counts,
        "identity_pairs": len(selected_identity),
        "identity_policy": {"fraction": identity_fraction, "limit": identity_limit,
                            "train_only": True, "excluded_from_accepted_pairs": True},
        "review_contract": review_contract(),
        "accepted_reviews_hash": digest(sorted(accepted_reviews, key=lambda r: r["id"])),
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
