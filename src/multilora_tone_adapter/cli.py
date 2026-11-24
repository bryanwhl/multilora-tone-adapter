"""CLI: all run artifacts go to private state, never the source checkout."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .common import (
    DEFAULT_MODEL,
    digest,
    private_dir,
    read_jsonl,
    state_root,
    write_json,
    write_jsonl,
)


def run_dir(name: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
        raise ValueError("Run name must be 1–64 letters, digits, underscores or hyphens")
    path = (state_root() / "runs" / name).resolve()
    # An override must not accidentally place messages in a Git worktree.
    if any((parent / ".git").exists() for parent in [path, *path.parents]):
        raise ValueError("Private state must be outside every Git checkout")
    return private_dir(path)


def output(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def import_options(args) -> dict:
    return {"max_episodes": args.limit, "allowed_chat_ids": args.chat_id,
            "since": args.since, "until": args.until,
            "author_names": args.author_name, "export_timezone": args.export_timezone}


def summary_report(report: dict) -> dict:
    """Keep file paths and sender evidence in private artifacts, not terminal output."""
    result = dict(report)
    if "html" in result:
        result["html"] = {key: value for key, value in result["html"].items()
                          if key not in {"source_manifest", "chat_manifest", "sender_evidence"}}
    return result


def prepare(args) -> dict:
    from .dataset import assign_splits
    from .telegram import import_telegram
    directory = run_dir(args.run)
    paths = [Path(p).expanduser().resolve() for p in args.exports]
    episodes, report = import_telegram(paths, args.author_id, **import_options(args))
    if not episodes:
        raise ValueError("No eligible authored response episodes found; verify export and author ID")
    source_hashes = ([report["html"]["snapshot_hash"]] if "html" in report
                     else [digest(p.read_text()) for p in paths])
    config = {"source_hashes": source_hashes,
              "author_hash": digest(args.author_name if args.author_name else args.author_id),
              "limit": args.limit,
              "allowed_chat_ids": args.chat_id, "since": args.since, "until": args.until,
              "episodes_hash": digest(episodes), "profile": "telegram", "task": "rewrite"}
    if "html" in report:
        config["html"] = {"export_timezone": args.export_timezone,
                          "identity_method": "explicit_name_mapping",
                          "parser_version": report["html"].get("parser_version", 1),
                          "source_manifest": report["html"]["source_manifest"]}
    config_path = directory / "run.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run already exists with different inputs; choose a new run name")
    write_json(config_path, config)
    write_jsonl(directory / "episodes.jsonl", episodes)
    write_json(directory / "import-report.json", report)
    write_json(directory / "splits.json", assign_splits(episodes))
    return {"run": args.run, "episodes": len(episodes), "report": summary_report(report),
            "private_directory": str(directory)}


def build(args) -> dict:
    from .dataset import build_dataset
    directory = run_dir(args.run)
    return build_dataset(read_jsonl(directory / "episodes.jsonl"),
                         read_jsonl(directory / "labels.jsonl"), directory / "dataset",
                         min_episodes=args.min_episodes)


def resolve_model(directory: Path) -> Path:
    from .runtime import download_model
    record = directory / "model.json"
    if record.exists():
        return Path(json.loads(record.read_text())["path"])
    path = download_model()
    write_json(record, {"repo_id": DEFAULT_MODEL, "path": str(path)})
    return path


def train_run(args) -> dict:
    from .runtime import train
    directory = run_dir(args.run)
    adapters = directory / "adapter"
    if (adapters / "training.json").exists():
        raise ValueError("Training record exists; use a new run to preserve provenance")
    return train(directory / "dataset", resolve_model(directory), adapters,
                 iters=args.iters, max_seq_length=args.max_seq_length,
                 num_layers=args.num_layers, memory_gb=args.memory_gb)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="multilora", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("inspect", "prepare"):
        cmd = sub.add_parser(name, help="Inspect or normalize Telegram JSON exports or an HTML directory")
        cmd.add_argument("exports", nargs="+")
        identity = cmd.add_mutually_exclusive_group(required=True)
        identity.add_argument("--author-id", help="Stable Telegram sender ID for JSON, e.g. user12345")
        identity.add_argument("--author-name", action="append",
                              help="Confirmed exact sender name for HTML; repeat for aliases")
        cmd.add_argument("--export-timezone", help="Confirmed IANA timezone for HTML wall times")
        cmd.add_argument("--limit", type=int, default=2000)
        cmd.add_argument("--chat-id", action="append", help="Restrict to this exported chat ID; repeatable")
        cmd.add_argument("--since", help="Inclusive UTC date or timezone-aware ISO timestamp")
        cmd.add_argument("--until", help="Exclusive UTC date or timezone-aware ISO timestamp")
        if name == "prepare":
            cmd.add_argument("--run", required=True)
    cmd = sub.add_parser("preflight", help="Measure this machine's resource headroom")
    cmd.add_argument("--reserve-gb", type=float, default=4)
    cmd.add_argument("--memory-gb", type=float)
    cmd = sub.add_parser("download-model", help="Download pinned Qwen 8B model to local cache")
    cmd.add_argument("--revision")
    for name in ("status", "labels-request", "labels-import", "label", "build", "train",
                 "evaluate", "evaluate-import", "activate", "serve"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--run", required=True)
        if name == "labels-request":
            cmd.add_argument("--limit", type=int, default=25)
        elif name == "labels-import":
            cmd.add_argument("--input", type=Path, required=True)
        elif name == "label":
            cmd.add_argument("--batch-size", type=int, default=25)
            cmd.add_argument("--max-batches", type=int, default=100)
            cmd.add_argument("--model", help="Optional explicit Codex model override")
        elif name == "build":
            cmd.add_argument("--min-episodes", type=int, default=500,
                             help="Lower only for explicit smoke tests, never production qualification")
        elif name == "train":
            cmd.add_argument("--iters", type=int, default=100)
            cmd.add_argument("--max-seq-length", type=int, default=512)
            cmd.add_argument("--num-layers", type=int, default=4)
            cmd.add_argument("--memory-gb", type=float)
        elif name == "evaluate":
            cmd.add_argument("--cases", type=Path, required=True,
                             help="JSON array of independently prepared {draft,context} objects")
            cmd.add_argument("--memory-gb", type=float)
        elif name == "evaluate-import":
            cmd.add_argument("--input", type=Path, required=True)
        elif name == "serve":
            cmd.add_argument("--port", type=int, default=8765)
            cmd.add_argument("--memory-gb", type=float)
            cmd.add_argument("--candidate", action="store_true", help="Explicit unqualified smoke only")
    cmd = sub.add_parser("rewrite", help="Request style rewrite from local service")
    cmd.add_argument("--draft-file", type=Path, required=True)
    cmd.add_argument("--port", type=int, default=8765)
    cmd = sub.add_parser("audit-public", help="Audit tracked and staged files for private artifacts")
    cmd.add_argument("--repo", type=Path, default=Path.cwd())
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        cmd = args.command
        if cmd == "inspect":
            from .telegram import import_telegram
            episodes, report = import_telegram([Path(p).expanduser() for p in args.exports],
                                                args.author_id, **import_options(args))
            output({"episodes": len(episodes), "report": summary_report(report)})
        elif cmd == "prepare":
            output(prepare(args))
        elif cmd == "preflight":
            from .runtime import preflight
            output(preflight(memory_gb=args.memory_gb, reserve_gb=args.reserve_gb))
        elif cmd == "download-model":
            from .runtime import download_model
            output({"path": str(download_model(revision=args.revision))})
        elif cmd == "audit-public":
            from .publication import audit
            result = audit(args.repo)
            output(result)
            return 1 if result["findings"] else 0
        elif cmd == "rewrite":
            import httpx
            response = httpx.post(f"http://127.0.0.1:{args.port}/v1/rewrite", timeout=300,
                                  json={"profile": "telegram", "task": "rewrite",
                                        "draft": args.draft_file.read_text()})
            response.raise_for_status()
            output(response.json())
        else:
            directory = run_dir(args.run)
            if cmd == "status":
                output({"run": args.run, "private_directory": str(directory),
                        "episodes": len(read_jsonl(directory / "episodes.jsonl")) if (directory / "episodes.jsonl").exists() else 0,
                        "labels": len(read_jsonl(directory / "labels.jsonl")) if (directory / "labels.jsonl").exists() else 0,
                        "dataset": (directory / "dataset/manifest.json").exists(),
                        "training": (directory / "adapter/training.json").exists(),
                        "evaluation": (directory / "evaluation.json").exists(),
                        "active": (directory / "active.json").exists()})
            elif cmd == "labels-request":
                from .labeling import make_request
                output(make_request(directory, args.limit))
            elif cmd == "labels-import":
                from .labeling import import_labels
                output(import_labels(directory, args.input))
            elif cmd == "label":
                from .labeling import codex_label
                output(codex_label(directory, args.batch_size, args.model, args.max_batches))
            elif cmd == "build":
                output(build(args))
            elif cmd == "train":
                output(train_run(args))
            elif cmd in ("evaluate", "evaluate-import", "activate"):
                from .evaluation import activate, generate_comparison, import_evaluation
                if cmd == "evaluate":
                    output(generate_comparison(directory, args.cases, args.memory_gb))
                elif cmd == "evaluate-import":
                    output(import_evaluation(directory, args.input))
                else:
                    output(activate(directory))
            elif cmd == "serve":
                import uvicorn

                from .api import create_app
                from .evaluation import verify_active
                from .runtime import MLXEngine
                if not args.candidate:
                    verify_active(directory)
                engine = MLXEngine(resolve_model(directory), directory / "adapter",
                                   memory_gb=args.memory_gb)
                uvicorn.run(create_app(engine, adapter_version=args.run), host="127.0.0.1",
                            port=args.port, access_log=False)
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        # Errors contain operational metadata, never a repr of imported source records.
        print(f"multilora: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
