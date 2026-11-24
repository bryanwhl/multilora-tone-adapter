# multilora-tone-adapter

Turn private Telegram exports into a local writing-style adapter. Codex prepares the content of a reply; a quantized Qwen 8B model with your Telegram LoRA adapter rewrites the draft in your voice.

```text
Telegram JSON or HTML export → original reply + preceding history
                              ↓ Codex labeler
                    neutral draft → original reply
                              ↓ local QLoRA
                      Telegram style adapter

Your request → Codex draft → local model + adapter → meaning check → returned draft
```

The first version supports one app and one task: **Telegram draft rewriting**. It does not retrieve live messages, send replies, or infer your intended answer from conversation history alone. Separate app adapters can be added later against the same pinned base.

## Requirements

- Python 3.11 or later and [uv](https://docs.astral.sh/uv/).
- Apple Silicon macOS for local MLX training and inference. Data preparation and non-MLX tests also run on Linux.
- A Telegram Desktop export: machine-readable JSON or an existing HTML export directory. JSON carries stronger identity metadata; HTML requires confirmed exported author names and the export’s timezone. Telegram API credentials are unnecessary.
- Codex for the default labeling workflow. The invoking agent creates bounded structured batches using your existing setup; this does not extract Codex credentials or provide a general-purpose API key.

See [Telegram export instructions](docs/telegram-export.md) for existing HTML folders, new JSON exports, author identity, and timezone handling.

An 8B model fitting into memory for inference does not establish that QLoRA training will fit alongside your applications. Preflight checks use the actual machine and current memory load; speed optimization is deferred.

## Install

From a checkout:

```bash
uv sync --extra dev --extra mlx
uv run multilora --help
uv run multilora preflight
```

For data preparation or CI without MLX, omit `--extra mlx`. The base model defaults to `mlx-community/Qwen3-8B-4bit`; a download is pinned to an immutable Hugging Face revision before training.

To make the workflow discoverable in Codex, install the versioned skill folder `skills/multilora-tone-adapter` in your user skill directory, or invoke that file explicitly. The skill uses this project's `multilora` command. See [the skill](skills/multilora-tone-adapter/SKILL.md).

## First run

Run the workflow inside Codex so the agent can inspect quality reports and label batches. Supply export paths and explicit author identity: `--author-id` for JSON, or confirmed `--author-name` values plus `--export-timezone` for HTML. The initial review establishes selected chats and dates, identity, exclusions, memory headroom, and representative source/input/target pairs. After that review, routine stages continue automatically; genuine data, access, resource, and quality blockers stop the run.

The CLI provides separate resumable stages. The importer accepts individual chat or full-account JSON files, or one HTML export directory. Do not mix JSON files and an HTML directory in one import. Use repeatable `--chat-id` to select import chat IDs, `--since` for an inclusive lower bound, and `--until` for an exclusive upper bound. Calendar dates mean UTC midnight; ISO timestamps must include a timezone. Selection takes recent episodes round-robin across chats up to `--limit`, preventing one busy chat from consuming the whole sample. Inspect current command help before scripting stages:

```bash
uv run multilora inspect /path/to/telegram/result.json --author-id user123456
uv run multilora prepare /path/to/telegram/result.json --author-id user123456 --run telegram-pilot
uv run multilora labels-request --run telegram-pilot --limit 25
```

For an existing HTML export, confirm the exact sender name(s) and the timezone used when the export was produced. Repeat `--author-name` for known historical aliases; do not infer identity from message frequency:

```bash
uv run multilora inspect /path/to/telegram-html-export \
  --author-name "Your exported sender name" --export-timezone Europe/London
uv run multilora prepare /path/to/telegram-html-export \
  --author-name "Your exported sender name" --export-timezone Europe/London \
  --run telegram-html-pilot
```

The timezone above is illustrative, not a default. Use the confirmed IANA timezone of the export. HTML timestamps are interpreted in that zone and then converted to UTC for date filtering. Ambiguous/nonexistent local timestamps and uncertain sender attribution are quarantined. HTML chat/message identities are export-local, bound to a source snapshot manifest; they are not stable Telegram API IDs.

For a bounded selection, add the same filters to both inspection and preparation:

```bash
uv run multilora prepare /path/to/telegram/result.json --author-id user123456 \
  --chat-id 12345 --chat-id 67890 --since 2026-01-01 --until 2026-04-01 \
  --limit 2000 --run telegram-selected
```

Secret-pattern and media-bearing messages are excluded before labeling. This conservative screen may remove usable examples and cannot detect every secret. The private import report records exclusion counts without message bodies.

Codex reads the private request and schema paths returned by `labels-request`, generates neutral drafts with the same meaning as the authentic replies, and writes the structured response to a private file. Each label includes `id`, `draft`, `meaning_preserved`, and `reason`. Import and repeat until all selected episodes have labels:

```bash
uv run multilora labels-import --run telegram-pilot --input /private/path/labels.json
uv run multilora build --run telegram-pilot
uv run multilora download-model
uv run multilora train --run telegram-pilot
```

For explicit unattended labeling through the installed Codex CLI, `uv run multilora label --run telegram-pilot` uses the user's existing configuration. For example: `uv run multilora label --run telegram-pilot --batch-size 25 --max-batches 100`. Optional `--model MODEL` changes the Codex model; the batch flags bound work. The current CLI supports Codex model overrides, not an arbitrary hosted-provider API. This mode starts separate ephemeral Codex runs and remains subject to account limits. The invoking-agent batch workflow above is the default.

Evaluate on independently prepared drafts against the same base with and without a style prompt. Each case must declare `independently_authored: true`; this is a provenance assertion, not automatic verification. The evaluation command writes a private blind comparison request. Codex judges that request, then imports the judgments before activation:

```bash
uv run multilora evaluate --run telegram-pilot --cases /private/path/cases.json
# Codex reads the returned request_file and writes private judgments.json.
uv run multilora evaluate-import --run telegram-pilot --input /private/path/judgments.json
uv run multilora activate --run telegram-pilot
uv run multilora serve --run telegram-pilot --port 8765
```

See [evaluation inputs and judgment schema](docs/evaluation.md#cli-evaluation-contract). `activate` rejects unqualified or stale artifacts. `serve --candidate` is available only for an explicit unqualified smoke test; it does not activate the adapter.

A candidate smoke test is distinct from qualification. Successful model loading, one training step, or a successful HTTP response does not establish good personal style or faithful meaning preservation. See [evaluation](docs/evaluation.md).

## Daily use

Ask Codex to help craft a Telegram reply. Codex first resolves the intended content and then calls the local service with that draft. The service returns a styled message; Codex checks that the result preserves the original intent before presenting it. You review and send the reply yourself.

Example request using entirely fictional text:

```bash
curl http://127.0.0.1:8765/v1/rewrite \
  -H 'Content-Type: application/json' \
  --data '{"profile":"telegram","draft":"Yes, I can join tomorrow.","context":[],"max_output_tokens":128}'
```

Alternatively, put the draft in a private text file and use `uv run multilora rewrite --draft-file /private/path/draft.txt`.

The API binds to localhost by default. It serves one active generation initially and rejects unsupported profiles. See current `/docs` on the running service for the exact request and response schema.

## Private data stays outside Git

Private run artifacts default to:

```text
~/.local/share/multilora-tone-adapter/
```

This includes normalized messages, generated labels, datasets, reports, and adapters. `MULTILORA_STATE_DIR` can select another private location. Run commands reject private state inside a Git checkout. Do not use a shared folder for private state.

The repository contains source code, synthetic tests, documentation, and a skill. It excludes exports, dataset rows, model weights, adapters, credentials, and private reports. `.gitignore` is a defense, not a publication audit: inspect the staged files and run `uv run multilora audit-public` before every push. The audit scans staged blobs, tracked working-tree files, and reachable Git history. Never use `git add -f` for private artifacts.

Enable the versioned local commit and push checks after cloning:

```bash
git config core.hooksPath .githooks
```

Both hooks run `uv run --frozen multilora audit-public --repo .`. Git does not enable repository hooks automatically on clone. Hooks and scanners are defense in depth; inspect publication candidates and do not bypass a private-data finding.

Labeling through Codex or an explicitly selected hosted model can send the selected text to that model's service. Local storage, training, and style inference do not make the entire agent workflow offline. Local artifacts are retained indefinitely until explicitly removed. See [data handling](docs/data-handling.md).

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

The standard test suite uses synthetic data and fake inference engines. Real MLX model download, fine-tuning, and HTTP inference require a separate local run. See [recorded synthetic validation](docs/validation.md), [architecture](docs/architecture.md), and [evaluation](docs/evaluation.md).

## License

MIT for this project's source code. Base model weights and dependencies retain their own licenses. A dataset or adapter containing personal messages is private by default and is not covered by publishing this source repository.
