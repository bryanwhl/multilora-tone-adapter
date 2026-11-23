# Architecture

The local model is a style rewriter. Codex decides what the message should say and supplies the content; the adapter changes expression without changing facts or intent.

## Training path

1. Import selected Telegram Desktop JSON exports and identify the author by stable exported ID.
2. Apply selected exported chat IDs and the inclusive/exclusive date range; exclude secret-pattern and media-bearing messages before labeling. Reconstruct authored response episodes, including adjacent message bursts and only preceding context. Select recent episodes round-robin across selected chats up to the configured limit.
3. Split source groups without overlapping message windows or duplicates across train, validation, and test.
4. Give the labeler both preceding history and the authentic reply. Generate a neutral draft with the same meaning; keep the authentic reply as the unchanged target.
5. Validate structured labels, provenance, and protected content. Quarantine failed pairs rather than fabricating targets.
6. Export the versioned training dataset and manifest into private state.
7. Pin the base model revision, validate token lengths and prompt masking, and train a QLoRA adapter locally.
8. Compare against prompt-only baselines and activate only a qualified candidate.

The collection target is 2,000 unique usable episodes. A 500-episode run is exploratory, not a quality guarantee. Raw message count is not the dataset size: messages may be grouped, filtered, or unusable without context.

## Train and serve contract

`common.messages_for` constructs the same messages for training, evaluation, and serving:

```json
[
  {"role": "system", "content": "Rewrite the supplied draft ..."},
  {"role": "user", "content": "{\"context\": [], \"draft\": \"A neutral draft\"}"},
  {"role": "assistant", "content": "The authentic authored target"}
]
```

Inference omits the final assistant target. Training masks the prompt and supervises only the final target. The original reply is available to the labeler during pair creation, but never as an inference input to the trainable model.

## Runtime path

```text
User goal
  → Codex determines intended answer
  → POST /v1/rewrite with draft and optional context
  → validate profile and request
  → serialize common.messages_for
  → local Qwen 8B + Telegram adapter
  → protected-content checks
  → Codex checks meaning against the input draft
  → draft returned for manual sending
```

Version one uses a single active generation and one Telegram adapter. The base stays loaded during daily use when memory permits. Training and serving happen separately. Future app profiles must preserve adapter and conversation-cache isolation; they do not justify loading another full base model per app.

## Resource behavior

Preflight checks run on the actual Apple Silicon Mac. Total RAM, current load, available disk, and reserved app headroom determine whether the 8B workflow is a candidate. A newer chip with less memory may support less training headroom than an older chip with more memory.

Training must not silently truncate away required facts or the supervised target. A model that cannot fit within the memory policy is a checkpointed blocker. Do not silently downgrade the model or move training to a hosted service.

Each evaluation variant runs in a separate process so the base, prompted baseline, and adapter cannot retain one another's GPU allocations. The next worker waits up to 30 seconds for macOS to report enough headroom before loading. Cache cleanup inside a persistent service does not guarantee immediate physical memory release; restart the service after a memory guard stops generation if headroom does not recover.

## Responsibility boundaries

| Component | Owns |
| --- | --- |
| Codex skill | Initial review, batch labeling, report interpretation, orchestration, final meaning review |
| Telegram importer | Author identity, text entities, chronology, source references, response episodes |
| Dataset builder | Accepted pairs, grouped splits, provenance, hashes, atomic outputs |
| MLX runtime | Pinned model, token validation, local QLoRA and inference |
| Local API | Input validation, explicit profile routing, bounded concurrency, safe error responses |
| Private state | Exports' derivatives, datasets, labels, candidate adapters, evaluation records |

The default labeler is the invoking Codex agent. The workflow does not treat Codex authentication as an API credential for arbitrary subprocesses. The current optional override is a Codex model selected with `label --model`; arbitrary hosted-provider API connectors are outside this implementation. Do not silently change models or account/run limits.
