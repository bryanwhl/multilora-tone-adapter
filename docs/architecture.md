# Architecture

The local model is a style rewriter. Codex decides what the message should say and supplies only the completed draft; the adapter changes expression without changing facts or intent. Conversation history is available during curation and to Codex when composing a reply, but is never part of the local model's training, evaluation, or inference input.

## Training path

1. Import selected Telegram Desktop JSON exports using a stable exported author ID, or an HTML export using confirmed author names and export timezone.
2. Apply selected exported chat IDs and the inclusive/exclusive date range; exclude secret-pattern and media-bearing messages before labeling. Reconstruct authored response episodes, including adjacent message bursts and only preceding context. Select recent episodes round-robin across selected chats up to the configured limit.
3. Split source groups without overlapping message windows or duplicates across train, validation, and test.
4. Give the labeler both preceding history and the authentic reply. Generate a neutral draft with the same meaning; keep the authentic reply as the unchanged target.
5. Independently review each draft/target pair without history or the generator's judgment. Bind reviews to source, draft, target, and review-contract hashes. Quarantine failures, uncertainty, stale reviews and explicit typo/correction chains rather than fabricating targets.
6. Export reviewed draft-only inputs, unchanged authentic targets, and a versioned manifest into private state. Add a bounded, deduplicated identity subset only from training sources; it does not increase independent-episode counts. Keep history outside model messages.
7. Pin the base model, validate complete token sequences, and train locally with assistant-only loss that excludes all padding. Save intermediate checkpoints and measure the full validation split.
8. Compare checkpoint candidates on a fixed development suite against diverse paired prompt-only examples. Select by meaning before style; freeze the selection and run a disjoint final evaluation. Activate only a qualified candidate.

The collection target is 2,000 unique usable episodes. A 500-episode run is exploratory, not a quality guarantee. Raw message count is not the dataset size: messages may be grouped, filtered, or unusable without context.

## Train and serve contract

`common.messages_for` constructs the same messages for training, evaluation, and serving:

```json
[
  {"role": "system", "content": "Rewrite the supplied draft ..."},
  {"role": "user", "content": "{\"draft\": \"A neutral draft\"}"},
  {"role": "assistant", "content": "The authentic authored target"}
]
```

Inference omits the final assistant target. Training masks the prompt and supervises only the final target. The original reply and preceding history are available to the labeler during pair creation. Neither history nor a reference reply is passed to the adapter during inference. The local model receives the draft through the same serialization in training and daily use.

The evaluation-only prompted baseline adds deterministically selected, diverse draft/target pairs drawn from training data and a meaning-first style rubric to its system instructions. Those examples are a comparator, not conversation history or a production input field. Base and adapter variants use the canonical system instruction and draft-only user payload; the production API accepts neither context nor style-reference examples. Evaluation categories are reporting metadata and never enter the model's draft.

Dataset manifests use `schema_version: 3` and bind independent reviews, identity policy, row provenance and file hashes. The manifest, training record, and evaluation binding carry `input_contract` with `name: "draft-only-v1"` and a `sha256` covering the system instruction and sole user field `draft`. A separate training-objective contract binds the corrected loss. Training requires a complete reviewed dataset; qualification requires the current objective and evaluation protocol. Older records remain historical evidence, not proof of new qualification. Loading a context-trained adapter remains incompatible even for candidate serving.

For migration, preserve the old run and copy retained source episodes and labels into a fresh private run. Independently review the retained pairs before rebuilding. Keep authentic targets and source grouping unchanged; regenerate only invalid neutral drafts. Retrain with the corrected objective and qualify through the development/final protocol. Never relabel an old manifest as compatible.

## Runtime path

```text
User goal
  → Codex determines intended answer
  → POST /v1/rewrite/candidate with completed draft only
  → validate profile and request; reject context and unknown fields
  → serialize common.messages_for
  → local Qwen 8B + Telegram adapter
  → candidate + hash-bound review request, not final output
  → structured Codex review of candidate against completed draft
  → passing review and protected checks: styled reply
  → failed/uncertain/unavailable review: original draft with fallback status
  → user reviews and sends manually
```

`rewrite` stores a private review request. The invoking Codex agent fills its schema and calls `rewrite-import`; `rewrite --auto-review` explicitly opts into a fresh Codex process using existing configuration. A review must cover content, actors, negation scope, uncertainty, conditions, language, affect, and the qualified language scope. Review hashes prevent reusing an approval for a different draft, candidate or adapter version. Model-assisted review is not a correctness guarantee.

The final `/v1/rewrite` endpoint supports an injected semantic reviewer for embedded use. Without one, it returns the original draft with `fallback_original` and `review_required`, without loading the model. The candidate endpoint is explicitly separate and exposes no final `output` field. A failed review cannot silently trigger the same deterministic request again.

Version one uses a single active generation and one Telegram adapter. The base stays loaded during daily use when memory permits. Training and serving happen separately. Future app profiles must preserve adapter and conversation-cache isolation; they do not justify loading another full base model per app.

## Resource behavior

Preflight checks run on the actual Apple Silicon Mac. Total RAM, current load, available disk, and reserved app headroom determine whether the 8B workflow is a candidate. A newer chip with less memory may support less training headroom than an older chip with more memory.

Before LoRA startup, a dedicated training worker sets MLX's advisory memory limit to the agreed budget and limits the free-buffer cache to the smaller of 256 MiB or one sixteenth of that budget. These settings are not a hard OS allocation cap. The parent independently monitors process RSS, macOS memory pressure, and application headroom, stopping the run when measured limits are exceeded; brief allocation spikes can occur between checks. See the [MLX allocator API](https://github.com/ml-explore/mlx/blob/main/mlx/memory.h).

Training must not silently truncate away required facts or the supervised target. A model that cannot fit within the memory policy is a checkpointed blocker. Do not silently downgrade the model or move training to a hosted service.

The worker supplies the corrected loss explicitly to upstream training, validation and test evaluation; it does not modify installed dependency files. For original token positions, the supervised mask is `offset <= position < sequence_length`. Padding is excluded even for short examples. Checkpoint forks verify recorded weight and dataset hashes, write a fresh private candidate, retain the experiment-family link, and inherit no qualification.

Each evaluation variant runs in a separate process so the base, prompted baseline, and adapter cannot retain one another's GPU allocations. The next worker waits up to 30 seconds for macOS to report enough headroom before loading. Cache cleanup inside a persistent service does not guarantee immediate physical memory release; restart the service after a memory guard stops generation if headroom does not recover.

## Responsibility boundaries

| Component | Owns |
| --- | --- |
| Codex skill | Initial review, batch labeling, report interpretation, orchestration, final meaning review |
| Telegram importer | Author identity, text entities, chronology, source references, response episodes |
| Dataset builder | Draft-only inputs, authentic targets, grouped splits, provenance, hashes, atomic outputs |
| MLX runtime | Compatible input contract, pinned model, token validation, local QLoRA and inference |
| Local API | Input validation, explicit profile routing, bounded concurrency, safe error responses |
| Private state | Exports' derivatives, datasets, labels, candidate adapters, evaluation records |

The default labeler is the invoking Codex agent. The workflow does not treat Codex authentication as an API credential for arbitrary subprocesses. The current optional override is a Codex model selected with `label --model`; arbitrary hosted-provider API connectors are outside this implementation. Do not silently change models or account/run limits.
