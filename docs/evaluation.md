# Evaluation and qualification

A successful training process does not establish that an adapter writes well. Qualification evaluates the behavior used in daily work: Codex supplies only a completed draft and the local model rewrites it without changing meaning. Conversation history is excluded from every model variant's case input.

## Three distinct test levels

| Level | Establishes | Does not establish |
| --- | --- | --- |
| Synthetic automated suite | Parsing, grouping, split invariants, request validation, failure behavior | Real model compatibility or personal style quality |
| Local model smoke test | Real model load, a training/inference path, local API response | Sufficient training volume, style gains, or faithful everyday behavior |
| Private-data qualification | Held-out comparisons and meaning checks under the configured policy | A guarantee that every future draft is correct |

Report these separately. Do not present a one-step training smoke test as completed personal fine-tuning.

## Inputs and baselines

Use fresh, deployment-like neutral drafts as the primary evaluation inputs. Each draft contains the intended reply, not conversation history or a request to decide an answer. The draft generator must not receive the authentic reference wording. Neutral drafts generated from held-out authentic replies can form a separate reconstruction test; they are easier and cannot alone demonstrate deployment performance.

Compare the same pinned base under three conditions:

1. Base only.
2. Base with a Telegram style prompt and examples drawn only from training data.
3. Base with the Telegram adapter.

Every variant receives the same draft-only user payload. The prompted baseline alone adds fixed train-only style examples to its system instructions as an evaluation comparator; it receives no conversation history. These examples are not a production API feature or an input to the adapter variant.

Candidate letters are shuffled independently for each case using OS-backed randomness. The mode mapping stays in the private comparison artifact and is bound to the comparison hash. Give the judge only the blind evaluation request, never `comparison-private.json` or the worker outputs that identify each mode.

Select training settings with validation data. Reserve the locked test set for a final comparison; repeated tuning against it turns it into validation data.

## Meaning comes first

Check names, numbers, dates, URLs, negation, uncertainty, commitments, and who performs or receives each action. A refusal cannot become agreement, “might” cannot become “will,” and tomorrow cannot become today. Surface checks supplement semantic review; they do not replace it.

Evaluate style using the initial rubric and reference samples from training data. Include the relevant languages, short replies, message bursts, and code-switching patterns. Count ties and judge uncertainty rather than forcing a winner. Disclose when the same model generates and judges examples.

The starting policy proposes at least 50 fresh cases, an adapter preference rate of 60% over the strongest prompt-only baseline, a meaning-pass rate of 95%, and no changed critical facts in fixed probes. These thresholds are experimental release criteria, not evidence of perfect fidelity. A failed candidate remains inactive.

## Runtime reporting

Record the input contract, base revision, dataset hashes, adapter configuration, software versions, labeling provenance, evaluation inputs, judgments, peak memory, and duration in private state. Keep all user-run reports outside Git, including numeric summaries, run identifiers, hashes, timings, hardware observations, and output examples. Removing message bodies does not make a run report suitable for this source repository. Public documentation should contain reusable instructions and wholly synthetic examples, not records of individual training runs.

Timing is observed rather than optimized in the first version. Memory checks must still protect normal application use. Report real blockers with the last complete checkpoint; do not silently change the model, invent additional data, or claim qualification without the required evidence.

## CLI evaluation contract

Store evaluation inputs outside Git. `evaluate --run RUN --cases PATH` accepts a JSON array such as:

```json
[
  {
    "draft": "I cannot join on Monday, but Tuesday might work.",
    "independently_authored": true
  }
]
```

Use at least 50 distinct, genuinely independent cases for qualification. A one-case array is useful for a smoke test only. Do not assert independence for a draft reconstructed from the reference reply. The current flag is a caller assertion; the software cannot prove how a draft was authored.

Cases accept only `draft` and `independently_authored`. A `context` field is rejected even when empty. Earlier context-bearing cases may be retained as historical or development evidence; generate fresh draft-only cases for the new adapter's final qualification.

The command writes three-mode outputs, a private candidate mapping, and `evaluation-request.json`. Give Codex the returned request file, not the hidden mapping. Codex returns the exact `comparison_hash` and one judgment for every request case ID:

```json
{
  "comparison_hash": "copy the exact value from evaluation-request.json",
  "judgments": [
    {
      "id": "0",
      "winner": "tie",
      "meaning": {"A": true, "B": true, "C": true},
      "reason": "All candidates preserve the draft; no clear style preference."
    }
  ]
}
```

`winner` must be `A`, `B`, `C`, or `tie`. Judge each candidate's meaning separately before judging style. Use `false` for any changed fact, commitment, uncertainty, or language. Store this response privately, then run:

```bash
uv run multilora evaluate-import --run RUN --input /private/path/judgments.json
uv run multilora activate --run RUN
```

Import checks comparison identity, complete case IDs, meaning judgments, protected-content screens, the minimum 500 accepted training pairs, and the release thresholds. Activation binds the report to the input contract, training record, dataset, and adapter hashes. Changing those artifacts requires fresh qualification. A context-trained adapter or its evaluation report cannot be reused as a draft-only candidate. A smoke dataset with `build --min-episodes` below 500 cannot qualify for normal activation.

The current three-way preference rule requires the adapter to win 60% of all cases, counting ties separately; it is not a statistical significance test or a direct pairwise preference estimate. Automated labeling and judgment remain model-assisted evidence and require disclosure when the same model performs both.
