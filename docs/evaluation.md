# Evaluation and qualification

A successful training process does not establish that an adapter writes well. Qualification evaluates the behavior used in daily work: Codex supplies a draft and the local model rewrites it without changing meaning.

## Three distinct test levels

| Level | Establishes | Does not establish |
| --- | --- | --- |
| Synthetic automated suite | Parsing, grouping, split invariants, request validation, failure behavior | Real model compatibility or personal style quality |
| Local model smoke test | Real model load, a training/inference path, local API response | Sufficient training volume, style gains, or faithful everyday behavior |
| Private-data qualification | Held-out comparisons and meaning checks under the configured policy | A guarantee that every future draft is correct |

Report these separately. Do not present a one-step training smoke test as completed personal fine-tuning.

## Inputs and baselines

Use fresh, deployment-like neutral drafts as the primary evaluation inputs. The draft generator must not receive the authentic reference wording. Neutral drafts generated from held-out authentic replies can form a separate reconstruction test; they are easier and cannot alone demonstrate deployment performance.

Compare the same pinned base under three conditions:

1. Base only.
2. Base with a Telegram style prompt and examples drawn only from training data.
3. Base with the Telegram adapter.

Select training settings with validation data. Reserve the locked test set for a final comparison; repeated tuning against it turns it into validation data.

## Meaning comes first

Check names, numbers, dates, URLs, negation, uncertainty, and commitments. A refusal cannot become agreement, “might” cannot become “will,” and tomorrow cannot become today. Surface checks supplement semantic review; they do not replace it.

Evaluate style using the initial rubric and reference samples from training data. Include the relevant languages, short replies, message bursts, and code-switching patterns. Count ties and judge uncertainty rather than forcing a winner. Disclose when the same model generates and judges examples.

The starting policy proposes at least 50 fresh cases, an adapter preference rate of 60% over the strongest prompt-only baseline, a meaning-pass rate of 95%, and no changed critical facts in fixed probes. These thresholds are experimental release criteria, not evidence of perfect fidelity. A failed candidate remains inactive.

## Runtime reporting

Record the base revision, dataset hashes, adapter configuration, software versions, labeling provenance, evaluation inputs, judgments, peak memory, and duration. Keep private case text and outputs outside Git. Numeric summary reports can be published only after inspection for sensitive values.

Timing is observed rather than optimized in the first version. Memory checks must still protect normal application use. Report real blockers with the last complete checkpoint; do not silently change the model, invent additional data, or claim qualification without the required evidence.

## CLI evaluation contract

Store evaluation inputs outside Git. `evaluate --run RUN --cases PATH` accepts a JSON array such as:

```json
[
  {
    "draft": "I cannot join on Monday, but Tuesday might work.",
    "context": [],
    "independently_authored": true
  }
]
```

Use at least 50 distinct, genuinely independent cases for qualification. A one-case array is useful for a smoke test only. Do not assert independence for a draft reconstructed from the reference reply. The current flag is a caller assertion; the software cannot prove how a draft was authored.

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

Import checks comparison identity, complete case IDs, meaning judgments, protected-content screens, the minimum 500 accepted training pairs, and the release thresholds. Activation binds the report to the training record, dataset, and adapter hashes. Changing those artifacts requires fresh qualification. A smoke dataset with `build --min-episodes` below 500 cannot qualify for normal activation.

The current three-way preference rule requires the adapter to win 60% of all cases, counting ties separately; it is not a statistical significance test or a direct pairwise preference estimate. Automated labeling and judgment remain model-assisted evidence and require disclosure when the same model performs both.
