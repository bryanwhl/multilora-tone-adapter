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
2. Base with a meaning-first rubric and paired neutral-draft/authentic-target examples drawn only from training data.
3. Base with the Telegram adapter.

Every variant receives the same draft-only user payload. The prompted baseline alone adds fixed train-only paired examples to its system instructions; it receives no conversation history. Selection is deterministic and favors diversity in reply length, message bursts, questions, numbers, uncertainty and scripts. Examples are not a production API feature or an input to the adapter variant.

Initial selection retains at most eight complete pairs, 2,400 combined characters, and 1,200 characters per pair. A local tokenizer then checks every actual base and prompted input before a worker loads model weights. If needed, whole pairs are removed until every prompt fits the same 2,048-token limit. Drafts and examples are never truncated. The judge receives the actual retained examples; their hash and token budget are recorded. All variants allow up to 512 generated tokens and reject incomplete, length-stopped outputs.

Candidate letters are shuffled independently for each case using OS-backed randomness. The mode mapping stays in the private comparison artifact and is bound to the comparison hash. Give the judge only the blind evaluation request, never `comparison-private.json` or the worker outputs that identify each mode.

Select training settings with validation data and a fixed development evaluation. Reserve the locked final suite for one selected candidate. Repeated tuning against the final suite turns it into development data.

## Development and locked final evaluation

Evaluation defaults to `--purpose development`. Development reports never qualify an adapter. Compare checkpoint candidates on the same fixed development suite, ranking meaning preservation first, protected-value preservation second and usable style preference third.

```bash
uv run multilora evaluate --run CANDIDATE --cases /private/development.json --purpose development --language-scope english
uv run multilora evaluate-import --run CANDIDATE --input /private/development-judgments.json
uv run multilora checkpoint-rank --runs CANDIDATE_A CANDIDATE_B
uv run multilora checkpoint-select --run CHOSEN_CANDIDATE
uv run multilora evaluate --run CHOSEN_CANDIDATE --cases /private/final.json --purpose qualification --language-scope english
uv run multilora evaluate-import --run CHOSEN_CANDIDATE --input /private/final-judgments.json
uv run multilora activate --run CHOSEN_CANDIDATE
```

Checkpoint forks preserve the source run's `evaluation_family_root`. Its private ledger locks one development suite and records the selected artifact binding. Before final generation, it locks the final suite and selected candidate together. Development and final drafts must be disjoint after Unicode, whitespace and case normalization. This catches superficial duplicates, not every semantic paraphrase; authors must avoid near-duplicates and training-derived drafts too.

After final evaluation begins, the family is sealed against further development or candidate changes. Failed generation may retry only the same candidate and exact final suite. Further experiments need a new family and fresh final cases; never delete the ledger to reuse an exposed final test. Prior final failures can become explicitly declared development regressions in a later family.

Imported reports, comparisons and original judgments are archived privately. Ranking rejects candidates from different families, suites or language scopes and never ranks final reports. Activation checks the family's selected candidate and final-suite lock as well as artifact hashes. Legacy comparisons remain readable historical evidence but cannot newly qualify.

## Coverage and language scope

Cases may include one primary `category`; this metadata never enters model input. Qualification requires at least 50 cases and at least three in each required category:

| Category | Behavior to challenge |
| --- | --- |
| `actor` | Who acts, who receives the action, and speaker/recipient direction |
| `uncertainty` | Might, should, probably and confidence |
| `negation` | Refusal, absence and negation scope |
| `conditional` | Conditions attached to commitments or instructions |
| `protected_values` | Names, numbers, dates, URLs, units and exact facts |
| `long` | Multiple details and obligations; at least 40 whitespace-separated words |
| `no_op` | Drafts already in an appropriate style; unchanged copying may be correct |
| `multilingual` | Language and code-switching preservation; required only in multilingual scope |
| `ordinary` | Other representative drafts; no separate minimum |

Labels and independence flags are caller assertions, not proof. A case has one primary category even when it exercises several behaviors. Long-case word counts use whitespace, so they are not a language-independent measure; substantial English or mixed-language long cases can accompany separate multilingual probes.

`--language-scope english` does not claim multilingual support and does not require Chinese or another specific language. Choose `multilingual` only when training and final cases cover the intended languages and code-switching patterns. The category does not establish support for every language. Unsupported language requests use original-draft fallback; script detection alone cannot identify every non-English Latin-script message, so semantic language review is still required.

Candidate selection requires complete development coverage. Uncategorized historical cases remain useful for diagnosis but cannot satisfy selection or qualification coverage. Every required category must independently achieve at least 95% meaning preservation and 100% protected-value checks; with three cases, all three must pass. These are conservative experimental criteria, not statistical guarantees.

## Meaning comes first

Check names, numbers, dates, URLs, negation, uncertainty, commitments, and who performs or receives each action. A refusal cannot become agreement, “might” cannot become “will,” and tomorrow cannot become today. Surface checks supplement semantic review; they do not replace it.

Evaluate style using the initial rubric and reference samples from training data. Include the relevant languages, short replies, message bursts, and code-switching patterns. Count ties and judge uncertainty rather than forcing a winner. Disclose when the same model generates and judges examples.

Reports distinguish raw style wins from usable style wins: an adapter style win counts as usable only when its effective meaning judgment passes. Qualification requires at least 500 accepted training pairs, 50 final cases, 60% usable style wins, 95% meaning preservation, 100% protected-value checks, complete coverage and passing category scores. A failed candidate remains inactive. Small score differences do not establish that one model size is better.

The protected-value screen is supplementary. It cannot establish actor, uncertainty, omission or negation-scope correctness. It normalizes explicit zero-through-twenty cardinal spellings, Unicode numeric dashes and the fixed contraction `dunno`. It does not infer missing AM/PM, units, or contextual paraphrase equivalence. Inspect genuine flags and false positives separately rather than lowering thresholds to obtain a pass.

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

Use at least 50 distinct, genuinely independent cases with complete coverage for qualification. A one-case array is useful for a development smoke test only. Do not assert independence for a draft reconstructed from the reference reply. The current flag is a caller assertion; the software cannot prove how a draft was authored.

Cases accept `draft`, `independently_authored` and optional `category`, such as `"category": "uncertainty"`. A `context` field is rejected even when empty. Earlier context-bearing cases may be retained as historical evidence; prepare draft-only development cases and a separate fresh final suite.

The command writes three-mode outputs, a private candidate mapping, and `evaluation-request.json`. Give Codex the returned request file, not the hidden mapping. Codex returns the exact `comparison_hash` and one judgment for every request case ID:

```json
{
  "comparison_hash": "copy the exact value from evaluation-request.json",
  "judgments": [
    {
      "id": "0",
      "winner": "tie",
      "meaning": {"A": true, "B": true, "C": true},
      "uncertain": {"A": false, "B": false, "C": false},
      "reason_categories": [],
      "reason": "All candidates preserve the draft; no clear style preference."
    }
  ]
}
```

`winner` must be `A`, `B`, `C`, or `tie`. Judge each candidate's meaning separately before comparing style among meaning-preserving candidates. Use a tie when none qualifies or there is no clear preference. Boolean-only judgment rows remain compatible, but an explicitly uncertain candidate fails meaning even if `meaning` is true. Optional reason categories describe issues without changing scores.

Have a second independent reviewer adjudicate uncertain or disputed cases. Preserve the original row and attach an `adjudication` object containing `independently_reviewed: true`, complete `meaning` and `uncertain` maps, and a nonempty `reason`. This declaration does not prove reviewer independence. An uncertain adjudication still fails closed. Archive both assessments and disclose model-family overlap. Store responses privately, then run:

```bash
uv run multilora evaluate-import --run RUN --input /private/path/judgments.json
```

Import checks comparison identity, current family registration, complete case IDs, meaning and uncertainty judgments, coverage, protected-content screens, minimum training volume and release thresholds. Activation requires a passing locked qualification report and records language scope. Changing bound artifacts requires a new selection and fresh final qualification. A context-trained adapter, old training objective or legacy report cannot be reused as a current candidate. A smoke dataset below 500 accepted pairs cannot qualify.

The three-way usable preference rule counts ties separately; it is not a statistical significance test or direct pairwise estimate. Automated labeling and judgment remain model-assisted evidence. Passing qualification does not replace semantic review on each daily output. Greedy decoding is deterministic: repeating the same request is not a repair strategy. Fall back to the original draft when meaning cannot be confirmed.

All family ledgers and archives remain private. Evaluation rejects both run and family paths inside Git checkouts.
