# Validation guide

Use portable tests to verify software behavior, a separate local model smoke test to verify MLX execution, and private evaluation to assess a trained adapter. Passing one level does not establish the next.

## Portable checks

Run from a source checkout:

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

The suite uses fictional messages, synthetic fixtures, and fake inference engines. It covers parsing, identity and timezone requirements, grouped splits, draft-only serialization, API validation, resource guards, artifact compatibility, and qualification failures. It does not load a language model or establish personal style quality.

## Local model smoke test

Real training and inference require Apple Silicon and the optional MLX dependencies:

```bash
uv sync --extra dev --extra mlx
uv run multilora preflight
```

Use a separate private run with wholly fictional data to exercise model loading, a short training job, and the local HTTP API. Follow the [workflow commands](../README.md#first-run) and inspect each command's help before choosing settings. Run names, file paths, and resource budgets must come from that local setup.

Check that:

- Training and inference use the same draft-only input contract and reject incompatible artifacts.
- Token and prompt-masking checks pass without truncating required content.
- A valid draft-only HTTP request reaches the candidate adapter; a `context` field is rejected, including when empty.
- The resource guard preserves the configured application headroom and reports a failure if it cannot.
- An unqualified candidate cannot be activated. `serve --candidate` is for an explicit smoke test only.

A small fictional dataset or one successful HTTP response cannot qualify an adapter for normal use. Keep the smoke run's datasets, outputs, logs, weights, and report outside Git.

## Evaluation isolation

Evaluation runs the base, prompted baseline, and adapter in separate child processes. This prevents one variant from retaining another's MLX or Metal allocations. Each worker checks current headroom and waits at most 30 seconds for released pages to become available; persistent pressure stops the run.

Portable regressions cover worker cleanup, unchanged inputs, transient and persistent memory pressure, malformed or stale worker outputs, and failure without comparison or qualification artifacts.

## Private adapter qualification

Evaluate a completed adapter on fresh, independently prepared drafts using the [evaluation protocol](evaluation.md). Report style preference and meaning preservation separately. The qualification gate binds judgments to the dataset, input contract, base model, and adapter; a stale report cannot qualify changed artifacts.

Store run-specific results in private state. The public repository contains reusable validation instructions and wholly synthetic tests, not individual training-run reports, hardware measurements, or personal output examples. See [data handling](data-handling.md) for the publication boundary.
