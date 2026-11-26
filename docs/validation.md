# Validation results

The checks below used fictional messages and synthetic fixtures. They verify that the workflow executes and fails safely; they do not establish a useful personal writing style or qualify an adapter trained on private chats.

## Local validation on 23 November 2026

This historical run predates the draft-only input contract and used the earlier context-capable workflow. Its datasets and adapters are not compatible with the current runtime. These observations do not validate the new contract or a newly trained adapter; rerun the portable checks and a separate local model evaluation for the current implementation.

| Check | Result |
| --- | --- |
| Automated tests | 89 passed; one upstream Starlette test-client deprecation warning |
| Ruff | Passed |
| Codex skill validator | Passed |
| Telegram-shaped synthetic import and dataset | 18 accepted episodes; 14 training, 2 validation, 2 test |
| Real local MLX fine-tuning | Qwen3 8B 4-bit; 2 training steps, batch size 1, 4 trainable layers, maximum sequence length 256 |
| Local HTTP smoke test | HTTP 200 from a candidate adapter |
| Real model comparison | Base, style-prompt baseline, and adapter completed 3 fresh synthetic cases in separate processes under a 7 GiB model budget |
| Qualification import | Not qualified; 18 pairs and 3 cases are below the required minimums |
| Activation | Correctly rejected the unqualified candidate |

The three-case comparison produced two prompted-baseline wins and one tie. The adapter preserved meaning and screened protected values on all three cases. These are smoke-test observations, not a statistically meaningful quality result. Automated judgments are a proxy and are not human validation.

The smoke run deliberately used fewer than 500 accepted pairs and fewer than 50 independent evaluation cases. No private-chat adapter is qualified or activated by these checks. Raw datasets, model outputs, judgments, logs, and weights remain outside this repository.

## Evaluation isolation regression

Sequential model variants in one process could retain MLX or Metal allocations after Python references were released. Evaluation now runs each variant in a separate child process and checks available headroom before loading. The worker waits for at most 30 seconds when macOS has not yet reported released pages; persistent pressure still stops the run.

Automated regressions cover worker cleanup, unchanged inputs, transient and persistent memory pressure, malformed or stale worker outputs, and failure without comparison or qualification artifacts. A successful candidate HTTP response remains separate from production activation.

Run the portable checks with:

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

Real training and inference require Apple Silicon and the optional MLX dependencies. See [evaluation](evaluation.md) for qualification requirements and [data handling](data-handling.md) for private artifact boundaries.
