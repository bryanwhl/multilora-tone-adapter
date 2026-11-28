"""Set MLX allocator guidance before handing training to the upstream LoRA CLI.

MLX's memory limit is advisory. The parent still independently supervises RSS,
macOS memory pressure and available application headroom, and may stop training.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from functools import wraps

from .common import digest


def training_objective_contract() -> dict:
    """Version the objective independently of the draft serialization contract."""
    definition = {
        "version": 1,
        "target_positions": "prompt_offset <= original_token_index < sequence_length",
        "reduction": "mean_cross_entropy_over_unmasked_target_tokens",
        "includes_real_eos": True,
        "includes_padding": False,
    }
    return {"name": "assistant-target-no-padding-v1", "sha256": digest(definition)}


def target_token_mask(positions, offsets, lengths):
    """Positions are original token indices after the next-token target shift."""
    return (positions >= offsets) & (positions < lengths)


def assistant_target_loss(model, batch, lengths):
    """Score actual assistant tokens, including EOS, without scoring any padding."""
    import mlx.core as mx
    import mlx.nn as nn

    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    positions = mx.arange(1, targets.shape[1] + 1)
    mask = target_token_mask(positions, lengths[:, 0:1], lengths[:, 1:])
    losses = nn.losses.cross_entropy(logits, targets)
    count = mask.sum()
    return (losses * mask).astype(mx.float32).sum() / count, count


@contextmanager
def corrected_objective(lora_module):
    """Inject an explicit loss into this worker's upstream entry points only.

    MLX train forwards its loss callable to validation. Its separate evaluate
    entry point is wrapped too, so --test uses the same objective. No installed
    files or function defaults are modified.
    """
    original = {name: getattr(lora_module, name) for name in ("train", "evaluate")}

    def bind(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            kwargs["loss"] = assistant_target_loss
            return function(*args, **kwargs)
        return wrapped

    try:
        for name, function in original.items():
            setattr(lora_module, name, bind(function))
        yield
    finally:
        for name, function in original.items():
            setattr(lora_module, name, function)


def cache_limit_bytes(memory_budget_bytes: int) -> int:
    """Keep reusable allocator buffers small relative to the worker budget."""
    if type(memory_budget_bytes) is not int or memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be a positive integer")
    return min(256 * 1024**2, memory_budget_bytes // 16)


def _positive_bytes(value: str) -> int:
    try:
        budget = int(value)
        cache_limit_bytes(budget)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "memory budget must be a positive integer in bytes"
        ) from None
    return budget


class _SingleBudget(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error("memory budget must be supplied exactly once")
        setattr(namespace, self.dest, values)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--memory-budget-bytes", type=_positive_bytes, required=True, action=_SingleBudget
    )
    options, upstream_args = parser.parse_known_args(argv)
    if upstream_args[:1] == ["--"]:
        upstream_args = upstream_args[1:]
    # No MLX/model imports happen until the required allocator budget is valid.
    import mlx.core as mx

    mx.set_memory_limit(options.memory_budget_bytes)
    mx.set_cache_limit(cache_limit_bytes(options.memory_budget_bytes))
    from mlx_lm.lora import main as lora_main

    original_argv = sys.argv
    try:
        sys.argv = ["mlx_lm.lora", *upstream_args]
        with corrected_objective(sys.modules["mlx_lm.lora"]):
            lora_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
