"""Set MLX allocator guidance before handing training to the upstream LoRA CLI.

MLX's memory limit is advisory. The parent still independently supervises RSS,
macOS memory pressure and available application headroom, and may stop training.
"""

from __future__ import annotations

import argparse
import sys


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
        lora_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
