"""Private, single-variant MLX process. No source text is emitted to stdout."""

import json
import platform
import sys
import time
from pathlib import Path

from .common import write_json
from .evaluation import validate_cases
from .runtime import MLXEngine, preflight


def wait_for_headroom(memory_gb, seconds: float = 30) -> None:
    """macOS can report released GPU pages late even after the prior worker exits."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return  # The engine reports unsupported hardware without a retry delay.
    deadline = time.monotonic() + seconds
    while True:
        try:
            preflight(memory_gb=memory_gb)
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)


def generate(request: dict) -> list[str]:
    mode = request["mode"]
    if mode not in {"base", "prompt", "adapter"}:
        raise ValueError("Unsupported evaluation mode")
    validate_cases(request["cases"])
    options = {"memory_gb": request.get("memory_gb")}
    if mode == "prompt":
        options["evaluation_style_references"] = request["references"]
        options["evaluation_style_rubric"] = request.get("style_rubric")
    engine = MLXEngine(Path(request["model_path"]),
                       Path(request["adapter_path"]) if mode == "adapter" else None,
                       **options)
    try:
        outputs = []
        for case in request["cases"]:
            outputs.append(engine.rewrite(case["draft"], max_output_tokens=512))
        return outputs
    finally:
        engine.close()


if __name__ == "__main__":
    try:
        request = json.loads(Path(sys.argv[1]).read_text())
        wait_for_headroom(request.get("memory_gb"))
        write_json(Path(sys.argv[2]), generate(request))
    except Exception as exc:
        # Full third-party exception strings can contain private text.
        print(f"Evaluation worker stopped: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2) from None
