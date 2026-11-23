import json
import subprocess
import sys
import types
from unittest.mock import Mock

import pytest

from multilora_tone_adapter import runtime
from multilora_tone_adapter.common import messages_for, write_jsonl


def test_import_is_lazy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import multilora_tone_adapter.runtime; "
                "assert 'mlx.core' not in sys.modules; assert 'mlx_lm' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    snapshot = {
        "total_bytes": 32 * runtime.GIB,
        "available_bytes": 12 * runtime.GIB,
        "pressure_level": 1,
        "swap_used_bytes": runtime.GIB,
    }
    monkeypatch.setattr(runtime, "memory_snapshot", lambda: snapshot.copy())
    monkeypatch.delenv("MULTILORA_RESERVE_GB", raising=False)
    return snapshot


def test_budget_adapts_to_available_memory_and_app_reserve(mac):
    first = runtime.preflight()
    mac["available_bytes"] -= 3 * runtime.GIB
    second = runtime.preflight()
    assert first["budget_bytes"] - second["budget_bytes"] == 3 * runtime.GIB
    assert first["reserve_bytes"] == int(32 * 0.12 * runtime.GIB)
    with pytest.raises(RuntimeError, match="headroom"):
        runtime.preflight(memory_gb=10)
    assert runtime.preflight(memory_gb=4, reserve_gb=5)["budget_bytes"] == 4 * runtime.GIB


def test_budget_refuses_pressure_and_nonfinite(mac):
    with pytest.raises(ValueError):
        runtime.preflight(float("nan"))
    mac["pressure_level"] = 2
    with pytest.raises(RuntimeError, match="pressure"):
        runtime.preflight()


def test_download_pins_sha_and_private_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(tmp_path / "private"))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    download = Mock(return_value=str(snapshot))
    api = Mock()
    api.model_info.return_value.sha = "a" * 40
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: api, snapshot_download=download),
    )
    assert runtime.download_model("org/model", "main") == snapshot
    assert download.call_args.kwargs["revision"] == "a" * 40
    assert download.call_args.kwargs["cache_dir"].startswith(str(tmp_path / "private"))
    assert json.loads(runtime._model_record(snapshot).read_text())["revision"] == "a" * 40


def test_fingerprint_detects_changed_weights_and_tokenizer(monkeypatch, tmp_path):
    monkeypatch.setenv("MULTILORA_STATE_DIR", str(tmp_path / "private"))
    model = tmp_path / ".cache" / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    weights = model / "model.safetensors"
    weights.write_bytes(b"original")
    original = runtime.model_fingerprint(model)
    weights.write_bytes(b"changed")
    assert runtime.model_fingerprint(model)["fingerprint"] != original["fingerprint"]


class Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        has_target = messages[-1]["role"] == "assistant"
        prompt_messages = messages[:-1] if has_target else messages
        text = json.dumps(prompt_messages, sort_keys=True) + "<assistant>"
        if has_target:
            text += messages[-1]["content"] + "<end>"
        return list(text.encode())


@pytest.fixture
def fake_mlx_dataset(monkeypatch):
    class ChatDataset:
        def __init__(self, rows, tokenizer, mask_prompt):
            self.tokenizer = tokenizer

        def process(self, row):
            messages = row["messages"]
            return (
                self.tokenizer.apply_chat_template(messages),
                len(self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)),
            )

    monkeypatch.setitem(
        sys.modules, "mlx_lm.tuner.datasets", types.SimpleNamespace(ChatDataset=ChatDataset)
    )


def test_actual_mask_boundary_and_no_silent_truncation(fake_mlx_dataset, tmp_path):
    write_jsonl(tmp_path / "train.jsonl", [{"messages": messages_for("A draft.", [], "A reply.")}])
    result = runtime.verify_training_rows(tmp_path, Tokenizer(), Tokenizer(), 4096)
    assert result["masking_verified"]
    with pytest.raises(ValueError, match="truncate"):
        runtime.verify_training_rows(tmp_path, Tokenizer(), Tokenizer(), 32)


def test_mask_boundary_mismatch_is_rejected(fake_mlx_dataset, tmp_path):
    class WrongTokenizer(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return super().apply_chat_template(*args, **kwargs) + [999]

    write_jsonl(tmp_path / "train.jsonl", [{"messages": messages_for("Draft", [], "Reply")}])
    with pytest.raises(ValueError, match="masking boundary"):
        runtime.verify_training_rows(tmp_path, Tokenizer(), WrongTokenizer(), 4096)


def test_training_failure_is_persisted_and_never_qualified(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "preflight", Mock(side_effect=RuntimeError("pressure")))
    adapter = tmp_path / "candidate"
    with pytest.raises(RuntimeError):
        runtime.train(tmp_path / "data", tmp_path / "model", adapter)
    metadata = json.loads((adapter / "training.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["qualified"] is False
    assert not (adapter / "adapters.safetensors").exists()
    with pytest.raises(ValueError, match="fresh"):
        runtime.train(tmp_path / "data", tmp_path / "model", adapter)


def test_supervisor_stops_child_when_apps_need_memory(monkeypatch):
    child = Mock()
    child.poll.return_value = None
    process = Mock()
    process.memory_info.return_value.rss = runtime.GIB
    process.children.return_value = []
    monkeypatch.setattr(runtime.psutil, "Process", lambda pid: process)
    monkeypatch.setattr(
        runtime, "memory_snapshot", lambda: {"pressure_level": 1, "available_bytes": runtime.GIB}
    )
    stop = Mock()
    monkeypatch.setattr(runtime, "_stop", stop)
    with pytest.raises(RuntimeError, match="headroom"):
        runtime._supervise(
            child, {"budget_bytes": 8 * runtime.GIB, "reserve_bytes": 3 * runtime.GIB}
        )
    stop.assert_called_once_with(child)


def fake_generation(monkeypatch, *, finish="stop", pressure=1):
    closed = []

    def stream(*args, **kwargs):
        try:
            yield types.SimpleNamespace(text="partial rewrite", finish_reason=finish)
        finally:
            closed.append(True)

    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(stream_generate=stream))
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.sample_utils",
        types.SimpleNamespace(make_sampler=lambda **kwargs: None),
    )
    monkeypatch.setattr(
        runtime,
        "memory_snapshot",
        lambda: {"pressure_level": pressure, "available_bytes": 10 * runtime.GIB},
    )
    engine = runtime.MLXEngine(".")
    engine._model = object()
    engine._tokenizer = Tokenizer()
    engine._budget = {"budget_bytes": 8 * runtime.GIB, "reserve_bytes": 3 * runtime.GIB}
    return engine, closed


def test_truncated_generation_is_discarded(monkeypatch):
    engine, closed = fake_generation(monkeypatch, finish="length")
    with pytest.raises(ValueError, match="partial rewrite discarded"):
        engine.rewrite("draft")
    assert closed == [True]


def test_memory_guard_unloads_and_closes_generator(monkeypatch):
    engine, closed = fake_generation(monkeypatch, pressure=2)
    with pytest.raises(RuntimeError, match="memory"):
        engine.rewrite("draft")
    assert not engine.loaded
    assert closed == [True]


@pytest.mark.parametrize("change", ["mutated", "unbound"])
def test_adapter_config_mutation_or_missing_binding_rejected_before_mlx(
    monkeypatch, tmp_path, change
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapters.safetensors"
    weights.write_bytes(b"adapter-weights")
    config = adapter / "adapter_config.json"
    config.write_text('{"lora_parameters":{"scale":20}}')
    metadata = {
        "status": "succeeded",
        "model": {"fingerprint": "original-model"},
        "adapter_sha256": runtime.file_sha256(weights),
        "adapter_config_sha256": runtime.file_sha256(config),
    }
    if change == "mutated":
        config.write_text('{"lora_parameters":{"scale":99}}')
    else:
        metadata.pop("adapter_config_sha256")
    (adapter / "training.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(runtime, "preflight", lambda *args: {"budget_bytes": 8 * runtime.GIB})
    monkeypatch.setattr(
        runtime,
        "model_fingerprint",
        lambda *args: {"fingerprint": "original-model", "weight_bytes": runtime.GIB},
    )
    engine = runtime.MLXEngine(tmp_path / "model", adapter)
    with pytest.raises(ValueError, match="incomplete or does not match"):
        engine.load()
    assert not engine.loaded
