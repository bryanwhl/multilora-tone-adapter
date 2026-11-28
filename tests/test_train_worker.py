"""Allocator setup precedes all upstream LoRA imports and model allocations."""

import builtins
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from checkpoint_fixtures import curated_dataset

from multilora_tone_adapter import runtime, train_worker


@pytest.fixture
def fake_mlx(monkeypatch):
    events = []
    core = ModuleType("mlx.core")
    core.set_memory_limit = lambda value: events.append(("memory", value))
    core.set_cache_limit = lambda value: events.append(("cache", value))
    mlx = ModuleType("mlx")
    mlx.__path__ = []
    mlx.core = core
    lora = ModuleType("mlx_lm.lora")
    lora.train = lambda *args, **kwargs: None
    lora.evaluate = lambda *args, **kwargs: None
    lora.main = lambda: events.append(("main", list(sys.argv)))
    package = ModuleType("mlx_lm")
    package.__path__ = []
    for name, module in [
        ("mlx", mlx),
        ("mlx.core", core),
        ("mlx_lm", package),
        ("mlx_lm.lora", lora),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    real_import = builtins.__import__

    def watched_import(name, *args, **kwargs):
        if name == "mlx_lm.lora":
            events.append(("lora_import", None))
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", watched_import)
    return events, lora


def test_allocator_caps_precede_upstream_import_and_args_are_forwarded(fake_mlx):
    events, _ = fake_mlx
    original_argv = sys.argv
    budget = 7 * 1024**3
    upstream = [
        "--model",
        "synthetic-model",
        "--train",
        "--batch-size",
        "1",
        "--mask-prompt",
        "--grad-checkpoint",
    ]
    train_worker.main(["--memory-budget-bytes", str(budget), "--", *upstream])
    assert events == [
        ("memory", budget),
        ("cache", 256 * 1024**2),
        ("lora_import", None),
        ("main", ["mlx_lm.lora", *upstream]),
    ]
    assert sys.argv is original_argv


def test_forwarded_budget_like_option_cannot_change_worker_cap(fake_mlx):
    events, _ = fake_mlx
    train_worker.main(["--memory-budget-bytes", "1024", "--", "--memory-budget-bytes", "999999"])
    assert events[:2] == [("memory", 1024), ("cache", 64)]
    # Upstream receives the unknown option and rejects it through its own parser.
    assert events[-1] == ("main", ["mlx_lm.lora", "--memory-budget-bytes", "999999"])


def test_lora_failure_restores_original_process_args(fake_mlx):
    _, lora = fake_mlx
    original_argv = sys.argv

    def fail():
        raise RuntimeError("synthetic training failure")

    lora.main = fail
    with pytest.raises(RuntimeError, match="synthetic"):
        train_worker.main(["--memory-budget-bytes", "1024", "--train"])
    assert sys.argv is original_argv


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--memory-budget-bytes"],
        ["--memory-budget-bytes", "0"],
        ["--memory-budget-bytes", "-1"],
        ["--memory-budget-bytes", "nan"],
        ["--memory-budget-bytes", "inf"],
        ["--memory-budget-bytes", "1.5"],
        ["--memory-budget-bytes", "true"],
        ["--memory-budget-bytes", "1", "--memory-budget-bytes", "2"],
        ["--memory-budget", "1024"],
    ],
)
def test_invalid_or_ambiguous_caps_fail_before_any_mlx_import(monkeypatch, args):
    real_import = builtins.__import__
    imports = []

    def watched_import(name, *positional, **kwargs):
        if name == "mlx" or name.startswith(("mlx.", "mlx_lm")):
            imports.append(name)
            raise AssertionError("invalid CLI imported a model runtime")
        return real_import(name, *positional, **kwargs)

    monkeypatch.setattr(builtins, "__import__", watched_import)
    with pytest.raises(SystemExit) as error:
        train_worker.main(args)
    assert error.value.code == 2
    assert imports == []


@pytest.mark.parametrize(
    "budget,expected", [(1, 0), (16, 1), (1024, 64), (8 * 1024**3, 256 * 1024**2)]
)
def test_cache_budget_is_small_but_can_be_disabled_for_tiny_budget(budget, expected):
    assert train_worker.cache_limit_bytes(budget) == expected


@pytest.mark.parametrize("budget", [True, 0, -1, 1.5, float("nan"), float("inf")])
def test_programmatic_noninteger_or_nonpositive_budgets_rejected(budget):
    with pytest.raises(ValueError):
        train_worker.cache_limit_bytes(budget)


@pytest.mark.parametrize("real_child_permissions", [False, True])
def test_parent_wires_budget_worker_metadata_without_changing_upstream_help_or_guards(
    monkeypatch, tmp_path, real_child_permissions
):
    if real_child_permissions and runtime.os.name != "posix":
        pytest.skip("Training child umask is a POSIX contract")
    # Fail if training ever mutates the caller-wide mask instead of the child mask.
    parent_umask = Mock(side_effect=AssertionError("caller umask must not be modified"))
    monkeypatch.setattr(runtime.os, "umask", parent_umask)
    data, model, adapter = (tmp_path / name for name in ("data", "model", "adapter"))
    data.mkdir()
    model.mkdir()
    curated_dataset(data)
    budget = {
        "budget_bytes": 7 * runtime.GIB,
        "reserve_bytes": 3 * runtime.GIB,
        "pressure_level": 1,
    }
    monkeypatch.setattr(runtime, "preflight", lambda memory_gb: budget)
    monkeypatch.setattr(
        runtime,
        "model_fingerprint",
        lambda path: {"weight_bytes": 100, "fingerprint": "synthetic-model"},
    )

    def view(model_path, adapter_dir):
        directory = adapter_dir / "training_model"
        directory.mkdir()
        (directory / "tokenizer_config.json").write_text("{}")
        return directory

    monkeypatch.setattr(runtime, "_training_view", view)
    tokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer))
    monkeypatch.setattr(runtime, "verify_training_rows", lambda *args: {"masking_verified": True})
    help_check = Mock(
        return_value=SimpleNamespace(
            returncode=0, stdout="--mask-prompt --grad-checkpoint --max-seq-length --num-layers"
        )
    )
    monkeypatch.setattr(runtime.subprocess, "run", help_check)
    launched = []
    child = object()
    real_popen = runtime.subprocess.Popen

    def launch(command, **kwargs):
        launched.append((command, kwargs))
        if real_child_permissions:
            # Exercise the exact captured child settings without importing a model.
            script = (
                "from pathlib import Path; "
                "Path('adapters.safetensors').write_bytes(b'synthetic-adapter'); "
                "Path('0000001_adapters.safetensors').write_bytes(b'synthetic-checkpoint'); "
                "Path('adapter_config.json').write_text('{}')"
            )
            with real_popen([sys.executable, "-c", script], **kwargs) as permission_child:
                assert permission_child.wait(timeout=10) == 0
        else:
            (adapter / "adapters.safetensors").write_bytes(b"synthetic-adapter")
            (adapter / "0000001_adapters.safetensors").write_bytes(b"synthetic-checkpoint")
            (adapter / "adapter_config.json").write_text("{}")
        return child

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    supervise = Mock(return_value={"exit_code": 0, "peak_rss_bytes": 1024})
    monkeypatch.setattr(runtime, "_supervise", supervise)
    result = runtime.train(data, model, adapter, iters=2, memory_gb=7)
    command, kwargs = launched[0]
    assert command[:6] == [
        sys.executable,
        "-m",
        "multilora_tone_adapter.train_worker",
        "--memory-budget-bytes",
        str(budget["budget_bytes"]),
        "--",
    ]
    assert "--mask-prompt" in command and "--grad-checkpoint" in command
    assert command[command.index("--save-every") + 1] == "1"
    assert command[command.index("--val-batches") + 1] == "-1"
    assert help_check.call_args.args[0] == [sys.executable, "-m", "mlx_lm.lora", "--help"]
    supervise.assert_called_once_with(child, budget)
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["umask"] == 0o077
    parent_umask.assert_not_called()
    if real_child_permissions:
        for name in ("adapters.safetensors", "adapter_config.json"):
            assert (adapter / name).stat().st_mode & 0o777 == 0o600
        assert adapter.stat().st_mode & 0o777 == 0o700
    assert result["worker"] == {
        "module": "multilora_tone_adapter.train_worker",
        "memory_budget_bytes": budget["budget_bytes"],
        "cache_limit_bytes": 256 * 1024**2,
        "memory_limit_advisory": True,
    }
    assert result["status"] == "succeeded" and result["qualified"] is False
    assert result["training_objective"] == train_worker.training_objective_contract()
    assert [row["step"] for row in result["checkpoints"]] == [1, 2]
    assert json.loads((adapter / "training.json").read_text())["worker"] == result["worker"]
    assert kwargs["cwd"] == Path(adapter)


@pytest.mark.parametrize(
    "offset,length,padded_length,expected",
    [(3, 5, 33, [3, 4]), (2, 3, 513, [2]), (1, 2, 2, [1]), (0, 4, 8, [1, 2, 3])],
)
def test_target_mask_excludes_padding_and_keeps_final_real_token(
    offset, length, padded_length, expected
):
    # Includes a one-token assistant target and padding much longer than the reply.
    positions = [
        position for position in range(1, padded_length)
        if train_worker.target_token_mask(position, offset, length)
    ]
    assert positions == expected


def test_objective_injected_for_training_and_validation_without_persistent_mutation(fake_mlx):
    _, lora = fake_mlx
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(("evaluate", kwargs["loss"]))

    def train(*args, **kwargs):
        calls.append(("train", kwargs["loss"]))
        # This is the upstream training contract: forward loss into validation.
        evaluate(loss=kwargs["loss"])

    lora.train, lora.evaluate = train, evaluate
    with train_worker.corrected_objective(lora):
        lora.train()
        lora.evaluate()
    assert calls == [
        ("train", train_worker.assistant_target_loss),
        ("evaluate", train_worker.assistant_target_loss),
        ("evaluate", train_worker.assistant_target_loss),
    ]
    assert lora.train is train and lora.evaluate is evaluate


def test_actual_mlx_loss_and_gradients_ignore_padding_and_prompt():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = mx.zeros((2, 32, 7))

        def __call__(self, inputs):
            return self.logits[:, :inputs.shape[1], :]

    model = TinyModel()
    # Genuine EOS uses token 6. Row 2 has exactly one target token (EOS).
    batch = mx.array([[1, 2, 3, 4, 6] + [0] * 28, [1, 2, 6] + [0] * 30])
    lengths = mx.array([[3, 5], [2, 3]])
    (loss, count), gradients = nn.value_and_grad(model, train_worker.assistant_target_loss)(
        model, batch, lengths
    )
    mx.eval(loss, count, gradients)
    assert count.item() == 3
    assert loss.item() == pytest.approx(1.945910, abs=1e-5)
    nonzero = (mx.abs(gradients["logits"]).sum(axis=-1) > 0).tolist()
    assert [[i for i, flag in enumerate(row) if flag] for row in nonzero] == [[2, 3], [1]]
    # Poison only prompt/padded predictions; the objective must remain identical.
    changed = mx.array(model.logits)
    changed[0, :2, 0] = 100
    changed[0, 4:, 0] = 100
    changed[1, :1, 0] = 100
    changed[1, 2:, 0] = 100
    model.logits = changed
    second_loss, second_count = train_worker.assistant_target_loss(model, batch, lengths)
    assert second_loss.item() == pytest.approx(loss.item(), abs=1e-6)
    assert second_count.item() == 3


def test_actual_upstream_training_and_validation_use_corrected_objective(monkeypatch, tmp_path):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    optim = pytest.importorskip("mlx.optimizers")
    lora = pytest.importorskip("mlx_lm.lora")
    trainer = pytest.importorskip("mlx_lm.tuner.trainer")

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(7, 4)
            self.projection = nn.Linear(4, 7)

        def __call__(self, inputs):
            return self.projection(self.embedding(inputs))

    calls = []
    actual_loss = train_worker.assistant_target_loss

    def observed_loss(model, batch, lengths):
        calls.append(model.training)
        return actual_loss(model, batch, lengths)

    monkeypatch.setattr(train_worker, "assistant_target_loss", observed_loss)
    dataset = [([1, 2, 6], 2)]  # One genuine assistant token (EOS).
    args = trainer.TrainingArgs(
        iters=1, batch_size=1, val_batches=-1, steps_per_eval=1,
        steps_per_report=1, steps_per_save=1, max_seq_length=64,
        adapter_file=tmp_path / "adapters.safetensors",
    )
    model = TinyModel()
    mx.eval(model.parameters())
    with train_worker.corrected_objective(lora):
        lora.train(model=model, optimizer=optim.SGD(learning_rate=0.0),
                   train_dataset=dataset, val_dataset=dataset, args=args)
        result = lora.evaluate(model=model, dataset=dataset, batch_size=1,
                               num_batches=-1, max_seq_length=64)
    assert True in calls and calls.count(False) >= 2
    assert result >= 0
    assert (tmp_path / "adapters.safetensors").is_file()
