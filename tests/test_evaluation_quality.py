"""Development/final isolation and meaning-first evaluation on wholly synthetic data."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from multilora_tone_adapter import evaluation
from multilora_tone_adapter.common import messages_for, write_json, write_jsonl
from multilora_tone_adapter.labeling import validate_meaning


def covered_cases(prefix="Development", *, multilingual=False):
    cases = []
    categories = [*evaluation.REQUIRED_CATEGORIES]
    if multilingual:
        categories.append("multilingual")
    for category in categories:
        for index in range(3):
            draft = f"{prefix} {category} case {index}: please preserve this synthetic detail."
            if category == "long":
                draft += " " + " ".join(f"detail{word}" for word in range(45))
            cases.append({"draft": draft, "independently_authored": True, "category": category})
    while len(cases) < 50:
        cases.append({"draft": f"{prefix} ordinary case {len(cases)}: send 12 units.",
                      "independently_authored": True, "category": "ordinary"})
    return cases


@pytest.fixture
def environment(tmp_path, monkeypatch):
    binding = {"accepted_pairs": 500, "adapter_sha256": "synthetic-weights",
               "evaluation_family_root": str(tmp_path), "training_objective": "synthetic-objective"}
    monkeypatch.setattr(evaluation, "_artifact_binding", lambda _: dict(binding))
    monkeypatch.setattr(evaluation, "_fit_style_examples", lambda model, cases, references: (references, {}))
    monkeypatch.setattr(evaluation, "_run_variant",
                        lambda directory, model, mode, cases, references, memory_gb:
                        [case["draft"] for case in cases])
    write_json(tmp_path / "model.json", {"path": "synthetic-model"})
    write_jsonl(tmp_path / "dataset/train.jsonl",
                [{"messages": messages_for("Synthetic reference draft", target="synthetic style")}])
    return tmp_path, binding


def run_comparison(directory, cases, *, purpose="development", language_scope="english"):
    write_json(directory / "cases.json", cases)
    evaluation.generate_comparison(directory, directory / "cases.json",
                                   purpose=purpose, language_scope=language_scope)
    comparison = json.loads((directory / "comparison-private.json").read_text())
    judgments = []
    for key, mapping in comparison["mapping"].items():
        judgments.append({"id": key, "winner": next(k for k, v in mapping.items() if v == "adapter"),
                          "meaning": dict.fromkeys("ABC", True),
                          "uncertain": dict.fromkeys("ABC", False), "reason": "Synthetic review"})
    return comparison, judgments


def import_judgments(directory, comparison, judgments):
    write_json(directory / "judgments.json",
               {"comparison_hash": evaluation.digest(comparison), "judgments": judgments})
    return evaluation.import_evaluation(directory, directory / "judgments.json")


def prepare_selected(environment):
    directory, binding = environment
    comparison, judgments = run_comparison(directory, covered_cases())
    report = import_judgments(directory, comparison, judgments)
    assert report["qualified"] is False
    assert report["blocking_reasons"] == ["development_only"]
    evaluation.select_candidate(directory)
    return directory, binding


def test_development_never_qualifies_but_disjoint_locked_final_can(environment):
    directory, _ = prepare_selected(environment)
    with pytest.raises(ValueError, match="qualification"):
        evaluation.activate(directory)
    comparison, judgments = run_comparison(directory, covered_cases("Final"), purpose="qualification")
    report = import_judgments(directory, comparison, judgments)
    assert report["qualified"] is True
    assert report["usable_adapter_wins"] == 50
    assert report["language_scope"] == "english"
    assert evaluation.activate(directory)["active"] is True
    evaluation.verify_active(directory)
    active = json.loads((directory / "active.json").read_text())
    assert active["language_scope"] == "english"


def test_completed_final_generation_is_idempotent_and_never_reshuffles(environment, monkeypatch):
    directory, _ = prepare_selected(environment)
    comparison, _ = run_comparison(directory, covered_cases("Final"), purpose="qualification")
    before = {name: (directory / name).read_bytes()
              for name in ("comparison-private.json", "evaluation-request.json")}

    def unexpected(*args, **kwargs):
        raise AssertionError("Completed final comparison must not run again")

    monkeypatch.setattr(evaluation, "_run_variant", unexpected)
    monkeypatch.setattr(evaluation, "_fit_style_examples", unexpected)
    monkeypatch.setattr(evaluation.secrets, "SystemRandom", unexpected)
    again = evaluation.generate_comparison(directory, directory / "cases.json", purpose="qualification")
    assert again["comparison_hash"] == evaluation.digest(comparison)
    assert all((directory / name).read_bytes() == content for name, content in before.items())


def test_first_final_judgments_and_report_cannot_be_replaced(environment):
    directory, _ = prepare_selected(environment)
    comparison, judgments = run_comparison(directory, covered_cases("Final"), purpose="qualification")
    judgments[0]["meaning"][judgments[0]["winner"]] = False
    failed = import_judgments(directory, comparison, judgments)
    assert failed["qualified"] is False
    assert import_judgments(directory, comparison, judgments) == failed
    judgments[0]["meaning"][judgments[0]["winner"]] = True
    with pytest.raises(ValueError, match="immutable"):
        import_judgments(directory, comparison, judgments)
    judgments[0]["meaning"][judgments[0]["winner"]] = False
    judgments[0]["adjudication"] = {
        "independently_reviewed": True, "meaning": dict.fromkeys("ABC", True),
        "uncertain": dict.fromkeys("ABC", False), "reason": "Late adjudication",
    }
    with pytest.raises(ValueError, match="before importing"):
        import_judgments(directory, comparison, judgments)
    assert json.loads((directory / "evaluation.json").read_text()) == failed
    with pytest.raises(ValueError, match="qualification"):
        evaluation.activate(directory)


def test_failed_worker_can_retry_identical_final_suite(environment, monkeypatch):
    directory, _ = prepare_selected(environment)
    original = evaluation._run_variant

    def failed(*args, **kwargs):
        raise RuntimeError("Synthetic worker interruption")

    monkeypatch.setattr(evaluation, "_run_variant", failed)
    with pytest.raises(RuntimeError, match="interruption"):
        run_comparison(directory, covered_cases("Final"), purpose="qualification")
    ledger = json.loads((directory / "evaluation-family.json").read_text())
    assert "comparison_hash" not in ledger["qualification"]
    monkeypatch.setattr(evaluation, "_run_variant", original)
    comparison, judgments = run_comparison(directory, covered_cases("Final"), purpose="qualification")
    assert import_judgments(directory, comparison, judgments)["qualified"]


def test_concurrent_final_generation_runs_workers_once(environment, monkeypatch):
    directory, _ = prepare_selected(environment)
    write_json(directory / "final-cases.json", covered_cases("Final"))
    entered, release = Event(), Event()
    modes = []

    def worker(directory, model, mode, cases, references, memory_gb):
        modes.append(mode)
        if mode == "base":
            entered.set()
            assert release.wait(5)
        return [case["draft"] for case in cases]

    monkeypatch.setattr(evaluation, "_run_variant", worker)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(evaluation.generate_comparison, directory, directory / "final-cases.json",
                            purpose="qualification")
        assert entered.wait(5)
        second = pool.submit(evaluation.generate_comparison, directory, directory / "final-cases.json",
                             purpose="qualification")
        release.set()
        assert first.result(timeout=5) == second.result(timeout=5)
    assert modes == ["base", "prompt", "adapter"]


@pytest.mark.parametrize("adjudication", [False, True])
def test_missing_uncertainty_is_rejected_instead_of_assumed_false(adjudication):
    row = {"meaning": dict.fromkeys("ABC", True)}
    if adjudication:
        row["uncertain"] = dict.fromkeys("ABC", False)
        row["adjudication"] = {"independently_reviewed": True, "meaning": dict.fromkeys("ABC", True),
                               "reason": "Incomplete second review"}
    with pytest.raises(ValueError, match="uncertainty"):
        evaluation._effective_meaning(row)


def test_final_requires_prior_selection_and_coverage(environment):
    directory, _ = environment
    with pytest.raises(ValueError, match="Select this candidate"):
        run_comparison(directory, covered_cases("Final"), purpose="qualification")
    with pytest.raises(ValueError, match="complete category coverage"):
        run_comparison(directory, [{"draft": str(i), "independently_authored": True}
                                   for i in range(50)], purpose="qualification")


def test_final_cannot_reuse_development_draft_after_case_or_whitespace_changes(environment):
    directory, _ = prepare_selected(environment)
    final = covered_cases("Final")
    final[0]["draft"] = "  " + covered_cases()[0]["draft"].upper().replace(" ", "  ") + "  "
    with pytest.raises(ValueError, match="disjoint"):
        run_comparison(directory, final, purpose="qualification")


def test_fixed_development_suite_and_final_lock_apply_to_sibling_bindings(environment):
    directory, binding = prepare_selected(environment)
    with pytest.raises(ValueError, match="fixed"):
        run_comparison(directory, covered_cases("Another development suite"))
    run_comparison(directory, covered_cases("Final"), purpose="qualification")
    with pytest.raises(ValueError, match="sealed"):
        run_comparison(directory, covered_cases())
    with pytest.raises(ValueError, match="already locked"):
        run_comparison(directory, covered_cases("Different final"), purpose="qualification")
    binding["adapter_sha256"] = "different-checkpoint"
    with pytest.raises(ValueError, match="Select this candidate"):
        run_comparison(directory, covered_cases("Final"), purpose="qualification")


def test_style_win_is_unusable_when_meaning_fails_or_is_uncertain(environment):
    directory, _ = environment
    comparison, judgments = run_comparison(directory, covered_cases())
    judgments[0]["meaning"][judgments[0]["winner"]] = False
    judgments[1]["uncertain"] = dict.fromkeys("ABC", False)
    judgments[1]["uncertain"][judgments[1]["winner"]] = True
    report = import_judgments(directory, comparison, judgments)
    assert report["adapter_wins"] == 50
    assert report["usable_adapter_wins"] == report["meaning_pass"] == 48
    assert report["uncertain_cases"] == 1
    assert "actor" in report["failing_categories"]


def test_independent_adjudication_retains_original_but_can_resolve_uncertainty(environment):
    directory, _ = environment
    comparison, judgments = run_comparison(directory, covered_cases())
    judgments[0]["uncertain"] = dict.fromkeys("ABC", True)
    judgments[0]["adjudication"] = {"independently_reviewed": True,
                                    "meaning": dict.fromkeys("ABC", True),
                                    "uncertain": dict.fromkeys("ABC", False),
                                    "reason": "Second synthetic reviewer confirmed the actors."}
    report = import_judgments(directory, comparison, judgments)
    assert report["meaning_pass"] == 50
    assert report["independently_adjudicated_cases"] == 1
    archived = directory / "evaluation-history" / report["comparison_hash"] / report["judgments_hash"] / "judgments.json"
    assert json.loads(archived.read_text())["judgments"][0]["uncertain"]["A"] is True


def test_unverified_adjudication_is_rejected(environment):
    directory, _ = environment
    comparison, judgments = run_comparison(directory, covered_cases())
    judgments[0]["adjudication"] = {"independently_reviewed": False,
                                    "meaning": dict.fromkeys("ABC", True), "reason": "Self review"}
    with pytest.raises(ValueError, match="independent review"):
        import_judgments(directory, comparison, judgments)


def test_category_failure_blocks_final_even_when_global_meaning_rate_passes(environment):
    directory, _ = prepare_selected(environment)
    comparison, judgments = run_comparison(directory, covered_cases("Final"), purpose="qualification")
    judgments[0]["meaning"][judgments[0]["winner"]] = False
    report = import_judgments(directory, comparison, judgments)
    assert report["meaning_pass"] == 49
    assert report["qualified"] is False
    assert report["failing_categories"] == ["actor"]


def test_multilingual_scope_requires_its_own_coverage(environment):
    directory, _ = environment
    comparison, judgments = run_comparison(directory, covered_cases(), language_scope="multilingual")
    report = import_judgments(directory, comparison, judgments)
    assert report["coverage"]["missing_categories"] == ["multilingual"]
    with pytest.raises(ValueError, match="complete development"):
        evaluation.select_candidate(directory)


def test_long_category_requires_actual_length_and_duplicates_ignore_category():
    with pytest.raises(ValueError, match="at least"):
        evaluation.validate_cases([{"draft": "short", "independently_authored": True, "category": "long"}])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.validate_cases([
            {"draft": "Same draft", "independently_authored": True, "category": "actor"},
            {"draft": " SAME   DRAFT ", "independently_authored": True, "category": "ordinary"},
        ])


def test_style_example_selection_is_paired_deterministic_and_diverse():
    rows = [{"messages": messages_for(f"Synthetic {i}", target=f"short {i}")} for i in range(20)]
    rows += [
        {"messages": messages_for("Synthetic uncertainty", target="maybe later?\nnot sure")},
        {"messages": messages_for("Synthetic mixed script", target="可以，see you later")},
        {"messages": messages_for("Synthetic long reply", target=" ".join(["synthetic"] * 40))},
    ]
    selected = evaluation.select_style_examples(rows)
    assert selected == evaluation.select_style_examples(list(reversed(rows)))
    assert len(selected) == 8
    assert any("可以" in pair["target"] for pair in selected)
    assert any("maybe" in pair["target"] for pair in selected)
    assert any(len(pair["target"].split()) > 25 for pair in selected)
    assert all(set(pair) == {"draft", "target"} for pair in selected)


def test_style_example_selector_rejects_history_inputs():
    rows = [{"messages": messages_for("Synthetic draft", target="synthetic target")}]
    rows[0]["messages"][1]["content"] = json.dumps({"draft": "Synthetic draft", "context": []})
    with pytest.raises(ValueError, match="only draft and target"):
        evaluation.select_style_examples(rows)


def test_rank_development_prefers_meaning_over_style_and_never_uses_final(environment, monkeypatch):
    directory, binding = environment
    common = {"purpose": "development", "case_hash": "shared", "language_scope": "english",
              "binding": binding, "cases": 50, "failing_categories": []}
    reports = {"a": {**common, "meaning_pass": 49, "protected_values_pass": 50, "usable_adapter_wins": 31},
               "b": {**common, "meaning_pass": 48, "protected_values_pass": 50, "usable_adapter_wins": 45}}
    monkeypatch.setattr(evaluation, "_development_report", lambda path: reports[path.name])
    ranked = evaluation.rank_development_candidates([directory / "b", directory / "a"])
    assert Path(ranked["candidates"][0]["run"]).name == "a"
    reports["b"]["case_hash"] = "different"
    with pytest.raises(ValueError, match="same family"):
        evaluation.rank_development_candidates([directory / "a", directory / "b"])


@pytest.mark.parametrize("draft,output", [
    ("Send four boxes.", "send 4 boxes"),
    ("Please arrive between 9–11.", "please arrive between 9-11"),
    ("I do not know yet.", "i dunno yet"),
])
def test_obvious_protected_value_equivalents(draft, output):
    assert validate_meaning(draft, output) == []


@pytest.mark.parametrize("draft,output", [
    ("Send four boxes.", "send 5 boxes"),
    ("Please arrive between 9–11.", "please arrive between 9-12"),
    ("I do not know yet.", "i know already"),
])
def test_normalization_does_not_hide_actual_changes(draft, output):
    assert validate_meaning(draft, output)


def test_ledger_rejects_git_checkout_even_if_metadata_redirects_family(environment):
    directory, binding = environment
    public = directory / "public-repository"
    (public / ".git").mkdir(parents=True)
    binding["evaluation_family_root"] = str(public / "private-looking-child")
    with pytest.raises(ValueError, match="outside Git"):
        run_comparison(directory, covered_cases())
    assert not (public / "private-looking-child").exists()


def test_stale_final_family_lock_invalidates_activation(environment):
    directory, _ = prepare_selected(environment)
    comparison, judgments = run_comparison(directory, covered_cases("Final"), purpose="qualification")
    import_judgments(directory, comparison, judgments)
    evaluation.activate(directory)
    ledger = json.loads((directory / "evaluation-family.json").read_text())
    ledger["qualification"]["suite_hash"] = "changed"
    write_json(directory / "evaluation-family.json", ledger)
    with pytest.raises(ValueError, match="locked candidate"):
        evaluation.verify_active(directory)
    with pytest.raises(ValueError, match="locked candidate"):
        evaluation.activate(directory)


def test_prompt_budget_drops_complete_pairs_without_loading_model(monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return list(range(sum(len(message["content"]) for message in messages)))

        def encode(self, value):
            return list(value)

    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return Tokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    pairs = [{"draft": "x" * 1800, "target": "y" * 1800}, {"draft": "neutral", "target": "casual"}]
    selected, budget = evaluation._fit_style_examples(Path("local-model"), [{"draft": "synthetic"}], pairs)
    assert selected == [pairs[1]]
    assert budget["dropped_whole_examples"] == 1
    assert calls == [("local-model", {"local_files_only": True, "trust_remote_code": False})]
    with pytest.raises(ValueError, match="no input was truncated"):
        evaluation._fit_style_examples(Path("local-model"), [{"draft": "x" * 3000}], pairs)
