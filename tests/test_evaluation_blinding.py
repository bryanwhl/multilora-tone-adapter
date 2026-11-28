"""Candidate letters depend on OS-backed shuffling, never predictable case indices."""

import itertools
import json

from multilora_tone_adapter import evaluation
from multilora_tone_adapter.common import digest, messages_for, write_json, write_jsonl


def test_per_case_shuffle_controls_letters_and_only_private_mapping_identifies_variants(
    monkeypatch, tmp_path
):
    modes = ["base", "prompt", "adapter"]
    permutations = list(itertools.permutations(modes))
    cases = [
        {"draft": f"Synthetic draft {index}", "independently_authored": True}
        for index in range(len(permutations))
    ]
    write_json(tmp_path / "cases.json", cases)
    write_json(tmp_path / "model.json", {"path": "synthetic-model"})
    write_jsonl(
        tmp_path / "dataset/train.jsonl",
        [{"messages": messages_for("Synthetic reference draft", target="Synthetic style reference")}],
    )
    binding = {"synthetic_binding": True}
    monkeypatch.setattr(evaluation, "_artifact_binding", lambda _: binding)
    monkeypatch.setattr(evaluation, "_fit_style_examples", lambda model, cases, references: (references, {}))
    generated_modes = []

    def run_variant(directory, model, mode, worker_cases, references, memory_gb):
        generated_modes.append(mode)
        assert worker_cases == cases
        assert references == [{"draft": "Synthetic reference draft", "target": "Synthetic style reference"}]
        return [f"{mode}-synthetic-result-{index}" for index in range(len(cases))]

    monkeypatch.setattr(evaluation, "_run_variant", run_variant)
    shuffled_lists = []

    class ScriptedShuffler:
        def shuffle(self, order):
            assert order == modes  # Every case starts from a fresh complete set.
            shuffled_lists.append(order)
            order[:] = permutations[len(shuffled_lists) - 1]

    created = []

    def system_random():
        created.append(True)
        return ScriptedShuffler()

    monkeypatch.setattr(evaluation.secrets, "SystemRandom", system_random)
    result = evaluation.generate_comparison(tmp_path, tmp_path / "cases.json")
    private_path = tmp_path / "comparison-private.json"
    private = json.loads(private_path.read_text())
    request = json.loads((tmp_path / "evaluation-request.json").read_text())
    assert created == [True]
    assert len(shuffled_lists) == len(cases)
    assert len({id(order) for order in shuffled_lists}) == len(cases)
    assert generated_modes == modes  # Model-generation sequence is unchanged.
    assert private["binding"] == binding
    assert set(request) == {
        "comparison_hash",
        "style_references_train_only",
        "style_rubric",
        "instructions",
        "cases",
    }
    for index, blind_case in enumerate(request["cases"]):
        assert set(blind_case) == {"id", "draft", "candidates"}
        assert blind_case["id"] == str(index)
        expected_mapping = dict(zip("ABC", permutations[index], strict=True))
        assert private["mapping"][str(index)] == expected_mapping
        assert set(blind_case["candidates"]) == set("ABC")
        for letter, mode in expected_mapping.items():
            assert blind_case["candidates"][letter] == private["variants"][mode][index]
    assert result["comparison_hash"] == request["comparison_hash"] == digest(private)
    assert private_path.stat().st_mode & 0o777 == 0o600
    private["mapping"]["0"]["A"], private["mapping"]["0"]["B"] = (
        private["mapping"]["0"]["B"],
        private["mapping"]["0"]["A"],
    )
    assert digest(private) != request["comparison_hash"]
