"""Synthetic polarity checks for explicit English auxiliary contractions."""

import pytest

from multilora_tone_adapter.labeling import validate_meaning


@pytest.mark.parametrize(
    "expanded,contracted",
    [
        ("I cannot attend.", "I can't attend."),
        ("I will not attend.", "I won't attend."),
        ("I do not agree.", "I don't agree."),
        ("It does not fit.", "It doesn't fit."),
        ("I did not mail it.", "I didn't mail it."),
        ("It is not ready.", "It isn't ready."),
        ("We are not ready.", "We aren't ready."),
        ("It was not ready.", "It wasn't ready."),
        ("We were not ready.", "We weren't ready."),
        ("I have not mailed the parcel yet.", "I haven't mailed the parcel yet."),
        ("It has not arrived.", "It hasn't arrived."),
        ("I had not seen it.", "I hadn't seen it."),
        ("I could not attend.", "I couldn't attend."),
        ("I would not agree.", "I wouldn't agree."),
        ("I should not send it.", "I shouldn't send it."),
        ("I must not send it.", "I mustn't send it."),
        ("I might not attend.", "I mightn't attend."),
        ("I need not attend.", "I needn't attend."),
        ("I ought not attend.", "I oughtn't attend."),
        ("I dare not ask.", "I daren't ask."),
        ("I shall not attend.", "I shan't attend."),
        ("I am not ready.", "I ain't ready."),
    ],
)
@pytest.mark.parametrize("apostrophe", ["'", "’", "‘", ""])
def test_equivalent_negation_contractions_preserve_detected_polarity(
    expanded, contracted, apostrophe
):
    variant = contracted.replace("'", apostrophe)
    assert "changed_negation" not in validate_meaning(expanded, variant)
    assert "changed_negation" not in validate_meaning(variant, expanded)


@pytest.mark.parametrize(
    "negative,positive",
    [
        ("I can't attend.", "I can attend."),
        ("I won't attend.", "I will attend."),
        ("I haven't mailed the parcel yet.", "I have mailed the parcel."),
        ("It hasn't arrived.", "It has arrived."),
        ("I shouldn't send it.", "I should send it."),
        ("We aren't ready.", "We are ready."),
        ("I don't want to attend.", "I want to attend."),
    ],
)
@pytest.mark.parametrize("apostrophe", ["'", "’", "‘", ""])
def test_opposite_polarities_remain_flagged(negative, positive, apostrophe):
    variant = negative.replace("'", apostrophe)
    assert "changed_negation" in validate_meaning(variant, positive)
    assert "changed_negation" in validate_meaning(positive, variant)


@pytest.mark.parametrize(
    "word", ["want", "plant", "restaurant", "important", "constant", "patient"]
)
def test_ordinary_positive_words_do_not_count_as_negation(word):
    positive = f"The synthetic word is {word}."
    assert "changed_negation" not in validate_meaning(positive, "The synthetic word is ordinary.")
    assert "changed_negation" in validate_meaning(positive, "The synthetic word is not ordinary.")


def test_case_insensitivity_and_other_protected_value_checks_are_preserved():
    draft = "I have not mailed 12 parcels yet."
    output = "I HAVEN’T mailed 13 parcels yet."
    assert validate_meaning(draft, output) == ["changed_numbers"]
    assert draft == "I have not mailed 12 parcels yet."
    assert output == "I HAVEN’T mailed 13 parcels yet."


@pytest.mark.parametrize(
    "negative,positive",
    [
        ("I cannot attend.", "I can attend."),
        ("No parcels arrived.", "Parcels arrived."),
        ("I never mailed it.", "I mailed it."),
        ("It is not ready.", "It is ready."),
    ],
)
def test_standalone_negative_forms_keep_existing_polarity_policy(negative, positive):
    assert "changed_negation" in validate_meaning(negative, positive)
    assert "changed_negation" in validate_meaning(positive, negative)
