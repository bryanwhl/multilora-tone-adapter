"""Small entirely synthetic reviewed datasets for training boundary tests."""

from multilora_tone_adapter.dataset import build_dataset
from multilora_tone_adapter.pair_review import REVIEW_CHECKS, review_binding, review_contract


def curated_dataset(directory):
    episodes, labels, reviews = [], [], []
    pairs = [("Thank you.", "thanks!"), ("See you soon.", "see you soon!"),
             ("That works.", "that works!")]
    for index, (draft, target) in enumerate(pairs):
        episode = {
            "id": str(index), "context": [], "target": target,
            "group_id": f"synthetic-session-{index}", "source_ids": [f"synthetic-{index}"],
            "timestamp": f"2026-01-0{index + 1}T00:00:00Z",
        }
        label = {"id": str(index), "draft": draft, "meaning_preserved": True}
        episodes.append(episode)
        labels.append(label)
        reviews.append({
            **review_binding(episode, label), "review_contract": review_contract(),
            "request_hash": "synthetic-request", "verdict": "pass",
            "checks": {name: "pass" for name in REVIEW_CHECKS}, "reason": "Equivalent greeting.",
        })
    return build_dataset(episodes, labels, directory, min_episodes=3, reviews=reviews,
                         identity_fraction=0)
