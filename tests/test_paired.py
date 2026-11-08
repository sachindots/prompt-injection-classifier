import json

from injclf.paired import build_pairs, paired_counts


def test_pairs_use_separate_documents_and_test_attack_templates(tmp_path):
    folder = tmp_path / "artifacts/bipia-upstream/rev"
    (folder / "email").mkdir(parents=True)
    for phase in ("train", "test"):
        for kind in ("text", "code"):
            (folder / f"{kind}_attack_{phase}.json").write_text(
                json.dumps({"family": [f"{phase} payload"]})
            )
        documents = [
            "duplicate",
            "reserved",
            *[f"{phase} letter {i}" for i in range(4)],
        ]
        (folder / f"email/{phase}.jsonl").write_text(
            "\n".join(json.dumps({"context": d}) for d in documents)
        )
    cfg = {
        "revision": "rev",
        "seed": 42,
        "domains": ["email"],
        "train_documents_per_domain": 3,
        "validation_documents_per_domain": 1,
    }
    splits = build_pairs(tmp_path, cfg, {"reserved"})
    documents = [clean for split in splits.values() for _, clean, _ in split]
    assert len(set(documents)) == len(documents)
    assert "reserved" not in [
        clean for phase in ("train", "validation") for _, clean, _ in splits[phase]
    ]
    assert {k: len(v) for k, v in splits.items()} == {
        "train": 3,
        "validation": 1,
        "test": 6,
    }
    for partition, pairs in splits.items():
        phase = "test" if partition == "test" else "train"
        for _, clean, attacked in pairs:
            assert clean in attacked and f"{phase} payload" in attacked


def test_both_members_must_be_classified_correctly():
    metrics = paired_counts([[0.5, 0.6], [0.8, 0.9], [0.1, 0.5]], 0.5)
    assert metrics == {
        "pairs": 3,
        "both_correct": 1,
        "attacks_detected": 2,
        "benign_flagged": 1,
        "attack_scored_higher": 3,
    }
