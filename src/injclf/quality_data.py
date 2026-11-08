"""Load pinned public rows and reserve the external exam before fitting."""

import hashlib

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def rows(root, name, repo, revision, file):
    path = root / "artifacts/quality-data" / (name + ".parquet")
    if not path.exists():
        path = hf_hub_download(repo, file, revision=revision, repo_type="dataset")
    return pq.read_table(path, columns=["text", "label"]).to_pylist()


def sample(records, seen, limit, seed):
    grouped = {0: [], 1: []}
    for row in records:
        text, label = row["text"], row["label"]
        key = " ".join(text.casefold().split())
        if key and key not in seen:
            seen.add(key)
            order = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
            grouped[label].append((order, text))
    return [
        (text, label)
        for label, group in grouped.items()
        for _, text in sorted(group)[:limit]
    ]


def quality_rows(root, c, corpus):
    seen = {
        " ".join(text.casefold().split())
        for group in corpus.values()
        for _, text, _ in group
    }
    exam = {}
    for name in c["exam"]:
        records = rows(
            root, name, c["dataset"], c["dataset_revision"], f"data/{name}.parquet"
        )
        exam[name] = sample(records, seen, c["max_per_label"], c["seed"])
    training = {}
    for name, source in c["training"].items():
        training[name] = sample(
            rows(root, name, **source), seen, c["train_per_label_source"], c["seed"]
        )
    return training, exam
