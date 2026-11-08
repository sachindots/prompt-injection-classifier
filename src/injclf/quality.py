"""Benchmark a frozen detector on the reserved public exam and existing gateway sets."""

import argparse
import hashlib
import json
import tomllib
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.special import expit
from transformers import AutoModel, AutoTokenizer

from injclf.policy import choose_cutoff, counts, policy_corpus
from injclf.probe import features
from injclf.quality_data import quality_rows


def checkpoint_scorer(folder: Path):
    model = AutoModel.from_pretrained(
        folder, attn_implementation="sdpa", reference_compile=False
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(folder)
    head = json.loads((folder / "head.json").read_text())
    weights = np.asarray(head["weights"])
    depth = head["layers"]

    def score(text):
        v = features(model, tokenizer, text, [depth])[depth]
        return float(expit(v @ weights + head["bias"]).max())

    return score, head["threshold"]


def horizon_scorer(root, c, corpus, budget):
    from injclf.challenger import Challenger

    folder = root / "artifacts/horizon"
    for name in (
        "tokenizer.json",
        "onnx/model_quantized.onnx",
        "SHA256SUMS",
        "LICENSE",
        "NOTICE",
    ):
        if not (folder / name).exists():
            hf_hub_download(c["model"], name, revision=c["revision"], local_dir=folder)
    hashes = {
        name: digest
        for digest, name in (
            line.split() for line in (folder / "SHA256SUMS").read_text().splitlines()
        )
    }
    for name in ("tokenizer.json", "onnx/model_quantized.onnx"):
        with (folder / name).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != hashes[name]:
                raise ValueError("Challenger artifact checksum mismatch")
    score = Challenger(folder, c).score
    negatives = defaultdict(list)
    parts = []
    for name in ("validation", "gateway_validation"):
        values = []
        for kind, text, label in corpus[name]:
            value = score(text)
            values.append((label, value))
            if not label:
                negatives[kind].append(value)
        parts.append(values)
    floor = choose_cutoff(list(negatives.values()), budget)
    options = {floor, *[s for part in parts for _, s in part if s >= floor]}
    cutoff = max(
        options, key=lambda t: (sum(counts(part, t)["f1"] for part in parts), t)
    )
    return score, cutoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["gateway", "refined", "horizon"])
    name = parser.parse_args().model
    root = Path(__file__).resolve().parents[2]
    c = tomllib.loads((root / "configs/quality.toml").read_text())
    policy = tomllib.loads((root / "configs/gateway.toml").read_text())
    corpus, _, _ = policy_corpus(root, policy)
    _, exam = quality_rows(root, c, corpus)
    digests = {
        k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in exam.items()
    }
    if (
        digests
        != json.loads((root / "results/refinement.json").read_text())["exam_sha256"]
    ):
        raise ValueError("Exam differs from the frozen training manifest")
    torch.set_num_threads(c["threads"])
    if name == "horizon":
        score, cutoff = horizon_scorer(root, c, corpus, policy["benign_flag_budget"])
    else:
        score, cutoff = checkpoint_scorer(root / "artifacts" / name)
    result = {
        "model": name,
        "config": c,
        "threshold": cutoff,
        "exam_sha256": digests,
        "sets": {},
    }
    for part in (
        "validation",
        "gateway_validation",
        "gateway_audit",
        "test",
        "held_out",
        "extraction_challenge",
    ):
        exam[part] = [(text, label) for _, text, label in corpus[part]]
    score(corpus["train"][0][1])
    for part, rows in exam.items():
        values = []
        times = []
        for text, label in rows:
            start = perf_counter_ns()
            values.append((label, score(text)))
            times.append((perf_counter_ns() - start) / 1e6)
        m = counts(values, cutoff)
        m["accuracy"] = (
            m["benign"] - m["false_positives"] + m["detected_attacks"]
        ) / len(rows)
        m.update(
            p50_ms=float(np.percentile(times, 50)),
            p95_ms=float(np.percentile(times, 95)),
        )
        result["sets"][part] = m
        print(part, m, flush=True)
    (root / f"results/quality_{name}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
