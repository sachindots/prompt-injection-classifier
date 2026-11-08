"""Choose a gateway threshold on independent security text, then audit it."""

import hashlib
import json
import math
import random
import re
import tomllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

from injclf.data import load_data
from injclf.predict import load_detector
from injclf.probe import features


def choose_cutoff(groups: list[list[float]], budget: float) -> float:
    """Lowest strict cutoff meeting the empirical false-positive budget in every group."""
    return max(
        sorted(scores, reverse=True)[math.floor(len(scores) * budget)]
        for scores in groups
    )


def counts(rows, threshold):
    y = np.array([label for label, _ in rows])
    scores = np.array([score for _, score in rows])
    flagged = scores > threshold
    return {
        "attacks": int(y.sum()),
        "detected_attacks": int(sum(flagged & (y == 1))),
        "benign": int(sum(y == 0)),
        "false_positives": int(sum(flagged & (y == 0))),
        "f1": float(f1_score(y, flagged)) if len(set(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, scores))
        if len(set(y)) == 2
        else None,
    }


def main():
    root = Path(__file__).resolve().parents[2]
    c = tomllib.loads((root / "configs/gateway.toml").read_text())
    folder = root / "artifacts/policy-sources"
    folder.mkdir(exist_ok=True, parents=True)
    for file, url in [("purple.json", c["attack_url"])]:
        if not (folder / file).exists():
            urlretrieve(url, folder / file)
    rows = []
    for item in json.loads((folder / "purple.json").read_text()):
        rows.append(("attack", item["injection_variant"], item["user_input"], 1))
    for q in json.loads((root / "configs/security_questions.json").read_text()):
        rows.append(("question", str(q["id"]), q["title"], 0))

    def passages(sheet):
        filename = sheet + "_Cheat_Sheet.md"
        target = folder / filename
        if not target.exists():
            urlretrieve(c["passages_url"] + filename, target)
        paragraphs = [
            p.strip()
            for p in target.read_text().split("\n\n")
            if re.match(r"^[A-Z]", p) and 120 <= len(p) <= 1500 and "\n" not in p
        ]
        return [("passage", sheet, p, 0) for p in paragraphs[:15]]

    with ThreadPoolExecutor(max_workers=6) as pool:
        for group_rows in pool.map(passages, c["sheets"]):
            rows.extend(group_rows)
    splits, stats = load_data(root / "configs/data.toml")
    seen = {r.key for part in vars(splits).values() for r in part}
    partitions = {}
    for source in ("attack", "question", "passage"):
        groups = sorted({group for kind, group, _, _ in rows if kind == source})
        random.Random(c["seed"]).shuffle(groups)
        half = len(groups) // 2
        for index, group in enumerate(groups):
            partitions[source, group] = (
                "gateway_fit"
                if index < half // 2
                else "gateway_validation"
                if index < half
                else "gateway_audit"
            )
    corpus = defaultdict(list)
    for kind, group, text, label in rows:
        key = " ".join(text.casefold().split())
        if key and key not in seen:
            seen.add(key)
            corpus[partitions[kind, group]].append((kind, text, label))
    for name in ("train", "validation", "test", "held_out"):
        corpus[name] = [
            ("deepset" if name != "held_out" else "rag", r.text, r.label)
            for r in getattr(splits, name)
        ]
    # Reserve the extraction challenge before adding human hijacking attempts to training.
    for suite, part in (
        ("extraction", "extraction_challenge"),
        ("hijacking", "human_fit"),
    ):
        path = folder / (suite + ".jsonl")
        if not path.exists():
            urlretrieve(
                c["human_attacks_url"]
                + f"{suite}-robustness/v1/{suite}_robustness_dataset.jsonl",
                path,
            )
        for line in path.read_text().splitlines():
            text = json.loads(line)["attack"]
            key = " ".join(text.casefold().split())
            if key and key not in seen:
                seen.add(key)
                corpus[part].append((suite, text, 1))
    model, tokenizer = load_detector()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = model.base_model.to(device).eval()
    torch.set_num_threads(4)
    depths = c["encoder_depths"]
    cache = {}

    def vectors(part):
        if part not in cache:
            cache[part] = [
                features(encoder, tokenizer, text, depths)
                for _, text, _ in corpus[part]
            ]
        return cache[part]

    fit = vectors("train") + vectors("gateway_fit")
    fit_labels = [row[2] for row in corpus["train"] + corpus["gateway_fit"]]
    candidates = []
    best_key = None
    for include_human in (False, True):
        for depth in depths:
            head = LogisticRegression(
                C=c["regularization"], random_state=c["seed"], max_iter=1000
            )
            inputs = fit + (vectors("human_fit") if include_human else [])
            labels = fit_labels + (
                [1] * len(corpus["human_fit"]) if include_human else []
            )
            head.fit(np.stack([v[depth][0] for v in inputs]), labels)
            dev = []
            negatives = defaultdict(list)
            for name in ("validation", "gateway_validation"):
                scored = []
                for (kind, _, label), v in zip(
                    corpus[name], vectors(name), strict=True
                ):
                    score = float(
                        expit(
                            v[depth] @ head.coef_[0].astype(float)
                            + float(head.intercept_[0])
                        ).max()
                    )
                    scored.append((label, score))
                    if not label:
                        negatives[kind].append(score)
                dev.append(scored)
            threshold = choose_cutoff(list(negatives.values()), c["benign_flag_budget"])
            dev_metrics = [counts(records, threshold) for records in dev]
            quality = sum(m["f1"] for m in dev_metrics) / len(dev_metrics)
            candidates.append(
                {
                    "human_training": include_human,
                    "layers": depth,
                    "cutoff": threshold,
                    "mean_validation_f1": quality,
                    "validation": dev_metrics,
                }
            )
            if best_key is None or (quality, -depth) > best_key:
                best_key = (quality, -depth)
                use_human = include_human
                selected, cutoff, weights, bias = (
                    depth,
                    threshold,
                    head.coef_[0].astype(float),
                    float(head.intercept_[0]),
                )
    metrics = {}
    for name in (
        "validation",
        "gateway_validation",
        "test",
        "gateway_audit",
        "held_out",
        "extraction_challenge",
    ):
        scores = [
            float(expit(v[selected] @ weights + bias).max()) for v in vectors(name)
        ]
        metrics[name] = counts(
            [(row[2], s) for row, s in zip(corpus[name], scores, strict=True)], cutoff
        )
    model_path = root / "artifacts/gateway"
    encoder.layers = encoder.layers[:selected]
    encoder.config.num_hidden_layers = selected
    encoder.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    head_config = {
        "weights": weights.tolist(),
        "bias": bias,
        "threshold": cutoff,
        "layers": selected,
    }
    (model_path / "head.json").write_text(json.dumps(head_config))
    with (model_path / "model.safetensors").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    result = {
        "model": f"ModernBERT · {selected}-layer gateway",
        "threshold": cutoff,
        "config": c,
        "validation_candidates": candidates,
        "human_training": use_human,
        "data": stats,
        "case": {
            r.origin.split("/")[-1]: r.text
            for r in splits.held_out
            if r.origin.startswith("attacks.jsonl/a003/")
        },
        "head_sha256": hashlib.sha256(
            (model_path / "head.json").read_bytes()
        ).hexdigest(),
        "weights_sha256": digest,
        "splits": metrics,
        "fit_counts": {
            k: len(corpus[k]) for k in ("train", "gateway_fit", "human_fit")
        },
    }
    (root / "results/gateway.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["model"], json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
