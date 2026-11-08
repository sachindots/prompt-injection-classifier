"""Adapt the six-layer detector on published training data; select on validation only."""

import hashlib
import json
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import f1_score
from torch.nn.functional import binary_cross_entropy_with_logits, normalize
from transformers import AutoModel, AutoTokenizer, set_seed

from injclf.policy import choose_cutoff, counts, policy_corpus
from injclf.probe import features
from injclf.quality_data import quality_rows


def main():
    root = Path(__file__).resolve().parents[2]
    c = tomllib.loads((root / "configs/quality.toml").read_text())
    policy = tomllib.loads((root / "configs/gateway.toml").read_text())
    corpus, _, _ = policy_corpus(root, policy)
    new, exam = quality_rows(root, c, corpus)
    train = [
        (text, label)
        for name in ("train", "human_fit", "gateway_fit")
        for _, text, label in corpus[name]
    ]
    for group in new.values():
        train.extend(group)
    set_seed(c["seed"])
    folder = root / "artifacts/gateway"
    enc = AutoModel.from_pretrained(
        folder, attn_implementation="sdpa", reference_compile=False
    ).cuda()
    tok = AutoTokenizer.from_pretrained(folder)
    old = json.loads((folder / "head.json").read_text())
    depth = old["layers"]
    head = torch.nn.Linear(enc.config.hidden_size, 1).cuda()
    with torch.no_grad():
        head.weight.copy_(torch.tensor(old["weights"]))
        head.bias.fill_(old["bias"])
    enc.embeddings.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            p
            for p in list(enc.parameters()) + list(head.parameters())
            if p.requires_grad
        ],
        lr=c["learning_rate"],
    )
    target = root / "artifacts/refined"
    target.mkdir(exist_ok=True)
    history = []
    best = -1
    for epoch in range(1, c["epochs"] + 1):
        enc.train()
        for ids in torch.randperm(len(train)).split(c["train_batch"]):
            batch = [train[i] for i in ids.tolist()]
            tokens = tok(
                [r[0] for r in batch],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to("cuda")
            mask = tokens.attention_mask.unsqueeze(-1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = enc(**tokens, output_hidden_states=True).hidden_states[depth]
                pooled = normalize((hs * mask).sum(1) / mask.sum(1))
                logits = head(pooled).flatten().float()
                loss = binary_cross_entropy_with_logits(
                    logits,
                    torch.tensor(
                        [r[1] for r in batch], device="cuda", dtype=torch.float32
                    ),
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            optimizer.step()
        enc.eval()
        w = head.weight.detach().cpu().numpy().flatten().astype(float)
        bias = head.bias.item()
        parts = {}
        negatives = defaultdict(list)
        for name in ("validation", "gateway_validation"):
            scored = []
            for kind, text, label in corpus[name]:
                v = features(enc, tok, text, [depth])[depth]
                score = float(expit(v @ w + bias).max())
                scored.append((label, score))
                if not label:
                    negatives[kind].append(score)
            parts[name] = scored
        cutoff = choose_cutoff(list(negatives.values()), policy["benign_flag_budget"])

        def metric(t, partitions=parts):
            return np.mean(
                [
                    f1_score([y for y, s in v], [s > t for y, s in v])
                    for v in partitions.values()
                ]
            )

        threshold = max(
            {cutoff, *[s for v in parts.values() for _, s in v if s >= cutoff]},
            key=lambda t: (metric(t), t),
        )
        quality = float(metric(threshold))
        history.append(
            {
                "epoch": epoch,
                "mean_validation_f1": quality,
                "threshold": threshold,
                "splits": {k: counts(v, threshold) for k, v in parts.items()},
            }
        )
        print(history[-1], flush=True)
        if quality > best:
            best = quality
            enc.save_pretrained(target)
            tok.save_pretrained(target)
            (target / "head.json").write_text(
                json.dumps(
                    {
                        "layers": depth,
                        "bias": bias,
                        "weights": w.tolist(),
                        "threshold": threshold,
                    }
                )
            )
    result = {
        "config": c,
        "epochs": history,
        "new_training_counts": {k: len(v) for k, v in new.items()},
        "exam_sha256": {
            k: hashlib.sha256(json.dumps(v).encode()).hexdigest()
            for k, v in exam.items()
        },
    }
    (root / "results/refinement.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
