"""Fit ModernBERT adapters, select a validation checkpoint, and evaluate the saved model."""

import json
import tomllib
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from injclf.baselines import choose_threshold
from injclf.data import load_data
from injclf.predict import score_texts


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    c = tomllib.loads((root / "configs/modernbert.toml").read_text())
    set_seed(c["seed"])
    splits, stats = load_data(root / "configs/data.toml")
    base = snapshot_download(
        c["base"],
        revision=c["revision"],
        allow_patterns=["*.json", "model.safetensors"],
    )
    tokenizer = AutoTokenizer.from_pretrained(base)
    backbone = AutoModelForSequenceClassification.from_pretrained(
        base, attn_implementation="sdpa", reference_compile=False
    )
    model = get_peft_model(
        backbone,
        LoraConfig(
            task_type="SEQ_CLS",
            r=c["rank"],
            lora_alpha=c["alpha"],
            target_modules=["Wqkv"],
            modules_to_save=["head", "classifier"],
        ),
    ).to("cuda")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=c["learning_rate"]
    )
    y = [r.label for r in splits.validation]
    history = []
    best = (-1.0, -1.0)
    best_path = root / "artifacts/modernbert_adapter"
    window = c["max_length"]
    size = c["batch_size"]
    for epoch in range(1, c["epochs"] + 1):
        model.train()
        for ids in torch.randperm(len(splits.train)).split(size):
            rows = [splits.train[i] for i in ids.tolist()]
            tokens = tokenizer(
                [r.text for r in rows],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=window,
            ).to("cuda")
            labels = torch.tensor([r.label for r in rows], device="cuda")
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**tokens, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scores = np.array(
            score_texts(
                model,
                tokenizer,
                [r.text for r in splits.validation],
                window,
                c["stride"],
                size,
            )
        )
        threshold = choose_threshold(y, scores)
        key = (f1_score(y, scores > threshold), average_precision_score(y, scores))
        history.append(
            {
                "epoch": epoch,
                "f1": key[0],
                "average_precision": key[1],
                "threshold": threshold,
            }
        )
        print(history[-1], flush=True)
        if key > best:
            best = key
            best_epoch = epoch
            best_threshold = threshold
            model.save_pretrained(best_path)
    from peft import PeftModel

    merged = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained(
            base, attn_implementation="sdpa", reference_compile=False
        ),
        best_path,
    ).merge_and_unload()
    merged.config.window_length = window
    merged.config.window_stride = c["stride"]
    merged.config.injection_threshold = best_threshold
    path = root / "artifacts/modernbert"
    merged.save_pretrained(path)
    tokenizer.save_pretrained(path)
    del model, backbone, optimizer
    merged.to("cuda")
    result = {
        "model": "ModernBERT-base + LoRA",
        "config": c,
        "threshold": best_threshold,
        "best_epoch": best_epoch,
        "validation": history,
        "data": stats,
        "splits": {},
    }
    for name in ("validation", "test", "held_out"):
        rows = getattr(splits, name)
        labels = [r.label for r in rows]
        scores = np.array(
            score_texts(
                merged, tokenizer, [r.text for r in rows], window, c["stride"], size
            )
        )
        pred = scores > best_threshold
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel().tolist()
        result["splits"][name] = {
            "f1": f1_score(labels, pred),
            "precision": precision_score(labels, pred, zero_division=0),
            "recall": recall_score(labels, pred),
            "roc_auc": roc_auc_score(labels, scores),
            "average_precision": average_precision_score(labels, scores),
            "benign": tn + fp,
            "false_positives": fp,
            "false_positive_rate": fp / (tn + fp),
            "detected_attacks": tp,
            "attacks": tp + fn,
        }
    (root / "results/modernbert_lora.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result["splits"], flush=True)


if __name__ == "__main__":
    main()
