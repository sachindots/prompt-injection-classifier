"""Inference for the locally trained classifier."""

from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@torch.inference_mode()
def score_texts(
    model, tokenizer, texts: list[str], length: int, stride: int, batch_size: int
) -> list[float]:
    """Keep the highest positive-class score across each text's overlapping windows."""
    model.eval()
    scores = []
    for start in range(0, len(texts), batch_size):
        group = texts[start : start + batch_size]
        encoded = tokenizer(
            group,
            truncation=True,
            max_length=length,
            stride=stride,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        owners = encoded.pop("overflow_to_sample_mapping").tolist()
        maxima = [0.0] * len(group)
        for offset in range(0, len(owners), batch_size):
            batch = {
                k: v[offset : offset + batch_size].to(model.device)
                for k, v in encoded.items()
            }
            probs = model(**batch).logits.float().softmax(-1)[:, 1].tolist()
            for i, p in enumerate(probs):
                owner = owners[offset + i]
                maxima[owner] = max(maxima[owner], p)
        scores.extend(maxima)
    return scores


@lru_cache(maxsize=1)
def load_detector():
    path = Path(__file__).resolve().parents[2] / "artifacts/modernbert"
    model = AutoModelForSequenceClassification.from_pretrained(
        path, attn_implementation="sdpa"
    )
    return model.eval(), AutoTokenizer.from_pretrained(path)


def injection_score(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter nonempty text")
    model, tokenizer = load_detector()
    return score_texts(
        model,
        tokenizer,
        [text],
        model.config.window_length,
        model.config.window_stride,
        1,
    )[0]
