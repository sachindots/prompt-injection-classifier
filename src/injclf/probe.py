"""Pool contextual token features and apply the gateway's fitted linear head."""

import torch
from torch.nn.functional import normalize


@torch.inference_mode()
def features(encoder, tokenizer, text, depths):
    tokens = tokenizer(
        text,
        max_length=256,
        stride=64,
        truncation=True,
        padding=True,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )
    tokens.pop("overflow_to_sample_mapping")
    output = {d: [] for d in depths}
    for start in range(0, len(tokens["input_ids"]), 8):
        batch = {k: v[start : start + 8].to(encoder.device) for k, v in tokens.items()}
        states = encoder(**batch, output_hidden_states=True).hidden_states
        mask = batch["attention_mask"].unsqueeze(-1)
        for d in depths:
            # All candidate depths use the intermediate layer output before the final norm.
            hidden = states[d]
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            output[d].append(normalize(pooled).cpu())
    return {d: torch.cat(parts).numpy() for d, parts in output.items()}
