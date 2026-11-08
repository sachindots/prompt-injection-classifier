"""Score text with the published, quantized Horizon Labs detector."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.special import expit
from tokenizers import Tokenizer


class Challenger:
    def __init__(self, folder: Path, config: dict):
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(config["window"], stride=config["stride"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = config["threads"]
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(folder / "onnx/model_quantized.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.inputs = {item.name for item in self.session.get_inputs()}

    def score(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter nonempty text")
        encoded = self.tokenizer.encode(text)
        best = 0.0
        for window in [encoded, *encoded.overflowing]:
            fields = {
                "input_ids": window.ids,
                "attention_mask": window.attention_mask,
                "token_type_ids": window.type_ids,
            }
            batch = {
                key: np.asarray([fields[key]], dtype=np.int64) for key in self.inputs
            }
            logits = self.session.run(None, batch)[0][0].astype(float)
            best = max(best, float(expit(logits[1] - logits[0])))
        return best
