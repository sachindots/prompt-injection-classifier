from types import SimpleNamespace

import pytest
import torch

from injclf.predict import injection_score, score_texts


def test_window_scores_stay_with_their_source_text():
    class WindowModel:
        device = "cpu"

        def eval(self):
            return self

        def __call__(self, input_ids):
            return SimpleNamespace(
                logits=torch.cat([torch.zeros_like(input_ids), input_ids], dim=1)
            )

    def tokenize(texts, **kwargs):
        assert texts == ["first", "second"]
        assert kwargs["return_overflowing_tokens"]
        return {
            "input_ids": torch.tensor([[-3.0], [4.0], [-2.0]]),
            "overflow_to_sample_mapping": torch.tensor([0, 0, 1]),
        }

    assert score_texts(
        WindowModel(), tokenize, ["first", "second"], 256, 64, 2
    ) == pytest.approx(torch.tensor([4.0, -2.0]).sigmoid().tolist())


def test_empty_input_rejected_before_loading_checkpoint():
    with pytest.raises(ValueError, match="nonempty"):
        injection_score("  ")
