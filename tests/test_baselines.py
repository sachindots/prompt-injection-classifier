import numpy as np

from injclf.baselines import choose_threshold


def test_strict_threshold_separates_tied_positive_scores():
    scores = np.array([0.1, 0.4, 0.6, 0.6])
    labels = [0, 0, 1, 1]
    assert (scores > choose_threshold(labels, scores)).tolist() == labels
