"""Train the lexical baseline and export the offline workbench."""

import json
from itertools import pairwise
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from injclf.data import load_data


def choose_threshold(labels, scores) -> float:
    values = sorted(set(scores))
    boundaries = [0.0, *((a + b) / 2 for a, b in pairwise(values)), 1.0]
    return float(max(boundaries, key=lambda t: (f1_score(labels, scores > t), t)))


def build(root: Path) -> None:
    splits, stats = load_data(root / "configs/data.toml")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([r.text for r in splits.train])
    model = LogisticRegression(random_state=stats["seed"]).fit(
        X, [r.label for r in splits.train]
    )

    def scores(rows):
        return model.predict_proba(vectorizer.transform([r.text for r in rows]))[:, 1]

    threshold = choose_threshold(
        [r.label for r in splits.validation], scores(splits.validation)
    )
    metrics = {
        "model": "TF-IDF + logistic regression",
        "threshold": threshold,
        "splits": {},
    }
    for name in ("test", "held_out"):
        rows = getattr(splits, name)
        labels = [r.label for r in rows]
        probs = scores(rows)
        pred = probs > threshold
        metrics["splits"][name] = {
            "f1": f1_score(labels, pred),
            "roc_auc": roc_auc_score(labels, probs),
            "average_precision": average_precision_score(labels, probs),
            "false_positives": sum(int(p and not y) for p, y in zip(pred, labels)),
            "benign": labels.count(0),
        }
    (root / "results/tfidf.json").write_text(json.dumps(metrics, indent=2) + "\n")
    case = {
        r.origin.split("/")[-1]: r.text
        for r in splits.held_out
        if r.origin.startswith("attacks.jsonl/a003/")
    }
    assert case.keys() == {"question", "injected_document"}
    payload = {
        "terms": {
            term: [vectorizer.idf_[i], model.coef_[0, i]]
            for term, i in vectorizer.vocabulary_.items()
        },
        "intercept": model.intercept_[0],
        "metrics": metrics,
        "case": case,
        "stats": stats,
    }
    page = (
        (root / "web/workbench.html")
        .read_text()
        .replace("__PAYLOAD__", json.dumps(payload).replace("<", "\\u003c"))
    )
    output = root / "artifacts/workbench.html"
    output.parent.mkdir(exist_ok=True)
    output.write_text(page)


if __name__ == "__main__":
    build(Path.cwd())
