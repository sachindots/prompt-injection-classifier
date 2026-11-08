"""Fit a lexical document detector and audit matched clean/injected BIPIA pairs."""

import hashlib
import json
import random
import tarfile
import tomllib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from injclf.baselines import choose_threshold
from injclf.data import external_rows
from injclf.policy import policy_corpus
from injclf.quality import checkpoint_scorer
from injclf.quality_data import rows

type Pair = tuple[str, str, str]


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def source(root: Path, revision: str, name: str) -> str:
    target = root / "artifacts/bipia-upstream" / revision / name
    if not target.exists():
        url = f"https://raw.githubusercontent.com/microsoft/BIPIA/{revision}/benchmark/{name}"
        with urlopen(url, timeout=60) as response:
            content = response.read()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return target.read_text()


def paired_counts(scores, threshold: float) -> dict[str, int]:
    clean, attack = np.asarray(scores).T
    return {
        "pairs": len(clean),
        "both_correct": int(((clean <= threshold) & (attack > threshold)).sum()),
        "attacks_detected": int((attack > threshold).sum()),
        "benign_flagged": int((clean > threshold).sum()),
        "attack_scored_higher": int((attack > clean).sum()),
    }


def inject(clean: str, payloads: list[str], rng: random.Random) -> str:
    payload = rng.choice(payloads)
    lines = clean.splitlines(keepends=True)
    idx = rng.choice([0, len(lines) // 2, len(lines)])
    return (
        "".join(lines[:idx]).rstrip("\n") + "\n" + payload + "\n" + "".join(lines[idx:])
    ).strip()


def fresh_pairs(root: Path, c: dict, seen: set[str]) -> list[Pair]:
    cfg = c["fresh_test"]
    folder = root / "artifacts/paired-sources"
    folder.mkdir(exist_ok=True)
    archive = folder / cfg["python_url"].rsplit("/", 1)[-1]
    if not archive.exists():
        with urlopen(cfg["python_url"], timeout=90) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != cfg["python_sha256"]:
            raise ValueError("Incomplete or unexpected CPython download")
        archive.write_bytes(content)
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != cfg["python_sha256"]:
            raise ValueError("CPython archive checksum mismatch")
    sources = {"cpython": [], "usenet": []}
    with tarfile.open(archive, "r|xz") as tar:
        for member in tar:
            path = Path(member.name)
            if (
                member.isfile()
                and len(path.parts) >= 3
                and path.parts[1] == "Lib"
                and not any(p in ("test", "tests") for p in path.parts)
                and path.suffix == ".py"
                and cfg["min_chars"] <= member.size <= cfg["max_chars"]
            ):
                sources["cpython"].append(
                    tar.extractfile(member).read().decode("utf-8")
                )
    news = fetch_20newsgroups(
        subset="test",
        data_home=folder,
        shuffle=False,
        remove=("headers", "footers", "quotes"),
    )
    sources["usenet"] = news.data
    rng = random.Random(c["seed"])
    pairs = []
    for domain, docs in sources.items():
        kind = "code" if domain == "cpython" else "text"
        families = json.loads(source(root, c["revision"], f"{kind}_attack_test.json"))
        pool = [s for family in families.values() for s in family]
        candidates = sorted(
            docs, key=lambda s: hashlib.sha256(f"{c['seed']}:{s}".encode()).digest()
        )
        selected = 0
        for doc in candidates:
            clean = doc.strip()
            key = normalized(clean)
            if not cfg["min_chars"] <= len(clean) <= cfg["max_chars"] or key in seen:
                continue
            injected = inject(clean, pool, rng)
            if normalized(injected) in seen:
                continue
            seen.update([key, normalized(injected)])
            pairs.append((domain, clean, injected))
            selected += 1
            if selected == cfg["documents_per_source"]:
                break
        if selected != cfg["documents_per_source"]:
            raise ValueError(f"Too few eligible documents from {domain}")
    return pairs


def build_pairs(root: Path, c: dict, reserved: set[str]) -> dict[str, list[Pair]]:
    rng = random.Random(c["seed"])
    splits = {key: [] for key in ("test", "validation", "train")}
    seen_text = set()
    seen_url = set()
    templates = {}
    for partition in ("test", "train"):
        for kind in ("text", "code"):
            families = json.loads(
                source(root, c["revision"], f"{kind}_attack_{partition}.json")
            )
            templates[partition, kind] = [
                t for family in families.values() for t in family
            ]
            if partition == "train":
                assert not set(templates["test", kind]).intersection(
                    templates["train", kind]
                )
        for domain in c["domains"]:
            lines = source(
                root, c["revision"], f"{domain}/{partition}.jsonl"
            ).splitlines()
            rng.shuffle(lines)
            used = 0
            for line in lines:
                row = json.loads(line)
                clean = row["context"]
                clean = ("\n".join(clean) if isinstance(clean, list) else clean).strip()
                key = normalized(clean)
                url = row.get("context_url")
                duplicate = key in seen_text or (url and url in seen_url)
                seen_text.add(key)
                if url:
                    seen_url.add(url)
                if (
                    duplicate
                    or not key
                    or (
                        partition == "train"
                        and any(key in known or known in key for known in reserved)
                    )
                ):
                    continue
                split = partition
                if partition == "train":
                    if (
                        used
                        >= c["train_documents_per_domain"]
                        + c["validation_documents_per_domain"]
                    ):
                        continue
                    if used < c["validation_documents_per_domain"]:
                        split = "validation"
                used += 1
                pool = templates[partition, "code" if domain == "code" else "text"]
                splits[split].append((domain, clean, inject(clean, pool, rng)))
    return splits


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    c = tomllib.loads((root / "configs/paired.toml").read_text())
    q = tomllib.loads((root / "configs/quality.toml").read_text())
    # Prior BIPIA reports remain reserved, including shortened copies of passages.
    old = rows(root, "bipia", q["dataset"], q["dataset_revision"], "data/bipia.parquet")
    reserved = {
        normalized(r["text"]) for r in old if not r["label"] and r["text"].strip()
    }
    data_config = tomllib.loads((root / "configs/data.toml").read_text())
    sibling = root / "configs" / data_config["owasp_data"]
    external = {
        r.key
        for r in external_rows(
            {
                name: (sibling / name).read_bytes()
                for name in ("attacks.jsonl", "golden.jsonl")
            }
        )
    }
    splits = build_pairs(root, c, reserved | external)
    if any(
        normalized(text) in external
        for pairs in splits.values()
        for _, clean, injected in pairs
        for text in (clean, injected)
    ):
        raise ValueError("Sibling RAG text found in paired experiment")
    splits["bipia_test"] = splits.pop("test")
    policy = tomllib.loads((root / "configs/gateway.toml").read_text())
    corpus, _, _ = policy_corpus(root, policy)
    seen = {normalized(text) for group in corpus.values() for _, text, _ in group}
    seen.update(normalized(row["text"]) for row in old)
    seen.update(
        normalized(text)
        for group in splits.values()
        for _, clean, attack in group
        for text in (clean, attack)
    )
    splits["test"] = fresh_pairs(root, c, seen)
    train = splits["train"]
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), norm=None, binary=True),
        LogisticRegression(
            C=c["regularization"], random_state=c["seed"], max_iter=1000
        ),
    )
    clf.fit(
        [s for _, clean, injected in train for s in (clean, injected)],
        [0, 1] * len(train),
    )

    def lexical(part: list[Pair]) -> np.ndarray:
        return clf.predict_proba(
            [s for _, clean, attack in part for s in (clean, attack)]
        )[:, 1].reshape(-1, 2)

    validation = lexical(splits["validation"])
    threshold = choose_threshold([0, 1] * len(validation), validation.ravel())
    gateway, deployed_threshold = checkpoint_scorer(root / "artifacts/gateway")
    report = {
        "config": c,
        "splits": {
            k: {
                "documents": len(v),
                "domains": {
                    domain: sum(d == domain for d, _, _ in v)
                    for domain in (
                        ["cpython", "usenet"] if k == "test" else c["domains"]
                    )
                },
                "pairs_sha256": hashlib.sha256(json.dumps(v).encode()).hexdigest(),
            }
            for k, v in splits.items()
        },
        "validation": paired_counts(validation, threshold),
        "test": {},
    }
    for partition in ("test", "bipia_test"):
        test = splits[partition]
        scores = {
            "paired_tfidf": (threshold, lexical(test)),
            "deployed_gateway": (
                deployed_threshold,
                np.asarray(
                    [[gateway(clean), gateway(attack)] for _, clean, attack in test]
                ),
            ),
        }
        report[partition] = {}
        for name, (cutoff, predictions) in scores.items():
            report[partition][name] = {
                "threshold": cutoff,
                **paired_counts(predictions, cutoff),
                "domains": {
                    domain: paired_counts(
                        predictions[[d == domain for d, _, _ in test]], cutoff
                    )
                    for domain in sorted({d for d, _, _ in test})
                },
            }
    (root / "results/paired.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
