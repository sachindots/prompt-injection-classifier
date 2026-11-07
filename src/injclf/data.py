"""Build classifier partitions from deepset and the external RAG evaluation."""

import argparse
import hashlib
import json
import tomllib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Example:
    text: str
    label: int
    origin: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(f"{self.origin}: expected nonempty text")
        if type(self.label) is not int or self.label not in (0, 1):
            raise ValueError(f"{self.origin}: expected binary label")

    @property
    def key(self) -> str:
        return " ".join(self.text.casefold().split())


@dataclass
class Splits:
    train: list[Example]
    validation: list[Example]
    test: list[Example]
    held_out: list[Example]


def external_rows(sources: Mapping[str, bytes]) -> Iterator[Example]:
    for filename, content in sources.items():
        for line in content.splitlines():
            row = json.loads(line)
            origin = f"{filename}/{row['id']}"
            document = row.get("injected_document")
            if filename == "golden.jsonl":
                yield Example(row["question"], 0, origin + "/question")
            elif document is not None:
                yield Example(document, 1, origin + "/injected_document")
                yield Example(row["question"], 0, origin + "/question")
            elif row["category"] == "indirect_injection":
                raise ValueError(f"{origin}: missing injected document")
            else:
                yield Example(row["question"], 1, origin + "/question")


def partition(
    train: Iterable[Example],
    test: Iterable[Example],
    external: Iterable[Example],
    seed: int,
    validation_fraction: float,
) -> Splits:
    seen: dict[str, Example] = {}

    def reserve(rows: Iterable[Example]) -> list[Example]:
        unique = []
        for row in rows:
            prior = seen.get(row.key)
            if prior is None:
                seen[row.key] = row
                unique.append(row)
            elif row.label != prior.label:
                raise ValueError(
                    f"conflicting labels at {prior.origin} and {row.origin}"
                )
        return unique

    held_out = reserve(external)
    test_rows = reserve(test)
    eligible = reserve(train)
    train_rows, validation = train_test_split(
        eligible,
        test_size=validation_fraction,
        random_state=seed,
        stratify=[row.label for row in eligible],
    )
    return Splits(train_rows, validation, test_rows, held_out)


def load_data(config_path: Path) -> tuple[Splits, dict]:
    config = tomllib.loads(config_path.read_text())
    dataset = load_dataset(
        "deepset/prompt-injections", revision=config["dataset_revision"]
    )
    directory = config_path.parent / config["owasp_data"]
    sources = {
        name: (directory / name).read_bytes()
        for name in ("golden.jsonl", "attacks.jsonl")
    }

    def upstream(split: str) -> Iterator[Example]:
        for i, row in enumerate(dataset[split]):
            yield Example(row["text"], row["label"], f"deepset/{split}/{i}")

    splits = partition(
        upstream("train"),
        upstream("test"),
        external_rows(sources),
        config["seed"],
        config["validation_fraction"],
    )
    stats = {
        "dataset_revision": config["dataset_revision"],
        "seed": config["seed"],
        "validation_fraction": config["validation_fraction"],
        "upstream_rows": {name: len(rows) for name, rows in dataset.items()},
        "source_sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()
        },
        "splits": {},
    }
    for name, rows in vars(splits).items():
        labels = Counter(row.label for row in rows)
        payload = json.dumps([(r.origin, r.text, r.label) for r in rows]).encode()
        stats["splits"][name] = {
            "total": len(rows),
            "benign": labels[0],
            "injection": labels[1],
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return splits, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.toml"))
    args = parser.parse_args()
    _, stats = load_data(args.config)
    output = args.config.resolve().parents[1] / "results/data_stats.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
