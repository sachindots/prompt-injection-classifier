"""Verify the shipped checkpoint on CPU and measure single-text inference."""

import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from sklearn.metrics import f1_score

from injclf.data import load_data
from injclf.predict import injection_score, load_detector


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    report = json.loads((root / "results/modernbert_lora.json").read_text())
    model, _ = load_detector()
    threshold = model.config.injection_threshold
    if threshold != report["threshold"]:
        raise ValueError("Saved model and training report disagree on the threshold")
    splits, stats = load_data(root / "configs/data.toml")
    if stats != report["data"]:
        raise ValueError("Data fingerprints differ from the training run")
    torch.set_num_threads(4)
    injection_score(splits.train[0].text)
    elapsed = []
    checks = {}
    for name in ("test", "held_out"):
        rows = getattr(splits, name)
        predicted = []
        for row in rows:
            start = perf_counter_ns()
            score = injection_score(row.text)
            elapsed.append((perf_counter_ns() - start) / 1e6)
            predicted.append(score > threshold)
        checks[name] = f1_score([row.label for row in rows], predicted)
        if not np.isclose(checks[name], report["splits"][name]["f1"]):
            raise ValueError(
                f"CPU inference differs from the evaluated model on {name}"
            )
    checkpoint = root / "artifacts/modernbert"
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    result = {
        "model": report["model"],
        "cpu": cpu,
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "texts": len(elapsed),
        "p50_ms": float(np.percentile(elapsed, 50)),
        "p95_ms": float(np.percentile(elapsed, 95)),
        "model_directory_bytes": sum(
            f.stat().st_size for f in checkpoint.iterdir() if f.is_file()
        ),
        "cpu_f1": checks,
    }
    (root / "results/modernbert_cpu.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
