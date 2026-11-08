"""Measure warmed end-to-end loopback decisions, one request at a time."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter_ns
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from injclf.data import load_data


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    splits, _ = load_data(root / "configs/data.toml")
    url = "http://127.0.0.1:8765/v1/check"

    def request(text):
        req = Request(
            url,
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=10) as response:
            return json.load(response)

    request(splits.train[0].text)
    samples = []
    actions = {}
    for name in ("test", "held_out"):
        rows = getattr(splits, name)
        detected = flagged = 0
        for row in rows:
            start = perf_counter_ns()
            result = request(row.text)
            samples.append((perf_counter_ns() - start) / 1e6)
            if result["action"] == "block":
                detected += row.label
                flagged += 1 - row.label
        actions[name] = {"detected_attacks": detected, "false_positives": flagged}
    policy = json.loads((root / "results/gateway.json").read_text())
    for part, observed in actions.items():
        assert all(v == policy["splits"][part][k] for k, v in observed.items()), part
    metrics = {
        "device": result["device"],
        "head_sha256": policy["head_sha256"],
        "transport": "loopback HTTP, sequential, warmed",
        "requests": len(samples),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "actions": actions,
    }

    def timed(text):
        start = perf_counter_ns()
        try:
            request(text)
        except HTTPError as exc:
            if exc.code == 503:
                return None
            raise
        return (perf_counter_ns() - start) / 1e6

    start = perf_counter_ns()
    with ThreadPoolExecutor(max_workers=4) as pool:
        batch = list(pool.map(timed, [r.text for r in splits.test]))
    duration = (perf_counter_ns() - start) / 1e9
    completed = [s for s in batch if s is not None]
    metrics["four_clients"] = {
        "requests": len(batch),
        "completed": len(completed),
        "busy": len(batch) - len(completed),
        "completed_per_second": len(completed) / duration,
        "p95_ms": float(np.percentile(completed, 95)),
    }
    (root / "results/gateway_latency.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(metrics)


if __name__ == "__main__":
    main()
