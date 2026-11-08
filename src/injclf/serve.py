"""Serve the trained classifier and workbench on loopback."""

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore

import numpy as np
import torch
from scipy.special import expit
from transformers import AutoModel, AutoTokenizer

from injclf.probe import features


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    metrics = json.loads((root / "results/gateway.json").read_text())
    stats = metrics["data"]
    model_path = root / "artifacts/gateway"
    with (model_path / "model.safetensors").open("rb") as weights:
        if (
            hashlib.file_digest(weights, "sha256").hexdigest()
            != metrics["weights_sha256"]
        ):
            raise ValueError("Recalibrate the gateway policy for these weights")
    model = AutoModel.from_pretrained(
        model_path, attn_implementation="sdpa", reference_compile=False
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    head_bytes = (model_path / "head.json").read_bytes()
    if hashlib.sha256(head_bytes).hexdigest() != metrics["head_sha256"]:
        raise ValueError("Gateway policy and classifier head disagree")
    head = json.loads(head_bytes)
    if (
        head["threshold"] != metrics["threshold"]
        or head["layers"] != model.config.num_hidden_layers
    ):
        raise ValueError("Gateway head and saved policy disagree")
    coefficients = np.asarray(head["weights"])

    def injection_score(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter nonempty text")
        depth = head["layers"]
        vectors = features(model, tokenizer, text, [depth])[depth]
        return float(expit(vectors @ coefficients + head["bias"]).max())

    if torch.cuda.is_available():
        model.to("cuda")
    injection_score("Warm up inference")
    inference_slot = BoundedSemaphore(1)
    payload = {
        "endpoint": "/v1/check",
        "metrics": metrics,
        "stats": stats,
        "case": metrics["case"],
    }
    comparison = {}
    for key in ("gateway", "refined", "horizon"):
        report = root / f"results/quality_{key}.json"
        if report.exists():
            comparison[key] = json.loads(report.read_text())
    payload["quality"] = comparison
    benchmark = root / "results/gateway_latency.json"
    if benchmark.exists():
        latency = json.loads(benchmark.read_text())
        if latency.get("head_sha256") == metrics["head_sha256"]:
            payload["latency"] = latency
    page = (
        (root / "web/workbench.html")
        .read_text()
        .replace("__PAYLOAD__", json.dumps(payload).replace("<", "\\u003c"))
        .encode()
    )

    class Handler(BaseHTTPRequestHandler):
        timeout = 5

        def reply(self, status: int, mime: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/":
                self.send_error(404)
                return
            self.reply(200, "text/html; charset=utf-8", page)

        def do_POST(self) -> None:
            if self.path not in ("/score", "/v1/check"):
                self.send_error(404)
                return
            if self.headers.get_content_type() != "application/json":
                self.send_error(415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    self.reply(413, "text/plain", b"Use a request smaller than 64 KiB.")
                    return
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise TypeError("Expected a JSON object")
                if not inference_slot.acquire(timeout=0.1):
                    self.reply(503, "text/plain", b"Classifier busy. Retry later.")
                    return
                try:
                    if self.path == "/score":
                        if not (
                            root / "artifacts/modernbert/model.safetensors"
                        ).is_file():
                            self.reply(
                                503,
                                "text/plain",
                                b"Legacy checkpoint unavailable. Use /v1/check.",
                            )
                            return
                        from injclf.predict import injection_score as legacy_score

                        score = legacy_score(body.get("text"))
                    else:
                        score = injection_score(body.get("text"))
                finally:
                    inference_slot.release()
            except (TypeError, ValueError) as exc:
                self.reply(400, "text/plain", str(exc).encode())
                return
            decision = {"score": score}
            if self.path == "/v1/check":
                cutoff = metrics["threshold"]
                decision.update(
                    action="block" if score > cutoff else "allow",
                    threshold=cutoff,
                    model=metrics["model"],
                    device=torch.cuda.get_device_name()
                    if model.device.type == "cuda"
                    else "CPU",
                )
            self.reply(200, "application/json", json.dumps(decision).encode())

    with ThreadingHTTPServer(("127.0.0.1", 8765), Handler) as server:
        print("Workbench running at http://127.0.0.1:8765", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
