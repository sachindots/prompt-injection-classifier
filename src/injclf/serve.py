"""Serve the trained classifier and workbench on loopback."""

import json
import tomllib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

from injclf.predict import injection_score, load_detector


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    metrics = json.loads((root / "results/modernbert_lora.json").read_text())
    model, _ = load_detector()
    if model.config.injection_threshold != metrics["threshold"]:
        raise ValueError("Checkpoint threshold differs from the saved evaluation")
    config = root / "configs/data.toml"
    source = (
        config.parent
        / tomllib.loads(config.read_text())["owasp_data"]
        / "attacks.jsonl"
    )
    case = next(
        r
        for line in source.read_text().splitlines()
        if (r := json.loads(line))["id"] == "a003"
    )
    payload = {
        "endpoint": "/score",
        "metrics": metrics,
        "stats": metrics["data"],
        "case": case,
    }
    page = (
        (root / "web/workbench.html")
        .read_text()
        .replace("__PAYLOAD__", json.dumps(payload).replace("<", "\\u003c"))
        .encode()
    )

    class Handler(BaseHTTPRequestHandler):
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
            if self.path != "/score":
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
                score = injection_score(body.get("text"))
            except (TypeError, ValueError) as exc:
                self.reply(400, "text/plain", str(exc).encode())
                return
            self.reply(200, "application/json", json.dumps({"score": score}).encode())

    with HTTPServer(("127.0.0.1", 8765), Handler) as server:
        print("Workbench running at http://127.0.0.1:8765", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
