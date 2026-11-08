"""Call a model provider only after the local inspection service returns allow."""

import json
from collections.abc import Callable
from urllib.request import Request, urlopen


def guarded_call[T](
    text: str,
    call_model: Callable[[str], T],
    *,
    endpoint: str = "http://127.0.0.1:8765/v1/check",
    timeout: float = 2.0,
) -> T:
    request = Request(
        endpoint,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError("Inspection service unavailable")
        decision = json.load(response)
    if decision.get("action") == "block":
        raise PermissionError("Prompt flagged for review")
    if decision.get("action") != "allow":
        raise RuntimeError("Inspection service returned an invalid decision")
    return call_model(text)
