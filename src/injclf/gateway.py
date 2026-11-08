"""Inspect text before calling a model provider."""

import json
import logging
from collections.abc import Callable
from http.client import HTTPException
from urllib.request import Request, urlopen

_ENDPOINT = "http://127.0.0.1:8765/v1/check"
_log = logging.getLogger(__name__)


def _inspect(text: str, endpoint: str, timeout: float) -> str:
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
    if not isinstance(decision, dict) or decision.get("action") not in (
        "block",
        "allow",
    ):
        raise RuntimeError("Inspection service returned an invalid decision")
    return decision["action"]


def guarded_call[T](
    text: str,
    call_model: Callable[[str], T],
    *,
    endpoint: str = _ENDPOINT,
    timeout: float = 2.0,
) -> T:
    if _inspect(text, endpoint, timeout) == "block":
        raise PermissionError("Prompt flagged for review")
    return call_model(text)


def shadow_call[T](
    text: str,
    call_model: Callable[[str], T],
    *,
    endpoint: str = _ENDPOINT,
    timeout: float = 2.0,
) -> T:
    """Log the inspection outcome and forward the original text even on a block."""
    try:
        action = _inspect(text, endpoint, timeout)
    except (OSError, HTTPException, ValueError, RuntimeError):
        _log.warning("shadow inspection_error")
    else:
        _log.info("shadow %s", action)
    return call_model(text)
