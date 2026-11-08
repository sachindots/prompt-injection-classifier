import io
import json
from unittest.mock import Mock

import pytest

from injclf.gateway import guarded_call, shadow_call
from injclf.policy import choose_cutoff


@pytest.mark.parametrize("action", ["block", "unknown", None])
def test_provider_is_unreachable_without_allow(monkeypatch, action):
    response = io.BytesIO(json.dumps({"action": action}).encode())
    response.status = 200
    monkeypatch.setattr("injclf.gateway.urlopen", lambda *args, **kwargs: response)
    provider = Mock()
    with pytest.raises((PermissionError, RuntimeError)):
        guarded_call("text", provider)
    provider.assert_not_called()


def test_allow_calls_provider_once_with_inspected_text(monkeypatch):
    response = io.BytesIO(b'{"action":"allow"}')
    response.status = 200
    monkeypatch.setattr("injclf.gateway.urlopen", lambda *args, **kwargs: response)
    provider = Mock(return_value="reply")
    assert guarded_call("original", provider) == "reply"
    provider.assert_called_once_with("original")


def test_cutoff_respects_each_benign_group_and_strict_comparison():
    groups = [[0.9, 0.5, 0.5, 0.1], [0.8, 0.8, 0.2, 0.1]]
    threshold = choose_cutoff(groups, 0.25)
    assert threshold == 0.8
    assert all(sum(s > threshold for s in group) <= 1 for group in groups)


def test_timeout_never_calls_provider(monkeypatch):
    def unavailable(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr("injclf.gateway.urlopen", unavailable)
    provider = Mock()
    with pytest.raises(TimeoutError):
        guarded_call("text", provider)
    provider.assert_not_called()


@pytest.mark.parametrize(
    ("body", "event"),
    [
        (b'{"action":"allow"}', "allow"),
        (b'{"action":"block"}', "block"),
        (b'{"action":"unknown"}', "inspection_error"),
        (b"null", "inspection_error"),
        (b"not-json", "inspection_error"),
        (None, "inspection_error"),
    ],
)
def test_shadow_logs_outcome_and_forwards_once(monkeypatch, caplog, body, event):
    if body is None:
        send = Mock(side_effect=TimeoutError)
    else:
        response = io.BytesIO(body)
        response.status = 200
        send = Mock(return_value=response)
    monkeypatch.setattr("injclf.gateway.urlopen", send)
    provider = Mock(return_value="reply")
    with caplog.at_level("INFO", logger="injclf.gateway"):
        assert shadow_call("private prompt", provider) == "reply"
    assert caplog.messages == [f"shadow {event}"]
    provider.assert_called_once_with("private prompt")
