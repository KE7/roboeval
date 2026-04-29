"""Regression tests for robo_eval.server_runner.

Focus areas:
  - _poll_health readiness handling
  - install_signal_handlers: ensure importing the module is side-effect free
"""
from __future__ import annotations

import signal
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: tiny in-process HTTP server
# ---------------------------------------------------------------------------

def _make_handler(response_body: bytes, content_type: str = "application/json"):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, *args):  # suppress server logs in test output
            pass

    return Handler


def _start_server(handler, host="127.0.0.1"):
    server = HTTPServer((host, 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


# ---------------------------------------------------------------------------
# Test: import is side-effect free
# ---------------------------------------------------------------------------

def test_import_does_not_install_signal_handlers():
    """Importing server_runner must NOT change SIGINT/SIGTERM handlers."""
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    import robo_eval.server_runner  # noqa: F401 (side-effect free import check)

    assert signal.getsignal(signal.SIGINT) is original_sigint, (
        "Importing server_runner changed the SIGINT handler — "
        "signal.signal() must not be called at module level."
    )
    assert signal.getsignal(signal.SIGTERM) is original_sigterm, (
        "Importing server_runner changed the SIGTERM handler — "
        "signal.signal() must not be called at module level."
    )


# ---------------------------------------------------------------------------
# Test: _poll_health with {"ready": true}
# ---------------------------------------------------------------------------

def test_poll_health_ready_true():
    """_poll_health returns True when the server replies {"ready": true}."""
    from robo_eval.server_runner import _poll_health

    handler = _make_handler(b'{"ready": true}')
    server, port = _start_server(handler)
    try:
        result = _poll_health(f"http://127.0.0.1:{port}", timeout=5.0, interval=0.1)
    finally:
        server.shutdown()

    ready, err = result
    assert ready is True
    assert err == ""


def test_poll_health_status_ok():
    """_poll_health returns (True, '') when the server replies {"status": "ok"} (no 'ready' key).

    Covers fallback readiness behavior for servers that expose status but not ready.
    """
    from robo_eval.server_runner import _poll_health

    handler = _make_handler(b'{"status": "ok"}')
    server, port = _start_server(handler)
    try:
        result = _poll_health(f"http://127.0.0.1:{port}", timeout=5.0, interval=0.1)
    finally:
        server.shutdown()

    ready, err = result
    assert ready is True
    assert err == ""


def test_poll_health_ready_false_does_not_return_early():
    """_poll_health keeps polling when the server replies {"ready": false}."""
    from robo_eval.server_runner import _poll_health

    # Server always returns {"ready": false}; poll should time out and return (False, "timeout").
    handler = _make_handler(b'{"ready": false}')
    server, port = _start_server(handler)
    t0 = time.time()
    try:
        result = _poll_health(f"http://127.0.0.1:{port}", timeout=0.5, interval=0.1)
    finally:
        server.shutdown()

    elapsed = time.time() - t0
    ready, err = result
    assert ready is False
    # Should have polled for at least the timeout duration.
    assert elapsed >= 0.4


def test_poll_health_unreachable_server():
    """_poll_health returns (False, ...) when no server is listening on the given port."""
    from robo_eval.server_runner import _poll_health

    # Port 1 is typically unreachable (privileged) or closed.
    # Use a high-numbered port that is almost certainly unbound.
    result = _poll_health("http://127.0.0.1:19999", timeout=0.3, interval=0.1)
    ready, err = result
    assert ready is False


def test_poll_health_no_ready_no_status():
    """_poll_health does NOT return True for a generic non-ready JSON body.

    {"some_key": "some_value"} has neither 'ready' nor 'status=="ok"'.
    """
    from robo_eval.server_runner import _poll_health

    handler = _make_handler(b'{"some_key": "some_value"}')
    server, port = _start_server(handler)
    try:
        result = _poll_health(f"http://127.0.0.1:{port}", timeout=0.4, interval=0.1)
    finally:
        server.shutdown()

    # Should time out rather than falsely returning True.
    ready, err = result
    assert ready is False


# ---------------------------------------------------------------------------
# Test: install_signal_handlers
# ---------------------------------------------------------------------------

def test_install_signal_handlers_replaces_sigint():
    """install_signal_handlers() should change the SIGINT handler."""
    from robo_eval.server_runner import install_signal_handlers, _signal_handler

    original = signal.getsignal(signal.SIGINT)
    try:
        install_signal_handlers()
        assert signal.getsignal(signal.SIGINT) is _signal_handler
        assert signal.getsignal(signal.SIGTERM) is _signal_handler
    finally:
        # Restore original handler so we don't affect other tests.
        signal.signal(signal.SIGINT, original)
        signal.signal(signal.SIGTERM, original)
