"""LiteLLM VLM configuration for sim evaluation.

Provides helper functions to configure the VLM endpoint used by
``vlm_hl.vlm_methods`` to route through a litellm proxy server.
"""

import logging

import requests

import vlm_hl.vlm_methods as vlmi

# Default port for the litellm proxy server
_DEFAULT_PORT = 4000
_REACHABILITY_TIMEOUT = 3.0

logger = logging.getLogger(__name__)


def _assert_litellm_reachable(host: str, port: int) -> None:
    """Probe the litellm proxy at startup so users don't burn a sim init
    only to fail on the first VLM call mid-episode.
    """
    url = f"http://{host}:{port}/v1/models"
    try:
        resp = requests.get(url, timeout=_REACHABILITY_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"litellm proxy not reachable at http://{host}:{port}.\n"
            f"  Start it with `bash scripts/start_vlm.sh`, or pass --no-vlm "
            f"if you don't need VLM planning.\n"
            f"  Underlying error: {exc}"
        ) from exc
    if resp.status_code >= 500:
        raise RuntimeError(
            f"litellm proxy at http://{host}:{port} is reachable but returned "
            f"HTTP {resp.status_code}. Check `litellm` logs."
        )


def setup_litellm_client(host="localhost", port=_DEFAULT_PORT, api_key="not-needed", model_override=None):
    """Configure vlm_methods to use a litellm proxy at ``host:port``.

    Args:
        host: Hostname of the litellm proxy (default: localhost).
        port: Port of the litellm proxy (default: 4000).
        api_key: API key for the proxy (default: "not-needed" for local proxies).
        model_override: If set, overrides the VLM model name for all calls.
    """
    _assert_litellm_reachable(host, port)
    api_base = f"http://{host}:{port}/v1"
    vlmi.setup_litellm(api_base=api_base, api_key=api_key, model_override=model_override)


def setup_litellm_from_endpoint(endpoint, model_override=None):
    """Configure vlm_methods from an endpoint string like ``host:port``.

    Args:
        endpoint: Either ``"host:port"`` or just ``"host"`` (uses default port 4000).
        model_override: If set, overrides the VLM model name for all calls.
    """
    if ":" in endpoint:
        host, port = endpoint.rsplit(":", 1)
        port = int(port)
    else:
        host = endpoint
        port = _DEFAULT_PORT
    setup_litellm_client(host=host, port=port, model_override=model_override)
