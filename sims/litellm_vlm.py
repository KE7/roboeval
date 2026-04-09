"""LiteLLM VLM configuration for sim evaluation.

Provides helper functions to configure the VLM endpoint used by
``vlm_hl.vlm_methods`` to route through a litellm proxy server.
"""

import vlm_hl.vlm_methods as vlmi

# Default port for the litellm proxy server
_DEFAULT_PORT = 4000


def setup_litellm_client(host="localhost", port=_DEFAULT_PORT, api_key="not-needed", model_override=None):
    """Configure vlm_methods to use a litellm proxy at ``host:port``.

    Args:
        host: Hostname of the litellm proxy (default: localhost).
        port: Port of the litellm proxy (default: 4000).
        api_key: API key for the proxy (default: "not-needed" for local proxies).
        model_override: If set, overrides the VLM model name for all calls.
    """
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
