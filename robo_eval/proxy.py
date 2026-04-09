"""
VLA Round-Robin Proxy Server.

Lightweight HTTP reverse proxy that distributes VLA requests across N identical
backend servers. Provides transparent load balancing with health checking and
automatic backend failover.

All eval processes, sim workers, and orchestrators talk to a single proxy URL.
The proxy forwards requests round-robin to healthy backends.

Usage:
    python -m robo_eval.proxy --backends http://localhost:5100 http://localhost:5101
    python -m robo_eval.proxy --backends http://localhost:5100 http://localhost:5101 --port 5200

Endpoints (proxied):
    GET  /health   -> aggregated health across all backends
    GET  /info     -> forwarded to first healthy backend (cached)
    POST /predict  -> round-robin to next healthy backend
    POST /reset    -> broadcast to ALL backends

Proxy-only endpoints:
    GET  /backends -> list all backends and their health status
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend tracking
# ---------------------------------------------------------------------------

@dataclass
class Backend:
    """Tracks a single VLA backend server."""
    url: str
    healthy: bool = False
    last_check: float = 0.0
    last_error: str = ""
    requests_served: int = 0
    # Grace period: don't mark as unhealthy until this time
    grace_until: float = field(default_factory=lambda: time.time() + 30.0)


class BackendPool:
    """Manages a pool of VLA backends with round-robin selection.

    Thread-safe: uses an asyncio lock for the round-robin counter and health
    state mutations. The counter is a simple integer modulo len(backends).
    """

    def __init__(self, urls: list[str]):
        self.backends = [Backend(url=u.rstrip("/")) for u in urls]
        self._rr_index = 0
        self._lock = asyncio.Lock()
        self._info_cache: Optional[dict] = None
        self._info_cache_time: float = 0.0
        self._info_cache_ttl: float = 60.0  # seconds

    @property
    def healthy_backends(self) -> list[Backend]:
        return [b for b in self.backends if b.healthy]

    @property
    def any_healthy(self) -> bool:
        return any(b.healthy for b in self.backends)

    async def next_healthy(self) -> Optional[Backend]:
        """Select the next healthy backend using round-robin.

        Returns None if no healthy backends are available.
        """
        async with self._lock:
            n = len(self.backends)
            if n == 0:
                return None

            # Try each backend starting from current index
            for _ in range(n):
                backend = self.backends[self._rr_index % n]
                self._rr_index = (self._rr_index + 1) % n
                if backend.healthy:
                    backend.requests_served += 1
                    return backend

            return None

    async def mark_unhealthy(self, backend: Backend, error: str = "") -> None:
        """Mark a backend as unhealthy (e.g., after a failed request)."""
        async with self._lock:
            backend.healthy = False
            backend.last_error = error
            logger.warning("Backend %s marked unhealthy: %s", backend.url, error)

    async def check_health(self, client: httpx.AsyncClient) -> None:
        """Poll all backends for health status."""
        now = time.time()
        for backend in self.backends:
            try:
                resp = await client.get(f"{backend.url}/health", timeout=30.0)
                data = resp.json()
                was_healthy = backend.healthy
                backend.healthy = data.get("ready", False)
                backend.last_check = now
                backend.last_error = "" if backend.healthy else data.get("error", "not ready")
                if backend.healthy and not was_healthy:
                    logger.info("Backend %s became healthy", backend.url)
                elif not backend.healthy and was_healthy:
                    logger.warning("Backend %s became unhealthy: %s", backend.url, backend.last_error)
            except Exception as e:
                # During grace period, don't mark as unhealthy (backend may still be starting)
                if now < backend.grace_until:
                    continue
                if backend.healthy:
                    logger.warning("Backend %s health check failed: %s", backend.url, e)
                backend.healthy = False
                backend.last_check = now
                backend.last_error = str(e)

    def get_cached_info(self) -> Optional[dict]:
        """Return cached /info response if still valid."""
        if self._info_cache and (time.time() - self._info_cache_time) < self._info_cache_ttl:
            return self._info_cache

    def set_cached_info(self, info: dict) -> None:
        """Cache an /info response."""
        self._info_cache = info
        self._info_cache_time = time.time()


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_pool: Optional[BackendPool] = None
_client: Optional[httpx.AsyncClient] = None
_health_task: Optional[asyncio.Task] = None

# CLI args
_cli_backends: list[str] = []
_cli_port: int = 5200
_cli_health_interval: float = 10.0


# ---------------------------------------------------------------------------
# Background health checker
# ---------------------------------------------------------------------------

async def _health_check_loop(pool: BackendPool, client: httpx.AsyncClient, interval: float):
    """Periodically check backend health."""
    while True:
        try:
            await pool.check_health(client)
        except Exception as e:
            logger.error("Health check loop error: %s", e)
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool, _client, _health_task

    _pool = BackendPool(_cli_backends)
    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )

    # Initial health check
    await _pool.check_health(_client)

    # Start background health checker
    _health_task = asyncio.create_task(
        _health_check_loop(_pool, _client, _cli_health_interval)
    )

    healthy_count = len(_pool.healthy_backends)
    total_count = len(_pool.backends)
    logger.info(
        "VLA Proxy started: %d/%d backends healthy, port %d",
        healthy_count, total_count, _cli_port,
    )

    yield

    # Cleanup
    if _health_task:
        _health_task.cancel()
        try:
            await _health_task
        except asyncio.CancelledError:
            pass
    if _client:
        await _client.aclose()


app = FastAPI(title="VLA Round-Robin Proxy", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Proxy endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Aggregated health: ready if any backend is healthy."""
    healthy = _pool.healthy_backends
    total = len(_pool.backends)

    if not healthy:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "backends_healthy": 0,
                "backends_total": total,
                "error": "No healthy backends",
            },
        )

    # Return model_id from cached /info if available, else from first healthy backend
    cached_info = _pool.get_cached_info()
    model_id = cached_info.get("model_id", "") if cached_info else ""

    return {
        "ready": True,
        "model_id": model_id,
        "backends_healthy": len(healthy),
        "backends_total": total,
    }


@app.get("/info")
async def info():
    """Forward to first healthy backend. Response is cached for 60s."""
    # Check cache first
    cached = _pool.get_cached_info()
    if cached:
        return cached

    backend = await _pool.next_healthy()
    if not backend:
        return JSONResponse(
            status_code=503,
            content={"error": "No healthy backends available"},
        )

    try:
        resp = await _client.get(f"{backend.url}/info", timeout=5.0)
        data = resp.json()
        _pool.set_cached_info(data)
        return data
    except Exception as e:
        await _pool.mark_unhealthy(backend, str(e))
        return JSONResponse(
            status_code=502,
            content={"error": f"Backend {backend.url} failed: {e}"},
        )


@app.post("/predict")
async def predict(request: Request):
    """Round-robin /predict to the next healthy backend.

    No retry on failure: /predict is idempotent but retrying would add
    latency and the observation may already be stale. Better to let the
    caller (env_wrapper.py) handle the error and abort the episode.
    """
    backend = await _pool.next_healthy()
    if not backend:
        return JSONResponse(
            status_code=503,
            content={"error": "No healthy backends available"},
        )

    # Read the raw request body once and forward it
    body = await request.body()

    try:
        resp = await _client.post(
            f"{backend.url}/predict",
            content=body,
            headers={"Content-Type": "application/json"},
            timeout=60.0,  # VLA inference can take up to ~30s
        )
        # Catch non-JSON responses so backend errors aren't masked
        try:
            content = resp.json()
        except (ValueError, UnicodeDecodeError):
            raw = resp.text
            await _pool.mark_unhealthy(backend, f"Non-JSON response: {raw}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": f"Backend {backend.url} returned non-JSON response "
                             f"(HTTP {resp.status_code}): {raw}"
                },
            )
        return JSONResponse(
            status_code=resp.status_code,
            content=content,
        )
    except Exception as e:
        await _pool.mark_unhealthy(backend, str(e))
        return JSONResponse(
            status_code=502,
            content={"error": f"Backend {backend.url} failed: {e}"},
        )


@app.post("/reset")
async def reset():
    """Broadcast /reset to ALL backends.

    SmolVLA's select_action() manages an internal action queue that must be
    cleared at episode boundaries. Since round-robin means we don't know
    which backend served the last /predict, we broadcast to all.

    Returns success if at least one backend acknowledged the reset.
    """
    if not _pool.any_healthy:
        return JSONResponse(
            status_code=503,
            content={"error": "No healthy backends available"},
        )

    results = []
    for backend in _pool.healthy_backends:
        try:
            resp = await _client.post(
                f"{backend.url}/reset",
                timeout=5.0,
            )
            results.append({"url": backend.url, "status": resp.status_code})
        except Exception as e:
            results.append({"url": backend.url, "error": str(e)})

    any_success = any(r.get("status") == 200 for r in results)
    return {
        "success": any_success,
        "backends": results,
    }


@app.post("/reload")
async def reload_model(request: Request):
    """Broadcast /reload to ALL backends.

    Used to hot-swap model checkpoints across all VLA replicas when
    switching LIBERO suites with OpenVLA (which has separate fine-tuned
    checkpoints per suite).

    Blocks until all backends have completed the reload. May take several
    minutes if models need to be loaded from disk.
    """
    body = await request.body()

    results = []
    for backend in _pool.backends:
        try:
            resp = await _client.post(
                f"{backend.url}/reload",
                content=body,
                headers={"Content-Type": "application/json"},
                timeout=600.0,  # Model loading can take several minutes
            )
            data = resp.json()
            results.append({"url": backend.url, "status": resp.status_code, **data})
        except Exception as e:
            results.append({"url": backend.url, "error": str(e)})

    any_success = any(r.get("success") for r in results)

    # Invalidate info cache since model may have changed
    _pool._info_cache = None

    return {
        "success": any_success,
        "backends": results,
    }


@app.get("/backends")
async def list_backends():
    """List all backends and their health status (proxy-only endpoint)."""
    return {
        "backends": [
            {
                "url": b.url,
                "healthy": b.healthy,
                "last_check": b.last_check,
                "last_error": b.last_error,
                "requests_served": b.requests_served,
            }
            for b in _pool.backends
        ],
        "healthy_count": len(_pool.healthy_backends),
        "total_count": len(_pool.backends),
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    parser = argparse.ArgumentParser(
        description="VLA Round-Robin Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Proxy two local VLA servers
  python -m robo_eval.proxy --backends http://localhost:5100 http://localhost:5101

  # Custom port and health interval
  python -m robo_eval.proxy --backends http://localhost:5100 http://localhost:5101 \\
      --port 5200 --health-interval 5
""",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        required=True,
        help="URLs of VLA backend servers (e.g., http://localhost:5100 http://localhost:5101)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5200,
        help="Port to serve on (default: 5200)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--health-interval",
        type=float,
        default=10.0,
        dest="health_interval",
        help="Seconds between health checks (default: 10)",
    )
    args = parser.parse_args()

    global _cli_backends, _cli_port, _cli_health_interval
    _cli_backends = args.backends
    _cli_port = args.port
    _cli_health_interval = args.health_interval

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    n = len(args.backends)
    logger.info("[vla-proxy] Starting VLA proxy on %s:%d", args.host, args.port)
    logger.info("[vla-proxy] Backends (%d):", n)
    for url in args.backends:
        logger.info("  - %s", url)
    logger.info("[vla-proxy] Health check interval: %ss", args.health_interval)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
