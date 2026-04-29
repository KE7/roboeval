"""ReplicaOrchestrator — run policy-server replicas across available GPUs.

GPU Mode Decision
-----------------
``decide_gpu_mode()`` examines the available CUDA devices and the model's
estimated memory footprint to choose between:

* **SINGLE**   — one GPU (or no multi-GPU benefit): run policy server directly.
* **REPLICAS** — model fits comfortably on each GPU: spawn N subprocesses with
                 ``CUDA_VISIBLE_DEVICES=i``, one per GPU, for N× throughput.

Tensor parallelism is outside the scope of this orchestrator; models that do
not fit comfortably in replica mode fall back to single-process execution.

Environment Variables
---------------------
``FORCE_REPLICAS``   "1" / "true" / "yes" — force replica mode even if auto
                     detection would choose SINGLE.
``MAX_REPLICAS``     integer — cap number of replicas (0 = no cap).
``GPU_IDS``          comma-separated GPU indices — override which GPUs are used
                     (e.g. "0,1,2,3").

Usage
-----
::

    # Auto-mode (typical):
    mode, cfg = decide_gpu_mode(model_size_gb=1.0)  # SmolVLA ~1 GB

    if mode == GPUMode.REPLICAS:
        orch = ReplicaOrchestrator(
            script_module="sims.vla_policies.smolvla_policy",
            model_id="HuggingFaceVLA/smolvla_libero",
            base_port=5200,
            gpu_ids=cfg["gpu_ids"],
        )
        app = make_replica_app(orch, title="SmolVLA Replica Orchestrator")
        uvicorn.run(app, port=5102)
    else:
        policy = SmolVLAPolicy()
        app = make_app(policy, model_id, device="cuda")
        uvicorn.run(app, port=5102)
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
import traceback
from contextlib import asynccontextmanager
from enum import Enum
from itertools import cycle
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU mode enum
# ---------------------------------------------------------------------------


class GPUMode(Enum):
    SINGLE = "single"
    REPLICAS = "replicas"
    # Tensor-parallel execution is not exposed by this orchestrator.


# ---------------------------------------------------------------------------
# decide_gpu_mode
# ---------------------------------------------------------------------------


def decide_gpu_mode(
    model_size_gb: float,
    *,
    force_replicas: bool = False,
    max_replicas: int | None = None,
) -> tuple[GPUMode, dict]:
    """Decide whether to run in single-GPU or multi-GPU replica mode.

    Parameters
    ----------
    model_size_gb:
        Estimated model size in GB (typically measured in bfloat16).
        Use ``sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9``
        after loading, or estimate from parameter count × 2 bytes.
    force_replicas:
        Override auto-detection: always use replica mode (if > 1 GPU available).
        Also read from env var ``FORCE_REPLICAS``.
    max_replicas:
        Maximum number of replicas to spawn.  ``None`` = use all available GPUs.
        Also read from env var ``MAX_REPLICAS`` (0 = no cap).

    Returns
    -------
    (mode, config_dict)

    config_dict keys:
        ``gpu_ids``    — list of CUDA device indices to use.
        ``n_replicas`` — number of replicas (REPLICAS mode) or 1 (SINGLE mode).

    Decision logic
    --------------
    * n_gpus == 1 → SINGLE (no choice)
    * model_size_gb < 0.5 × min_vram_per_gpu AND n_gpus > 1 → REPLICAS
    * otherwise → SINGLE  (model too large for comfortable replication; TP not implemented)
    """
    # --- Read env-var overrides ---
    env_force = os.environ.get("FORCE_REPLICAS", "").lower()
    force_replicas = force_replicas or env_force in ("1", "true", "yes")

    env_max = os.environ.get("MAX_REPLICAS", "")
    if env_max and max_replicas is None:
        max_replicas = int(env_max) or None  # "0" → None (no cap)

    # GPU_IDS env var (comma-separated): override which GPUs are used
    gpu_ids_env = os.environ.get("GPU_IDS", "")
    if gpu_ids_env:
        gpu_ids = [int(x.strip()) for x in gpu_ids_env.split(",") if x.strip()]
    else:
        gpu_ids = None  # determined after device_count() check below

    try:
        import torch
        n_gpus = torch.cuda.device_count()
    except ImportError:
        logger.warning("PyTorch not importable — defaulting to SINGLE mode.")
        return GPUMode.SINGLE, {"gpu_ids": [0], "n_replicas": 1}

    if n_gpus == 0:
        logger.warning("No CUDA GPUs found — defaulting to SINGLE mode (CPU).")
        return GPUMode.SINGLE, {"gpu_ids": [], "n_replicas": 1}

    if gpu_ids is None:
        gpu_ids = list(range(n_gpus))

    # Apply max_replicas cap
    if max_replicas is not None:
        gpu_ids = gpu_ids[:max_replicas]

    # Single GPU — no parallelism possible
    if n_gpus == 1:
        return GPUMode.SINGLE, {"gpu_ids": [gpu_ids[0]], "n_replicas": 1}

    # Query VRAM per GPU
    vram_per_gpu = [
        torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        for i in range(n_gpus)
    ]
    min_vram = min(vram_per_gpu)

    # Model fits comfortably (< 50% of smallest GPU VRAM) → replicate
    fits_comfortably = model_size_gb < 0.5 * min_vram

    if fits_comfortably or force_replicas:
        n_replicas = len(gpu_ids)
        logger.info(
            "decide_gpu_mode → REPLICAS  (model=%.1f GB, min_vram=%.1f GB, "
            "n_replicas=%d, force=%s)",
            model_size_gb, min_vram, n_replicas, force_replicas,
        )
        return GPUMode.REPLICAS, {"gpu_ids": gpu_ids, "n_replicas": n_replicas}

    # Model does not fit comfortably for replication; use single-process mode.
    logger.info(
        "decide_gpu_mode → SINGLE  (model=%.1f GB, min_vram=%.1f GB — "
        "%.0f%% utilisation per GPU; TP not implemented)",
        model_size_gb, min_vram, 100.0 * model_size_gb / min_vram,
    )
    return GPUMode.SINGLE, {"gpu_ids": [gpu_ids[0]], "n_replicas": 1}


# ---------------------------------------------------------------------------
# ReplicaOrchestrator
# ---------------------------------------------------------------------------


class ReplicaOrchestrator:
    """Spins up N policy-server replicas (one OS subprocess per GPU) and routes requests.

    Each replica is a full ``python -m <script_module>`` process with its own CUDA
    context (``CUDA_VISIBLE_DEVICES=<gpu_id>``).  Memory is fully isolated across
    replicas — no NCCL communication, no shared tensors.

    Request routing uses **round-robin**: each ``/predict`` call is forwarded to
    the next replica in sequence.  ``/reset`` is **broadcast** to all replicas
    because each maintains independent per-episode state.

    Parameters
    ----------
    script_module:
        Python module path for the replica server, e.g.
        ``"sims.vla_policies.smolvla_policy"``.  Run as ``python -m <module>``.
    model_id:
        Model identifier passed as ``--model-id`` to each replica.
    base_port:
        Replicas listen on ``base_port``, ``base_port+1``, …, ``base_port+N-1``.
    gpu_ids:
        List of CUDA device indices.  Replica *i* is assigned
        ``CUDA_VISIBLE_DEVICES=gpu_ids[i]``.
    """

    def __init__(
        self,
        script_module: str,
        model_id: str,
        base_port: int,
        gpu_ids: list[int],
    ) -> None:
        self.script_module = script_module
        self.model_id = model_id
        self.base_port = base_port
        self.gpu_ids = list(gpu_ids)
        self.n_replicas = len(self.gpu_ids)
        self.replica_urls: list[str] = [
            f"http://127.0.0.1:{base_port + i}" for i in range(self.n_replicas)
        ]

        self._procs: list[subprocess.Popen] = []
        self._rr_cycle = cycle(range(self.n_replicas))
        # asyncio objects are created lazily on first async use
        self._lock: asyncio.Lock | None = None
        self._client: httpx.AsyncClient | None = None

        logger.info(
            "ReplicaOrchestrator: %d replicas on ports %d–%d (GPUs %s)",
            self.n_replicas,
            base_port,
            base_port + self.n_replicas - 1,
            ", ".join(str(g) for g in self.gpu_ids),
        )

    # ------------------------------------------------------------------
    # Lifecycle (synchronous — safe to call from __init__ / lifespan)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn N replica subprocesses.

        Each subprocess runs::

            CUDA_VISIBLE_DEVICES=<gpu_id> python -m <script_module> \\
                --model-id <model_id> --port <base_port+i> --device cuda
        """
        env_base = os.environ.copy()
        for i, gpu_id in enumerate(self.gpu_ids):
            port = self.base_port + i
            env = {**env_base, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            proc = subprocess.Popen(
                [
                    "python", "-m", self.script_module,
                    "--model-id", self.model_id,
                    "--port", str(port),
                    "--device", "cuda",
                ],
                env=env,
            )
            self._procs.append(proc)
            logger.info(
                "Spawned replica %d: PID=%d  port=%d  CUDA_VISIBLE_DEVICES=%d",
                i, proc.pid, port, gpu_id,
            )

    def stop(self) -> None:
        """Terminate all replica subprocesses and wait for them to exit."""
        for proc in self._procs:
            proc.terminate()
        for proc in self._procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Replica PID=%d did not terminate cleanly; killing.", proc.pid)
                proc.kill()
                proc.wait()
        self._procs.clear()
        logger.info("All %d replicas stopped.", self.n_replicas)

    # ------------------------------------------------------------------
    # Async helpers (must be called from within an asyncio event loop)
    # ------------------------------------------------------------------

    def _ensure_async_state(self) -> None:
        """Lazily initialise asyncio lock and httpx client on first call."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)

    async def wait_ready(self, timeout: float = 300.0) -> None:
        """Poll ``/health`` on all replicas until all report ``ready: true``.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait before raising ``TimeoutError``.

        Raises
        ------
        TimeoutError
            If not all replicas become ready within *timeout* seconds.
        """
        self._ensure_async_state()
        deadline = time.monotonic() + timeout
        logger.info("Waiting for %d replicas to become ready (timeout=%.0fs)…", self.n_replicas, timeout)
        while time.monotonic() < deadline:
            try:
                statuses = await asyncio.gather(
                    *[self._client.get(f"{url}/health") for url in self.replica_urls],
                    return_exceptions=True,
                )
                all_ready = all(
                    isinstance(s, httpx.Response)
                    and s.status_code == 200
                    and s.json().get("ready")
                    for s in statuses
                )
                if all_ready:
                    logger.info("All %d replicas ready.", self.n_replicas)
                    return
            except Exception as exc:
                logger.debug("wait_ready poll error: %s", exc)
            await asyncio.sleep(3.0)
        raise TimeoutError(
            f"Replicas not all ready after {timeout:.0f}s. "
            f"Check replica logs on ports {self.base_port}–{self.base_port + self.n_replicas - 1}."
        )

    async def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Forward a predict request to the next replica (round-robin).

        Parameters
        ----------
        payload:
            The raw JSON body of the ``/predict`` request (a ``PredictRequest`` dict).
        """
        self._ensure_async_state()
        async with self._lock:
            idx = next(self._rr_cycle)
        url = self.replica_urls[idx]
        resp = await self._client.post(f"{url}/predict", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Broadcast a reset to ALL replicas.

        Each replica has independent per-episode state (action chunk buffer,
        etc.), so reset must reach every replica, not just the one that will
        serve the next predict call.
        """
        self._ensure_async_state()
        results = await asyncio.gather(
            *[self._client.post(f"{url}/reset", json=payload) for url in self.replica_urls],
            return_exceptions=True,
        )
        errors = [str(r) for r in results if isinstance(r, Exception)]
        if errors:
            logger.warning("reset: %d/%d replicas errored: %s", len(errors), self.n_replicas, errors)
        return {
            "success": len(errors) == 0,
            "n_replicas": self.n_replicas,
            "errors": errors,
        }

    async def info(self) -> dict[str, Any]:
        """Fetch ``/info`` from replica 0 and annotate with orchestrator metadata."""
        self._ensure_async_state()
        resp = await self._client.get(f"{self.replica_urls[0]}/info")
        data: dict[str, Any] = resp.json()
        data["gpu_mode"] = GPUMode.REPLICAS.value
        data["n_replicas"] = self.n_replicas
        data["gpu_ids"] = self.gpu_ids
        return data


# ---------------------------------------------------------------------------
# Thin FastAPI app wrapping the orchestrator
# ---------------------------------------------------------------------------


def make_replica_app(
    orchestrator: ReplicaOrchestrator,
    title: str = "VLA Replica Orchestrator",
) -> FastAPI:
    """Build a FastAPI app that delegates all requests to *orchestrator*.

    Endpoints
    ---------
    ``GET  /health``  — aggregated ready status across all replicas.
    ``GET  /info``    — metadata from replica 0 + orchestrator fields.
    ``POST /reset``   — broadcast to all replicas.
    ``POST /predict`` — forwarded to next replica (round-robin).
    """

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        # Start replicas synchronously, then wait for them asynchronously
        orchestrator.start()
        try:
            await orchestrator.wait_ready(timeout=300.0)
        except TimeoutError as exc:
            logger.error("Replica startup timed out: %s", exc)
        yield
        # Shutdown
        orchestrator.stop()
        if orchestrator._client is not None:
            await orchestrator._client.aclose()

    app = FastAPI(title=title, lifespan=_lifespan)

    @app.get("/health")
    async def health():
        """Aggregate health from all replicas.

        Returns::

            {
                "ready": bool,
                "gpu_mode": "replicas",
                "n_replicas": int,
                "gpu_ids": list[int]
            }
        """
        orchestrator._ensure_async_state()
        try:
            statuses = await asyncio.gather(
                *[
                    orchestrator._client.get(f"{url}/health")
                    for url in orchestrator.replica_urls
                ],
                return_exceptions=True,
            )
            all_ready = all(
                isinstance(s, httpx.Response)
                and s.status_code == 200
                and s.json().get("ready")
                for s in statuses
            )
        except Exception:
            all_ready = False
        return {
            "ready": all_ready,
            "gpu_mode": GPUMode.REPLICAS.value,
            "n_replicas": orchestrator.n_replicas,
            "gpu_ids": orchestrator.gpu_ids,
        }

    @app.get("/info")
    async def info():
        try:
            return await orchestrator.info()
        except Exception as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})

    @app.post("/reset")
    async def reset_all(request: Request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        return await orchestrator.reset(payload)

    @app.post("/predict")
    async def predict(request: Request):
        try:
            payload = await request.json()
            return await orchestrator.predict(payload)
        except httpx.HTTPStatusError as exc:
            return JSONResponse(
                status_code=exc.response.status_code,
                content={"error": str(exc)},
            )
        except Exception as exc:
            logger.exception("Orchestrator predict failed")
            return JSONResponse(
                status_code=500,
                content={"error": str(exc), "traceback": traceback.format_exc()},
            )

    return app
