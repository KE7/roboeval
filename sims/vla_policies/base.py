"""Abstract base class and FastAPI factory for VLA policy servers.

All servers expose four standard endpoints::

    GET  /health  → {status, ready, model_id, gpu_mode, device}
    GET  /info    → {name, model_id, action_space, state_dim, action_chunk_size, ...}
    POST /reset   → {success}           # per-episode state reset
    POST /predict → {actions, ...}      # inference

To add a policy server, subclass VLAPolicyBase, implement
load_model/predict/get_info, and call make_app().

Batching
--------
Pass ``max_batch_size > 1`` to ``make_app()`` to enable transparent server-side
batching.  Incoming ``/predict`` requests are accumulated for up to
``max_wait_ms`` milliseconds (or until ``max_batch_size`` requests are
pending) and dispatched together via ``policy.predict_batch()``.
Callers see no API change — single ``/predict`` requests work as before.

When ``max_batch_size == 1`` (the default), no queue overhead is added.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import traceback
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from sims.vla_policies.vla_schema import PredictRequest, VLAObservation

try:
    from roboeval.specs import ActionObsSpec
except ImportError:
    ActionObsSpec = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def detect_lerobot_image_transform(model_id: str, policy: Any = None) -> str:
    """Return ``'flip_hw'`` when the model needs a LIBERO 180° flip, else ``'none'``.

    Two conditions must both hold:
    1. ``lerobot.processor.env_processor.LiberoProcessorStep`` is importable.
    2. The model ID or config contains ``"libero"`` (case-insensitive).
    """
    short = model_id.split("/")[-1] if "/" in model_id else model_id
    try:
        from lerobot.processor.env_processor import LiberoProcessorStep
    except ImportError:
        logger.info("[%s] LiberoProcessorStep unavailable — image_transform=none", short)
        return "none"

    indicators = ["libero" in model_id.lower()]
    if policy is not None and hasattr(policy, "config"):
        cfg = policy.config
        for attr in ("pretrained_path", "repo_id"):
            val = getattr(cfg, attr, "") or ""
            indicators.append("libero" in val.lower())

    if any(indicators):
        logger.info("[%s] Detected image_transform=flip_hw (LIBERO 180° cameras)", short)
        return "flip_hw"

    logger.info("[%s] No LIBERO indicators — image_transform=none", short)
    return "none"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class VLAPolicyBase(ABC):
    """Abstract base for all VLA policy HTTP servers.

    Subclasses **must** implement:
      * ``load_model(model_id, device, **kwargs)`` — load weights; set
        ``self.ready = True`` on success.
      * ``predict(obs: VLAObservation) → list[list[float]]`` — inference.
      * ``get_info() → dict`` — model metadata (returned from ``GET /info``).

    Subclasses **may** override:
      * ``reset()``              — per-episode reset (default: no-op).
      * ``predict_batch()``      — true GPU-batched inference (default: serial
                                   fallback over ``predict()``).
      * ``load_in_background``   — set ``True`` to load in a daemon thread so
                                   the server is immediately reachable while the
                                   model loads (returns 503 until ready).
      * ``supports_batching``    — set ``True`` when ``predict_batch()`` is
                                   overridden with a true batched implementation.
    """

    load_in_background: bool = False
    #: Set True in subclasses that override predict_batch() with a real implementation.
    supports_batching: bool = False

    def __init__(self) -> None:
        self.ready: bool = False
        self.load_error: str = ""
        self.model_id: str = ""

    @abstractmethod
    def load_model(self, model_id: str, device: str, **kwargs) -> None:
        """Load model weights.  Must set ``self.ready = True`` on success."""

    @abstractmethod
    def predict(self, obs: VLAObservation) -> list[list[float]]:
        """Run inference; return a list of action vectors (each a list of floats)."""

    @abstractmethod
    def get_info(self) -> dict:
        """Return the ``/info`` metadata dict."""

    def reset(self) -> None:
        """Reset per-episode state (default: no-op)."""
        return None

    def get_action_spec(self) -> dict[str, ActionObsSpec] | None:
        """Return the action spec (what this VLA produces).

        Returns a mapping of component name → ``ActionObsSpec``, or ``None`` if this
        server does not declare a typed spec (legacy mode).  Declared specs are
        serialized into ``GET /info`` under the ``action_spec`` key.
        """
        return None

    def get_observation_spec(self) -> dict[str, ActionObsSpec] | None:
        """Return the observation spec (what this VLA expects as input).

        Returns a mapping of component name → ``ActionObsSpec``, or ``None`` if this
        server does not declare a typed spec (legacy mode).  Declared specs are
        serialized into ``GET /info`` under the ``observation_spec`` key.
        """
        return None

    def predict_batch(self, obs_list: list[VLAObservation]) -> list[list[list[float]]]:
        """Batched inference.  Default: serial fallback over ``predict()``.

        Subclasses should override this method (and set ``supports_batching = True``)
        to implement true GPU-batched inference with a single forward pass for
        all observations in *obs_list*.

        Parameters
        ----------
        obs_list:
            List of ``VLAObservation`` objects to process together.

        Returns
        -------
        List of per-sample action lists — same format as ``predict()`` but for
        each element of *obs_list*.  Length == ``len(obs_list)``.
        """
        return [self.predict(obs) for obs in obs_list]

    # -- internal background-load wrapper --

    def _load_bg(self, model_id: str, device: str, **kwargs) -> None:
        try:
            self.load_model(model_id, device, **kwargs)
        except Exception as exc:
            self.load_error = str(exc)
            logger.error("Model load failed: %s\n%s", exc, traceback.format_exc())


# ---------------------------------------------------------------------------
# Async batch queue
# ---------------------------------------------------------------------------


class BatchQueue:
    """Asyncio-based request accumulator for transparent server-side batching.

    Incoming observations are held for up to *max_wait_ms* milliseconds (or
    until *max_batch_size* requests are pending) then dispatched together via
    ``policy.predict_batch()``.

    Usage::

        queue = BatchQueue(policy, max_batch_size=8, max_wait_ms=15.0)
        # Start in the asyncio event loop:
        asyncio.create_task(queue.drain_loop())
        # In /predict endpoint:
        actions = await queue.submit(req.obs)
    """

    def __init__(
        self,
        policy: VLAPolicyBase,
        max_batch_size: int = 8,
        max_wait_ms: float = 20.0,
    ) -> None:
        self._policy = policy
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[tuple[VLAObservation, asyncio.Future]] = asyncio.Queue()

    async def submit(self, obs: VLAObservation) -> list[list[float]]:
        """Enqueue *obs* and await its result.

        Returns the same value as ``policy.predict(obs)`` would.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[list[list[float]]] = loop.create_future()
        await self._queue.put((obs, fut))
        return await fut

    async def drain_loop(self) -> None:
        """Background coroutine — runs until cancelled.

        Waits for the first request, then collects additional requests for up to
        *max_wait_ms* ms (or until *max_batch_size* items are pending), then
        calls ``policy.predict_batch()`` in a thread executor and routes results
        back to each caller's future.
        """
        loop = asyncio.get_running_loop()
        while True:
            # Block until at least one observation is available
            first_obs, first_fut = await self._queue.get()
            batch: list[tuple[VLAObservation, asyncio.Future]] = [(first_obs, first_fut)]

            # Collect more items within the deadline
            deadline = loop.time() + self.max_wait_ms / 1000.0
            while len(batch) < self.max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except TimeoutError:
                    break

            obs_list = [obs for obs, _ in batch]
            futs = [fut for _, fut in batch]

            logger.info("BatchQueue.drain_loop: dispatching batch_size=%d", len(batch))
            # Run batched inference in thread pool so the event loop stays free
            try:
                actions_list: list[list[list[float]]] = await loop.run_in_executor(
                    None, self._policy.predict_batch, obs_list
                )
                for fut, actions in zip(futs, actions_list, strict=False):
                    if not fut.done():
                        fut.set_result(actions)
            except Exception as exc:
                logger.exception(
                    "BatchQueue.drain_loop: predict_batch failed (batch=%d)", len(batch)
                )
                for fut in futs:
                    if not fut.done():
                        fut.set_exception(exc)


# ---------------------------------------------------------------------------
# FastAPI factory
# ---------------------------------------------------------------------------


def make_app(
    policy: VLAPolicyBase,
    model_id: str,
    device: str = "cuda",
    title: str = "VLA Policy Server",
    max_batch_size: int = 1,
    max_wait_ms: float = 20.0,
    **load_kwargs: Any,
) -> FastAPI:
    """Build and return a FastAPI app with the four standard endpoints wired to *policy*.

    Model loading happens inside the ASGI lifespan (synchronously or in a background
    daemon thread, controlled by ``policy.load_in_background``).

    Parameters
    ----------
    policy:
        ``VLAPolicyBase`` instance to wrap.
    model_id:
        Model identifier forwarded to ``policy.load_model()``.
    device:
        Device string forwarded to ``policy.load_model()`` and reported in ``/health``.
    title:
        FastAPI app title (shown in ``/docs``).
    max_batch_size:
        When ``> 1``, enables transparent server-side batching via a
        ``BatchQueue``.  Incoming ``/predict`` requests are accumulated up to
        this many before being dispatched together.  Default ``1`` = no batching.
    max_wait_ms:
        Maximum time (milliseconds) to wait before dispatching a partial batch.
        Only relevant when ``max_batch_size > 1``.
    **load_kwargs:
        Extra keyword arguments forwarded to ``policy.load_model()``.

    Extra model-specific endpoints can be added to the returned app after this call::

        app = make_app(policy, model_id, ...)

        @app.post("/reload")
        def reload(req: ReloadSchema): ...
    """

    # Build optional batch queue (None when batching is disabled)
    _batch_queue: BatchQueue | None = None
    if max_batch_size > 1:
        _batch_queue = BatchQueue(
            policy=policy,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
        )
        logger.info(
            "BatchQueue enabled: max_batch_size=%d, max_wait_ms=%.1f (supports_batching=%s)",
            max_batch_size,
            max_wait_ms,
            policy.supports_batching,
        )

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        # --- Model loading ---
        if policy.load_in_background:
            t = threading.Thread(
                target=policy._load_bg,
                args=(model_id, device),
                kwargs=load_kwargs,
                daemon=True,
            )
            t.start()
        else:
            try:
                policy.load_model(model_id, device, **load_kwargs)
            except Exception as exc:
                policy.load_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Failed to load model: %s", policy.load_error)

        # --- Start batch drain loop if batching is enabled ---
        drain_task: asyncio.Task | None = None
        if _batch_queue is not None:
            drain_task = asyncio.create_task(_batch_queue.drain_loop())
            logger.info("BatchQueue drain_loop started.")

        yield

        # --- Cleanup ---
        if drain_task is not None:
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title=title, lifespan=_lifespan)

    # ------------------------------------------------------------------
    # /health — reports gpu_mode and device in addition to ready status
    # ------------------------------------------------------------------

    @app.get("/health")
    def health():
        if policy.load_error and not policy.ready:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "ready": False,
                    "model_id": policy.model_id or model_id,
                    "error": policy.load_error,
                    "gpu_mode": "single",
                    "device": device,
                },
            )
        # Return 503 while the model is still loading in background.
        # Clients that check HTTP status will not
        # prematurely conclude the server is ready.  The body still includes
        # ``ready: false`` so Python callers using _poll_health() also wait.
        if not policy.ready:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "loading",
                    "ready": False,
                    "model_id": policy.model_id or model_id,
                    "gpu_mode": "single",
                    "device": device,
                },
            )
        return {
            "status": "ok",
            "ready": True,
            "model_id": policy.model_id,
            "gpu_mode": "single",
            "device": device,
        }

    @app.get("/info")
    def info():
        data = dict(policy.get_info())
        if policy.load_error:
            data["error"] = policy.load_error
        if _batch_queue is not None:
            data["batching"] = {
                "enabled": True,
                "max_batch_size": max_batch_size,
                "max_wait_ms": max_wait_ms,
                "supports_batching": policy.supports_batching,
            }
        # Serialize typed ActionObsSpec contracts if the policy declares them.
        # Legacy obs_requirements / action_space keys are left untouched for
        # backward compatibility.  New keys action_spec / observation_spec are
        # added alongside them.
        action_spec = policy.get_action_spec()
        if action_spec is not None:
            data["action_spec"] = {k: v.to_dict() for k, v in action_spec.items()}
        observation_spec = policy.get_observation_spec()
        if observation_spec is not None:
            data["observation_spec"] = {k: v.to_dict() for k, v in observation_spec.items()}
        return data

    @app.post("/reset")
    def reset_policy():
        if not policy.ready:
            return JSONResponse(
                status_code=503,
                content={"error": policy.load_error or "Model not ready yet"},
            )
        try:
            policy.reset()
            return {"success": True}
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    # ------------------------------------------------------------------
    # /predict — batching path (async) or direct path (sync, unchanged)
    # ------------------------------------------------------------------

    if _batch_queue is not None:
        # Batching-enabled: accumulate requests in BatchQueue, dispatch via predict_batch()
        @app.post("/predict")
        async def predict_batched(req: PredictRequest):
            try:
                if not policy.ready:
                    return JSONResponse(
                        status_code=503,
                        content={"error": policy.load_error or "Model not ready yet"},
                    )
                actions = await _batch_queue.submit(req.obs)
                info_data = policy.get_info()
                return {
                    "actions": actions,
                    "chunk_size": len(actions),
                    "action_chunk_size": info_data.get("action_chunk_size", len(actions)),
                    "action_space": info_data.get("action_space", {}),
                    "model_id": policy.model_id,
                }
            except Exception as exc:
                logger.exception("Prediction failed (batched)")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(exc), "traceback": traceback.format_exc()},
                )
    else:
        # Default path: unchanged synchronous behaviour, no queue overhead.
        @app.post("/predict")
        def predict(req: PredictRequest):
            try:
                if not policy.ready:
                    return JSONResponse(
                        status_code=503,
                        content={"error": policy.load_error or "Model not ready yet"},
                    )
                actions = policy.predict(req.obs)
                info_data = policy.get_info()
                return {
                    "actions": actions,
                    "chunk_size": len(actions),
                    "action_chunk_size": info_data.get("action_chunk_size", len(actions)),
                    "action_space": info_data.get("action_space", {}),
                    "model_id": policy.model_id,
                }
            except Exception as exc:
                logger.exception("Prediction failed")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(exc), "traceback": traceback.format_exc()},
                )

    return app
