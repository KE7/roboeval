#!/usr/bin/env python
"""
run_openvla_native_eval.py

Native OpenVLA evaluation on LIBERO benchmark suites.
Loads the model directly and runs the standard LIBERO evaluation loop:
init states, warmup, image flip, and action unnormalization.

Usage:
    MUJOCO_GL=egl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
        python scripts/run_openvla_native_eval.py \\
        --suite libero_spatial --n-episodes 20 \\
        --out-dir results/openvla_native

Supported suites:
    libero_spatial  -> openvla/openvla-7b-finetuned-libero-spatial
    libero_10       -> openvla/openvla-7b-finetuned-libero-10

Per-suite max steps:
    libero_spatial: 280
    libero_object:  280
    libero_goal:    300
    libero_10:      520
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Offline model loading
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ---------------------------------------------------------------------------
# PyTorch 2.10 compat: torch.load defaults to weights_only=True, but LIBERO
# init-state files are numpy-pickle files.  Allow loading them safely.
# ---------------------------------------------------------------------------
_orig_torch_load = torch.load


def _permissive_torch_load(*args, **kwargs):
    """Wrapper that defaults weights_only=False for LIBERO numpy-pickle init-state files."""
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _permissive_torch_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Suite → model mapping
# ---------------------------------------------------------------------------
SUITE_MODEL = {
    "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "libero_10": "openvla/openvla-7b-finetuned-libero-10",
    # Add model IDs here to enable additional suites.
}

SUITE_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

NUM_WARMUP_STEPS = 10  # no-op steps after reset


# ---------------------------------------------------------------------------
# Action processing
# ---------------------------------------------------------------------------


def process_action(action: np.ndarray) -> np.ndarray:
    """Post-process OpenVLA action for LIBERO.

    OpenVLA outputs actions in [-1, +1] (from 256-bin quantization).
    Gripper convention: OpenVLA uses RLDS (1=close, -1=open);
    LIBERO uses opposite (-1=close, 1=open).

    This matches the gripper convention used by the OpenVLA policy wrapper.
    """
    a = action.copy()
    # Binarize gripper based on sign, then invert to match LIBERO convention
    a[-1] = -(1.0 if a[-1] > 0.0 else -1.0)
    return a


# ---------------------------------------------------------------------------
# Image extraction + flip
# ---------------------------------------------------------------------------


def get_agentview_image(obs) -> np.ndarray:
    """Extract third-person image and flip 180° (matches lerobot/openvla training)."""
    img = obs["agentview_image"]
    return img[::-1, ::-1]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_id: str, device: str = "cuda"):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    log.info("Loading processor from %s …", model_id)
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
    )

    log.info("Loading model from %s on %s …", model_id, device)
    t0 = time.time()
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",  # avoid _supports_sdpa compat issue w/ transformers>=4.50
    )
    model.eval()
    log.info("Model loaded in %.1fs", time.time() - t0)

    dev = next(model.parameters()).device
    log.info("Model on device: %s", dev)
    return model, processor, dev


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------


def predict(
    model, processor, device, img_np: np.ndarray, instruction: str, unnorm_key: str
) -> np.ndarray:
    """Run one forward pass; returns raw 7-dim unnormalized action array."""
    pil_img = Image.fromarray(img_np)
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    inputs = processor(prompt, pil_img).to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        action = model.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
        )

    return np.array(action, dtype=np.float64).flatten()[:7]


# ---------------------------------------------------------------------------
# LIBERO environment setup
# ---------------------------------------------------------------------------


def build_libero_env(task, resolution: int = 256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env_args = {
        "bddl_file_name": task_bddl,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------


def run_episode(
    env,
    model,
    processor,
    device,
    task_description: str,
    unnorm_key: str,
    initial_state,
    max_steps: int,
) -> bool:
    """Run one episode. Returns True on success."""
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Warmup: no-op for NUM_WARMUP_STEPS steps
    warmup_action = [0.0] * 6 + [-1.0]  # all zeros, gripper open
    for _ in range(NUM_WARMUP_STEPS):
        obs, _, done, _ = env.step(warmup_action)
        if done:
            return False

    # Rollout
    for _ in range(max_steps):
        img = get_agentview_image(obs)

        raw_action = predict(model, processor, device, img, task_description, unnorm_key)
        action = process_action(raw_action)

        obs, reward, done, info = env.step(action.tolist())
        if done:
            return True

    return False


# ---------------------------------------------------------------------------
# Task-level evaluation
# ---------------------------------------------------------------------------


def eval_task(
    task_suite,
    task_id: int,
    model,
    processor,
    device,
    unnorm_key: str,
    max_steps: int,
    n_episodes: int,
    log_file,
) -> tuple[int, int]:
    """Returns (successes, total_episodes)."""
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)
    desc = task.language

    env = build_libero_env(task)

    successes = 0
    for ep_idx in tqdm(range(n_episodes), desc=f"Task {task_id}", leave=False):
        init_state = init_states[ep_idx % len(init_states)]
        success = run_episode(
            env,
            model,
            processor,
            device,
            desc,
            unnorm_key,
            init_state,
            max_steps,
        )
        successes += int(success)
        msg = f"  task={task_id} ep={ep_idx} success={success} running={successes}/{ep_idx + 1}"
        log.info(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    env.close()
    return successes, n_episodes


# ---------------------------------------------------------------------------
# Suite-level evaluation
# ---------------------------------------------------------------------------


def eval_suite(
    suite_name: str,
    model_id: str,
    n_episodes: int,
    out_dir: Path,
    device: str = "cuda",
    start_task: int = 0,
    prior_results: list | None = None,
) -> dict:
    from libero.libero import benchmark

    max_steps = SUITE_MAX_STEPS.get(suite_name, 300)
    unnorm_key = suite_name  # model's norm_stats key matches suite name

    log.info("=" * 60)
    log.info(
        "Suite: %s | model: %s | %d eps/task | start_task=%d",
        suite_name,
        model_id,
        n_episodes,
        start_task,
    )
    log.info("=" * 60)

    # Load model
    model, processor, dev = load_model(model_id, device)

    # Verify unnorm key
    if hasattr(model, "norm_stats"):
        ns_keys = list(model.norm_stats.keys())
        log.info("Available norm_stats keys: %s", ns_keys)
        if unnorm_key not in ns_keys:
            # try _no_noops variant
            alt = unnorm_key + "_no_noops"
            if alt in ns_keys:
                unnorm_key = alt
                log.info("Using norm key: %s", unnorm_key)
            else:
                log.warning("unnorm_key '%s' not in norm_stats! Actions may be wrong.", unnorm_key)

    # Load benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    n_tasks = task_suite.n_tasks
    log.info("Tasks: %d (running %d-%d)", n_tasks, start_task, n_tasks - 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Append mode so resumption doesn't erase completed tasks
    log_path = out_dir / f"{suite_name}.log"
    results_path = out_dir / f"{suite_name}.json"

    # Seed totals from prior completed tasks
    task_results = list(prior_results) if prior_results else []
    total_suc = sum(t["success"] for t in task_results)
    total_eps = sum(t["total"] for t in task_results)

    open_mode = "a" if start_task > 0 else "w"
    with open(log_path, open_mode) as log_file:
        if start_task == 0:
            log_file.write(f"Suite: {suite_name}\n")
            log_file.write(f"Model: {model_id}\n")
            log_file.write(f"Episodes per task: {n_episodes}\n\n")
        else:
            log_file.write(f"\n--- Resuming from task {start_task} ---\n\n")

        for task_id in range(start_task, n_tasks):
            task = task_suite.get_task(task_id)
            log.info("--- Task %d: %s", task_id, task.language)
            log_file.write(f"\nTask {task_id}: {task.language}\n")
            log_file.flush()

            suc, eps = eval_task(
                task_suite,
                task_id,
                model,
                processor,
                dev,
                unnorm_key,
                max_steps,
                n_episodes,
                log_file,
            )

            total_suc += suc
            total_eps += eps
            rate = suc / eps if eps > 0 else 0.0
            task_results.append(
                {
                    "task_id": task_id,
                    "description": task.language,
                    "success": suc,
                    "total": eps,
                    "rate": rate,
                }
            )

            summary = (
                f"Task {task_id} result: {suc}/{eps} = {rate * 100:.1f}% | "
                f"overall so far: {total_suc}/{total_eps} = "
                f"{total_suc / total_eps * 100:.1f}%"
            )
            log.info(summary)
            log_file.write(summary + "\n\n")
            log_file.flush()

        overall_rate = total_suc / total_eps if total_eps > 0 else 0.0
        final = f"\nFINAL: {suite_name} | {total_suc}/{total_eps} = {overall_rate * 100:.1f}%\n"
        log.info(final.strip())
        log_file.write(final)

    result = {
        "suite": suite_name,
        "model": model_id,
        "n_episodes_per_task": n_episodes,
        "tasks": task_results,
        "total_success": total_suc,
        "total_episodes": total_eps,
        "success_rate": overall_rate,
    }
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    log.info("Results saved to %s", results_path)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="OpenVLA native LIBERO eval")
    p.add_argument(
        "--suite",
        default="libero_spatial",
        choices=list(SUITE_MAX_STEPS.keys()),
        help="LIBERO task suite to evaluate",
    )
    p.add_argument(
        "--model-id", default=None, help="Override HuggingFace model ID (default: auto from suite)"
    )
    p.add_argument("--n-episodes", type=int, default=20, help="Episodes per task (default: 20)")
    p.add_argument(
        "--out-dir", default="results/openvla_native", help="Output directory for logs and JSON"
    )
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    p.add_argument(
        "--start-task", type=int, default=0, help="Resume from this task ID (0-indexed, default: 0)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    suite = args.suite
    model_id = args.model_id or SUITE_MODEL.get(suite)
    if model_id is None:
        log.error("No model cached for suite '%s'. Available: %s", suite, list(SUITE_MODEL.keys()))
        sys.exit(1)

    # Load prior completed task results if resuming
    out_dir = Path(args.out_dir)
    prior_results = None
    if args.start_task > 0:
        prior_json = out_dir / f"{suite}.json"
        if prior_json.exists():
            with open(prior_json) as f:
                prior_data = json.load(f)
            prior_results = prior_data.get("tasks", [])
            log.info("Loaded %d prior task results from %s", len(prior_results), prior_json)

    result = eval_suite(
        suite_name=suite,
        model_id=model_id,
        n_episodes=args.n_episodes,
        out_dir=out_dir,
        device=args.device,
        start_task=args.start_task,
        prior_results=prior_results,
    )

    print("\n" + "=" * 60)
    print(f"OPENVLA NATIVE DONE: {suite}")
    print(
        f"Success rate: {result['success_rate'] * 100:.1f}%  "
        f"({result['total_success']}/{result['total_episodes']})"
    )
    for t in result["tasks"]:
        print(
            f"  Task {t['task_id']:2d}: {t['success']:2d}/{t['total']:2d} "
            f"= {t['rate'] * 100:.0f}%  {t['description']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
