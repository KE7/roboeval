"""
LITEN Simulator Evaluation Script.

Runs the LITEN iterative reasoning/assessment loop against a robotic simulator
(LIBERO, RoboCasa, RoboTwin, LIBERO-Pro) via a sim_worker HTTP server and a
litellm VLM proxy.

Prerequisites:
    1. Start the simulator server: ``bash scripts/start_sim.sh --sim libero --port 5001``
    2. Start the litellm proxy:    ``bash scripts/start_vlm.sh``
    3. Start a VLA policy server:  ``bash scripts/start_pi05_policy.sh``

Usage:
    python run_sim_eval.py eval --sim libero --task 0 --suite libero_spatial \\
        --sim-url http://localhost:5001 --delta-actions
"""

import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import typer
from PIL import Image

import vlm_hl.vlm_methods as vlmi
from robo_eval.episode_logger import EpisodeResult, save_episode_result
from vlm_hl.vlm_methods import LLMStats
from ica.reasoning_ica import TaskICADir
from run import (
    get_reasoning_steps,
    get_top_level_task_assessment,
    save_video,
)
from run_utils import save_reasoning_ica_dir, save_top_level_ica_dir
from sims.env_wrapper import SimWrapper, VALID_SIMS
from sims.litellm_vlm import setup_litellm_from_endpoint

app = typer.Typer(
    help="LITEN Simulator Evaluation: Run LITEN against robotic simulators."
)


def save_episode_video(frames, output_path, fps=30, text_overlay=None):
    """Save a list of RGB frames as an MP4 video file.

    Args:
        frames: List of numpy arrays (H, W, 3) in RGB format.
        output_path: Path to the output .mp4 file.
        fps: Frames per second (default 30).
        text_overlay: Optional text to overlay on each frame.
    """
    import cv2

    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    try:
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if text_overlay:
                cv2.putText(
                    bgr, text_overlay, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
            writer.write(bgr)
    finally:
        writer.release()


def _slugify_filename_component(text: str, max_len: int = 80) -> str:
    """Convert free-form task text into a filesystem-friendly slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        return "task"
    return slug[:max_len].rstrip("_")


@app.command("eval")
def eval_sim(
    sim: str = typer.Option(
        ...,
        help=f"Simulator name (for action dims/horizon). One of: {VALID_SIMS}",
    ),
    task: str = typer.Option(
        ...,
        help="Task name or index within the simulator.",
    ),
    sim_url: str = typer.Option(
        ...,
        help="URL of the sim_worker HTTP server (e.g. http://localhost:5001).",
    ),
    suite: str = typer.Option(
        None,
        help="Task suite (for LIBERO: libero_spatial, libero_10, etc.).",
    ),
    vlm_endpoint: str = typer.Option(
        "localhost:4000",
        help="litellm proxy endpoint (host:port).",
    ),
    vlm_model: str = typer.Option(
        None,
        help="Override VLM model name (if litellm serves a specific model).",
    ),
    max_episodes: int = typer.Option(
        1,
        help="Number of episodes to evaluate.",
    ),
    start_episode: int = typer.Option(
        0,
        help="Episode index to start from (for resuming or per-episode runs).",
    ),
    max_steps: int = typer.Option(
        None,
        help="Max action steps per subtask (overrides sim default).",
    ),
    camera_resolution: int = typer.Option(
        256,
        help="Camera image resolution (square).",
    ),
    experience_dir: str = typer.Option(
        "sim_experience",
        help="Directory to store experience for LITEN context.",
    ),
    save_videos: bool = typer.Option(
        True,
        help="Save rollout videos.",
    ),
    headless: bool = typer.Option(
        False,
        help=(
            "Enable headless EGL offscreen rendering (no display required). "
            "Use this for CI/servers. Default: windowed GLFW rendering. "
            "The sim worker must also be started with --headless for this to take effect."
        ),
    ),
    no_think: bool = typer.Option(
        False, "--no-think", is_flag=True, help="Disable VLM thinking tokens"
    ),
    no_vlm: bool = typer.Option(
        False, "--no-vlm", is_flag=True,
        help="Skip VLM planning and use the raw task description as the instruction.",
    ),
    delta_actions: bool = typer.Option(
        False, "--delta-actions", is_flag=True,
        help="Set robot.controller.use_delta=True after reset (for Pi0.5 which outputs relative actions).",
    ),
    record_video: bool = typer.Option(
        False, "--record-video", is_flag=True,
        help="Record episode videos from simulator.",
    ),
    record_video_n: int = typer.Option(
        3, "--record-video-n",
        help="Max episodes per task to record (default 3).",
    ),
    results_dir: str = typer.Option(
        None, "--results-dir",
        help="Results directory for video output.",
    ),
    seed: int = typer.Option(
        None, "--seed",
        help="Random seed for reproducibility.",
    ),
    sim_config: str = typer.Option(
        None, "--sim-config",
        help="Optional path to a YAML file with extra sim configuration (passed to sim_worker).",
    ),
):
    """Run LITEN evaluation on a simulator environment."""
    # Set random seeds for reproducibility
    import random as _random
    if seed is not None:
        _random.seed(seed)
        try:
            import numpy as _np
            _np.random.seed(seed)
        except ImportError:
            pass
        typer.echo(f"Random seed set to {seed}")
    else:
        seed = _random.randint(0, 2**31 - 1)
        _random.seed(seed)
        try:
            import numpy as _np
            _np.random.seed(seed)
        except ImportError:
            pass
        typer.echo(f"Generated random seed: {seed}")

    if sim not in VALID_SIMS:
        typer.echo(f"Error: --sim must be one of {VALID_SIMS}")
        raise typer.Exit(1)

    # Load sim config YAML if provided
    import yaml
    if sim_config:
        try:
            with open(sim_config) as _f:
                sim_config_dict = yaml.safe_load(_f) or {}
        except FileNotFoundError:
            typer.echo(f"Error: sim config file not found: {sim_config}")
            raise typer.Exit(1)
        except yaml.YAMLError as e:
            typer.echo(f"Error: failed to parse sim config YAML: {e}")
            raise typer.Exit(1)
    else:
        sim_config_dict = {}

    # Forward the run-level seed into sim_config_dict so that backends
    # such as LiberoInfinityBackend use the actual run seed (not a hardcoded default).
    sim_config_dict["seed"] = seed

    # Patch VLM to use litellm proxy (skip in no-vlm mode)
    if not no_vlm:
        typer.echo(f"Connecting VLM to litellm proxy at {vlm_endpoint}...")
        setup_litellm_from_endpoint(vlm_endpoint, model_override=vlm_model)
    else:
        typer.echo("Running in --no-vlm mode: skipping VLM proxy setup.")

    # Create experience directory
    os.makedirs(experience_dir, exist_ok=True)

    # Copy sim config YAML into experience dir for reproducibility
    if sim_config:
        shutil.copy(sim_config, os.path.join(experience_dir, "sim_config.yaml"))

    # Load existing experience if available
    task_icadirs = []
    for subdir in os.listdir(experience_dir):
        subdir_path = os.path.join(experience_dir, subdir)
        if os.path.isdir(subdir_path):
            try:
                task_icadirs.append(TaskICADir(subdir_path))
            except Exception:
                pass  # Skip malformed experience dirs

    typer.echo(f"Loaded {len(task_icadirs)} prior experience entries.")

    for episode in range(start_episode, start_episode + max_episodes):
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Episode {episode - start_episode + 1}/{max_episodes} (init state {episode})")
        typer.echo(f"{'='*60}")

        # Connect to simulator server
        typer.echo(f"Connecting to {sim} simulator at {sim_url} for task '{task}'...")
        try:
            wrapper = SimWrapper(
                sim_server_url=sim_url,
                sim_name=sim,
                task_name=task,
                camera_resolution=camera_resolution,
                suite=suite,
                max_steps=max_steps,
                headless=headless,
                delta_actions=delta_actions,
                no_vlm=no_vlm,
                sim_config=sim_config_dict,
            )
        except Exception as e:
            typer.echo(f"Failed to initialize simulator (skipping episode): {e}")
            continue

        try:
            ep_start = time.time()

            # Reset to the correct episode init_state before anything else.
            # LiberoBackend.reset() (triggered via /reset) applies set_init_state(),
            # use_delta=True, and 10 physics warmup steps — all skipped by /init alone.
            typer.echo(f"Resetting simulator to episode {episode} init state...")
            wrapper.physical_reset(episode_index=episode)

            llm_stats = LLMStats()

            # Get initial observation
            initial_image = wrapper.current_image.copy()
            instruction = wrapper.task_instruction

            typer.echo(f"Task instruction: {instruction}")

            typer.echo("Generating LITEN plan program...")
            new_programplan, justification = vlmi.generate_planner_program(
                initial_image,
                instruction,
                object_uids=getattr(wrapper, "manipulable_object_uids", []),
                tuple_icadirs=task_icadirs,
                llm_stats=llm_stats,
                no_think=no_think,
                no_vlm=no_vlm,
            )
            typer.echo("Generated program:")
            typer.echo(new_programplan)
            typer.echo(f"Justification: {justification}")
            typer.echo(str(llm_stats))

            # Execute the plan
            typer.echo("Executing generated program...")
            exec(new_programplan.strip(), {"world": wrapper})
            final_image = wrapper.current_image.copy()

            # Check task success via simulator
            sim_success = wrapper.check_success()
            typer.echo(f"Simulator reports success: {sim_success}")

            # Write structured episode JSON immediately after check_success
            # so it's saved even if the VLM assessment step fails.
            ep_duration = time.time() - ep_start
            total_steps = sum(
                len(frames) for _, frames in wrapper.subtask_frame_tuples
            )
            subtask_names = [
                cmd for cmd, _ in wrapper.subtask_frame_tuples
            ]
            ep_result = EpisodeResult(
                task=int(task) if str(task).isdigit() else 0,
                episode=episode,
                success=sim_success,
                steps=total_steps,
                duration_s=round(ep_duration, 2),
                vla_calls=total_steps,
                subtasks=subtask_names,
            )
            ep_results_dir = results_dir
            if not ep_results_dir:
                ep_results_dir = str(
                    Path(experience_dir).parent.parent
                )
            if suite:
                ep_json_path = save_episode_result(
                    ep_results_dir, suite, int(task) if str(task).isdigit() else 0, episode, ep_result,
                )
                typer.echo(f"  Episode JSON: {ep_json_path}")

            # Save experience and VLM assessment (skip when --no-vlm)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_dir = os.path.join(
                experience_dir,
                f"{sim}_{task}_{timestamp}",
            )
            os.makedirs(episode_dir, exist_ok=True)

            assessment = None
            if not no_vlm:
                typer.echo(
                    f"Processing {len(wrapper.subtask_frame_tuples)} subtasks..."
                )
                subtask_reasoning_tuples = []
                for i, (subtask_cmd, frames) in enumerate(
                    wrapper.subtask_frame_tuples
                ):
                    subtask_dir = os.path.join(episode_dir, f"subtask_{i}")
                    os.makedirs(subtask_dir, exist_ok=True)

                    if not frames:
                        continue

                    init_img = Image.fromarray(frames[0])
                    final_img = Image.fromarray(frames[-1])

                    success, whathappened, reasoning = get_reasoning_steps(
                        init_img, final_img, subtask_cmd
                    )
                    save_reasoning_ica_dir(
                        subtask_dir,
                        init_img,
                        subtask_cmd,
                        success,
                        whathappened,
                        reasoning,
                    )
                    subtask_reasoning_tuples.append(
                        (subtask_cmd, success, whathappened, reasoning)
                    )

                # Assess overall task
                typer.echo("Assessing overall task outcome...")
                assessment = get_top_level_task_assessment(
                    initial_image,
                    final_image,
                    instruction,
                    sim_success,
                    subtask_reasoning_tuples,
                )
                episode_dir = save_top_level_ica_dir(
                    episode_dir,
                    initial_image,
                    final_image,
                    instruction,
                    sim_success,
                    assessment,
                )
                task_icadirs.append(TaskICADir(episode_dir))

            # Save video (legacy demo_videos/ path)
            if save_videos:
                all_frames = []
                for _, frames in wrapper.subtask_frame_tuples:
                    all_frames.extend(frames)
                if all_frames:
                    prompt_slug = _slugify_filename_component(instruction)
                    video_name = f"{sim}_{task}_{prompt_slug}_{timestamp}"
                    save_video(all_frames, video_name)

            # Save structured episode video to results_dir/videos/
            if record_video and (episode - start_episode) < record_video_n:
                all_frames = []
                for _, frames in wrapper.subtask_frame_tuples:
                    all_frames.extend(frames)
                if all_frames and results_dir:
                    videos_dir = os.path.join(results_dir, "videos")
                    os.makedirs(videos_dir, exist_ok=True)
                    prompt_slug = _slugify_filename_component(instruction)
                    video_path = os.path.join(
                        videos_dir,
                        f"{suite}_task{task}_ep{episode}_{prompt_slug}.mp4",
                    )
                    save_episode_video(
                        all_frames, video_path, text_overlay=instruction,
                    )
                    typer.echo(f"  Video saved: {video_path}")

            typer.echo(f"Episode {episode + 1} complete.")
            typer.echo(f"  Simulator success: {sim_success}")
            if assessment:
                typer.echo(f"  Assessment: {assessment}")
            typer.echo(f"  Experience saved to: {episode_dir}")

        except Exception as e:
            typer.echo(f"Episode {episode - start_episode + 1} failed with error: {e}")
            import traceback as _tb
            typer.echo(_tb.format_exc())
        finally:
            wrapper.close()

    typer.echo(f"\nAll {max_episodes} episodes complete.")


@app.command("list-tasks")
def list_tasks(
    sim: str = typer.Option(
        ...,
        help=f"Simulator to list tasks for. One of: {VALID_SIMS}",
    ),
    suite: str = typer.Option(
        None,
        help="Task suite (for LIBERO: libero_spatial, libero_10, etc.).",
    ),
):
    """List available tasks for a simulator (LIBERO only for now)."""
    if sim in ("libero", "libero_pro"):
        typer.echo(
            f"To list {sim} tasks, run inside the {sim} venv:\n"
            f"  python -c \"from libero.libero.benchmark import get_benchmark; "
            f"b = get_benchmark('{suite or 'libero_spatial'}')(); "
            f"print(b.get_task_names())\""
        )
    else:
        typer.echo(f"Task listing not yet implemented for {sim}.")


if __name__ == "__main__":
    app()
