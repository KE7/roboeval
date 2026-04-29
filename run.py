"""
Physical Robot CLI — Typer-based interface for running planner methods.

Provides subcommands for the full planner pipeline and ablation baselines:
  - ``planner``: Full planner with reasoning + in-context experience
  - ``ablation-nor``: No-Reasoning ablation
  - ``ablation-who``: What-Happened-Only ablation
  - ``positive-icl``: Positive-only ICL baseline
  - ``reflexion-like``: Reflexion-style baseline
  - ``test-planner``: Plan generation test with existing experience

Each command loops interactively, prompting for task instructions, generating
plan programs, executing them via the world stub, and saving experience.
"""

import os
from datetime import datetime
import cv2
import typer
from PIL import Image

import vlm_hl.vlm_methods as vlmi
from vlm_hl.vlm_methods import LLMStats
from run_utils import (
    save_reasoning_ica_dir,
    save_top_level_ica_dir,
    save_icl_dir,
    load_icl_dir,
    save_ablation_dir,
    load_ablation_dir,
    save_who_ablation_dir,
    load_who_ablation_dir,
    load_reflexion_dir,
    save_reflexion_dir,
)
from ica.reasoning_ica import TaskICADir

app = typer.Typer(
    help="Physical Robot CLI: Typer-based interface for running planner methods."
)

# --- Constants and Task Registry ---

EVAL_TASKS = {
    "stacktask": {
        "instruction": "Create a stack of any three objects on the table.",
        "example_image_path": "images/stackingtask.png",
    },
    "emptytask": {
        "instruction": "Move the contents of the bowls until two are empty at the same time. Objects should only be moved from one bowl to another bowl.",
        "example_image_path": "images/emptybowlstask.png",
    },
    "offtabletask": {
        "instruction": "Move the objects on the table on top one another so that at most three objects are directly in contact with the table. Do not count the clamps or bases as objects.",
        "example_image_path": "images/offtabletask.png",
    },
}

# --- Helper Functions ---


def save_video(frames, videoname):
    """Save a list of RGB numpy frames as an MP4 video in ``demo_videos/``.

    Args:
        frames: List of numpy arrays (H, W, 3) in RGB order.
        videoname: Filename stem (without extension or directory).
    """
    if not frames or not videoname:
        return
    os.makedirs("demo_videos", exist_ok=True)
    shape = (frames[0].shape[1], frames[0].shape[0])
    out = cv2.VideoWriter(
        f"demo_videos/{videoname}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, shape
    )
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved video to demo_videos/{videoname}.mp4")


def collect_all_frames(world) -> list:
    """Collect all frames from a world's subtask_frame_tuples into a flat list.

    Keeps the frame accumulator separate from each subtask's frame list.
    """
    all_frames = []
    for _subtask_cmd, subtask_frames in world.subtask_frame_tuples:
        all_frames.extend(subtask_frames)
    return all_frames


def get_task_info(task_name: str) -> tuple:
    """Look up a task by name in EVAL_TASKS.

    Returns:
        ``(instruction, image_path)`` if found, else ``(task_name, None)``.
    """
    if task_name in EVAL_TASKS:
        task = EVAL_TASKS[task_name]
        return task["instruction"], task["example_image_path"]
    return task_name, None


def setup_world(**kwargs):
    """Instantiate and return a world stub for your robot hardware.

    Override this function to connect to your specific robot setup.
    """
    raise NotImplementedError(
        "Please set up your own world stub here to enable physical execution."
    )


def prompt_video_save(frames):
    """Interactively ask the user whether to save frames as a video."""
    videoname = typer.prompt(
        "Do you want to save the most recent trajectory as a video? "
        "If yes, provide a name for the video. If not, press enter:",
        default="",
    )
    if videoname.strip():
        save_video(frames, videoname)


def _assess_success_and_critique(
    prior_image, final_image, instruction, video_frames=None, use_human=False
):
    """Determine success and generate a failure critique if needed.

    Shared logic between ``get_reasoning_steps`` and ``get_who_steps``.

    Returns:
        Tuple of ``(success: bool, whathappened: str | None)``.
    """
    if use_human:
        success = input("Was the task successful? (1/0): ").strip().lower() == "1"
    else:
        success = vlmi.determine_vla_success(prior_image, final_image, instruction)

    whathappened = None
    if not success:
        if video_frames is None:
            whathappened = vlmi.critique_vla_failure(
                prior_image, final_image, instruction
            )
        else:
            whathappened = vlmi.critique_vla_video_failure(video_frames, instruction)
    return success, whathappened


def get_reasoning_steps(
    prior_image, final_image, instruction, video_frames=None, use_human=False
):
    """Assess subtask outcome and generate reasoning about success/failure.

    Returns:
        Tuple of ``(success, whathappened, reasoning)``.
    """
    success, whathappened = _assess_success_and_critique(
        prior_image, final_image, instruction, video_frames, use_human
    )
    if success:
        reasoning = vlmi.describe_vla_success(prior_image, instruction)
    else:
        reasoning = vlmi.reason_about_vla_failure(
            prior_image, instruction, whathappened
        )
    return success, whathappened, reasoning


def get_who_steps(
    prior_image, final_image, instruction, video_frames=None, use_human=False
):
    """Assess subtask outcome and generate a "what happened" critique (no reasoning).

    Returns:
        Tuple of ``(success, whathappened)``.
    """
    return _assess_success_and_critique(
        prior_image, final_image, instruction, video_frames, use_human
    )


def get_top_level_task_assessment(
    prior_image, final_image, hl_instruction, success, reasoning_tuples
):
    """Generate a VLM assessment of the overall task outcome.

    Args:
        prior_image: Scene image before execution.
        final_image: Scene image after execution.
        hl_instruction: High-level task instruction.
        success: Whether the task succeeded.
        reasoning_tuples: List of per-subtask ``(cmd, success, critique, reasoning)`` tuples.

    Returns:
        Assessment text from the VLM.
    """
    if success:
        return vlmi.assess_hl_success(
            prior_image, final_image, hl_instruction, reasoning_tuples
        )
    return vlmi.assess_hl_failure(
        prior_image, final_image, hl_instruction, reasoning_tuples
    )


def _load_experience_subdirs(dir_path: str, loader_fn) -> list:
    """Load experience from all subdirectories using the given loader function."""
    entries = []
    for subdir in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir)
        if os.path.isdir(subdir_path):
            entries.append(loader_fn(subdir_path))
    return entries


# --- CLI Subcommands ---

@app.command("planner")
def planner(
    experiment_name: str = typer.Argument(
        help="Name of experiment (will be timestamped also)."
    ),
    experience_dir_path: str = typer.Option(
        "planner_context",
        help="Path to past experience storage for in-context inclusion.",
    ),
    ask_save_video: bool = typer.Option(
        False, help="Offer to save higher-quality videos."
    )
):
    """
    Runs the full planner method, consisting of
    (1) generating a plan,
    (2) executing that plan,
    and (3) assessing the result of that plan.
    
    The method will take in as arguments the following:
    - experiment_name: name of the experiment (for use in saving your execution results.)
    - experience_dir_path: path to the directory containing experience subdirectories that were previously generated from this method.
        you can provide existing experience to use as context, or an empty directory to start fresh.
    - ask_save_video: whether to offer to save a higher-quality video of the execution at the end of the run.
    """
    typer.echo("Beginning planner method.")
    world = setup_world()
    llm_stats = LLMStats()
    # Load all TaskICADirs (top-level experience dirs) from subdirectories
    task_icadirs = _load_experience_subdirs(experience_dir_path, TaskICADir)
    while True:
        typer.echo(
            "Input: A task instruction to execute by the VLA. Input 'exit' to break:"
        )
        instruction_input = typer.prompt("Instruction")
        if instruction_input == "exit":
            break
        if instruction_input in EVAL_TASKS:
            typer.echo("Found eval task in stored tasks.")
            instruction, _ = get_task_info(instruction_input)
        else:
            instruction = instruction_input
        typer.echo("Generating new plan program...")
        typer.echo(
            "Press Enter once the environment has been re-configured to begin plan generation, or type 'exit' to break."
        )
        is_ready = typer.prompt("Ready?", default="")
        if is_ready == "exit":
            break
        image = world.reset(instruction, refresh_objects=True)
        initial_image = image.copy()
        new_programplan, justification = vlmi.generate_planner_program(
            initial_image,
            instruction,
            object_uids=world.manipulable_object_uids,
            tuple_icadirs=task_icadirs,
            llm_stats=llm_stats,
        )
        print("Generated program:")
        print(new_programplan)
        print("Justification of generated program:")
        print(justification)
        print(llm_stats)
        typer.echo(
            "Press Enter to execute the generated program, or type 'exit' to break."
        )
        is_ready = typer.prompt("Ready?", default="")
        if is_ready == "exit":
            break
        else:
            exec(new_programplan.strip(), {"world": world})
            final_image = world.current_image.copy()
            world.arm_reset()
        typer.echo("Enter 0 if the execution was a failure, 1 if it was a success:")
        execution_success = typer.prompt("Success?", default="")
        execution_success = execution_success == "1"
        # Save the attempt in TaskICADir/ReasoningICADir format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        overalltaskdir = os.path.join(
            experience_dir_path,
            f"{experiment_name}_{timestamp}",
        )
        os.makedirs(overalltaskdir, exist_ok=True)
        print(f"Found {len(world.subtask_frame_tuples)} tuples")
        subtask_reasoning_tuples = []
        for i, tupl in enumerate(world.subtask_frame_tuples):
            unique_dir_pathname = os.path.join(
                overalltaskdir,
                f"subtask_{i}",
            )
            os.makedirs(unique_dir_pathname, exist_ok=True)
            task, frames = tupl
            init_img = Image.fromarray(frames[0])
            final_img = Image.fromarray(frames[-1])
            # Save image0.png (initial), image1.png (final)
            success, whathappened, reasoning = get_reasoning_steps(
                init_img, final_img, task
            )  # Pass video_frames=frames to critique using the full subtask video.
            subtask_icadirpath = save_reasoning_ica_dir(
                unique_dir_pathname, init_img, task, success, whathappened, reasoning
            )
            subtask_reasoning_tuples.append((task, success, whathappened, reasoning))
        print("Creating reasoning tuple for overall task.")
        assessment = get_top_level_task_assessment(
            initial_image,
            final_image,
            instruction,
            execution_success,
            subtask_reasoning_tuples,
        )
        overalltaskdir = save_top_level_ica_dir(
            overalltaskdir,
            initial_image,
            final_image,
            instruction,
            execution_success,
            assessment,
        )
        task_icadirs.append(TaskICADir(overalltaskdir))
        if ask_save_video:
            prompt_video_save(collect_all_frames(world))

@app.command("ablation-nor")
def ablation_nor(
    experiment_name: str = typer.Argument(
        help="Name of experiment (will be timestamped also)."
    ),
    ablation_dir_path: str = typer.Option(
        "ablation_tuples",
        help="Path to save subtask tuples for no-reasoning ablation.",
    ),
    ask_save_video: bool = typer.Option(
        False, help="Offer to save higher-quality videos."
    )
):
    """
    Run No-Reasoning ablation method.
    """
    typer.echo("Beginning No-Reasoning ablation method.")
    world = setup_world()
    llm_stats = LLMStats()
    # Load past experience, if available
    image_task_succ_tuples = _load_experience_subdirs(ablation_dir_path, load_ablation_dir)
    while True:
        typer.echo(
            "Input: A task instruction to execute by the VLA. Input 'exit' to break:"
        )
        instruction_input = typer.prompt("Instruction")
        if instruction_input == "exit":
            break
        if instruction_input in EVAL_TASKS:
            typer.echo("Found eval task in stored tasks.")
            instruction, _ = get_task_info(instruction_input)
        else:
            instruction = instruction_input
        typer.echo("Generating new program plan...")
        typer.echo(
            "Press Enter once the environment has been re-configured to begin plan generation, or type 'exit' to break."
        )
        is_ready = typer.prompt("Ready?", default="")
        if is_ready == "exit":
            break
        image = world.reset(instruction, refresh_objects=True)
        initial_image = image.copy()
        new_programplan, justification = vlmi.generate_program_with_nor_ablation(
            initial_image,
            instruction,
            object_uids=world.manipulable_object_uids,
            subtask_tuples=image_task_succ_tuples,
            llm_stats=llm_stats,
        )
        print("Generated program:")
        print(new_programplan)
        print("Justification of generated program:")
        print(justification)
        print(llm_stats)
        typer.echo(
            "Press Enter to execute the generated program, or type 'exit' to break."
        )
        is_ready = typer.prompt("Ready?", default="")
        if is_ready == "exit":
            break
        else:
            exec(new_programplan.strip(), {"world": world})
            world.arm_reset()
        typer.echo("Enter 0 if the execution was a failure, 1 if it was a success:")
        execution_success = typer.prompt("Success?", default="")
        execution_success = execution_success == "1"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("Saving ablation tuples.")
        for i, tupl in enumerate(world.subtask_frame_tuples):
            unique_dir_pathname = os.path.join(
                ablation_dir_path,
                f"{experiment_name}_{timestamp}_{i}",
            )
            task, frames = tupl
            init_img = Image.fromarray(frames[0])
            final_img = Image.fromarray(frames[-1])
            # Save image0.png (initial), task, and whether it was successful
            outcome = vlmi.determine_vla_success(init_img, final_img, task)
            os.makedirs(unique_dir_pathname, exist_ok=True)
            save_ablation_dir(unique_dir_pathname, init_img, task, outcome)
            image_task_succ_tuples.append((init_img, task, outcome))
        if ask_save_video:
            prompt_video_save(collect_all_frames(world))


@app.command("ablation-who")
def ablation_who(
    experiment_name: str = typer.Argument(
        help="Name of experiment (will be timestamped also)."
    ),
    ablation_dir_path: str = typer.Argument(
        help="Path to save subtask tuples for what-happened only ablation.",
    ),
    ask_save_video: bool = typer.Option(
        False, help="Offer to save higher-quality videos."
    ),
):
    """
    Run What-Happened-Only ablation method.
    """
    typer.echo("Beginning What-Happened Only Ablation method.")
    world = setup_world()
    llm_stats = LLMStats()
    # Load past experience, if available
    image_task_succ_wh_tuples = _load_experience_subdirs(ablation_dir_path, load_who_ablation_dir)
    while True:
        typer.echo(
            "Input: A task instruction to execute by the VLA. Input 'exit' to break:"
        )
        instruction_input = typer.prompt("Instruction")
        if instruction_input == "exit":
            break
        if instruction_input in EVAL_TASKS:
            typer.echo("Found eval task in stored tasks.")
            instruction, _ = get_task_info(instruction_input)
        else:
            instruction = instruction_input
        typer.echo("Generating new plan program...")
        typer.echo(
            "Press Enter once the environment has been re-configured to begin plan generation, or type 'exit' to break."
        )
        is_ready = typer.prompt("Ready?", default="")
        if is_ready == "exit":
            break
        image = world.reset(instruction, refresh_objects=True)
        initial_image = image.copy()
        new_programplan, justification = vlmi.generate_program_with_who_ablation(
            initial_image,
            instruction,
            object_uids=world.manipulable_object_uids,
            subtask_tuples=image_task_succ_wh_tuples,
            llm_stats=llm_stats,
        )
        print("Generated program:")
        print(new_programplan)
        print("Justification of generated program:")
        print(justification)
        print(llm_stats)
        typer.echo(
            "Press Enter to execute the generated program, or type 'exit' to break."
        )
        is_ready = typer.prompt("Ready?", default="")
        if is_ready == "exit":
            break
        else:
            exec(new_programplan.strip(), {"world": world})
            world.arm_reset()
        typer.echo("Enter 0 if the execution was a failure, 1 if it was a success:")
        execution_success = typer.prompt("Success?", default="")
        execution_success = execution_success == "1"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("Saving ablation tuples.")
        for i, tupl in enumerate(world.subtask_frame_tuples):
            unique_dir_pathname = os.path.join(
                ablation_dir_path,
                f"{experiment_name}_{timestamp}_{i}",
            )
            task, frames = tupl
            init_img = Image.fromarray(frames[0])
            final_img = Image.fromarray(frames[-1])
            # Save image0.png (initial), task, and whether it was successful
            outcome, whathappened = get_who_steps(init_img, final_img, task)
            os.makedirs(unique_dir_pathname, exist_ok=True)
            save_who_ablation_dir(
                unique_dir_pathname, init_img, task, outcome, whathappened
            )
            image_task_succ_wh_tuples.append((init_img, task, outcome, whathappened))
        if ask_save_video:
            prompt_video_save(collect_all_frames(world))


@app.command("positive-icl")
def positive_icl(
    experiment_name: str = typer.Argument(
        help="Name of experiment (will be timestamped also)."
    ),
    icl_dir_path: str = typer.Argument(
        help="Path to positive ICL examples for in-context inclusion.",
    ),
    ask_save_video: bool = typer.Option(
        False, help="Offer to save higher-quality videos."
    )
):
    """
    Run baseline method for positive-only ICL.
    """
    typer.echo("Beginning positive ICL baseline method.")
    world = setup_world()
    llm_stats = LLMStats()
    # Load past experience, if available
    image_task_tuples = _load_experience_subdirs(icl_dir_path, load_icl_dir)
    while True:
        typer.echo(
            "Input: A task instruction to execute by the VLA. Input 'exit' to break:"
        )
        instruction_input = typer.prompt("Instruction")
        if instruction_input == "exit":
            break
        if instruction_input in EVAL_TASKS:
            typer.echo("Found eval task in stored tasks.")
            instruction, _ = get_task_info(instruction_input)
        else:
            instruction = instruction_input
        typer.echo("Generating new plan program...")
        typer.echo(
            "Press Enter once the environment has been re-configured to begin plan generation, or type 'exit' to break."
        )
        is_ready = typer.prompt("Ready?", default="")
        if is_ready == "exit":
            break
        image = world.reset(instruction, refresh_objects=True)
        initial_image = image.copy()
        new_programplan, justification = vlmi.generate_program_with_icl_baseline(
            initial_image,
            instruction,
            object_uids=world.manipulable_object_uids,
            icl_tuples=image_task_tuples,
            llm_stats=llm_stats,
        )
        print("Generated program:")
        print(new_programplan)
        print("Justification of generated program:")
        print(justification)
        print(llm_stats)
        typer.echo(
            "Press Enter to execute the generated program, or type 'exit' to break."
        )
        is_ready = typer.prompt("Ready?", default="")
        if is_ready == "exit":
            break
        else:
            exec(new_programplan.strip(), {"world": world})
            world.arm_reset()
        typer.echo("Enter 0 if the execution was a failure, 1 if it was a success:")
        execution_success = typer.prompt("Success?", default="")
        execution_success = execution_success == "1"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("Checking to save success tuples")
        for i, tupl in enumerate(world.subtask_frame_tuples):
            unique_dir_pathname = os.path.join(
                icl_dir_path,
                f"{experiment_name}_{timestamp}_{i}",
            )
            task, frames = tupl
            init_img = Image.fromarray(frames[0])
            final_img = Image.fromarray(frames[-1])
            # Save image0.png (initial) and task if success
            if vlmi.determine_vla_success(init_img, final_img, task):
                print("Found successful subtask, saving to ICL examples.")
                os.makedirs(unique_dir_pathname, exist_ok=True)
                save_icl_dir(unique_dir_pathname, init_img, task)
                image_task_tuples.append((init_img, task))
        if ask_save_video:
            prompt_video_save(collect_all_frames(world))


@app.command("reflexion-like")
def reflexion_like(
    experiment_name: str = typer.Argument(
        help="Name of experiment (will be timestamped also)."
    ),
    reflexion_dir_path: str = typer.Argument(
        help="Path to reflexion examples for in-context inclusion.",
    ),
    ask_save_video: bool = typer.Option(
        False, help="Offer to save higher-quality videos."
    )
):
    """
    Run Reflexion-like baseline method.
    """
    typer.echo("Beginning reflexion-style baseline method.")
    world = setup_world()
    llm_stats = LLMStats()
    # Load past experience, if available
    image_task_reflection_tuples = _load_experience_subdirs(reflexion_dir_path, load_reflexion_dir)
    while True:
        typer.echo(
            "Input: A task instruction to execute by the VLA. Input 'exit' to break:"
        )
        instruction_input = typer.prompt("Instruction")
        if instruction_input == "exit":
            break
        if instruction_input in EVAL_TASKS:
            typer.echo("Found eval task in stored tasks.")
            instruction, _ = get_task_info(instruction_input)
        else:
            instruction = instruction_input
        typer.echo("Generating new plan program...")
        typer.echo(
            "Press Enter once the environment has been re-configured to begin plan generation, or type 'exit' to break."
        )
        is_ready = typer.prompt("Ready?", default="")
        if is_ready == "exit":
            break
        image = world.reset(instruction, refresh_objects=True)
        initial_image = image.copy()
        new_programplan, justification = vlmi.generate_program_with_reflexion_baseline(
            initial_image,
            instruction,
            object_uids=world.manipulable_object_uids,
            reflexion_tuples=image_task_reflection_tuples,
            llm_stats=llm_stats,
        )
        print("Generated program:")
        print(new_programplan)
        print("Justification of generated program:")
        print(justification)
        print(llm_stats)
        typer.echo(
            "Press Enter to execute the generated program, or type 'exit' to break."
        )
        is_ready = typer.prompt("Ready?", default="")
        if is_ready == "exit":
            break
        else:
            exec(new_programplan.strip(), {"world": world})
            world.arm_reset()
        typer.echo("Enter 0 if the execution was a failure, 1 if it was a success:")
        execution_success = typer.prompt("Success?", default="")
        execution_success = execution_success == "1"
        # Save the reflexion feedback for the overall task
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("Creating reflexion feedback for overall task.")
        overalltaskdir = os.path.join(
            reflexion_dir_path,
            f"{experiment_name}_{timestamp}",
        )
        os.makedirs(overalltaskdir, exist_ok=True)
        print(f"Found {len(world.subtask_frame_tuples)} tuples")
        all_frames = []
        all_tasks = []
        for i, tupl in enumerate(world.subtask_frame_tuples):
            task, frames = tupl
            all_frames.extend(frames)
            all_tasks.append(task)
        print("Generating reflexion feedback...")
        reflection = vlmi.generate_reflexion_feedback(
            instruction, all_tasks, all_frames
        )
        save_reflexion_dir(
            overalltaskdir,
            initial_image,
            instruction,
            reflection,
        )
        image_task_reflection_tuples.append((initial_image, instruction, reflection))
        if ask_save_video:
            prompt_video_save(collect_all_frames(world))


@app.command("test-planner")
def test_planner(
    experience_dir_path: str = typer.Argument(
        help="Path to directory with previous experience to test plan generation on.",
    ),
):
    """
    Program-level ICA tester: generates a full planner program using existing experience.

    Loads experience from ``experience_dir_path``, prompts for a task instruction,
    identifies objects, and generates a plan program (without executing it).
    """
    typer.echo("Beginning test-only method for planner with example experience.")
    llm_stats = LLMStats()
    # Load all TaskICADirs from experience subdirectories
    icadirs = _load_experience_subdirs(experience_dir_path, TaskICADir)
    typer.echo("Input: A task instruction to execute by the VLA. Defaults to emptytask:")
    instruction_input = typer.prompt("Instruction")
    if instruction_input == "exit":
        return
    if instruction_input in EVAL_TASKS:
        typer.echo("Found eval task in stored tasks.")
    else:
        instruction_input = "emptytask"
        typer.echo("Defaulting to emptytask.")
    task_instruction, example_image_path = get_task_info(
        instruction_input
    )
    initial_image = Image.open(example_image_path)
    manipulable_objects = vlmi.get_object_uids_from_scene(
        initial_image, task_instruction
    )
    typer.echo("Generating new plan program...")
    new_programplan, justification = vlmi.generate_planner_program(
        initial_image,
        task_instruction,
        object_uids=manipulable_objects,
        tuple_icadirs=icadirs,
        llm_stats=llm_stats,
    )
    print("Generated program:")
    print(new_programplan)
    print("Justification of generated program:")
    print(justification)
    print(llm_stats)

# --- Main Entry Point ---

if __name__ == "__main__":
    app()
