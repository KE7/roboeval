"""
World interfaces for plan execution.

This module defines ``BaseWorldStub``, the abstract base class for execution
environments. Subclasses implement ``act()`` and ``physical_reset()`` for their
target environment.
"""

from PIL import Image


def _vlmi():
    """Lazy import so cv2 (a vlm_hl dep) is not pulled in on non-VLM code paths."""
    import vlm_hl.vlm_methods as _m  # noqa: PLC0415

    return _m


class BaseWorldStub:
    """Abstract base class for world execution environments.

    Manages scene images, manipulable object tracking, subtask frame recording,
    and VLM-based reasoning (true/false, multiple choice, open questions).

    Subclasses must implement:
      - ``act(command)`` — execute a subtask command
      - ``physical_reset()`` — reset the physical environment

    Attributes:
        current_image: Latest scene observation as a PIL Image.
        task_instruction: Natural language description of the current task.
        subtask_frame_tuples: List of ``(command, frames)`` tuples recorded
            during plan execution.
        eval_len: Total number of observation frames across all subtasks.
        manipulable_object_uids: List of object identifiers detected by the VLM.
    """

    def __init__(
        self,
        initial_image: Image.Image | None = None,
        task_instruction: str | None = None,
    ):
        self.subtask_frame_tuples: list = []
        self.eval_len: int = 0
        self.current_image = initial_image
        self.task_instruction = task_instruction
        self.manipulable_object_uids: list = []
        self.execution_trace = None
        self.refresh_objects(initial_image)

    def refresh_objects(self, image: Image.Image | None):
        """Update the list of manipulable objects by querying the VLM.

        Args:
            image: Scene image to identify objects in. Skipped if None.
        """
        if image is not None:
            instruction = self.task_instruction or "No task instruction provided."
            self.manipulable_object_uids = _vlmi().get_object_uids_from_scene(image, instruction)

    def ask_tf(self, question: str) -> bool:
        """Ask a true/false question about the current scene.

        Args:
            question: Natural language yes/no question.

        Returns:
            Boolean answer from the VLM.
        """
        answer = _vlmi().evaluate_tf_question(question, self.current_image)
        if self.execution_trace is not None:
            self.execution_trace.record_reasoning(self.current_image, "ask_tf", question, answer)
        return answer

    def ask_mc(self, question: str, options: list[str]) -> str:
        """Ask a multiple-choice question about the current scene.

        Args:
            question: Natural language question.
            options: List of answer choices.

        Returns:
            The selected option string from the VLM.
        """
        answer = _vlmi().evaluate_mc_question(question, self.current_image, options_list=options)
        if self.execution_trace is not None:
            self.execution_trace.record_reasoning(
                self.current_image, "ask_mc", question, answer, options=options
            )
        return answer

    def ask_question(self, question: str, options: list[str] | None = None):
        """Ask a question about the current scene (open-ended or multiple-choice).

        Args:
            question: Natural language question.
            options: If provided, constrains the answer to these choices.

        Returns:
            VLM response (string for open, selected option for MC).
        """
        if options is None:
            return _vlmi().evaluate_open_question(question, self.current_image)
        return self.ask_mc(question, options)

    def act(self, command: str):
        """Execute a subtask command through the configured controller.

        Must be implemented by subclasses. Should:
          1. Generate actions and execute them.
          2. Append ``(command, frames)`` to ``self.subtask_frame_tuples``.
          3. Update ``self.current_image`` to the final observation.

        Args:
            command: Natural language subtask instruction.
        """
        raise NotImplementedError("act() must be implemented by subclasses.")

    def reset(self, new_task: str | None = None, keep_frames: bool = False):
        """Reset world state and re-identify manipulable objects.

        Clears frame history (unless ``keep_frames=True``), updates the task
        instruction if provided, calls ``physical_reset()``, and refreshes the
        object list from the new scene.

        Returns:
            The current image after reset.
        """
        if not keep_frames:
            self.subtask_frame_tuples = []
        if new_task is not None:
            self.task_instruction = new_task
        self.physical_reset()
        self.refresh_objects(self.current_image)
        return self.current_image

    def physical_reset(self):
        """Reset the physical environment (hardware or simulator).

        Must be implemented by subclasses. Should update ``self.current_image``
        to reflect the post-reset scene.
        """
        raise NotImplementedError("physical_reset() must be implemented by subclasses.")
