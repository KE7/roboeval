"""VLM utility functions for prompt formatting and data models."""

from pydantic import BaseModel


def format_action_sequence(action_sequence: list[str]) -> str:
    """Format a list of actions into a numbered sequence for prompt inclusion.

    Example: ["pick up cup", "place on table"] -> "1. pick up cup 2. place on table\\n"
    """
    parts = [f"{i + 1}. {action}" for i, action in enumerate(action_sequence)]
    return " ".join(parts) + "\n"


def format_question_list(question_list: list[str]) -> str:
    """Format a list of questions into a newline-separated string for prompt inclusion."""
    return "".join(f"{question}\n" for question in question_list)


class SceneObject(BaseModel):
    """A manipulable object identified in the scene by the VLM."""

    name: str
    color: str
