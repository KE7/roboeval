"""
Experience I/O Utilities.

Provides save/load helpers for the various experience directory formats used
by the CLI commands (planner, ablation-nor, ablation-who, positive-icl,
reflexion-like).  Each "experience directory" is a flat folder containing a
subset of: image0.png, image1.png, task.txt, success.txt, whathappened.txt,
reasoning.txt, assessment.txt, reflection.txt.

The low-level helpers ``_save_experience_dir`` and ``_load_experience_dir``
eliminate the per-format boilerplate; public wrappers keep the original call
signatures for backward compatibility.
"""

from __future__ import annotations

import os
from typing import Optional

from PIL import Image


# ======================================================================
# Generic save / load helpers
# ======================================================================

def _save_experience_dir(
    parent_dir: str,
    *,
    image0: Optional[Image.Image] = None,
    image1: Optional[Image.Image] = None,
    task: Optional[str] = None,
    success: Optional[bool] = None,
    whathappened: Optional[str] = None,
    reasoning: Optional[str] = None,
    assessment: Optional[str] = None,
    reflection: Optional[str] = None,
) -> str:
    """Save an experience directory with the specified fields.

    Creates ``parent_dir`` if needed.  Only writes files for non-None fields.

    Args:
        parent_dir: Directory path to write into.
        image0: Initial scene image (saved as ``image0.png``).
        image1: Final scene image (saved as ``image1.png``).
        task: Task description text (saved as ``task.txt``).
        success: Whether the task succeeded (saved as ``success.txt``).
        whathappened: Failure critique text (saved as ``whathappened.txt``).
        reasoning: Reasoning text (saved as ``reasoning.txt``).
        assessment: Top-level assessment text (saved as ``assessment.txt``).
        reflection: Reflexion feedback text (saved as ``reflection.txt``).

    Returns:
        The ``parent_dir`` path (for chaining).
    """
    os.makedirs(parent_dir, exist_ok=True)

    if image0 is not None:
        image0.save(os.path.join(parent_dir, "image0.png"))
    if image1 is not None:
        image1.save(os.path.join(parent_dir, "image1.png"))

    _text_fields = {
        "task.txt": task,
        "success.txt": str(success) if success is not None else None,
        "whathappened.txt": whathappened,
        "reasoning.txt": reasoning,
        "assessment.txt": assessment,
        "reflection.txt": reflection,
    }
    for filename, value in _text_fields.items():
        if value is not None:
            with open(os.path.join(parent_dir, filename), "w") as f:
                f.write(value)

    return parent_dir


def _load_text_field(dir_path: str, filename: str) -> Optional[str]:
    """Load a text file from an experience directory, returning None if missing."""
    path = os.path.join(dir_path, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return None


def _load_image(dir_path: str, filename: str = "image0.png") -> Optional[Image.Image]:
    """Load a PNG image from an experience directory, returning None if missing."""
    path = os.path.join(dir_path, filename)
    return Image.open(path) if os.path.exists(path) else None


def _load_success(dir_path: str) -> bool:
    """Load the success flag from an experience directory (defaults to False)."""
    text = _load_text_field(dir_path, "success.txt")
    return text.lower() == "true" if text else False


# ======================================================================
# Public save functions (backward-compatible signatures)
# ======================================================================

def save_top_level_ica_dir(parent_dir, image0, image1, task, success, assessment):
    """Save top-level task ICA directory with before/after images and assessment."""
    return _save_experience_dir(
        parent_dir,
        image0=image0, image1=image1,
        task=task, success=success, assessment=assessment,
    )


def save_icl_dir(parent_dir, image0, task):
    """Save a positive ICL example directory (image + task description)."""
    return _save_experience_dir(parent_dir, image0=image0, task=task)


def save_ablation_dir(parent_dir, image0, task, success):
    """Save a no-reasoning ablation directory (image, task, success flag)."""
    return _save_experience_dir(
        parent_dir, image0=image0, task=task, success=success,
    )


def save_who_ablation_dir(parent_dir, image0, task, success, whathappened):
    """Save a what-happened-only ablation directory (image, task, success, critique)."""
    return _save_experience_dir(
        parent_dir,
        image0=image0, task=task, success=success, whathappened=whathappened,
    )


def save_reflexion_dir(parent_dir, image0, task, reflection):
    """Save a reflexion-style experience directory (image, task, reflection)."""
    return _save_experience_dir(
        parent_dir, image0=image0, task=task, reflection=reflection,
    )


def save_reasoning_ica_dir(parent_dir, image, task, success, whathappened, reasoning):
    """Save an ICA reasoning directory (image, task, success, critique, reasoning).

    Args:
        parent_dir: Path to the directory.
        image: Initial scene image (saved as ``image0.png``).
        task: Task description text.
        success: Whether the subtask succeeded.
        whathappened: VLM critique of what went wrong (None if success).
        reasoning: VLM reasoning about the outcome.

    Returns:
        The ``parent_dir`` path.
    """
    return _save_experience_dir(
        parent_dir,
        image0=image, task=task,
        success=success, whathappened=whathappened, reasoning=reasoning,
    )


# ======================================================================
# Public load functions (backward-compatible signatures)
# ======================================================================

def load_icl_dir(dir_path):
    """Load a positive ICL example directory.

    Returns:
        Tuple of ``(image0, task)`` where each may be None.
    """
    return _load_image(dir_path), _load_text_field(dir_path, "task.txt")


def load_ablation_dir(dir_path):
    """Load a no-reasoning ablation directory.

    Returns:
        Tuple of ``(image0, task, success)``.
    """
    return (
        _load_image(dir_path),
        _load_text_field(dir_path, "task.txt"),
        _load_success(dir_path),
    )


def load_who_ablation_dir(dir_path):
    """Load a what-happened-only ablation directory.

    Returns:
        Tuple of ``(image0, task, success, whathappened)``.
    """
    return (
        _load_image(dir_path),
        _load_text_field(dir_path, "task.txt"),
        _load_success(dir_path),
        _load_text_field(dir_path, "whathappened.txt"),
    )


def load_reflexion_dir(dir_path):
    """Load a reflexion-style experience directory.

    Returns:
        Tuple of ``(image0, task, reflection)``.
    """
    return (
        _load_image(dir_path),
        _load_text_field(dir_path, "task.txt"),
        _load_text_field(dir_path, "reflection.txt"),
    )
