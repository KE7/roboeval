"""VLM methods for plan generation, assessment, and reasoning.

This module provides VLM (Vision-Language Model) interaction functions for
object identification, plan program generation, subtask assessment,
success/failure reasoning, and reflexion-style feedback.

All VLM calls are routed through litellm for provider-agnostic model access.
Configure the endpoint with ``setup_litellm()`` or via the litellm proxy.
"""

from __future__ import annotations

import base64
import json
import os
import re
from enum import Enum
from io import BytesIO

import cv2
import litellm
from PIL import Image
from pydantic import BaseModel

from ica.reasoning_ica import ReasoningICADir, TaskICADir

# API key resolution for direct API calls and proxy-compatible setups.
_key_path = os.path.join(os.path.dirname(__file__), "..", "utils", "openaikey.txt")
_openai_key = os.environ.get("OPENAI_API_KEY")
if not _openai_key and os.path.exists(_key_path):
    with open(_key_path) as _f:
        _openai_key = _f.readline().strip()
if not _openai_key:
    _openai_key = "not-needed"

# VLM configuration is set by setup_litellm() or by the defaults below.
vlm_api_base = None
vlm_api_key = _openai_key

# Default OpenAI model names (used when NOT routed through litellm proxy).
openai_vision_model = "gpt-4o-mini"
openai_text_model = "gpt-4o-mini"
openai_reasoning_model = "gpt-4o"

ica_model = openai_reasoning_model
vision_model = openai_vision_model
text_model = openai_text_model

token_usage = 0


def setup_litellm(api_base, api_key="not-needed", model_override=None):
    """Configure litellm proxy connection for all VLM calls.

    Args:
        api_base: Base URL of the litellm proxy (e.g. ``http://localhost:4000/v1``).
        api_key: API key for the proxy (default: ``"not-needed"`` for local proxies).
        model_override: If set, overrides the VLM model name for all call types
            (vision, text, and ICA/reasoning).
    """
    global vlm_api_base, vlm_api_key, vision_model, text_model, ica_model
    global openai_vision_model, openai_text_model, openai_reasoning_model
    vlm_api_base = api_base
    vlm_api_key = api_key
    if model_override:
        vision_model = text_model = ica_model = model_override
        openai_vision_model = openai_text_model = openai_reasoning_model = model_override


def _extract_json(text):
    """Extract JSON from LLM response text."""
    # Try markdown fenced block first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try raw JSON object
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _parse_structured(text, pydantic_model):
    """Parse LLM text output into a Pydantic model."""
    raw = _extract_json(text)
    data = json.loads(raw)
    return pydantic_model.model_validate(data)


def _schema_prompt(pydantic_model):
    """Generate a JSON schema instruction string for the model."""
    schema = pydantic_model.model_json_schema()
    return f"\n\nYou MUST respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}\nRespond ONLY with the JSON object, no other text."


def _append_schema_to_messages(messages, pydantic_model):
    """Append schema instruction to the last user message content."""
    schema_text = _schema_prompt(pydantic_model)
    msgs = [dict(m) for m in messages]  # shallow copy
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            content = msgs[i]["content"]
            if isinstance(content, str):
                msgs[i]["content"] = content + schema_text
            elif isinstance(content, list):
                msgs[i]["content"] = list(content) + [{"type": "text", "text": schema_text}]
            break
    return msgs


def _call_litellm(model, messages, no_think=False):
    """Make a litellm.completion call with configured api_base and api_key.

    When routing through a litellm proxy (vlm_api_base is set), we prefix the
    model with ``openai/`` so that litellm SDK uses the generic OpenAI-compatible
    provider instead of trying to use a native provider (e.g. Ollama).

    no_think behavior:
    - Ollama models: appends ``/no_think`` suffix to model name (Ollama-native)
    - All other models (Gemini, etc.): passes ``thinking={"thinking_budget": 0}``
      to disable reasoning tokens (Gemini thinking budget API)
    """
    effective_model = model
    if vlm_api_base is not None and not model.startswith("openai/"):
        effective_model = f"openai/{model}"
    if no_think and model.startswith("ollama/"):
        effective_model = f"{effective_model}/no_think"
    kwargs = {"model": effective_model, "messages": messages}
    if vlm_api_base is not None:
        kwargs["api_base"] = vlm_api_base
    if vlm_api_key is not None:
        kwargs["api_key"] = vlm_api_key
    if no_think and not model.startswith("ollama/"):
        # Gemini thinking models: budget=0 disables reasoning tokens.
        # Use extra_body (raw passthrough) to avoid litellm's OpenAI param validation.
        kwargs["extra_body"] = {"thinking": {"thinking_budget": 0}}
    if vlm_api_base is not None and "Qwen" in model:
        extra_body = dict(kwargs.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        kwargs["extra_body"] = extra_body
    return litellm.completion(**kwargs)


class LLMStats(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class TFAnswer(BaseModel):
    answer: bool


def format_obj_list(objects_list):
    """Format a list of objects into a comma-separated string for prompt inclusion."""
    if not objects_list:
        return ""
    return ", ".join(str(obj) for obj in objects_list)


def get_hardware_specific_instruction_space():
    """
    Load hardware-specific instruction space details from the configured prompt file.
    """
    with open("vlm_hl/prompts/plan_reasoning/droid_specific_instruction_space.txt") as f:
        sp_instruction_space = f.readlines()
    return format_obj_list(sp_instruction_space)


def vlm_call_with_image(
    image: Image.Image, prompt: str | None = None, model: str | None = None, tf: bool = False
):
    """Call the VLM with an image and text prompt.

    Args:
        image: PIL Image to include in the query.
        prompt: Text prompt for the VLM.
        model: Model name to use (defaults to the configured vision model).
        tf: If True, parse the response as a boolean (TFAnswer schema).

    Returns:
        Boolean if ``tf=True``, otherwise the raw text response.
    """
    if model is None:
        model = vision_model
    image_b64 = encode_image_to_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
    ]
    if tf:
        msgs_with_schema = _append_schema_to_messages(messages, TFAnswer)
        response = _call_litellm(model=model, messages=msgs_with_schema)
        text = response.choices[0].message.content
        parsed = _parse_structured(text, TFAnswer)
        return parsed.answer
    else:
        response = _call_litellm(model=model, messages=messages)
        vlm_response = response.choices[0].message.content.strip()
        return vlm_response


def vlm_call_with_text(prompt: str, model: str | None = None, tf: bool = False):
    """Call the VLM with a text-only prompt (no image).

    Args:
        prompt: Text prompt for the VLM.
        model: Model name to use (defaults to the configured text model).
        tf: If True, parse the response as a boolean (TFAnswer schema).

    Returns:
        Boolean if ``tf=True``, otherwise the raw text response.
    """
    if model is None:
        model = text_model
    messages = [{"role": "user", "content": prompt}]
    if tf:
        msgs_with_schema = _append_schema_to_messages(messages, TFAnswer)
        response = _call_litellm(model=model, messages=msgs_with_schema)
        text = response.choices[0].message.content
        parsed = _parse_structured(text, TFAnswer)
        return parsed.answer
    else:
        response = _call_litellm(model=model, messages=messages)
        vlm_response = response.choices[0].message.content.strip()
        return vlm_response


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode a PIL Image as a base64 JPEG string for VLM API calls."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def evaluate_tf_question(question, image):
    """Ask a true/false question about an image and return a boolean answer."""
    print("Asking VLM the following question: ", question)
    with open("vlm_hl/prompts/plan_reasoning/evaluate_tf_question.txt") as file:
        prompt = file.read().format(question=question)
    response = vlm_call_with_image(image, prompt, tf=True)
    print("VLM response: ", response)
    return response


def evaluate_mc_question(question, image, options_list):
    """Ask a multiple-choice question about an image and return the selected option.

    Args:
        question: Natural language question.
        image: PIL Image showing the scene.
        options_list: List of answer option strings.

    Returns:
        The name of the selected option (from the dynamically generated Enum).
    """
    print("Asking VLM the following question: ", question)
    with open("vlm_hl/prompts/plan_reasoning/evaluate_mc_question.txt") as file:
        prompt = file.read()
    prompt = prompt.format(question=question, options=format_obj_list(options_list))
    image_b64 = encode_image_to_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
    ]
    OptionsEnum = Enum("Options", options_list)

    class MCOptions(BaseModel):
        selection: OptionsEnum

    msgs_with_schema = _append_schema_to_messages(messages, MCOptions)
    response = _call_litellm(model=text_model, messages=msgs_with_schema)
    text = response.choices[0].message.content
    parsed = _parse_structured(text, MCOptions)
    print("VLM response: ", parsed.selection.name)
    return parsed.selection.name


def evaluate_open_question(question, image):
    """Ask an open-ended question about an image and return the VLM's text response."""
    print("Asking VLM the following question: ", question)
    with open("vlm_hl/prompts/plan_reasoning/evaluate_open_question.txt") as file:
        prompt = file.read().format(question=question)
    response = vlm_call_with_image(image, prompt)
    print("VLM response: ", response)
    return response


class ObjectUids(BaseModel):
    object_uids: list[str]


def get_object_uids_from_scene(current_image: Image, task_instruction: str):
    """
    Given an image, query a VLM to identify objects in the image and output a list of objects.
    """
    with open("vlm_hl/prompts/plan_reasoning/generate_uids.txt") as file:
        prompt = file.read()
    prompt = prompt.format(instruction=task_instruction)
    image_b64 = encode_image_to_base64(current_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
    ]
    msgs_with_schema = _append_schema_to_messages(messages, ObjectUids)
    response = _call_litellm(model=vision_model, messages=msgs_with_schema)
    text = response.choices[0].message.content
    parsed = _parse_structured(text, ObjectUids)
    print("Identified objects from scene: ", parsed.object_uids)
    return parsed.object_uids


def extract_frames(video_path, frame_rate=10):
    """
    Extract frames from a video.
    Always includes the first and last frame, plus every `frame_rate`th frame.
    Returns a list of base64-encoded JPEG strings.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames_b64 = []
    count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save first, every `frame_rate`th, and last frame
        if count == 0 or count % frame_rate == 0 or count == total_frames - 1:
            _, buffer = cv2.imencode(".jpg", frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            frames_b64.append(frame_b64)

        count += 1

    cap.release()
    return frames_b64


def extract_frames_from_list(frame_list, frame_rate=10):
    """
    Extract frames from a list of numpy array images.
    Always includes the first and last frame, plus every `frame_rate`th frame.
    Returns a list of base64-encoded JPEG strings.
    """
    frames_b64 = []
    total_frames = len(frame_list)
    for count, frame in enumerate(frame_list):
        if count == 0 or count % frame_rate == 0 or count == total_frames - 1:
            _, buffer = cv2.imencode(".jpg", frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            frames_b64.append(frame_b64)
    return frames_b64


def generate_reflexion_feedback(overall_instruction, subtasks, attempt_video_frames, frame_rate=20):
    """
    Using a video trajectory for an unsuccessful attempt, ask for reflexion-style feedback for use in future attempts.
    """
    frames_b64 = extract_frames_from_list(attempt_video_frames, frame_rate=frame_rate)
    with open("vlm_hl/prompts/assessment/reflexion_feedback.txt", encoding="utf-8") as f:
        prompt = f.read().format(overall_task=overall_instruction, subtask_instructions=subtasks)
    content = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in frames_b64
    )
    messages = [{"role": "user", "content": content}]
    response = _call_litellm(model=ica_model, messages=messages)
    return response.choices[0].message.content.strip()


def build_reasoning_tuple_subdir(overall_task_idx: int, attempt_idx: int, icadir: ReasoningICADir):
    """
    Build an API message structure for a robot attempt directory using ReasoningICADir.
    """
    att_pre = f"VLA_ATTEMPT_{overall_task_idx}_{attempt_idx}"

    # Load images
    reasoning_tuple_dir = icadir.get_reasoning_tuple()
    if reasoning_tuple_dir is None:
        return None
    initial_image = reasoning_tuple_dir["image0"]
    task = reasoning_tuple_dir["task"]
    success = reasoning_tuple_dir["success"]
    whathappened = reasoning_tuple_dir["whathappened"]
    reasoning = reasoning_tuple_dir["reasoning"]

    # Build message content
    content = [
        {"type": "text", "text": f"`[{att_pre}]`:"},
        {"type": "text", "text": f"`[{att_pre}_INITIAL_IMAGE]`:"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(initial_image)}"},
        },
        {
            "type": "text",
            "text": f"`[VLA_INSTRUCTION {overall_task_idx}_{attempt_idx}]`: {task}",
        },
        {
            "type": "text",
            "text": f"`[{att_pre}_SUCCESS]`: {success}",
        },
    ]
    if whathappened is not None:
        content.append(
            {
                "type": "text",
                "text": f"`[{att_pre}_WHAT_HAPPENED]`: {whathappened}",
            }
        )
    content.append({"type": "text", "text": f"`[{att_pre}_REASONING]`: {reasoning}"})

    user_msg = {"role": "user", "content": content}
    return user_msg


def build_top_level_reasoning_tuple(tlicadir: TaskICADir, overall_task_idx: int):
    exec_pre = f"EXECUTION_{overall_task_idx}"

    # Load images
    task_tuple_dict = tlicadir.get_task_tuple()
    initial_image = task_tuple_dict["image0"]
    final_image = task_tuple_dict["image1"]
    overall_task = task_tuple_dict["task"]
    success = task_tuple_dict["success"]
    assessment = task_tuple_dict["assessment"]
    subtask_ica_dirs = task_tuple_dict["subtasks"]

    if initial_image is None or final_image is None or overall_task is None:
        return []

    # build subtask messages
    subtask_attempt_messages = build_multi_reasoning_tuples(subtask_ica_dirs, overall_task_idx)

    # Build message content
    content = [
        {"type": "text", "text": f"`[{exec_pre}]`:"},
        {"type": "text", "text": f"`[{exec_pre}_INITIAL_IMAGE]`:"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(initial_image)}"},
        },
        {
            "type": "text",
            "text": f"`[OVERALL TASK {overall_task_idx}]`: {overall_task}",
        },
        {"type": "text", "text": f"`[{exec_pre}_FINAL_IMAGE]`:"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(final_image)}"},
        },
        {
            "type": "text",
            "text": f"`[SUCCESS {overall_task_idx}]`: {success}",
        },
        {
            "type": "text",
            "text": f"`[ASSESSMENT {overall_task_idx}]`: {assessment}",
        },
    ]

    top_lvl_msg = {"role": "user", "content": content}
    user_msgs = [top_lvl_msg] + subtask_attempt_messages
    return user_msgs


def build_multi_reasoning_tuples(icadirs: list[ReasoningICADir], overall_task_idx: int):
    """
    Build a combined API message structure for multiple robot attempts.

    Args:
        icadirs (list[ReasoningICADir]): List of ReasoningICADir instances.
    Returns:
        messages (list[dict]): A list suitable for chat completions API.
    """
    messages = []
    for idx, icadir in enumerate(icadirs):
        result = build_reasoning_tuple_subdir(overall_task_idx, idx, icadir)
        if result is None:
            continue
        messages.append(result)
    return messages


def build_multi_task_tuples(taskicadirs: list[TaskICADir]):
    """
    Build a combined API message structure for multiple robot attempts.

    Args:
        icadirs (list[ReasoningICADir]): List of ReasoningICADir instances.
    Returns:
        messages (list[dict]): A list suitable for chat completions API.
    """
    messages = []
    for idx, icadir in enumerate(taskicadirs):
        messages.extend(build_top_level_reasoning_tuple(icadir, idx))
    return messages


def build_positive_icl_examples(icl_tuples: list):
    """
    Build a combined API message structure for positive ICL examples.

    Args:
        icl_tuples (list): List of ICL tuples.
    Returns:
        messages (list[dict]): A list suitable for chat completions API.
    """
    messages = []
    for idx, (image, task) in enumerate(icl_tuples):
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"`[INSTRUCTION_{idx}_INITIAL_IMAGE]`:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_to_base64(image)}"
                        },
                    },
                    {"type": "text", "text": f"`[INSTRUCTION_{idx}]`: {task}"},
                ],
            }
        )
    return messages


def build_ablation_examples(ablation_tuples: list):
    """
    Build a combined API message structure for ablation examples.

    Args:
        icl_tuples (list): List of ICL tuples.
    Returns:
        messages (list[dict]): A list suitable for chat completions API.
    """
    messages = []
    for idx, (image, task, success) in enumerate(ablation_tuples):
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"`[ATTEMPT_{idx}_INITIAL_IMAGE]`:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_to_base64(image)}"
                        },
                    },
                    {"type": "text", "text": f"`[INSTRUCTION {idx}]`: {task}"},
                    {
                        "type": "text",
                        "text": f"`[ATTEMPT_{idx}_SUCCESS]`: {success}",
                    },
                ],
            }
        )
    return messages


def build_who_ablation_examples(ablation_tuples: list):
    """
    Build a combined API message structure for WHO ablation examples.

    Args:
        icl_tuples (list): List of ICL tuples.
    Returns:
        messages (list[dict]): A list suitable for chat completions API.
    """
    messages = []
    for idx, (image, task, success, whathappened) in enumerate(ablation_tuples):
        if whathappened is None:
            whathappened = "N/A"
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"`[ATTEMPT_{idx}_INITIAL_IMAGE]`:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_to_base64(image)}"
                        },
                    },
                    {"type": "text", "text": f"`[INSTRUCTION {idx}]`: {task}"},
                    {
                        "type": "text",
                        "text": f"`[ATTEMPT_{idx}_SUCCESS]`: {success}",
                    },
                    {
                        "type": "text",
                        "text": f"`[ATTEMPT_{idx}_WHAT_HAPPENED]`: {whathappened}",
                    },
                ],
            }
        )
    return messages


def build_reflexion_examples(reflexion_tuples: list):
    """
    Build a combined API message structure for reflexion-style examples.

    Args:
        icl_tuples (list): List of ICL tuples.
    Returns:
        messages (list[dict]): A list suitable for chat completions API.
    """
    messages = []
    for idx, (image, task, reflection) in enumerate(reflexion_tuples):
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"`[INSTRUCTION_{idx}_INITIAL_IMAGE]`:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_to_base64(image)}"
                        },
                    },
                    {"type": "text", "text": f"`[INSTRUCTION_{idx}]`: {task}"},
                    {
                        "type": "text",
                        "text": f"`[REFLECTION_{idx}]`: {reflection}",
                    },
                ],
            }
        )
    return messages


def generate_program_with_reflexion_baseline(
    initial_image: Image,
    task_description: str,
    object_uids: list[str],
    reflexion_tuples: list = None,
    llm_stats: LLMStats = LLMStats(),
):
    reflexion_tuples = [] if reflexion_tuples is None else reflexion_tuples
    with open(
        "vlm_hl/prompts/plan_reasoning/pizerofive_droid_instruction_space.txt", encoding="utf-8"
    ) as f:
        instruction_space = f.read()
    if len(reflexion_tuples) > 0:
        with open(
            "vlm_hl/prompts/plan_reasoning/reflexion_examples_card.txt",
            encoding="utf-8",
        ) as f:
            icl_examples_card = f.read()
    else:
        icl_examples_card = ""
    with open(
        "vlm_hl/prompts/plan_reasoning/generate_program_with_context.txt",
        encoding="utf-8",
    ) as f:
        unformatted_system_msg = f.read()
    formatted_system_msg = unformatted_system_msg.format(
        instruction_space=instruction_space, ica_examples_card=icl_examples_card
    )
    system_msg = {"role": "system", "content": formatted_system_msg}

    # Hardware-specific instruction addendum
    content = []
    with open("vlm_hl/prompts/plan_reasoning/format_task.txt", encoding="utf-8") as f:
        unformatted_task_prompt = f.read()
    sp_instruction_addendum = get_hardware_specific_instruction_space()
    formatted_prompt = unformatted_task_prompt.format(
        task_instruction=task_description,
        initial_manipulable_objects=format_obj_list(object_uids),
        sp_instruction_addendum=sp_instruction_addendum,
    )
    content.append({"type": "text", "text": formatted_prompt})

    enc_image = encode_image_to_base64(initial_image)
    content.append({"type": "text", "text": "`[INITIAL_ENVIRONMENT_IMAGE]`:"})
    content.append(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc_image}"}}
    )
    prev_attempt_messages = []
    if len(reflexion_tuples) > 0:
        prev_attempt_messages = build_reflexion_examples(reflexion_tuples)

    messages = [
        system_msg,
        *prev_attempt_messages,  # one user message per attempt
        {
            "role": "user",
            "content": content,
        },  # top-level prompt command + initial image
    ]

    msgs_with_schema = _append_schema_to_messages(messages, VLAPlanProgram)
    response = _call_litellm(model=ica_model, messages=msgs_with_schema)
    text = response.choices[0].message.content
    parsed = _parse_structured(text, VLAPlanProgram)
    llm_stats.input_tokens += response.usage.prompt_tokens
    llm_stats.output_tokens += response.usage.completion_tokens
    formatted_code = parsed.python_code.strip("`").replace("python", "").strip()
    return formatted_code, parsed.reasoning


def generate_program_with_icl_baseline(
    initial_image: Image,
    task_description: str,
    object_uids: list[str],
    icl_tuples: list = None,
    llm_stats: LLMStats = LLMStats(),
):
    icl_tuples = [] if icl_tuples is None else icl_tuples
    with open(
        "vlm_hl/prompts/plan_reasoning/pizerofive_droid_instruction_space.txt", encoding="utf-8"
    ) as f:
        instruction_space = f.read()
    if len(icl_tuples) > 0:
        with open(
            "vlm_hl/prompts/plan_reasoning/icl_baseline_examples_card.txt",
            encoding="utf-8",
        ) as f:
            icl_examples_card = f.read()
    else:
        icl_examples_card = ""
    with open(
        "vlm_hl/prompts/plan_reasoning/generate_program_with_context.txt",
        encoding="utf-8",
    ) as f:
        unformatted_system_msg = f.read()
    formatted_system_msg = unformatted_system_msg.format(
        instruction_space=instruction_space, ica_examples_card=icl_examples_card
    )
    system_msg = {"role": "system", "content": formatted_system_msg}

    # Top-level user message
    content = []
    with open("vlm_hl/prompts/plan_reasoning/format_task.txt", encoding="utf-8") as f:
        unformatted_task_prompt = f.read()
    sp_instruction_addendum = get_hardware_specific_instruction_space()
    formatted_prompt = unformatted_task_prompt.format(
        task_instruction=task_description,
        initial_manipulable_objects=format_obj_list(object_uids),
        sp_instruction_addendum=sp_instruction_addendum,
    )
    content.append({"type": "text", "text": formatted_prompt})

    enc_image = encode_image_to_base64(initial_image)
    content.append({"type": "text", "text": "`[INITIAL_ENVIRONMENT_IMAGE]`:"})
    content.append(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc_image}"}}
    )
    prev_attempt_messages = []
    if len(icl_tuples) > 0:
        prev_attempt_messages = build_positive_icl_examples(icl_tuples)

    messages = [
        system_msg,
        *prev_attempt_messages,  # one user message per attempt
        {
            "role": "user",
            "content": content,
        },  # top-level prompt command + initial image
    ]

    msgs_with_schema = _append_schema_to_messages(messages, VLAPlanProgram)
    response = _call_litellm(model=ica_model, messages=msgs_with_schema)
    text = response.choices[0].message.content
    parsed = _parse_structured(text, VLAPlanProgram)
    llm_stats.input_tokens += response.usage.prompt_tokens
    llm_stats.output_tokens += response.usage.completion_tokens
    formatted_code = parsed.python_code.strip("`").replace("python", "").strip()
    return formatted_code, parsed.reasoning


def generate_program_with_nor_ablation(
    initial_image: Image,
    task_description: str,
    object_uids: list[str],
    subtask_tuples: list = None,
    llm_stats: LLMStats = LLMStats(),
):
    subtask_tuples = [] if subtask_tuples is None else subtask_tuples
    with open(
        "vlm_hl/prompts/plan_reasoning/pizerofive_droid_instruction_space.txt", encoding="utf-8"
    ) as f:
        instruction_space = f.read()
    if len(subtask_tuples) > 0:
        with open(
            "vlm_hl/prompts/plan_reasoning/ablation_examples_card.txt",
            encoding="utf-8",
        ) as f:
            icl_examples_card = f.read()
    else:
        icl_examples_card = ""
    with open(
        "vlm_hl/prompts/plan_reasoning/generate_program_with_context.txt",
        encoding="utf-8",
    ) as f:
        unformatted_system_msg = f.read()
    formatted_system_msg = unformatted_system_msg.format(
        instruction_space=instruction_space, ica_examples_card=icl_examples_card
    )
    system_msg = {"role": "system", "content": formatted_system_msg}

    # Top-level user message
    content = []
    with open("vlm_hl/prompts/plan_reasoning/format_task.txt", encoding="utf-8") as f:
        unformatted_task_prompt = f.read()
    sp_instruction_addendum = get_hardware_specific_instruction_space()
    formatted_prompt = unformatted_task_prompt.format(
        task_instruction=task_description,
        initial_manipulable_objects=format_obj_list(object_uids),
        sp_instruction_addendum=sp_instruction_addendum,
    )
    content.append({"type": "text", "text": formatted_prompt})

    enc_image = encode_image_to_base64(initial_image)
    content.append({"type": "text", "text": "`[INITIAL_ENVIRONMENT_IMAGE]`:"})
    content.append(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc_image}"}}
    )
    prev_attempt_messages = []
    if len(subtask_tuples) > 0:
        prev_attempt_messages = build_ablation_examples(subtask_tuples)

    messages = [
        system_msg,
        *prev_attempt_messages,  # one user message per attempt
        {
            "role": "user",
            "content": content,
        },  # top-level prompt command + initial image
    ]

    msgs_with_schema = _append_schema_to_messages(messages, VLAPlanProgram)
    response = _call_litellm(model=ica_model, messages=msgs_with_schema)
    text = response.choices[0].message.content
    parsed = _parse_structured(text, VLAPlanProgram)
    llm_stats.input_tokens += response.usage.prompt_tokens
    llm_stats.output_tokens += response.usage.completion_tokens
    formatted_code = parsed.python_code.strip("`").replace("python", "").strip()
    return formatted_code, parsed.reasoning


def generate_program_with_who_ablation(
    initial_image: Image,
    task_description: str,
    object_uids: list[str],
    subtask_tuples: list = None,
    llm_stats: LLMStats = LLMStats(),
):
    subtask_tuples = [] if subtask_tuples is None else subtask_tuples
    with open(
        "vlm_hl/prompts/plan_reasoning/pizerofive_droid_instruction_space.txt", encoding="utf-8"
    ) as f:
        instruction_space = f.read()
    if len(subtask_tuples) > 0:
        with open(
            "vlm_hl/prompts/plan_reasoning/who_ablation_examples_card.txt",
            encoding="utf-8",
        ) as f:
            icl_examples_card = f.read()
    else:
        icl_examples_card = ""
    with open(
        "vlm_hl/prompts/plan_reasoning/generate_program_with_context.txt",
        encoding="utf-8",
    ) as f:
        unformatted_system_msg = f.read()
    formatted_system_msg = unformatted_system_msg.format(
        instruction_space=instruction_space, ica_examples_card=icl_examples_card
    )
    system_msg = {"role": "system", "content": formatted_system_msg}

    # Top-level user message
    content = []
    with open("vlm_hl/prompts/plan_reasoning/format_task.txt", encoding="utf-8") as f:
        unformatted_task_prompt = f.read()
    sp_instruction_addendum = get_hardware_specific_instruction_space()
    formatted_prompt = unformatted_task_prompt.format(
        task_instruction=task_description,
        initial_manipulable_objects=format_obj_list(object_uids),
        sp_instruction_addendum=sp_instruction_addendum,
    )
    content.append({"type": "text", "text": formatted_prompt})

    enc_image = encode_image_to_base64(initial_image)
    content.append({"type": "text", "text": "`[INITIAL_ENVIRONMENT_IMAGE]`:"})
    content.append(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc_image}"}}
    )
    prev_attempt_messages = []
    if len(subtask_tuples) > 0:
        prev_attempt_messages = build_who_ablation_examples(subtask_tuples)

    messages = [
        system_msg,
        *prev_attempt_messages,  # one user message per attempt
        {
            "role": "user",
            "content": content,
        },  # top-level prompt command + initial image
    ]

    msgs_with_schema = _append_schema_to_messages(messages, VLAPlanProgram)
    response = _call_litellm(model=ica_model, messages=msgs_with_schema)
    text = response.choices[0].message.content
    parsed = _parse_structured(text, VLAPlanProgram)
    llm_stats.input_tokens += response.usage.prompt_tokens
    llm_stats.output_tokens += response.usage.completion_tokens
    formatted_code = parsed.python_code.strip("`").replace("python", "").strip()
    return formatted_code, parsed.reasoning


class VLAPlanProgram(BaseModel):
    python_code: str
    reasoning: str


def generate_planner_program(
    initial_image: Image,
    task_description: str,
    object_uids: list[str],
    tuple_icadirs: list = None,
    llm_stats: LLMStats = LLMStats(),
    no_think: bool = False,
    no_vlm: bool = False,
):
    if no_vlm:
        return (
            f"world.act({repr(task_description)})",
            "no-vlm: bypassing VLM, using raw task description",
        )

    with open(
        "vlm_hl/prompts/plan_reasoning/pizerofive_droid_instruction_space.txt", encoding="utf-8"
    ) as f:
        instruction_space = f.read()
    if len(tuple_icadirs) > 0:
        with open(
            "vlm_hl/prompts/plan_reasoning/planner_examples_card.txt",
            encoding="utf-8",
        ) as f:
            planner_examples_card = f.read()
    else:
        planner_examples_card = ""

    with open(
        "vlm_hl/prompts/plan_reasoning/generate_program_with_context.txt",
        encoding="utf-8",
    ) as f:
        unformatted_system_msg = f.read()
    formatted_system_msg = unformatted_system_msg.format(
        instruction_space=instruction_space, ica_examples_card=planner_examples_card
    )
    system_msg = {"role": "system", "content": formatted_system_msg}

    # Top-level user message
    content = []
    with open("vlm_hl/prompts/plan_reasoning/format_task.txt", encoding="utf-8") as f:
        unformatted_task_prompt = f.read()
    sp_instruction_addendum = get_hardware_specific_instruction_space()
    formatted_prompt = unformatted_task_prompt.format(
        task_instruction=task_description,
        initial_manipulable_objects=format_obj_list(object_uids),
        sp_instruction_addendum=sp_instruction_addendum,
    )
    content.append({"type": "text", "text": formatted_prompt})

    enc_image = encode_image_to_base64(initial_image)
    content.append({"type": "text", "text": "`[INITIAL_ENVIRONMENT_IMAGE]`:"})
    content.append(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc_image}"}}
    )
    prev_attempt_messages = []
    if tuple_icadirs is not None:
        prev_attempt_messages = build_multi_task_tuples(tuple_icadirs)

    messages = [
        system_msg,
        *prev_attempt_messages,  # one user message per attempt
        {
            "role": "user",
            "content": content,
        },  # top-level prompt command + initial image
    ]

    msgs_with_schema = _append_schema_to_messages(messages, VLAPlanProgram)
    response = _call_litellm(model=ica_model, messages=msgs_with_schema, no_think=no_think)
    text = response.choices[0].message.content
    parsed = _parse_structured(text, VLAPlanProgram)
    llm_stats.input_tokens += response.usage.prompt_tokens
    llm_stats.output_tokens += response.usage.completion_tokens
    formatted_code = parsed.python_code.strip("`").replace("python", "").strip()
    return formatted_code, parsed.reasoning


def critique_vla_failure(initial_image: Image, final_image: Image, task_description: str):
    with open("vlm_hl/prompts/assessment/critique_failure_from_images.txt") as file:
        unformatted_prompt = file.read()
    formatted_prompt = unformatted_prompt.format(task_instruction=task_description)
    messages = format_two_image_message(initial_image, final_image, formatted_prompt)
    response = _call_litellm(model=vision_model, messages=messages)
    return response.choices[0].message.content.strip()


def critique_vla_video_failure(video_frames: list, task_description: str, frame_rate=20):
    with open("vlm_hl/prompts/assessment/critique_failure_from_video.txt") as file:
        unformatted_prompt = file.read()
    frames_b64 = extract_frames_from_list(video_frames, frame_rate=frame_rate)
    formatted_prompt = unformatted_prompt.format(task_instruction=task_description)
    content = [{"type": "text", "text": formatted_prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in frames_b64
    )
    messages = [{"role": "user", "content": content}]
    response = _call_litellm(model=ica_model, messages=messages)
    return response.choices[0].message.content.strip()


def format_reasoning_tuples(reasoning_tuples: list):
    formatted_tuples = ""
    for i, (subtask, success, what_happened, reasoning) in enumerate(reasoning_tuples):
        if what_happened is None:
            what_happened = "N/A"
        formatted_tuples += f"Subtask {i + 1}:\n Subtask Instruction: {subtask}\n Success?: {success}\n What Happened (if failure): {what_happened}\n Reasoning: {reasoning}\n"
    return formatted_tuples


def assess_hl_success(
    initial_image: Image,
    final_image: Image,
    task_description: str,
    reasoning_tuples: list,
):
    with open("vlm_hl/prompts/assessment/describe_hl_success_scene.txt") as file:
        unformatted_prompt = file.read()
    with open("vlm_hl/prompts/assessment/pizerofive_droid_vla_model_card.txt") as file:
        model_card = file.read()
    formatted_prompt = unformatted_prompt.format(
        model_card=model_card,
        task_instruction=task_description,
        subtask_reasoning_tuples=format_reasoning_tuples(reasoning_tuples),
    )
    messages = format_two_image_message(initial_image, final_image, formatted_prompt)
    response = _call_litellm(model=vision_model, messages=messages)
    return response.choices[0].message.content.strip()


def assess_hl_failure(
    initial_image: Image,
    final_image: Image,
    task_description: str,
    reasoning_tuples: list,
):
    with open("vlm_hl/prompts/assessment/critique_hl_failure_scene.txt") as file:
        unformatted_prompt = file.read()
    with open("vlm_hl/prompts/assessment/pizerofive_droid_vla_model_card.txt") as file:
        model_card = file.read()
    with open(
        "vlm_hl/prompts/plan_reasoning/pizerofive_droid_instruction_space.txt", encoding="utf-8"
    ) as f:
        instruction_space = f.read()
    formatted_prompt = unformatted_prompt.format(
        model_card=model_card,
        task_instruction=task_description,
        subtask_reasoning_tuples=format_reasoning_tuples(reasoning_tuples),
        instruction_space=instruction_space,
    )
    messages = format_two_image_message(initial_image, final_image, formatted_prompt)
    response = _call_litellm(model=vision_model, messages=messages)
    return response.choices[0].message.content.strip()


def describe_vla_success(initial_image: Image, task_description: str):
    with open("vlm_hl/prompts/assessment/describe_success_scene.txt") as file:
        unformatted_prompt = file.read()
    formatted_prompt = unformatted_prompt.format(task_instruction=task_description)
    response = vlm_call_with_image(initial_image, formatted_prompt, model=text_model)
    return response


def determine_vla_success(initial_image: Image, final_image: Image, task_description: str):
    with open("vlm_hl/prompts/assessment/determine_success_from_images.txt") as file:
        unformatted_prompt = file.read()
    formatted_prompt = unformatted_prompt.format(task_instruction=task_description)
    messages = format_two_image_message(initial_image, final_image, formatted_prompt)
    msgs_with_schema = _append_schema_to_messages(messages, TFAnswer)
    response = _call_litellm(model=vision_model, messages=msgs_with_schema)
    text = response.choices[0].message.content
    parsed = _parse_structured(text, TFAnswer)
    return parsed.answer


def reason_about_vla_failure(initial_image: Image, task_description: str, what_happened: str):
    with open("vlm_hl/prompts/assessment/reason_about_failure.txt") as file:
        unformatted_prompt = file.read()

    with open("vlm_hl/prompts/assessment/pizerofive_droid_vla_model_card.txt") as file:
        model_card = file.read()
    formatted_prompt = unformatted_prompt.format(
        model_card=model_card,
        task_instruction=task_description,
        what_happened=what_happened,
    )

    response = vlm_call_with_image(initial_image, formatted_prompt, model=text_model)
    return response


def format_two_image_message(initial_image: Image, final_image: Image, formatted_prompt: str):
    initial_image_b64 = encode_image_to_base64(initial_image)
    final_image_b64 = encode_image_to_base64(final_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": formatted_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{initial_image_b64}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{final_image_b64}"},
                },
            ],
        }
    ]
    return messages
