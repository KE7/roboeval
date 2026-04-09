import base64
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from sims.env_wrapper import SimWrapper
from world_stubs import BaseWorldStub


def _encode_image(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class _DummyResponse:
    status_code = 200

    def __init__(self, payload=None):
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_act_uses_reset_observation_for_first_vla_call(monkeypatch):
    init_img = Image.fromarray(np.full((8, 8, 3), 10, dtype=np.uint8))
    reset_img = Image.fromarray(np.full((8, 8, 3), 20, dtype=np.uint8))
    step_img = Image.fromarray(np.full((8, 8, 3), 30, dtype=np.uint8))

    def fake_world_init(self, initial_image=None, task_instruction=None):
        self.subtask_frame_tuples = []
        self.eval_len = 0
        self.current_image = initial_image
        self.task_instruction = task_instruction
        self.manipulable_object_uids = []
        self.execution_trace = None

    def fake_fetch_policy_info(self):
        self._policy_info = {"model_id": "fake"}
        self._policy_action_space = {"type": "eef_delta", "dim": 7}

    def fake_fetch_sim_info(self):
        self._sim_info = {
            "sim": "libero_infinity",
            "action_space": {"type": "eef_delta", "dim": 7},
            "obs_space": {"cameras": [{"role": "primary"}], "state": {"dim": 8}},
        }
        self._sim_action_space = self._sim_info["action_space"]

    def fake_post(self, path, json_data=None):
        if path == "/init":
            return {"success": True, "task_description": "demo task"}
        if path == "/reset":
            return {"image": _encode_image(reset_img)}
        if path == "/step":
            return {"image": _encode_image(step_img), "done": True}
        raise AssertionError(f"unexpected POST {path}")

    monkeypatch.setattr(BaseWorldStub, "__init__", fake_world_init)
    monkeypatch.setattr(SimWrapper, "_fetch_policy_info", fake_fetch_policy_info)
    monkeypatch.setattr(SimWrapper, "_fetch_sim_info", fake_fetch_sim_info)
    monkeypatch.setattr(SimWrapper, "_post", fake_post)
    monkeypatch.setattr(SimWrapper, "_get_obs_image", lambda self: init_img)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _DummyResponse())

    wrapper = SimWrapper(
        sim_server_url="http://fake-sim",
        sim_name="libero_infinity",
        task_name="0",
        no_vlm=True,
    )
    wrapper.physical_reset()

    def fail_get_obs():
        raise AssertionError("act() should use the reset image directly")

    captured = {}

    def fake_get_vla_actions(image, instruction, state=None, image2=None, image3=None, state_dict=None):
        captured["image"] = np.array(image)
        captured["instruction"] = instruction
        return [[0.0] * 7]

    monkeypatch.setattr(wrapper, "_get_obs_image", fail_get_obs)
    monkeypatch.setattr(wrapper, "_get_vla_actions", fake_get_vla_actions)

    wrapper.act("pick up the bowl")

    assert np.array_equal(captured["image"], np.array(reset_img))
    assert captured["instruction"] == "pick up the bowl"
