# Extending roboeval

This guide describes the public extension points for adding VLA policy servers, simulator backends, and new compatibility paths. The core contract is the same in each direction: every component exposes HTTP endpoints, declares action/observation metadata, and runs in its own environment.

See [docs/architecture.md#extension-cost](architecture.md#extension-cost) for the SLOC breakdown across v0.1.0 VLAs and simulators.

## Add a VLA

### 1. Implement a policy server

Start from the template:

```bash
cp examples/new_vla_template.py sims/vla_policies/my_model_policy.py
```

A VLA policy server must expose:

| Endpoint / method | Purpose |
|---|---|
| `GET /health` | Report whether model loading completed. |
| `GET /info` | Return model name, action space, state requirements, chunk size, and observation requirements. |
| `POST /reset` | Clear per-episode policy state, if any. |
| `POST /predict` | Convert a `VLAObservation` into one or more action vectors. |
| `get_action_spec()` | Return typed `ActionObsSpec` records for action components. |
| `get_observation_spec()` | Return typed `ActionObsSpec` records for required observations. |

The policy class should load weights in `load_model()` and set `self.ready = True` only after loading succeeds. For large models, set `load_in_background = True` so the server can answer `/health` while loading.

### 2. Return unnormalized actions

`predict()` returns real simulator-space actions. If the model predicts normalized or tokenized actions internally, convert them before returning:

```python
def predict(self, obs: VLAObservation) -> list[list[float]]:
    image = decode_primary_image(obs.images["primary"])
    state = obs.state.get("flat") or [0.0] * self._state_dim
    action = self._model.predict(image, obs.instruction, state)
    action = np.asarray(action, dtype=np.float64).reshape(-1)[: self._action_dim]
    return [action.tolist()]
```

The outer list length must match `action_chunk_size` from `get_info()`.

### 3. Declare the action and observation contract

Example LIBERO-style declaration:

```python
from roboeval.specs import ActionObsSpec, GRIPPER_CLOSE_NEG, IMAGE_RGB, LANGUAGE, POSITION_DELTA

def get_action_spec(self):
    return {
        "position": POSITION_DELTA,
        "rotation": ActionObsSpec("rotation", 3, "delta_axisangle", (-3.15, 3.15)),
        "gripper": GRIPPER_CLOSE_NEG,
    }

def get_observation_spec(self):
    return {
        "primary": IMAGE_RGB,
        "state": ActionObsSpec("state", 8, "libero_eef_pos3_aa3_grip2"),
        "instruction": LANGUAGE,
    }
```

Camera roles are semantic names such as `primary`, `wrist`, and `secondary`. State is usually passed under `obs.state["flat"]`, with structured state available under `obs.state["structured"]` for backends that expose named fields.

### 4. Register the VLA

Add the module, venv, and port in `roboeval/server_runner.py`:

```python
_VLA_MODULE_MAP["my_model"] = "sims.vla_policies.my_model_policy"
_VLA_DEFAULT_VENVS["my_model"] = ".venvs/my_model"
_VLA_DEFAULT_PORTS["my_model"] = 5106
```

Add setup support in `scripts/setup.sh` if the model requires a dedicated environment.

### 5. Add tests and a smoke config

Unit tests should check that `get_info()`, `get_action_spec()`, and `get_observation_spec()` are importable without loading model weights. Add a small YAML config under `configs/` once the policy has a simulator pairing that passes the contract gate.

```bash
roboeval serve --vla my_model --sim libero --headless
roboeval test --validate -c configs/my_model_libero_smoke.yaml
roboeval run -c configs/my_model_libero_smoke.yaml
```

## Add a Benchmark

### 1. Implement a simulator backend

Simulator backends are subclasses of `SimBackendBase`, defined in `sims/sim_worker.py`. A backend must implement:

| Method | Purpose |
|---|---|
| `init()` | Create the simulator task and return a task description. |
| `reset()` | Reset the simulator and return primary and optional wrist images. |
| `step()` | Apply an action and return images, reward, done flag, and info. |
| `get_obs()` | Return the current observation without stepping. |
| `check_success()` | Return the simulator's current success signal. |
| `close()` | Release simulator resources. |
| `get_info()` | Return action/observation metadata and optional typed specs. |

Images should be RGB `uint8` arrays with shape `H x W x 3`. `step()` should include a boolean success value in the returned info dict when the simulator provides one.

### 2. Declare backend metadata

`get_info()` is the simulator side of the contract:

```python
def get_info(self) -> dict:
    return {
        "action_space": {"type": "eef_delta", "dim": 7, "accepted_dims": [7]},
        "obs_space": {
            "cameras": [
                {"key": "agentview_image", "resolution": [256, 256], "role": "primary"},
                {"key": "wrist_image", "resolution": [256, 256], "role": "wrist"},
            ],
            "state": {"dim": 8, "format": "eef_pos3_axisangle3_gripper2"},
            "image_transform": "applied_in_sim",
        },
        "max_steps": 280,
        "action_spec": {
            "position": {"name": "position", "dims": 3, "format": "delta_xyz", "range": [-1, 1]},
            "rotation": {"name": "rotation", "dims": 3, "format": "delta_axisangle", "range": [-3.15, 3.15]},
            "gripper": {"name": "gripper", "dims": 1, "format": "binary_close_negative", "range": [-1, 1]},
        },
        "observation_spec": {
            "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
            "state": {"name": "state", "dims": 8, "format": "eef_pos3_axisangle3_gripper2"},
            "instruction": {"name": "language", "dims": 0, "format": "language"},
        },
    }
```

Use `obs_space.image_transform` to state whether the backend already applied any camera transform. LIBERO backends apply the 180-degree camera flip before sending images to the VLA; most non-LIBERO backends should use `none`.

### 3. Register suites and backend launch defaults

Register the backend class in `sims/sim_worker.py`:

```python
BACKENDS["my_sim"] = MySimBackend
```

Register launch defaults in `roboeval/server_runner.py`:

```python
_SIM_DEFAULT_PORTS["my_sim"] = 5309
_SIM_DEFAULT_VENVS["my_sim"] = ".venvs/my_sim"
```

If the simulator has named suites, add suite metadata and max-step defaults in the same places used by existing backends. Add a setup target in `scripts/setup.sh` for any external assets or non-PyPI dependencies.

### 4. Add tests and a smoke config

Backend tests should verify `get_info()` shape and typed-spec fields without requiring a full GPU rollout. Once paired with a VLA, add a config under `configs/` and validate it:

```bash
roboeval serve --vla pi05 --sim my_sim --headless
roboeval test --validate -c configs/pi05_my_sim_smoke.yaml
roboeval run -c configs/pi05_my_sim_smoke.yaml
```

## Extension Architecture

### Component model

Every evaluation run has three optional process groups:

| Process | Required | Interface |
|---|---|---|
| Orchestrator | Yes | `roboeval run`, config loading, contract validation, result writing. |
| VLA policy server | Yes | `/health`, `/info`, `/reset`, `/predict`. |
| Simulator worker | Yes | `/init`, `/reset`, `/step`, `/obs`, `/success`, `/info`, `/close`. |
| VLM planner proxy | Only for LITEN mode | LiteLLM-compatible endpoint used by the planner. |

Each process may use a different Python version and dependency stack. `roboeval serve` uses the registered component name to select the module, virtual environment, and port.

### Contract checks

The orchestrator compares VLA and simulator metadata before running episodes:

- Action dimensionality and action component formats must match.
- Required camera roles must be provided by the simulator.
- State dimensionality and state format must match when state is required.
- Gripper sign convention and rotation representation should be declared explicitly.
- One-sided typed specs are rejected under the default strict setting.

Use `roboeval test --validate -c <config>` as the fast preflight before running GPU rollouts.

### Common contract issues

| Issue | Resolution |
|---|---|
| Gripper opens when the policy intends close | Check `binary_close_negative` vs `binary_close_positive` declarations and conversions. |
| Camera view is rotated or mirrored | Confirm where the transform is applied and set `obs_space.image_transform` / `obs_requirements.image_transform` accordingly. |
| State vector length differs | Align `state_dim` and `ActionObsSpec` declarations before adding adapters. |
| Action chunks have the wrong length | Match `action_chunk_size` in `/info` to the outer list returned by `/predict`. |
| Simulator accepts multiple action dimensions | Declare `accepted_dims`, but keep the policy's emitted action spec exact. |

### LITEN-compatible extensions

LITEN mode calls the same VLA `/predict` endpoint as direct mode. A new VLA or simulator does not need a second implementation for hierarchical evaluation; it only needs to support repeated subtask instructions and normal reset/step semantics. Planner-specific changes belong under `vlm_hl/` and should preserve the `world.act(subtask_instruction)` boundary documented in [liten.md](liten.md).
