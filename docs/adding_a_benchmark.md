# Adding a New Benchmark (Simulator Backend)

This guide walks through adding a new simulator environment to robo-eval. By the end, you'll have a new sim backend that the robo-eval orchestrator can use for evaluation.

## Overview

robo-eval runs each simulator as an isolated **sim worker** process (in its own venv) that exposes a FastAPI HTTP interface. The orchestrator communicates with it via `/init`, `/reset`, `/step`, `/obs`, `/success`, and `/close` endpoints.

```
Orchestrator  --HTTP-->  Sim Worker  (/init, /reset, /step, /obs, /success)
     |                       |
     |                  YourBackend class (in sims/sim_worker.py)
     |
     +-----HTTP-->  VLA Policy Server (/predict)
```

## Step 1: Add a SimConfig Entry

Open `robo_eval/config.py` and add your simulator to `SIM_CONFIGS`:

```python
SIM_CONFIGS: Dict[str, SimConfig] = {
    "libero": SimConfig(name="libero", venv=".venvs/libero"),
    "libero_pro": SimConfig(name="libero_pro", venv=".venvs/libero_pro", env_vars={...}),
    # Add your simulator here:
    "mysim": SimConfig(
        name="mysim",
        venv=".venvs/mysim",        # relative to PROJECT_ROOT
        env_vars={                   # optional environment variables
            "MYSIM_DATA_PATH": "/path/to/assets",
        },
    ),
}
```

### SimConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | — | Short identifier, must match the dict key |
| `venv` | str | — | Relative path from project root to the Python venv with the simulator installed |
| `env_vars` | Dict[str, str] | `{}` | Environment variables set before launching the sim worker subprocess |

**Derived properties:**
- `venv_python` → `{PROJECT_ROOT}/{venv}/bin/python`

## Step 2: Add Suite Definitions

Suites define which task sets your simulator supports. Add them in `robo_eval/config.py`:

```python
# 1. Define the suite list
MYSIM_SUITES = ["mysim_easy", "mysim_hard"]

# 2. Add to SUITE_PRESETS so users can run them by name
SUITE_PRESETS: Dict[str, List[str]] = {
    "all": LIBERO_SUITES + LIBERO_PRO_SUITES + MYSIM_SUITES,
    "libero": LIBERO_SUITES,
    "libero_pro": LIBERO_PRO_SUITES,
    "mysim": MYSIM_SUITES,       # <-- add this
    ...
}

# 3. Add max steps per suite in SUITE_MAX_STEPS
SUITE_MAX_STEPS: Dict[str, int] = {
    "libero_spatial": 280,
    ...
    # Add your suites (rollout horizon = max steps per episode):
    "mysim_easy": 300,
    "mysim_hard": 500,
}

# 4. Update get_sim_for_suite() to route your suites to your sim
def get_sim_for_suite(suite: str) -> str:
    if suite in LIBERO_SUITES:
        return "libero"
    elif suite in MYSIM_SUITES:
        return "mysim"              # <-- add this
    else:
        return "libero_pro"
```

Also update the **duplicate** `SUITE_MAX_STEPS` in `sims/env_wrapper.py` (the sim venv uses Python 3.8 which can't import `robo_eval`):

```python
# sims/env_wrapper.py
SUITE_MAX_STEPS = {
    ...
    "mysim_easy": 300,
    "mysim_hard": 500,
}
```

Action spaces are **auto-discovered** at init time via `GET /info` on the sim worker — no
hardcoded dicts needed in `env_wrapper.py`. Your sim backend's `get_info()` method declares
its action space, and `SimWrapper` validates it matches the VLA at startup (see Step 3).

## Step 3: Implement `GET /info`

Every sim backend **must** implement a `get_info()` method that returns the simulator's action and observation space metadata. robo-eval calls `GET /info` on the sim worker at startup to auto-discover capabilities and perform compatibility checking against the VLA policy server.

### Expected Schema

```json
{
  "action_space": {
    "type": "eef_delta",
    "dim": 7
  },
  "obs_space": {
    "cameras": [
      {"key": "agentview_image", "resolution": [256, 256], "role": "primary"},
      {"key": "robot0_eye_in_hand_image", "resolution": [256, 256], "role": "wrist"}
    ],
    "state": {
      "dim": 8,
      "format": "eef_pos(3)+axisangle(3)+gripper_qpos(2)"
    }
  }
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `action_space.type` | str | `"eef_delta"` (end-effector delta) or `"joint_pos"` (joint positions) |
| `action_space.dim` | int | Action dimensionality (7 for LIBERO, 12 for RoboCasa, 14 for RoboTwin) |
| `obs_space.cameras` | list | Available cameras. Each entry has `key` (obs dict key), `resolution` ([H, W]), and `role` (`"primary"` or `"wrist"`) |
| `obs_space.state.dim` | int | Proprioceptive state dimensionality (0 if no state is provided) |
| `obs_space.state.format` | str | Human-readable description of the state vector layout |

### How robo-eval Uses `/info`

At startup, `SimWrapper` fetches `/info` from both the sim worker and the VLA policy server, then performs **compatibility negotiation**:

1. **Action space check:** The VLA's output action type and dim must exactly match the sim's expected action space. Mismatches raise a `ValueError` immediately — no automatic translation or padding is supported.
2. **Camera check:** Every camera role the VLA requires (declared in `obs_requirements.cameras`) must be provided by the sim. Missing cameras raise a `ValueError`.
3. **State dim check:** If the VLA requires proprioceptive state (`state_dim > 0`), the sim must provide matching `obs_space.state.dim`. VLAs with `state_dim=0` skip this check.

This fail-fast approach prevents silent mismatches that would otherwise cause 0% success rates with no error messages.

### Example Implementation

```python
class MySimBackend:
    def get_info(self) -> dict:
        """Return action/obs space metadata for auto-discovery."""
        cam_res = getattr(self, "_cam_res", 256)
        return {
            "action_space": {"type": "eef_delta", "dim": 7},
            "obs_space": {
                "cameras": [
                    {"key": "camera_rgb", "resolution": [cam_res, cam_res], "role": "primary"},
                ],
                "state": {"dim": 8, "format": "eef_pos(3)+axisangle(3)+gripper_qpos(2)"},
            },
        }
```

The sim worker's FastAPI app automatically routes `GET /info` to your backend's `get_info()` method — no additional wiring needed.

## Step 4: Implement the Backend Class (with `get_info`)

Open `sims/sim_worker.py` and add your backend class. Follow the established pattern:

```python
class MySimBackend:
    """Backend for MySim environments.

    Required methods: init(), reset(), step(), get_obs(), check_success(), close()
    Return conventions:
        reset()   -> (image, image2)     where image2 is None if no wrist camera
        step()    -> (image, image2, reward, done, info)
        get_obs() -> (image, image2)
    Images are numpy uint8 arrays, shape (H, W, 3), RGB.
    """

    def __init__(self):
        self.env = None

    def init(self, task_name: str, camera_resolution: int,
             suite: str = None, headless: bool = True):
        """Initialize the simulator for a specific task.

        Args:
            task_name: Task identifier (name or index).
            camera_resolution: Image size (e.g., 256 for 256x256).
            suite: Suite name (e.g., "mysim_easy"). Determines task set.
            headless: True for offscreen rendering (EGL), False for windowed (GLFW).

        Returns:
            Dict with at least {"task_description": "human-readable task string"}.
        """
        os.environ["MUJOCO_GL"] = "egl" if headless else "glfw"

        # TODO: Import your simulator and create the environment
        # from mysim import make_env
        # self.env = make_env(task_name, render_mode="offscreen")

        return {"task_description": "pick up the red block"}

    def reset(self, episode_index: int = None):
        """Reset environment. Returns (primary_image, wrist_image_or_None).

        Args:
            episode_index: Which initial state to load (for deterministic evals).
        """
        obs = self.env.reset()
        img = self._extract_image(obs)
        return img, None  # (image, image2=None if no wrist camera)

    def step(self, action):
        """Execute one action. Returns (image, image2, reward, done, info).

        Args:
            action: List of floats, length = action_dim.
        """
        obs, reward, done, info = self.env.step(np.array(action))
        img = self._extract_image(obs)
        return img, None, reward, done, info

    def get_obs(self):
        """Get current observation without stepping. Returns (image, image2)."""
        obs = self.env.get_observation()
        return self._extract_image(obs), None

    def check_success(self):
        """Check if the task success condition is met. Returns bool."""
        return self.env.check_success()

    def close(self):
        """Release resources."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def _extract_image(self, obs):
        """Extract RGB image as uint8 numpy array from observation dict."""
        img = obs["camera_image"]  # TODO: Use your sim's camera key
        return np.asarray(img, dtype=np.uint8)
```

### Register in the Backend Dict

In `sims/sim_worker.py`, add your class to the `BACKENDS` registry:

```python
BACKENDS = {
    "libero": LiberoBackend,
    "robocasa": RoboCasaBackend,
    "robotwin": RoboTwinBackend,
    "libero_pro": LiberoProBackend,
    "mysim": MySimBackend,          # <-- add this
}
```

That's all — the `/init` endpoint uses this dict to look up and instantiate backends.

## Step 5: Set Up the Virtual Environment

```bash
python3 -m venv .venvs/mysim
source .venvs/mysim/bin/activate
pip install numpy pillow fastapi uvicorn[standard] pydantic
pip install your-simulator-package
```

**Important:** The sim worker (`sims/sim_worker.py`) runs inside this venv. Make sure all simulator dependencies are installed here, not in the base robo-eval venv.

## Step 6: Test

### Manual testing

```bash
# 1. Start the sim worker
MUJOCO_GL=egl .venvs/mysim/bin/python sims/sim_worker.py --sim mysim --port 5001 --headless

# 2. Initialize a task
curl -X POST http://localhost:5001/init \
  -H "Content-Type: application/json" \
  -d '{"sim": "mysim", "task": "0", "suite": "mysim_easy"}'

# 3. Reset
curl -X POST http://localhost:5001/reset \
  -H "Content-Type: application/json" \
  -d '{"episode_index": 0}'

# 4. Step with a random action
curl -X POST http://localhost:5001/step \
  -H "Content-Type: application/json" \
  -d '{"action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}'

# 5. Check success
curl http://localhost:5001/success
```

### End-to-end with robo-eval

```bash
robo-eval run --vla pi05 --suites mysim_easy --episodes 1
```

## Backend Method Reference

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `init` | `(task_name, camera_resolution, suite?, headless?)` | `dict` | Create env, load task. Called once per task. |
| `reset` | `(episode_index?)` | `(image, image2)` | Reset to initial state. Called once per episode. |
| `step` | `(action)` | `(image, image2, reward, done, info)` | Execute action. Called every step. |
| `get_obs` | `()` | `(image, image2)` | Current observation without stepping. |
| `check_success` | `()` | `bool` | Task success condition. |
| `close` | `()` | `None` | Free resources. Called between tasks. |

**Image conventions:**
- numpy uint8 array, shape `(H, W, 3)`, RGB color order
- `image` = primary camera (agentview); `image2` = wrist camera (None if unavailable)
- Flip images to match training data orientation if needed (LIBERO flips 180 degrees)

## Common Pitfalls

### Rendering
- Always set `MUJOCO_GL=egl` for headless mode before any MuJoCo imports.
- The `init()` method should set `os.environ["MUJOCO_GL"]` based on the `headless` parameter.

### Memory Leaks
- The sim worker reuses a single process for multiple tasks. Your `close()` method must fully release the environment.
- The `/init` endpoint calls `close()` on the old backend before creating a new one.

### Action Conventions
- Actions arrive as a flat list of floats from the orchestrator.
- Ensure your sim's `get_info()` action space matches the VLA's (auto-checked at init).

### SUITE_MAX_STEPS Sync
- `SUITE_MAX_STEPS` is defined in **two** places: `robo_eval/config.py` and `sims/env_wrapper.py`.
- The sim venv (often Python 3.8) cannot import `robo_eval`, so `env_wrapper.py` has its own copy.
- If you update one, **update the other**.

### init_states
- If your simulator supports deterministic initial states per episode (like LIBERO's `set_init_state()`), load them in `init()` and apply them in `reset()`.
- If not, `episode_index` can be ignored in `reset()`.
