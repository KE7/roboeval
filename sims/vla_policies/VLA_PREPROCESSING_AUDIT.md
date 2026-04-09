# VLA Policy Server Preprocessing/Postprocessing Audit

## Executive Summary

All policy servers **CORRECTLY use official preprocessing/postprocessing stacks** from the original model implementers. No parity bugs detected from custom implementations.

---

## Detailed Audit Results

### 1. **openvla_policy.py**

#### Preprocessing (raw observation → model input)
- **Official processor:** `AutoProcessor.from_pretrained()`
- **Location:** Lines 91-95 (loading), Line 164 (usage)
- **Quote:**
  ```python
  _processor = AutoProcessor.from_pretrained(
      model_id,
      trust_remote_code=True,
      local_files_only=True,
  )
  ```
  ```python
  inputs = _processor(prompt, pil_img).to(_device, dtype=torch.bfloat16)
  ```
- **Status:** ✅ **OFFICIAL** — Uses HuggingFace AutoProcessor directly

#### Postprocessing (model output → action vector)
- **Official unnormalization:** Built into `model.predict_action(..., unnorm_key=_unnorm_key, ...)`
- **Location:** Lines 167-171
- **Quote:**
  ```python
  action = _model.predict_action(
      **inputs,
      unnorm_key=_unnorm_key,  # <-- official unnormalization key
      do_sample=False,
  )
  ```
- **Custom steps:** Lines 178-179 gripper convention inversion (RLDS→LIBERO)
  - This is **NOT a parity bug** — it's environment-specific convention conversion
- **Status:** ✅ **OFFICIAL** — Unnormalization delegated to model; only manual step is gripper convention (necessary)

#### Hardcoded stats
- None. Normalization stats come from model's `unnorm_key` parameter
- Line 238: image_resolution [256, 256] is for documentation only, not used by code

#### **Classification:** ✅ **OFFICIAL**

---

### 2. **cosmos_policy.py**

#### Preprocessing (raw observation → model input)
- **Manual image decode & flip:** Lines 211-222
  ```python
  primary = np.flipud(_decode(image_b64))
  ```
  - This matches NVIDIA's `prepare_observation()` in `run_robocasa_eval.py`
  - Flip is applied because model was trained on 180°-flipped RoboCasa images
- **Official inference function:** `get_action()` from `cosmos_policy.experiments.robot.cosmos_utils`
  - Location: Lines 234-246
  ```python
  result = get_action(
      cfg=_eval_cfg,
      model=_model,
      dataset_stats=_dataset_stats,
      obs=obs,
      task_label_or_embedding=instruction,
      ...
  )
  ```
  - This handles: JPEG compression, resize, center-crop, T5 embedding lookup, diffusion generation
- **Status:** ✅ **OFFICIAL** — All heavy lifting delegated to NVIDIA's `get_action()`

#### Postprocessing (model output → action vector)
- **Official unnormalization:** Inside `get_action()` (line 234)
- **Manual postprocessing:** Lines 254-259
  ```python
  actions = [
      a.tolist() if isinstance(a, np.ndarray) else list(a)
      for a in raw_actions[:EXEC_HORIZON]
  ]
  while len(actions) < EXEC_HORIZON:
      actions.append([0.0] * ACTION_DIM)
  ```
  - This is just format conversion + padding, not unnormalization
- **Status:** ✅ **OFFICIAL** — Unnormalization done by `get_action()`

#### Hardcoded stats
- None (ACTION_DIM=7, PROPRIO_DIM=9, etc. are config constants, not learned stats)
- image_resolution [224, 224] in /info is documentation only

#### **Classification:** ✅ **OFFICIAL**

---

### 3. **groot_policy.py**

#### Preprocessing (raw observation → model input)
- **Official policy wrapper:** `Gr00tPolicy` + `Gr00tSimPolicyWrapper` from upstream gr00t library
  - Location: Lines 61-62, 69-75
  ```python
  from gr00t.data.embodiment_tags import EmbodimentTag
  from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
  
  base_policy = Gr00tPolicy(
      embodiment_tag=emb_tag,
      model_path=model_id,
      device=device,
      strict=True,
  )
  _policy = Gr00tSimPolicyWrapper(base_policy, strict=True)
  ```
- **Manual observation assembly:** Lines 116-157 (`_build_flat_observation()`)
  - Constructs dict with "video.{key}", "state.{key}" keys based on modality config
  - This is just to match wrapper's expected input format — actual preprocessing happens inside wrapper
- **Inference:** Line 202
  ```python
  action_dict, _info = _policy.get_action(flat_obs)
  ```
- **Status:** ✅ **OFFICIAL** — Uses upstream wrapper; observation format is just input marshaling

#### Postprocessing (model output → action vector)
- **Wrapper output:** Already unnormalized (wrapper handles all normalization)
- **Manual flatten:** Lines 160-185 (`_flatten_action_chunk()`)
  ```python
  flat_action = np.concatenate(parts, axis=-1)
  if len(flat_action) > 0:
      gripper_value = float(flat_action[-1])
      binarized = 1.0 if gripper_value > 0.0 else -1.0
      flat_action[-1] = -binarized  # Negate for RoboCasa convention
  ```
  - Concatenates action keys and applies gripper convention (RLDS→RoboCasa)
  - **NOT a parity bug** — environment-specific convention conversion
- **Status:** ✅ **OFFICIAL** — Wrapper handles unnormalization; only manual step is convention conversion

#### Hardcoded stats
- None. All config loaded from modality_config (lines 79-82)

#### **Classification:** ✅ **OFFICIAL**

---

### 4. **pi05_policy.py**

#### Preprocessing (raw observation → model input)
- **Official preprocessor:** `make_pre_post_processors()` from lerobot
  - Location: Lines 126, 160-165
  ```python
  from lerobot.policies.factory import make_pre_post_processors
  
  _preprocessor, _postprocessor = make_pre_post_processors(
      cfg,
      pretrained_path=model_id,
      preprocessor_overrides={"device_processor": {"device": device}},
      postprocessor_overrides={"device_processor": {"device": "cpu"}},
  )
  ```
- **Frame construction:** Lines 220-240 (`_build_frame()`)
  - Just assembles frame dict with camera keys, state, task
  - Actual preprocessing happens inside `_preprocessor(frame)` (line 267)
- **Usage:** Line 267
  ```python
  batch = _preprocessor(frame)
  ```
- **Status:** ✅ **OFFICIAL** — Uses official lerobot preprocessor

#### Postprocessing (model output → action vector)
- **Official postprocessor:** Line 272
  ```python
  action = _postprocessor(action)
  ```
- **Manual list conversion:** Lines 243-255 (`_policy_action_to_list()`)
  - Just converts tensor/array to Python list, no unnormalization
- **Status:** ✅ **OFFICIAL** — Uses official lerobot postprocessor

#### Hardcoded stats
- None. All loaded from model config (lines 156-194)
- image_resolution [256, 256] in /info is documentation only

#### **Classification:** ✅ **OFFICIAL**

---

### 5. **smolvla_policy.py**

#### Preprocessing (raw observation → model input)
- **Official preprocessor:** `make_pre_post_processors()` from lerobot
  - Location: Lines 130, 144-149
  ```python
  from lerobot.policies.factory import make_pre_post_processors
  
  _preprocessor, _postprocessor = make_pre_post_processors(
      _policy.config,
      pretrained_path=model_id,
      preprocessor_overrides={"device_processor": {"device": device}},
      postprocessor_overrides={"device_processor": {"device": "cpu"}},
  )
  ```
- **Frame construction:** Lines 211-253 (`_predict()`)
  - Converts base64 PNG to tensor, assembles frame dict
  - Actual preprocessing handled by `_preprocessor(frame)` (line 256)
- **Usage:** Line 256
  ```python
  batch = _preprocessor(frame)
  ```
- **Status:** ✅ **OFFICIAL** — Uses official lerobot preprocessor

#### Postprocessing (model output → action vector)
- **Official postprocessor:** Line 263
  ```python
  action = _postprocessor(action)
  ```
- **Manual list conversion:** Lines 267-269
  - Just converts to list format, no unnormalization
- **Status:** ✅ **OFFICIAL** — Uses official lerobot postprocessor

#### Hardcoded stats
- None. All loaded from model config (lines 152-203)
- image_resolution [256, 256] in /info is documentation only

#### **Classification:** ✅ **OFFICIAL**

---

### 6. **internvla_policy.py**

#### Architecture
- **PURE HTTP PROXY (thin-client)** — does NOT load the model locally
- Lines 5-7: "This is a PURE HTTP PROXY (thin-client) that forwards requests to the InternVLA model server running in a Docker container."
- No preprocessing or postprocessing happens in this file

#### Preprocessing
- **Delegated to upstream server** (Docker container running actual InternVLA model)
- Lines 151-154: Forwards request to upstream
  ```python
  response = await _http_client.post(
      f"{_upstream_url}/predict",
      json=req.model_dump(),
  )
  ```

#### Postprocessing
- **Delegated to upstream server**

#### **Classification:** ⚠️ **N/A (Thin-Client Proxy)** — All preprocessing/postprocessing handled by upstream Docker server, not this file

---

## Summary Table

| Policy | Preprocessing | Postprocessing | Hardcoded Stats? | Verdict | Notes |
|--------|---------------|-----------------|-----------------|---------|-------|
| **openvla_policy.py** | ✅ OFFICIAL (AutoProcessor) | ✅ OFFICIAL (model.predict_action) | ❌ None | ✅ OFFICIAL | Only custom gripper flip (environment-specific) |
| **cosmos_policy.py** | ✅ OFFICIAL (get_action) | ✅ OFFICIAL (get_action) | ❌ None | ✅ OFFICIAL | Manual image flip matches NVIDIA's eval code |
| **groot_policy.py** | ✅ OFFICIAL (GR00TSimPolicyWrapper) | ✅ OFFICIAL (wrapper) | ❌ None | ✅ OFFICIAL | Manual gripper conversion (environment-specific) |
| **pi05_policy.py** | ✅ OFFICIAL (make_pre_post_processors) | ✅ OFFICIAL (make_pre_post_processors) | ❌ None | ✅ OFFICIAL | Clean separation of concerns |
| **smolvla_policy.py** | ✅ OFFICIAL (make_pre_post_processors) | ✅ OFFICIAL (make_pre_post_processors) | ❌ None | ✅ OFFICIAL | Clean separation of concerns |
| **internvla_policy.py** | ⚠️ DELEGATED | ⚠️ DELEGATED | ❌ None | ⚠️ N/A (proxy only) | Pure HTTP proxy to Docker container |

---

## Key Findings

### ✅ All Local Policies Use Official Stacks

1. **openvla_policy.py** ✅ OFFICIAL
   - AutoProcessor from transformers
   - predict_action() with unnorm_key

2. **cosmos_policy.py** ✅ OFFICIAL
   - NVIDIA's get_action() function (from cosmos_policy.experiments.robot.cosmos_utils)
   - Matches run_robocasa_eval.py reference implementation

3. **groot_policy.py** ✅ OFFICIAL
   - GR00T's official Gr00tPolicy + Gr00tSimPolicyWrapper
   - Wrapper handles all normalization

4. **pi05_policy.py** ✅ OFFICIAL
   - lerobot's make_pre_post_processors() factory
   - Preprocessing and postprocessing pipelines saved with model

5. **smolvla_policy.py** ✅ OFFICIAL
   - lerobot's make_pre_post_processors() factory
   - Preprocessing and postprocessing pipelines saved with model

6. **internvla_policy.py** ⚠️ N/A
   - Pure HTTP proxy; all heavy lifting in Docker container

### ⚠️ Minor Custom Touches (NOT Parity Bugs)

All custom code post-processing is **environment-specific convention conversion**, not parity-critical:

- **openvla_policy.py (lines 178-179):** Gripper sign flip (RLDS→LIBERO)
- **groot_policy.py (lines 172-183):** Gripper binarization + sign flip (RLDS→RoboCasa)

These are **necessary transformations** to match LIBERO/RoboCasa conventions, not bugs. The models output RLDS gripper convention but the simulators expect the opposite sign.

### ❌ No Hardcoded Normalization Stats Found

- No manual `mean = [...]` / `std = [...]` arrays
- No hardcoded image sizes (e.g., `cv2.resize(img, (224, 224))`) that bypass model config
- No manual tokenization or action decoding outside official libraries
- No reinvented action denormalization logic

---

## Conclusion

**Status: ✅ CLEAN**

All five local policy servers (openvla, cosmos, groot, pi05, smolvla) correctly use the official preprocessing/postprocessing stacks from their respective model libraries. No parity bugs detected from custom implementations.

The InternVLA server is a pure HTTP proxy, so preprocessing/postprocessing is delegated to the upstream Docker container.

