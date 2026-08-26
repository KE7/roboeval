# Troubleshooting

Use this guide when a setup, server, or run fails in a way that is not covered
by the install guide. Each entry is organized as symptom, likely cause, and
remedy.

## Server Does Not Become Ready

**Symptom:** `roboeval serve --vla ... --sim ...` waits for a component and then
reports that the server did not become ready.

**Likely cause:** The policy or simulator worker exited during startup, is still
loading weights, or cannot bind its configured port.

**Remedy:**

1. Check the component logs under `logs/`.
2. Confirm the selected component is installed:
   ```bash
   roboeval setup <vla> <sim>
   ```
3. Check whether the configured port is already in use:
   ```bash
   lsof -i :5100
   lsof -i :5300
   ```
4. Stop the existing server or choose a different port before retrying.

## Config Validation Fails

**Symptom:** `roboeval test --validate -c <config>` fails before any rollout
starts.

**Likely cause:** The config names a missing component, unsupported VLA/simulator
pairing, invalid server URL, or mismatched action/observation contract.

**Remedy:**

1. Run the validator and read the first reported error:
   ```bash
   roboeval test --validate -c configs/libero_spatial_pi05_smoke.yaml
   ```
2. Confirm the VLA and simulator are a supported pair in the docs.
3. If you are adding a new component, make sure both sides declare compatible
   `ActionObsSpec` metadata before running episodes.

## Model Weights Are Missing or Cannot Load

**Symptom:** The policy server starts but `/health` remains unavailable, reports
`ready: false`, or the log mentions missing weights, authentication, or a CUDA
allocation failure.

**Likely cause:** The checkpoint has not been downloaded, the model ID is wrong,
HuggingFace access is not configured for the selected model, or the GPU does not
have enough available memory.

**Remedy:**

1. Re-run the setup target for the policy:
   ```bash
   roboeval setup pi05
   ```
2. Start only the policy server and watch its log:
   ```bash
   roboeval serve --vla pi05
   ```
3. Confirm GPU availability with `nvidia-smi`.
4. Stop other policy servers before starting a large model.

## Simulator Worker Starts but Reset or Step Fails

**Symptom:** A run exits after `/init`, `/reset`, or `/step`, or the result JSON
contains a simulator connection error.

**Likely cause:** The simulator backend is missing assets or datasets, MuJoCo EGL
is not configured, or the simulator process crashed after startup.

**Remedy:**

1. Check the sim worker log under `logs/`.
2. Verify the simulator venv and datasets for the selected backend.
3. For MuJoCo backends on headless hosts, set EGL before launching:
   ```bash
   MUJOCO_GL=egl roboeval serve --vla pi05 --sim libero --headless
   ```
4. If the process died, restart the server pair before rerunning the config.

## Action or Observation Contract Mismatch

**Symptom:** Validation or run startup reports an action dimension, action type,
camera, state, gripper, or image-transform mismatch.

**Likely cause:** The selected policy was trained for a different simulator
contract, or a new integration is missing precise metadata.

**Remedy:**

1. Run:
   ```bash
   roboeval test --validate -c <config>
   ```
2. Check the policy server `/info` response and simulator `/info` response.
3. Pair policies only with simulators that use the same action type,
   dimensionality, gripper convention, and required observation roles.
4. For custom integrations, update the VLA and simulator `ActionObsSpec`
   declarations rather than relying on implicit adapters.

## Malformed Policy Actions

**Symptom:** The simulator fails during stepping, the run exits with a nonzero
status, or logs mention invalid numeric values from the policy.

**Likely cause:** The policy returned an action with the wrong shape, NaN/Inf
values, or values outside the simulator's expected range.

**Remedy:**

1. Validate the config first:
   ```bash
   roboeval test --validate -c <config>
   ```
2. Inspect the policy server log for `/predict` errors.
3. For custom policy servers, ensure `/predict` returns `list[list[float]]` with
   the declared `action_chunk_size` and action dimension.
4. Convert normalized or tokenized model outputs back to simulator-space actions
   before returning them.

## OpenGL or Blank Observation Errors

**Symptom:** MuJoCo reports an OpenGL initialization error, observations are blank,
or the simulator times out on a headless machine.

**Likely cause:** EGL libraries are missing or `MUJOCO_GL` is not set in the
environment that launches the simulator.

**Remedy:**

1. Install EGL dependencies as described in [install.md](install.md).
2. Start the server with:
   ```bash
   MUJOCO_GL=egl roboeval serve --vla pi05 --sim libero --headless
   ```
3. Confirm the driver is loaded with `nvidia-smi`.

## Wrong Python Environment

**Symptom:** Logs show `ModuleNotFoundError`, an unexpected package version, or a
policy/simulator import error.

**Likely cause:** The component was launched with the wrong virtual environment,
or setup did not complete for that component.

**Remedy:**

1. Re-run setup for the selected component:
   ```bash
   roboeval setup <component>
   ```
2. Prefer `roboeval serve --vla <name> --sim <name>` so the registered venvs are
   selected automatically.
3. For manual launches, use the component-specific Python under `.venvs/`.

## Interrupted Runs Leave Ports Busy

**Symptom:** A later run reports that a port is already in use or that a stale
server is still answering requests.

**Likely cause:** A previous interactive run was interrupted before all child
processes exited.

**Remedy:**

1. Find the process using the port:
   ```bash
   lsof -i :5100
   lsof -i :5300
   ```
2. Stop the old process or choose alternate `--vla-port` / `--sim-port` values.
3. Restart the server pair and rerun validation before launching a long run.

## Sharded Results Are Partial

**Symptom:** `roboeval merge` writes a result with `"partial": true`.

**Likely cause:** One or more shard files are missing, duplicated, or written for
a different shard count.

**Remedy:**

1. List the shard files and confirm the expected `shard{id}of{total}` names are
   present.
2. Re-run only the missing shard with the same config, `--shard-id`, and
   `--num-shards`.
3. Merge again:
   ```bash
   roboeval merge --pattern 'results/run/*shard*.json' --output results/run/final.json
   ```
