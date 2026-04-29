# Supported Pair Notes

This page records non-numeric notes for shipped VLA/simulator pairings. It is a
compatibility guide, not a benchmark report. It intentionally omits success
rates, leaderboard-style comparisons, and run-quality claims.

## Direct Pairs

## Pi 0.5 x LIBERO

- **Setup:** `roboeval setup pi05 libero`
- **Config:** `configs/libero_spatial_pi05_smoke.yaml`
- **Mode:** Direct.
- **Intended use:** Single-arm LIBERO evaluation with a LIBERO-tuned Pi 0.5 policy.
- **Server:** `roboeval serve --vla pi05 --sim libero --headless`
- **Action contract:** 7-dimensional end-effector delta action.
- **Observation contract:** Primary image, language instruction, and LIBERO-style state.
- **Caveats:** Pi 0.5 LIBERO checkpoints expect axis-angle state representation; gripper convention is declared through `ActionObsSpec` and must match the simulator side.

## Pi 0.5 x LIBERO-Pro

- **Setup:** `roboeval setup pi05 libero_pro`
- **Config:** `configs/libero_pro_pi05_smoke.yaml`
- **Mode:** Direct.
- **Intended use:** LIBERO-Pro compatibility runs using the Pi 0.5 policy-server interface.
- **Server:** `roboeval serve --vla pi05 --sim libero_pro --headless`
- **Action contract:** 7-dimensional end-effector delta action.
- **Observation contract:** LIBERO-compatible image, language, and state fields.
- **Caveats:** LIBERO-Pro uses its own setup target and external package path; confirm the data and library paths before long runs.

## Pi 0.5 x LIBERO-Infinity

- **Setup:** `roboeval setup pi05 libero_infinity`
- **Config:** `configs/libero_infinity_pi05_smoke.yaml`
- **Mode:** Direct.
- **Intended use:** Scenic-generated LIBERO perturbation studies with the Pi 0.5 low-level policy.
- **Server:** `roboeval serve --vla pi05 --sim libero_infinity --headless`
- **Action contract:** 7-dimensional end-effector delta action reused from LIBERO.
- **Observation contract:** LIBERO-compatible image, language, and state fields.
- **Caveats:** LIBERO-Infinity requires its separate Python environment and package discovery path; suite names use the `libero_infinity_` prefix.

## SmolVLA x LIBERO

- **Setup:** `roboeval setup smolvla libero`
- **Config:** `configs/libero_object_smolvla_smoke.yaml`
- **Mode:** Direct.
- **Intended use:** LIBERO object-suite runs with the SmolVLA policy server.
- **Server:** `roboeval serve --vla smolvla --sim libero --headless`
- **Action contract:** 7-dimensional end-effector delta action.
- **Observation contract:** RGB image, instruction, and LIBERO-style state.
- **Caveats:** Use the SmolVLA default port in the config or override both `roboeval serve` and YAML together.

## OpenVLA x LIBERO

- **Setup:** `roboeval setup openvla libero`
- **Config:** `configs/libero_spatial_openvla_smoke.yaml`
- **Mode:** Direct.
- **Intended use:** LIBERO spatial-suite evaluation with an OpenVLA policy server.
- **Server:** `roboeval serve --vla openvla --sim libero --headless`
- **Action contract:** 7-dimensional end-effector delta action.
- **Observation contract:** RGB image, instruction, and LIBERO-style state.
- **Caveats:** OpenVLA model loading can take longer than smaller policies; use readiness checks before starting the run.

## GR00T x LIBERO

- **Setup:** `roboeval setup groot libero`
- **Config:** `configs/libero_spatial_groot_smoke.yaml`
- **Mode:** Direct.
- **Intended use:** LIBERO runs with a GR00T checkpoint configured for the LIBERO action and observation convention.
- **Server:** `roboeval serve --vla groot --sim libero --headless`
- **Action contract:** 7-dimensional end-effector delta action.
- **Observation contract:** Head/primary and wrist camera inputs plus compatible robot state.
- **Caveats:** GR00T has multiple embodiment tags and model variants; set model and embodiment environment variables consistently with the config.

## InternVLA x RoboTwin

- **Setup:** `roboeval setup internvla robotwin`
- **Config:** `configs/robotwin_internvla_smoke.yaml`
- **Mode:** Direct.
- **Intended use:** Bimanual RoboTwin runs with InternVLA-A1.
- **Server:** `roboeval serve --vla internvla --sim robotwin --headless`
- **Action contract:** 14-dimensional absolute joint-position action.
- **Observation contract:** RoboTwin image and bimanual state fields expected by InternVLA.
- **Caveats:** RoboTwin has SAPIEN and asset requirements; import order and asset paths matter for this backend.

## ACT x ALOHA Gym

- **Setup:** `roboeval setup act aloha_gym`
- **Config:** `configs/aloha_gym_act_smoke.yaml`
- **Mode:** Direct.
- **Intended use:** ALOHA bimanual transfer-cube runs with an ACT checkpoint.
- **Server:** `roboeval serve --vla act --sim aloha_gym --headless`
- **Action contract:** 14-dimensional absolute joint-position action.
- **Observation contract:** ALOHA image and agent-position observations.
- **Caveats:** ACT is not a delta end-effector policy; keep `delta_actions: false`.

## Diffusion Policy x gym-pusht

- **Setup:** `roboeval setup diffusion_policy gym_pusht`
- **Config:** `configs/gym_pusht_diffusion_policy_smoke.yaml`
- **Mode:** Direct.
- **Intended use:** PushT runs with a Diffusion Policy checkpoint trained for the PushT action space.
- **Server:** `roboeval serve --vla diffusion_policy --sim gym_pusht --headless`
- **Action contract:** 2-dimensional absolute planar end-effector target.
- **Observation contract:** PushT image/state fields as exposed by the gym-pusht backend.
- **Caveats:** Do not reuse this config for LIBERO or ALOHA without a checkpoint and backend contract that match those domains.

## VQ-BeT x gym-pusht

- **Setup:** `roboeval setup vqbet gym_pusht`
- **Config:** `configs/gym_pusht_vqbet_smoke.yaml`
- **Mode:** Direct.
- **Intended use:** PushT runs with a vector-quantized behavior-transformer policy.
- **Server:** `roboeval serve --vla vqbet --sim gym_pusht --headless`
- **Action contract:** 2-dimensional absolute planar end-effector target.
- **Observation contract:** PushT image/state fields as exposed by the gym-pusht backend.
- **Caveats:** The pair is intentionally in the same PushT action space as the Diffusion Policy config; keep result comparisons protocol-controlled.

## TDMPC2 x Meta-World

- **Setup:** `roboeval setup tdmpc2 metaworld`
- **Config:** `configs/metaworld_tdmpc2_smoke.yaml`
- **Mode:** Direct.
- **Intended use:** Meta-World Sawyer control runs with a 4-dimensional TDMPC2 policy.
- **Server:** `roboeval serve --vla tdmpc2 --sim metaworld --headless`
- **Action contract:** 4-dimensional end-effector delta action `[dx, dy, dz, gripper]`.
- **Observation contract:** Meta-World state vector expected by the TDMPC2 checkpoint.
- **Caveats:** LIBERO-style 7-dimensional VLAs are intentionally rejected by the spec gate for Meta-World.

## LITEN-Style Hierarchical Pairs

The following pairs reuse the same low-level VLA and simulator contracts as
their direct counterparts. The added component is a VLM planner endpoint, and
the key YAML difference is `no_vlm: false`.

## Pi 0.5 x LIBERO x LITEN

- **Setup:** `roboeval setup pi05 libero`
- **Config:** `configs/libero_spatial_pi05_liten_smoke.yaml`
- **Mode:** LITEN-style hierarchical.
- **Intended use:** Planner-assisted LIBERO spatial runs with Pi 0.5 as the low-level policy.
- **Server:** `roboeval serve --vla pi05 --sim libero --headless`
- **Planner endpoint:** Configure `vlm_endpoint` in YAML; examples commonly use `localhost:8000`.
- **Caveats:** The planner changes instructions sent to the policy, not the low-level action contract.

## Pi 0.5 x LIBERO-Pro x LITEN

- **Setup:** `roboeval setup pi05 libero_pro`
- **Config:** `configs/libero_pro_pi05_liten_smoke.yaml`
- **Mode:** LITEN-style hierarchical.
- **Intended use:** Planner-assisted LIBERO-Pro runs using the Pi 0.5 low-level policy.
- **Server:** `roboeval serve --vla pi05 --sim libero_pro --headless`
- **Planner endpoint:** Configure `vlm_endpoint` in YAML.
- **Caveats:** Confirm LIBERO-Pro assets and shared library paths before launching planner-assisted runs.

## Pi 0.5 x LIBERO-Infinity x LITEN

- **Setup:** `roboeval setup pi05 libero_infinity`
- **Config:** `configs/libero_infinity_pi05_liten_smoke.yaml`
- **Mode:** LITEN-style hierarchical.
- **Intended use:** Planner-assisted perturbation studies on LIBERO-Infinity suites.
- **Server:** `roboeval serve --vla pi05 --sim libero_infinity --headless`
- **Planner endpoint:** Configure `vlm_endpoint` in YAML.
- **Caveats:** The planner endpoint can dominate runtime; tune VLM serving separately from the simulator.

## SmolVLA x LIBERO x LITEN

- **Setup:** `roboeval setup smolvla libero`
- **Config:** `configs/libero_object_smolvla_liten_smoke.yaml`
- **Mode:** LITEN-style hierarchical config path.
- **Intended use:** LIBERO object-suite runs with SmolVLA and planner fields present in the config.
- **Server:** `roboeval serve --vla smolvla --sim libero --headless`
- **Planner endpoint:** The config includes `vlm_endpoint`.
- **Caveats:** Check `no_vlm` in the config before assuming planner calls are active.

## OpenVLA x LIBERO x LITEN

- **Setup:** `roboeval setup openvla libero`
- **Config:** `configs/libero_spatial_openvla_liten_smoke.yaml`
- **Mode:** LITEN-style hierarchical.
- **Intended use:** Planner-assisted LIBERO spatial runs with OpenVLA as the low-level policy.
- **Server:** `roboeval serve --vla openvla --sim libero --headless`
- **Planner endpoint:** Configure `vlm_endpoint` in YAML.
- **Caveats:** OpenVLA readiness and planner readiness are separate; wait for both services before running.

## GR00T x LIBERO x LITEN

- **Setup:** `roboeval setup groot libero`
- **Config:** `configs/libero_spatial_groot_liten_smoke.yaml`
- **Mode:** LITEN-style hierarchical.
- **Intended use:** Planner-assisted LIBERO runs with GR00T as the low-level policy.
- **Server:** `roboeval serve --vla groot --sim libero --headless`
- **Planner endpoint:** Configure `vlm_endpoint` in YAML.
- **Caveats:** Keep GR00T model and embodiment configuration aligned with the LIBERO action contract.

## InternVLA x RoboTwin x LITEN

- **Setup:** `roboeval setup internvla robotwin`
- **Config:** `configs/robotwin_internvla_liten_smoke.yaml`
- **Mode:** LITEN-style hierarchical.
- **Intended use:** Planner-assisted bimanual RoboTwin runs with InternVLA.
- **Server:** `roboeval serve --vla internvla --sim robotwin --headless`
- **Planner endpoint:** Configure `vlm_endpoint` in YAML.
- **Caveats:** RoboTwin simulator setup and planner serving are independent readiness checks.

## ACT x ALOHA Gym x LITEN

- **Setup:** `roboeval setup act aloha_gym`
- **Config:** `configs/aloha_gym_act_liten_smoke.yaml`
- **Mode:** LITEN-style hierarchical.
- **Intended use:** Planner-assisted ALOHA transfer-cube runs with ACT as the low-level policy.
- **Server:** `roboeval serve --vla act --sim aloha_gym --headless`
- **Planner endpoint:** Configure `vlm_endpoint` in YAML.
- **Caveats:** ACT has no native language-conditioned planner semantics beyond the low-level instruction interface; keep `delta_actions: false`.

## CI Compatibility Pair

## InternVLA x ALOHA Gym

- **Setup:** `roboeval setup internvla aloha_gym`
- **Config:** `configs/ci/aloha_gym_internvla_smoke.yaml`
- **Mode:** Direct CI smoke path.
- **Intended use:** Compatibility coverage for a 14-dimensional bimanual policy on the pure-Python ALOHA backend.
- **Server:** `roboeval serve --vla internvla --sim aloha_gym --headless`
- **Action contract:** 14-dimensional absolute joint-position action.
- **Observation contract:** ALOHA transfer-cube observations mapped to InternVLA expectations.
- **Caveats:** This config lives under `configs/ci/`; include it intentionally when selecting CI coverage.

## Backend-Only Notes

## RoboCasa Backend

- **Setup:** `roboeval setup robocasa`
- **Config:** No shipped public direct-pair config in the support matrix.
- **Mode:** Simulator backend and registry support.
- **Intended use:** Future RoboCasa-compatible VLA pairings and custom experiments.
- **Action contract:** RoboCasa backend declares its own EEF/action and observation metadata.
- **Caveats:** Use [extending.md](extending.md) before adding a new public pair.

## ManiSkill2 Backend

- **Setup:** `roboeval setup maniskill2`
- **Config:** No shipped public direct-pair config in the support matrix.
- **Mode:** Backend scaffold and x86_64 execution path.
- **Intended use:** Future ManiSkill2-compatible policy experiments.
- **Action contract:** ManiSkill2 backend declares a 7-dimensional end-effector delta-style action contract for compatible tasks.
- **Caveats:** Platform support and simulator dependencies should be checked before adding pair configs.

## Shared Contract Reminders

- Direct and LITEN-style configs for the same pair share the same VLA and simulator contract.
- LITEN-style configs add planner fields; they do not change action dimensionality.
- `delta_actions: true` belongs to end-effector delta-style policies and simulators.
- `delta_actions: false` belongs to absolute joint-position or planar absolute-action policies.
- `vla_url` must match the policy server port actually used at launch.
- `sim_url` must match the simulator worker port actually used at launch.
- The config path should be treated as the source of truth for the shipped example.
- The setup command installs dependencies; it does not download every external dataset automatically.
- The spec gate is expected to reject unsupported cross-domain pairings.
- A supported pair can still require external assets, model weights, or environment variables.
- Use [install.md](install.md) for dependency details.
- Use [failure_modes.md](failure_modes.md) for troubleshooting startup and contract errors.
- Use [liten.md](liten.md) for planner-mode behavior.
- Use [extending.md](extending.md) before publishing a new pair.

## Reading This Page

- A "direct" pair means the VLA receives the task instruction directly.
- A "LITEN-style" pair means a VLM planner emits subtask instructions.
- The low-level action/observation contract remains the VLA/simulator contract.
- A config path is a runnable example, not a benchmark result.
- Caveats describe interface and setup concerns only.

## Pair Categories

### LIBERO-Family Single-Arm Pairs

- Pi 0.5 x LIBERO.
- Pi 0.5 x LIBERO-Pro.
- Pi 0.5 x LIBERO-Infinity.
- SmolVLA x LIBERO.
- OpenVLA x LIBERO.
- GR00T x LIBERO.

These pairs use single-arm manipulation conventions and rely on explicit state,
camera, gripper, and end-effector action declarations.

### Bimanual Pairs

- InternVLA x RoboTwin.
- ACT x ALOHA Gym.
- InternVLA x ALOHA Gym.

These pairs use 14-dimensional absolute joint-position actions rather than
LIBERO-style end-effector deltas.

### Planar Control Pairs

- Diffusion Policy x gym-pusht.
- VQ-BeT x gym-pusht.

These pairs use PushT's 2-dimensional planar absolute action space.

### State-Based Control Pair

- TDMPC2 x Meta-World.

This pair uses Meta-World's 4-dimensional Sawyer control action and a
state-based policy interface.

### Hierarchical Variants

LITEN-style variants are listed separately because they add a planner service,
but they should be read together with their direct pair entries.
