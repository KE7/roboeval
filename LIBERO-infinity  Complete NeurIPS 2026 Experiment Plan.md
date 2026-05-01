# LIBERO-infinity: Complete NeurIPS 2026 Experiment Plan

## Preamble: What the Repo Actually Supports

Based on the live README and docs, LIBERO-infinity has exactly **9 perturbation axes**:

| # | Axis | CLI flag | What varies | Unique to libero-infinity vs. competitors |
|---|------|----------|-------------|------------------------------------------|
| 1 | Position | `position` | Object (x,y) uniform over workspace, OOD-biased | Distribution-based (continuous), competitors use discrete levels |
| 2 | Object | `object` | Mesh + texture from 34 asset variant pools, BDDL rewriting | 34 classes; both competitors also have object perturbation |
| 3 | Robot | `robot` | Panda 7-DOF init qpos in joint-space radius 0.1–0.5 | Both competitors also have this |
| 4 | Camera | `camera` | agentview ±0.10m offset + ±15° tilt | LIBERO-Plus has this; LIBERO-PRO does not |
| 5 | Lighting | `lighting` | Intensity [0.4, 2.0], ambient [0.05, 0.6], position offset | LIBERO-Plus has this; LIBERO-PRO does not |
| 6 | Texture | `texture` | Table surface material swap | LIBERO-Plus has background/texture; LIBERO-PRO does not |
| 7 | Distractor | `distractor` | 1–5 clutter objects from 8-item pool, clearance-checked | Both competitors have distractors; yours auto-excludes task objects |
| 8 | Background | `background` | Wall + floor from 35 LIBERO texture assets | LIBERO-Plus has this; LIBERO-PRO does not |
| 9 | Articulation | `articulation` | Initial fixture state (doors, drawers, stoves); goal-reachability enforced | **Neither competitor supports this axis** |

There is **no language perturbation axis** — and this is the correct design decision. The question "what is infinite language?" is unanswerable; perturbing language requires an LLM paraphrase loop with no formal distribution, which violates the statistical grounding that makes LIBERO-infinity's contribution rigorous. Instead, language contribution is measured via a **counterfactual ablation** (run same scenes with `""` vs. real instruction), which is more principled than anything either competitor offers.

**Two compound presets:**
- `combined` = `position + object + robot + camera + lighting + distractor + background` (7 axes, no articulation, no texture)
- `full` = all 9 axes

***

## What to Cite Directly (Zero Re-Running)

### LIBERO-Plus Table 1: Single-Axis Results on LIBERO-Spatial

Cite verbatim from Liu et al. (LIBERO-Plus, arXiv 2510.13626). Present this in your paper's Related Work or Baselines section:

| Model | Baseline | Camera | Robot | Lighting | Background | Texture/Noise | Position |
|---|---|---|---|---|---|---|---|
| OpenVLA | 76.5% | 1.1% | 4.1% | 4.4% | 25.3% | 19.3% | 31.6% |
| OpenVLA-OFT | 97.1% | 59.7% | 37.2% | 85.8% | 92.4% | 76.7% | 77.1% |
| π₀ | 94.2% | 15.8% | 6.6% | 79.6% | 78.5% | 79.4% | 70.4% |
| π₀-fast | 85.5% | 66.4% | 24.8% | 73.0% | 67.7% | 75.8% | 70.3% |

**Do not re-run any of these.** They are the established baselines.

### LIBERO-PRO Headline Finding

Cite from Zhou et al. (LIBERO-PRO, arXiv 2510.03827): all tested models collapse to **0.0% under their combined 4-dimension generalization setting** (objects + instructions + environments + initial states). Use this in your intro to motivate the saturation problem.

### Language Blindness Finding

Cite from LIBERO-Plus Section 4.4: "VLA models are largely insensitive to language variations and tend to ignore language instructions completely." You will complement this, not replicate it.

### VLA-VA Language Counterfactual Baseline

Cite from VLA-VA (referenced in your own README): OpenVLA-OFT scores ~83% vision-only vs. ~97% with full instruction on LIBERO tasks. This gives the reviewer a prior-work anchor for your Experiment 4 language counterfactual results without you having to run a training ablation.

***

## The Four New Experiments: Complete Instructions

### Experiment 1 — Articulation Axis Characterization

**Purpose:** Prove that articulation perturbation exposes failures that neither competitor can detect. This is the single most important new empirical contribution because articulation is the only axis with zero overlap with any prior work.

**Task selection:** Pick 3 tasks from `libero_goal/` that each involve a different fixture family:
- Cabinet drawer: `open_the_middle_drawer_of_the_cabinet.bddl`
- Drawer + object placement: `open_the_top_drawer_and_put_the_bowl_inside.bddl`
- Stove: `turn_on_the_stove.bddl`

These three cover all three fixture families in your articulation model: drawers, cabinets (doors), and stoves. Do not use libero_spatial tasks here — those are object-on-object placement tasks with no fixture goals, so articulation perturbation does not apply (your pipeline auto-skips non-applicable axes).

**Models:** OpenVLA-OFT and π₀. These are the two best-performing models in LIBERO-Plus and have publicly available HuggingFace checkpoints. OpenVLA-OFT is at `openvla/openvla-oft-pretrained-prismatic-7b-libero-goal-lora` and π₀ is accessible via the Physical Intelligence API or its released checkpoint.

**Runs:**

```bash
# For each of the 3 tasks, run 3 conditions × 2 models = 6 runs × 3 tasks = 18 runs total
# Condition A: position-only (the competitor-equivalent baseline)
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/<task>.bddl \
  --perturbation position \
  --n-scenes 100 \
  --seed 42 \
  --output results/exp1_position_<task>_<model>.json \
  --verbose

# Condition B: articulation-only (your new axis)
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/<task>.bddl \
  --perturbation articulation \
  --n-scenes 100 \
  --seed 42 \
  --output results/exp1_articulation_<task>_<model>.json \
  --verbose

# Condition C: position + articulation (joint perturbation)
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/<task>.bddl \
  --perturbation position,articulation \
  --n-scenes 100 \
  --seed 42 \
  --output results/exp1_pos_art_<task>_<model>.json \
  --verbose
```

**N=100 per cell.** This gives a Wilson 95% CI of approximately ±9.8pp at 50% success rate, which is the worst case. At the ~70% success rates observed in prior work, CI width narrows to ±9.0pp — sufficient to resolve a 15pp gap, which is the minimum scientifically interesting effect.

**What to report:** A table with 3 tasks × 3 conditions × 2 models = 18 cells, each with success rate and Wilson CI. The key comparison is articulation alone vs. position alone for each model. If a model scores, say, 72% under position but 35% under articulation, this reveals that the model is relying on visual/positional cues to infer fixture state rather than actually reasoning about it — a new failure mode.

**If articulation makes no difference:** If scores under articulation-only are within CI of the baseline for all tasks, this is still a valid and publishable finding — it means VLA models have learned some robustness to initial fixture state, which is genuinely surprising given that LIBERO-PRO showed they collapse under far simpler perturbations. Report it honestly.

**Compute estimate:** 18 runs × 100 scenes × 300 steps × ~0.05s/step = ~27,000 steps × 0.05s ≈ ~23 GPU-hours with a single worker. With 4 parallel envs (`make_vec_env(n_envs=4)`): ~6 GPU-hours.

***

### Experiment 2 — `combined` Preset Head-to-Head Comparison Table

**Purpose:** Produce the paper's main comparison table. Evaluates LIBERO-infinity's 7-axis `combined` preset (which covers the same axis families as LIBERO-Plus) against prior single-axis results from LIBERO-Plus and shows that (a) the combined evaluation is harder than any single axis alone, (b) LIBERO-infinity produces calibrated CIs while LIBERO-Plus reports point estimates.

**Task selection:** Run all 10 tasks in `libero_spatial/` (the same suite LIBERO-Plus Table 1 uses):
```
pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl
pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl
pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate.bddl
pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate.bddl
pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate.bddl
pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate.bddl
pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl
pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate.bddl
pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl
pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate.bddl
```

**Models:** OpenVLA-OFT and π₀. Optionally add OpenVLA vanilla if the checkpoint is available (its camera=1.1% from LIBERO-Plus already tells a story; your combined result for it will also be near 0% and reinforces the narrative).

**Runs:**

```bash
# For each of the 10 tasks × 2 models = 20 runs
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_spatial/<task>.bddl \
  --perturbation combined \
  --n-scenes 200 \
  --seed 42 \
  --output results/exp2_combined_<task>_<model>.json \
  --verbose
```

**N=200 per cell.** At N=200, the Wilson CI width is ±6.9pp at 50% success rate and ±4.8pp at 85% — tight enough to resolve any 10pp gap. This is the recommended N for the paper's main table because it balances statistical power with compute cost.

**What to report:** A table with 10 tasks × 2 models, each with success rate ± CI. Also report the **suite-level mean** across all 10 tasks with a combined Wilson CI (see formula in the Statistics section below). In a second column for each model, show the corresponding LIBERO-Plus single-axis best score (cite from Table 1) as a reference — this contextualizes how much harder the combined evaluation is.

**Key claim to support:** LIBERO-Plus's hardest single axis is Camera (π₀: 15.8%, OpenVLA-OFT: 59.7%). If `combined` produces results at or below those numbers, it proves the combined evaluation is harder than the hardest single perturbation — establishing that LIBERO-infinity is more challenging by construction.

**Compute estimate:** 20 runs × 200 scenes × 300 steps × ~0.05s/step ≈ 600,000 steps. With 4 parallel envs: ~21 GPU-hours. This is the most expensive experiment and should be started first.

***

### Experiment 3 — Articulation Ablation Within `full` vs. `combined`

**Purpose:** Isolate articulation's contribution to difficulty by comparing `combined` (7 axes, no articulation) vs. `full` (all 9, including articulation). The delta is attributed entirely to articulation + texture, since those are the only axes added. This is a clean ablation that tells reviewers: "here is the quantified value of the new axis."

**Task selection:** Pick **2 tasks** from `libero_goal/` with fixture goals (use the same 2 drawer tasks from Experiment 1 since you already have baseline numbers):
- `open_the_middle_drawer_of_the_cabinet.bddl`
- `open_the_top_drawer_and_put_the_bowl_inside.bddl`

Do not use `libero_spatial/` tasks for this experiment — they have no articulated fixtures, so `full` vs. `combined` would only differ in the texture axis, making this a texture ablation rather than an articulation ablation.

**Runs:**

```bash
# For each of the 2 tasks × 2 models = 4 runs per condition
# Condition A: combined (baseline — same as Exp 2 but on libero_goal tasks)
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/<task>.bddl \
  --perturbation combined \
  --n-scenes 150 \
  --seed 42 \
  --output results/exp3_combined_<task>_<model>.json

# Condition B: full (adds articulation + texture)
MUJOCO_GL=egl libero-eval \
  --bddl src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/<task>.bddl \
  --perturbation full \
  --n-scenes 150 \
  --seed 42 \
  --output results/exp3_full_<task>_<model>.json
```

**N=150 per cell.** This is the minimum N needed to detect a 15pp gap with 80% statistical power at α=0.05, given expected success rates in the 40–70% range. The Wilson CI at N=150 is approximately ±8pp at 50% success.

**What to report:** A 2-task × 2-model × 2-condition = 8-cell table. Report \( \Delta_\text{articulation} = p_\text{full} - p_\text{combined} \) for each model–task pair, along with the Wilson CI on the difference (see formula in the Statistics section). If the delta is negative (full is harder), articulation is contributing to difficulty above and beyond the 7 axes that LIBERO-Plus-equivalent evaluation already captures.

**Compute estimate:** 8 runs × 150 scenes × 300 steps × ~0.05s/step ≈ 180,000 steps. With 4 parallel envs: ~2.5 GPU-hours.

***

### Experiment 4 — Language Counterfactual (Zero New Scenes)

**Purpose:** Provide a principled measure of language contribution without perturbing language syntax. This is more rigorous than LIBERO-Plus's language paraphrase approach because it measures language's *causal* contribution under identical visual conditions, not just sensitivity to paraphrase style.

**Reuse:** Use the exact same generated scenes from Experiment 2 (the `combined` run on `libero_spatial/`). The Scenic-sampled scene parameters are recorded per-episode in the JSON output. Replay each scene with a modified policy closure that passes `instruction=""` instead of the real instruction.

```python
import json
import numpy as np
from libero_infinity.eval import evaluate
from libero_infinity.task_config import TaskConfig
from libero_infinity.compiler import generate_scenic_file

# Load per-scene params from Experiment 2 output
with open("results/exp2_combined_<task>_<model>.json") as f:
    exp2_results = json.load(f)

cfg = TaskConfig.from_bddl("path/to/task.bddl")
scenic_path = generate_scenic_file(cfg, perturbation="combined")

# Re-run with real instruction (already done in Exp 2 — just reload)
p_lang = exp2_results["success_rate"]

# Run same scenes with empty instruction
results_no_lang = evaluate(
    scenic_path=scenic_path,
    bddl_path="path/to/task.bddl",
    policy=make_policy(""),          # empty string — no instruction
    n_scenes=200,
    seed=42,                         # SAME seed as Exp 2 → same scenes
    verbose=True,
)
p_vision = results_no_lang.success_rate
lang_contribution = p_lang - p_vision
```

**Critical implementation note:** Pass `seed=42` (same seed used in Experiment 2) to ensure the same scenes are sampled. The policy closure for the no-language condition passes `""` as the instruction. Do NOT modify the BDDL or Scenic program — the change is only in the policy input.

**What to report:** For each of the 10 libero_spatial tasks × 2 models: `p_lang`, `p_vision`, `lang_contribution = p_lang - p_vision`, and the Wilson CI on the difference. Compute a suite-level mean language contribution. Compare against the VLA-VA reported value (~14pp for OpenVLA-OFT). A key finding is whether language contribution under combined perturbation (novel visual conditions) is higher or lower than under fixed LIBERO (familiar conditions) — if it's lower, models regress to vision-only shortcuts when perturbed, which is a novel insight.

**Compute estimate:** 20 runs × 200 scenes × 300 steps × ~0.05s/step = 600,000 steps. With 4 parallel envs: ~21 GPU-hours. **However**, if compute is the bottleneck, reduce to N=100 per cell (200K steps total, ~10 GPU-hours). At N=100, the Wilson CI is ±9.8pp — sufficient to resolve language contributions above 20pp. Given VLA-VA's ~14pp finding, this may not resolve small contributions for π₀. Report this as a limitation if underpowered.

***

## Statistics: All Formulas Fully Specified

### Wilson Score Confidence Interval (Per-Cell)

For a cell with \(k\) successes out of \(n\) trials, the 95% Wilson score CI is:

\[ \hat{p} = \frac{k}{n} \]

\[ \tilde{p} = \frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \]

\[ \tilde{\sigma} = \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}} \]

\[ \text{CI} = \left( \tilde{p} - \tilde{\sigma},\ \tilde{p} + \tilde{\sigma} \right) \]

where \(z = 1.96\) for 95% confidence. This is the formula already implemented in `results.summary()` output. Do not use the normal approximation \( \hat{p} \pm 1.96\sqrt{\hat{p}(1-\hat{p})/n} \) — it breaks down at extreme proportions (i.e., near 0% or 100%), which is exactly where your results will be for perturbed OpenVLA.

**Practical CI widths at N=100, 150, 200:**

| N | CI half-width at p=0.50 | CI half-width at p=0.70 | CI half-width at p=0.90 |
|---|---|---|---|
| 100 | ±9.8pp | ±9.0pp | ±5.9pp |
| 150 | ±8.0pp | ±7.3pp | ±4.8pp |
| 200 | ±6.9pp | ±6.3pp | ±4.2pp |

### Wilson CI on a Difference (Between Two Conditions)

For comparing condition A (success \(k_A\) of \(n_A\)) vs. condition B (success \(k_B\) of \(n_B\)) — used in Experiments 3 and 4 to report \(\Delta\) values:

\[ \Delta = \hat{p}_A - \hat{p}_B \]

The Newcombe-Wilson interval for the difference is:

\[ \text{CI}_\Delta = \left( \Delta - \sqrt{(l_A - \hat{p}_A)^2 + (\hat{p}_B - u_B)^2},\ \Delta + \sqrt{(\hat{p}_A - l_A)^2 + (u_A - \hat{p}_A)^2 + \dots} \right) \]

In practice, compute both Wilson CIs independently and report \( \Delta \pm \sqrt{\sigma_A^2 + \sigma_B^2} \) where \(\sigma_A\) and \(\sigma_B\) are the individual Wilson half-widths. This is the standard approximation used in robotics benchmarking and is accurate when \(n \geq 50\).

### Suite-Level Aggregation

To report a **suite-level mean** across \(T\) tasks (e.g., across all 10 LIBERO-Spatial tasks):

1. Compute per-task Wilson CIs \( [\ell_t, u_t] \) for each task \(t\)
2. Report the mean success rate: \( \bar{p} = \frac{1}{T} \sum_{t=1}^{T} \hat{p}_t \)
3. Report the pooled CI half-width: \( \bar{\sigma} = \frac{1}{T} \sqrt{\sum_{t=1}^{T} \sigma_t^2} \) where \(\sigma_t\) is the Wilson half-width for task \(t\)

Do not pool episodes across tasks (i.e., do not compute \(k_\text{total}/n_\text{total}\)) — tasks have different difficulty levels and pooling treats them as exchangeable.

### Minimum Detectable Effect (MDE)

For a two-sided test at α=0.05, power=0.80, with a reference success rate of \(p_0\):

\[ n \geq \frac{(z_{\alpha/2} + z_\beta)^2 \cdot [p_0(1-p_0) + p_1(1-p_1)]}{(p_0 - p_1)^2} \]

At \(p_0=0.65\) (typical perturbed π₀ result), \(n=100\) gives 80% power to detect a 15pp gap. At \(n=200\), the MDE drops to ~10pp. Use N=200 for the main table (Experiment 2) where precision matters; N=100 is acceptable for secondary analyses (Experiments 1, 3).

***

## Full Experiment Master Table

| Exp | Condition(s) | Suite | Tasks | N/cell | Models | Runs | Total episodes | GPU-hours (4 env) | Citable from prior? |
|---|---|---|---|---|---|---|---|---|---|
| 1a | `position` only | libero_goal | 3 fixture tasks | 100 | OVA-OFT, π₀ | 6 | 600 | 0.9h | No — new axis comparison |
| 1b | `articulation` only | libero_goal | 3 fixture tasks | 100 | OVA-OFT, π₀ | 6 | 600 | 0.9h | No — novel axis |
| 1c | `position,articulation` | libero_goal | 3 fixture tasks | 100 | OVA-OFT, π₀ | 6 | 600 | 0.9h | No — novel combination |
| 2 | `combined` | libero_spatial | All 10 tasks | 200 | OVA-OFT, π₀ | 20 | 4,000 | 21h | Cite LP Table 1 as context |
| 3a | `combined` | libero_goal | 2 fixture tasks | 150 | OVA-OFT, π₀ | 4 | 600 | 1.3h | No — required for Exp 3 |
| 3b | `full` | libero_goal | 2 fixture tasks | 150 | OVA-OFT, π₀ | 4 | 600 | 1.3h | No — novel |
| 4 | `combined` + empty instr | libero_spatial | All 10 tasks | 200 | OVA-OFT, π₀ | 20 | 4,000 | 21h | Cite VLA-VA as anchor |
| **TOTAL** | | | | | | **66 runs** | **11,000 ep.** | **~47h** | |

**With 2 GPUs running in parallel: ~24 real-time hours total.** Start Experiments 2 and 4 in parallel on two GPUs; they share the same seed and suite so outputs are cross-comparable.

***

## Task-by-Task Assignment and Seed Policy

### Seeds

- All runs use `--seed 42` as the primary seed
- Experiment 4 **must** use `--seed 42` to match Experiment 2 scenes (same Scenic samples → valid counterfactual)
- For any replication runs or sensitivity checks, use seeds `0, 1, 2` to verify CI coverage

### Fixture Task Selection Rationale (Experiments 1 and 3)

The 3 fixture tasks chosen from `libero_goal/` each activate a different articulation fixture family:

| Task file | Fixture family | Articulation state varied |
|---|---|---|
| `open_the_middle_drawer_of_the_cabinet.bddl` | Cabinet drawer | Drawer openness: fully closed → partially open |
| `open_the_top_drawer_and_put_the_bowl_inside.bddl` | Drawer + object placement | Top drawer openness; bowl start position |
| `turn_on_the_stove.bddl` | Stove burner | Stove state: off (default) → goal-reachability range |

The remaining `libero_goal/` tasks (`put_the_bowl_on_the_plate.bddl`, `put_the_bowl_on_the_stove.bddl`, etc.) are object-placement tasks with no fixture articulation goal — articulation perturbation on those tasks only changes incidental fixture state, not the goal-relevant state. They are valid for Experiment 2 (`combined`) but not for Experiments 1 and 3 where the articulation signal must be causally relevant to task success.

***

## Paper Structure: Where Each Experiment Goes

| Section | Content | Experiments used |
|---|---|---|
| **Introduction** | Motivate with LIBERO-PRO's 0.0% collapse, LIBERO-Plus single-axis brittleness | Cite only |
| **Related Work** | LIBERO-Plus Table 1 reproduced in full; LIBERO-PRO headline; VLA-VA language finding | Cite only |
| **Benchmark Design** | 9 axes with specs; Scenic program auto-generation; Wilson CI methodology; articulation goal-reachability constraint | No experiments needed — describe design |
| **Main Results (§4.1)** | `combined` vs. prior single-axis (Experiment 2) | Exp 2 + cited LP Table 1 |
| **Articulation Analysis (§4.2)** | Per-fixture articulation characterization; `full` vs. `combined` ablation | Exps 1 + 3 |
| **Language Counterfactual (§4.3)** | Language contribution under novel scenes vs. VLA-VA anchor | Exp 4 + cite VLA-VA |
| **Statistical Methodology (§5)** | Wilson CI formulas; N sensitivity; reproducibility argument | CIs from all experiments |
| **Limitations** | No language axis; sim-only; articulation pool limited to LIBERO's fixture set | No experiments needed |

***

## Ambiguity Resolution Log

Every ambiguity identified during planning, resolved:

1. **Do I run all 9 axes simultaneously?** No. `combined` = 7 axes; `full` = 9. Use `combined` for the main table (matches LIBERO-Plus scope). Use `full` only in Experiment 3 to isolate articulation's marginal contribution.

2. **Do I need to ablate individual axes?** No, for axes shared with LIBERO-Plus (camera, lighting, background, robot, position, object, distractor). Their single-axis numbers are cited. You only run single-axis for `articulation` (Experiment 1b) because no prior paper has done it.

3. **Which LIBERO tasks for articulation experiments?** Confirmed: `open_the_middle_drawer_of_the_cabinet.bddl`, `open_the_top_drawer_and_put_the_bowl_inside.bddl`, `turn_on_the_stove.bddl` from `libero_goal/`. These are the only tasks where articulation perturbation is causally relevant to success.

4. **Do `libero_spatial` tasks even activate articulation?** No. The libero_spatial suite consists entirely of bowl-placement tasks. The compiler auto-skips fixture axes when BDDL has no fixture goals. Do not run `full` on libero_spatial — the output would be identical to `combined`.

5. **How do I implement the language counterfactual?** Pass `make_policy("")` with the same `seed=42` as Experiment 2. The Scenic sampler is deterministic given the seed, so the same 200 scenes are generated for both the `p_lang` (Exp 2) and `p_vision` (Exp 4) conditions.

6. **What is the correct CI formula?** Wilson score (not normal approximation). The normal approximation \(\hat{p} \pm 1.96\sqrt{\hat{p}(1-\hat{p})/n}\) produces invalid (negative lower bound) CIs at extreme proportions. Given that OpenVLA under camera perturbation scores 1.1%, the Wilson formula is required.

7. **What if π₀'s checkpoint requires the Physical Intelligence API?** Use π₀-fast instead (checkpoint publicly available, scores similar — 85.5% baseline vs. π₀'s 94.2%). Report which variant was used in the paper.

8. **Do I need a new BDDL task to prove "infinity"?** No — the reviewer argument is statistical (infinite samples from a formal distribution), not combinatorial (infinite tasks). The task-count claim is about *scenes*, not *tasks*. The ability to run any BDDL task is a separate feature already proven by the pipeline's auto-generation; a footnote demonstrating this on a custom BDDL (from libero_10) is sufficient, not a full experiment.

9. **What does "seed" control in the pipeline?** The seed is passed to Scenic's rejection sampler and controls the sequence of scene configurations drawn. Two runs with the same seed, same BDDL, same perturbation flags, and same model will produce the same episode sequence. This is the mechanism enabling Experiment 4's exact scene reuse.

10. **Can I run Experiments 2 and 4 simultaneously?** Yes — they use different policy callables but identical Scenic programs and seeds. Set up two separate Python processes: one with `make_policy(cfg.language)` and one with `make_policy("")`. They do not share state.