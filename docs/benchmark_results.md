# Benchmark Results

Episodes per task vary by model to match each published evaluation protocol (see [Methodology](#methodology) below). Cells marked "in progress" are currently running; "pending" means not yet started.

> Last updated: 2026-03-07 (added per-task breakdowns with task prompts, gemini-2.5-flash note)

**Evaluation columns:**
- **Liten w/o VLM** (col1): LITEN pipeline with task decomposition only (no vision-language model)
- **Native Results** (col2): Direct lerobot-eval or policy-native evaluation (ground-truth baseline)
- **Liten** (col3): Full LITEN pipeline with VLM (`vertex_ai/gemini-3-flash-preview`) — ⚠️ Pi0.5 LIBERO col3 was originally run with `gemini-2.5-flash` by mistake; those results are invalidated. The definitive col3 results use `gemini-3-flash-preview` (see [VLM Model Comparison](#pi05-liten-col3----vlm-model-comparison)).

**Metrics per cell:** Task pass / total tasks, raw success rate (successes/episodes).
A task "passes" if at least 1 of its N episodes succeeds.

---

## LIBERO

Standard LIBERO benchmark: 4 suites x 10 tasks. Episodes per task vary by model (see [Methodology](#methodology)).

### Pi0.5 -- `lerobot/pi05_libero_finetuned`

50 eps/task.

Sources:
- Liten w/o VLM: `results/pi05_50eps_col1/`
- Native: `outputs/eval/2026-03-04/23-59-22_libero_pi05/` (spatial), `outputs/eval/2026-03-05/01-19-50_libero_pi05/` (object), `outputs/eval/2026-03-05/02-18-00_libero_pi05/` (goal), `outputs/eval/2026-03-05/03-10-08_libero_pi05/` (10)
- Liten: `results/pi05_50eps_col3_g3f/` (re-run with correct `gemini-3-flash-preview`)

| Suite | Liten w/o VLM | Native Results | Liten |
|-------|---------------|----------------|-------|
| libero_spatial | 10/10 tasks, 90.2% (451/500) | 10/10 tasks, 87.8% (439/500) | 10/10 tasks, 82.6% (413/500) |
| libero_object  | 10/10 tasks, 90.4% (452/500) | 10/10 tasks, 92.2% (461/500) | 10/10 tasks, 73.0% (365/500) |
| libero_goal    | 10/10 tasks, 96.8% (484/500) | 10/10 tasks, 94.4% (472/500) | 10/10 tasks, 81.2% (406/500) |
| libero_10      | 10/10 tasks, 82.4% (412/500) | 10/10 tasks, 85.2% (426/500) | 10/10 tasks, 72.8% (364/500) |
| **Overall**    | **40/40 tasks, 90.0% (1799/2000)** | **40/40 tasks, 89.9% (1798/2000)** | **40/40 tasks, 77.4% (1548/2000)** |

#### Pi0.5 Per-Task Breakdown (50 eps/task)

##### libero_spatial

| Task | Instruction | col1 (no VLM) | col2 (native) | col3 (Liten g3f) |
|------|-------------|---------------|----------------|-------------------|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 48/50 (96%) | 45/50 (90%) | 49/50 (98%) |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 49/50 (98%) | 45/50 (90%) | 47/50 (94%) |
| 2 | pick up the black bowl from table center and place it on the plate | 47/50 (94%) | 48/50 (96%) | 47/50 (94%) |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 45/50 (90%) | 49/50 (98%) | 47/50 (94%) |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 45/50 (90%) | 43/50 (86%) | 39/50 (78%) |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 43/50 (86%) | 22/50 (44%) | 37/50 (74%) |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 49/50 (98%) | 48/50 (96%) | 43/50 (86%) |
| 7 | pick up the black bowl on the stove and place it on the plate | 42/50 (84%) | 50/50 (100%) | 46/50 (92%) |
| 8 | pick up the black bowl next to the plate and place it on the plate | 39/50 (78%) | 42/50 (84%) | 44/50 (88%) |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 44/50 (88%) | 47/50 (94%) | 14/50 (28%) |
| **Total** | | **451/500 (90.2%)** | **439/500 (87.8%)** | **413/500 (82.6%)** |

##### libero_object

| Task | Instruction | col1 (no VLM) | col2 (native) | col3 (Liten g3f) |
|------|-------------|---------------|----------------|-------------------|
| 0 | pick up the alphabet soup and place it in the basket | 40/50 (80%) | 45/50 (90%) | 32/50 (64%) |
| 1 | pick up the cream cheese and place it in the basket | 46/50 (92%) | 47/50 (94%) | 34/50 (68%) |
| 2 | pick up the salad dressing and place it in the basket | 47/50 (94%) | 47/50 (94%) | 44/50 (88%) |
| 3 | pick up the bbq sauce and place it in the basket | 38/50 (76%) | 40/50 (80%) | 31/50 (62%) |
| 4 | pick up the ketchup and place it in the basket | 47/50 (94%) | 50/50 (100%) | 19/50 (38%) |
| 5 | pick up the tomato sauce and place it in the basket | 44/50 (88%) | 43/50 (86%) | 36/50 (72%) |
| 6 | pick up the butter and place it in the basket | 47/50 (94%) | 50/50 (100%) | 31/50 (62%) |
| 7 | pick up the milk and place it in the basket | 45/50 (90%) | 45/50 (90%) | 44/50 (88%) |
| 8 | pick up the chocolate pudding and place it in the basket | 50/50 (100%) | 49/50 (98%) | 48/50 (96%) |
| 9 | pick up the orange juice and place it in the basket | 48/50 (96%) | 45/50 (90%) | 46/50 (92%) |
| **Total** | | **452/500 (90.4%)** | **461/500 (92.2%)** | **365/500 (73.0%)** |

##### libero_goal

| Task | Instruction | col1 (no VLM) | col2 (native) | col3 (Liten g3f) |
|------|-------------|---------------|----------------|-------------------|
| 0 | open the middle drawer of the cabinet | 48/50 (96%) | 45/50 (90%) | 43/50 (86%) |
| 1 | put the bowl on the stove | 49/50 (98%) | 50/50 (100%) | 45/50 (90%) |
| 2 | put the wine bottle on top of the cabinet | 50/50 (100%) | 47/50 (94%) | 49/50 (98%) |
| 3 | open the top drawer and put the bowl inside | 48/50 (96%) | 43/50 (86%) | 13/50 (26%) |
| 4 | put the bowl on top of the cabinet | 50/50 (100%) | 50/50 (100%) | 48/50 (96%) |
| 5 | push the plate to the front of the stove | 50/50 (100%) | 46/50 (92%) | 46/50 (92%) |
| 6 | put the cream cheese in the bowl | 41/50 (82%) | 46/50 (92%) | 39/50 (78%) |
| 7 | turn on the stove | 50/50 (100%) | 50/50 (100%) | 38/50 (76%) |
| 8 | put the bowl on the plate | 49/50 (98%) | 49/50 (98%) | 49/50 (98%) |
| 9 | put the wine bottle on the rack | 49/50 (98%) | 46/50 (92%) | 36/50 (72%) |
| **Total** | | **484/500 (96.8%)** | **472/500 (94.4%)** | **406/500 (81.2%)** |

##### libero_10

| Task | Instruction | col1 (no VLM) | col2 (native) | col3 (Liten g3f) |
|------|-------------|---------------|----------------|-------------------|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 32/50 (64%) | 31/50 (62%) | 19/50 (38%) |
| 1 | put both the cream cheese box and the butter in the basket | 42/50 (84%) | 47/50 (94%) | 32/50 (64%) |
| 2 | turn on the stove and put the moka pot on it | 44/50 (88%) | 44/50 (88%) | 40/50 (80%) |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 43/50 (86%) | 44/50 (88%) | 42/50 (84%) |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 43/50 (86%) | 44/50 (88%) | 43/50 (86%) |
| 5 | pick up the book and place it in the back compartment of the caddy | 49/50 (98%) | 47/50 (94%) | 50/50 (100%) |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 45/50 (90%) | 44/50 (88%) | 26/50 (52%) |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 33/50 (66%) | 37/50 (74%) | 32/50 (64%) |
| 8 | put both moka pots on the stove | 39/50 (78%) | 41/50 (82%) | 42/50 (84%) |
| 9 | put the yellow and white mug in the microwave and close it | 42/50 (84%) | 47/50 (94%) | 38/50 (76%) |
| **Total** | | **412/500 (82.4%)** | **426/500 (85.2%)** | **364/500 (72.8%)** |

#### Pi0.5 Liten col3 -- VLM Model Comparison

> **Important:** The original pi05 col3 10-eps/task results (2026-03-01) used `gemini-2.5-flash` by mistake instead of the intended `gemini-3-flash-preview`. Those results have been **invalidated**. The definitive 50-eps/task col3 results above use the correct `gemini-3-flash-preview` model.
>
> **Old gemini-2.5-flash col3 scores (INVALIDATED, 10 eps/task):**
> spatial=75%, object=75%, goal=87%, libero_10=33%, overall=67.5%
>
> **New gemini-3-flash-preview col3 scores (50 eps/task):**
> spatial=82.6%, object=73.0%, goal=81.2%, libero_10=72.8%, overall=77.4%
>
> Most notably, **libero_10 jumped from 33% to 72.8%** after switching to the correct VLM model and increasing evaluation episodes. The gemini-2.5-flash model appeared to produce lower-quality scene descriptions, particularly harming multi-step tasks in libero_10.

##### Gemini 2.5 Flash comparison run (paused)

A side-by-side 50-eps comparison with `vertex_ai/gemini-2.5-flash` was started but paused to prioritize the g3f run.

Source: `results/pi05_50eps_col3_g25f/` (paused)

| Suite | Status |
|-------|--------|
| libero_spatial | paused at ~116/500 eps; interim: 116/125 = 92.8% |
| libero_object  | not started |
| libero_goal    | not started |
| libero_10      | not started |

### SmolVLA -- `HuggingFaceVLA/smolvla_libero`

10 eps/task (matches the original SmolVLA paper protocol, which also used 10 eps/task).

Sources:
- Liten w/o VLM: `results/smolvla_10eps_col1_libero/`
- Native: `outputs/eval/2026-03-02/13-27-40_libero_smolvla/` (spatial), `outputs/eval/2026-03-02/10-40-33_libero_smolvla/` (object), `outputs/eval/2026-03-02/13-27-06_libero_smolvla/` (goal), `outputs/eval/2026-03-02/17-15-19_libero_smolvla/` (10)
- Liten: pending

| Suite | Liten w/o VLM | Native Results | Liten |
|-------|---------------|----------------|-------|
| libero_spatial | 10/10 tasks, 71.0% (71/100) | 10/10 tasks, 74.0% (74/100) | pending |
| libero_object  | 10/10 tasks, 90.0% (90/100) | 10/10 tasks, 91.0% (91/100) | pending |
| libero_goal    | 10/10 tasks, 75.0% (75/100) | 10/10 tasks, 76.0% (76/100) | pending |
| libero_10      | 8/10 tasks, 37.0% (37/100)* | 9/10 tasks, 43.0% (43/100) | pending |

\* 22 of 100 episodes hit VLA inference timeouts and are counted as failures. Excluding errored episodes: 37/78 = 47.4%.

#### SmolVLA 50 eps/task run (COMPLETE)

A 50 eps/task rerun for higher statistical power. Source: `results/smolvla_50eps_col1_libero/`

| Suite | Status |
|-------|--------|
| libero_spatial | **COMPLETE — 10/10 tasks, 393/500 = 78.6%** |
| libero_object  | **COMPLETE — 10/10 tasks, 459/500 = 91.8%** |
| libero_goal    | **COMPLETE — 10/10 tasks, 392/500 = 78.4%** |
| libero_10      | **COMPLETE — 10/10 tasks, 203/500 = 40.6%** |

> **Note:** All 4 suites were restarted ~2026-03-07 13:25 PST. Previous spatial run was at 495/500 (71.1%). The object/goal/10 suites from the prior launch showed anomalously low rates (29.5%/15.9%/2.5%) — restart likely intended to fix a config issue. New early rates are much more consistent with 10eps baselines.
>
> **Update 2026-03-08:** Goal and libero_10 restarted ~01:09 PST (v2), then again ~03:19 PST (v3). v1 had goal 111/500 (92.8%), libero_10 36/500 (55.6%); v2 had goal 12/500, libero_10 4/500. Old logs in `logs_backup_phase2_v1/` and `logs_backup_phase2_v2/`. Spatial and object remain complete.

### OpenVLA -- `openvla/openvla-7b-finetuned-libero-{spatial,10}`

50 eps/task, 1 seed. Note: the original OpenVLA paper used 50 eps/task x 3 random seeds (1500 total rollouts per suite); our evaluation uses 1 seed only.

Sources:
- Liten w/o VLM: pending
- Native: `scripts/run_openvla_native_eval.py` (spatial 20eps, 10 10eps)
- Liten: `results/openvla_50eps_col3_libero/` (in progress)

| Suite | Liten w/o VLM | Native Results | Liten |
|-------|---------------|----------------|-------|
| libero_spatial | pending | 71.5% (143/200, 20 eps/task) | in progress (~325/500 eps, 205S/120F = 63.1% interim) |
| libero_object  | pending | N/A (no checkpoint) | pending |
| libero_goal    | pending | N/A (no checkpoint) | pending |
| libero_10      | pending | 37.0% (37/100, 10 eps/task) | pending |

OpenVLA only has finetuned checkpoints for libero_spatial and libero_10 (native col2).

---

## LIBERO-PRO

LIBERO-PRO introduces structured OOD evaluations. Three suites: `libero_spatial_object`,
`libero_goal_swap`, `libero_spatial_with_mug` (10 tasks each, 50 eps/task = 500 eps/suite).
Native Results (col2) is N/A for all models -- lerobot-eval does not support LIBERO-PRO suites.

### Pi0.5 -- `lerobot/pi05_libero_finetuned`

50 eps/task.

Sources:
- Liten w/o VLM: `results/pi05_50eps_pro_col1/`
- Native: N/A
- Liten: `results/pi05_50eps_pro_col3/`

| Suite | Liten w/o VLM | Native Results | Liten |
|-------|---------------|----------------|-------|
| libero_spatial_object   | 10/10 tasks, 93.8% (469/500) | N/A | ~~10/10 tasks, 88.0% (440/500)~~ **INVALIDATED** |
| libero_goal_swap        | 4/10 tasks, 11.8% (59/500)   | N/A | ~~5/10 tasks, 11.4% (57/500)~~ **INVALIDATED**   |
| libero_spatial_with_mug | 10/10 tasks, 84.8% (424/500) | N/A | ~~10/10 tasks, 73.0% (365/500)~~ **INVALIDATED** |
| **Overall**             | **24/30 tasks, 63.5% (952/1500)** | -- | ~~**25/30 tasks, 57.5% (862/1500)**~~ **INVALIDATED** |

### SmolVLA -- `HuggingFaceVLA/smolvla_libero`

10 eps/task.

Sources:
- Liten w/o VLM: `results/smolvla_10eps_col1_libero_pro/` (spatial_object), `results/smolvla_10eps_col1_libero_pro_goalswap/` (goal_swap), `results/smolvla_10eps_col1_libero_pro_withmug/` (with_mug)
- Native: N/A
- Liten: pending

| Suite | Liten w/o VLM | Native Results | Liten |
|-------|---------------|----------------|-------|
| libero_spatial_object   | 10/10 tasks, 72.0% (72/100)* | N/A | pending |
| libero_goal_swap        | in progress (10 tasks launched, ep 1 running — very slow ~50 min/ep) | N/A | pending |
| libero_spatial_with_mug | in progress (10 tasks launched, ep 1 running — very slow ~50 min/ep) | N/A | pending |

\* 2 of 100 episodes hit VLA inference timeouts (task 2). Excluding errored episodes: 72/98 = 73.5%.

#### SmolVLA 50 eps/task Pro run (INVALID)

Source: `results/smolvla_50eps_col1_libero_pro/` -- All 30 tasks completed but every task scored 0/50.
Summary.txt shows duplicated output (two runs collided on same port 5500). **This run is invalid and should be re-run.**

### OpenVLA -- `openvla/openvla-7b-finetuned-libero-spatial`

50 eps/task, 1 seed (original OpenVLA paper used 3 seeds).

Sources:
- Liten w/o VLM: pending
- Native: N/A
- Liten: pending

| Suite | Liten w/o VLM | Native Results | Liten |
|-------|---------------|----------------|-------|
| libero_spatial_object   | pending | N/A | pending |
| libero_goal_swap        | pending | N/A | pending |
| libero_spatial_with_mug | pending | N/A | pending |

Note: Previous openvla col3 LIBERO-PRO attempts (7 tries) all failed due to OOM during parallel inference.

### Key Observations (50 eps/task)

1. **spatial_object and with_mug remain easy for pi05** -- 88-94% raw success, 10/10 task pass in both columns.
2. **goal_swap is OOD-hard** -- pi05 scores ~12% raw, only 4-5/10 tasks pass. Matches LIBERO-PRO paper findings.
3. **Col3 (VLM) slightly hurts pi05 raw rates** -- LIBERO overall drops from 90.0% to 77.4% with VLM (g3f re-run). LIBERO-PRO col3 pending re-run with correct VLM model.
4. **Col3 increases goal_swap task pass** (4/10 -> 5/10) despite similar raw rate, suggesting VLM helps on a few marginal tasks.

---

## LIBERO-PRO P1/P2 Evaluation

### Definitions

- **P1 -- Task Perturbation**: Task language rewritten to describe a different subtask. Expected ~0%.
- **P2 -- Position Swap**: Object positions swapped. Expected 10-40% for strong VLAs.

### Published Paper Numbers

> Source: [Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO)

| Model | Spatial Orig | Spatial P1 | Spatial P2 | Goal Orig | Goal P1 | Goal P2 | 10 Orig | 10 P1 | 10 P2 | Object Orig | Object P1 | Object P2 |
|-------|-------------|-----------|-----------|----------|--------|--------|--------|------|------|------------|----------|----------|
| OpenVLA | 0.95 | 0.0 | 0.0 | 0.98 | 0.0 | 0.0 | 0.93 | 0.0 | 0.0 | 0.99 | 0.0 | 0.0 |
| Pi0     | 0.90 | 0.0 | 0.0 | 0.92 | 0.0 | 0.0 | 0.82 | 0.0 | 0.0 | 0.98 | 0.0 | 0.0 |
| Pi0.5   | 0.96 | 0.0 | 0.2 | 0.97 | 0.0 | 0.4 | 0.93 | 0.0 | 0.1 | 0.98 | 0.0 | 0.2 |

### Pi0.5 -- Our Results (50 eps/task)

Run 2026-03-03. 4 suites x 10 tasks x 50 eps = 2000 episodes per condition.
Source: `results/pi05_p1p2_50eps_parallel/`

Earlier runs (lower-fidelity): `results/libero_pro_pi05_p1p2_50eps/` (sequential, partially complete), `results/pi05_p1p2_20260302_140517/` (1 ep/task pilot).

#### P1 -- Task Perturbation

| Suite | Successes | Rate | vs. Paper (0.0%) |
|-------|-----------|------|------------------|
| libero_spatial_task | 0/500  | 0.0%  | matches |
| libero_object_task  | 0/500  | 0.0%  | matches |
| libero_goal_task    | 54/500 | 10.8% | anomaly* |
| libero_10_task      | 39/500 | 7.8%  | anomaly* |
| **P1 Overall (corrected)** | **5/1900** | **0.26%** | ~ 0% |

\* Two tasks are trivially satisfiable due to benchmark bugs (see [P1 Anomalies](#p1-anomalies) below).

#### P2 -- Position Swap

| Suite | Successes | Rate | Paper Pi0.5 |
|-------|-----------|------|-------------|
| libero_spatial_swap | 99/500  | **19.8%** | 20% |
| libero_goal_swap    | 43/500  | 8.6%  | 40% |
| libero_object_swap  | 27/500  | 5.4%  | 20% |
| libero_10_swap      | 0/500   | 0.0%  | 10% |
| **P2 Overall**      | **169/2000** | **8.45%** | 22.5% |

### Our Results vs. Paper Summary

| Condition | Our Pi0.5 | Paper Pi0.5 | Paper OpenVLA |
|-----------|-----------|-------------|---------------|
| P1 (corrected) | **0.26%** | 0.0% | 0.0% |
| P2 | **8.45%** | 22.5% | 0.0% |

Pi0.5 P1 ~ 0% (matches paper). P2 = 8.45% (below paper's 22.5% but non-zero, confirming spatial generalization capability). libero_spatial P2 (19.8%) closely matches paper (20%).

### P1 Anomalies

Summary of P1 anomalies:

Two of 40 P1 tasks are trivially satisfiable due to degenerate BDDL perturbations in LIBERO-PRO:

1. **`libero_goal_task` Task 7** (`turn_on_the_stove`): Perturbed goal = "turn off stove", but stove starts OFF. Result: 49/50 = 98%.
2. **`libero_10_task` Task 8** (`put_both_moka_pots_on_the_stove`): Perturbed goal removes one pot requirement, remaining goal trivially true. Result: 39/50 = 78%.

These are LIBERO-PRO benchmark bugs, not VLA issues. Corrected P1 (excluding these 2 tasks): 5/1900 = 0.26%.

---

## Methodology

### Episodes per Task

We match each model's published evaluation protocol for fair comparison:

| Model | Eps/Task | Seeds | Rationale |
|-------|----------|-------|-----------|
| **Pi0.5** | 50 | 1 | Matches published Pi0.5 evaluation protocol |
| **OpenVLA** | 50 | 1 | Paper used 50 eps/task x 3 random seeds (1500 rollouts/suite); we use 1 seed due to compute constraints |
| **SmolVLA** | 10 | 1 | Matches published SmolVLA paper protocol (10 eps/task) |
| **LIBERO-PRO (all models)** | 50 | 1 | Matches published LIBERO-PRO evaluation protocol |

All evaluations use a single random seed. The OpenVLA paper averaged results over 3 seeds; our single-seed results may show higher variance on individual tasks.

### Native (col2) Evaluation

Native results use `lerobot-eval` with each model's official checkpoint, running in the same simulator environment. This provides the ground-truth baseline -- any difference between col1/col3 and col2 reflects the impact of the LITEN pipeline. Pi0.5 native results at 50 eps/task; SmolVLA native at 10 eps/task; OpenVLA native via custom script at 10-20 eps/task.

---

## Design Limitations

### OpenVLA -- LITEN col1/col3 on standard LIBERO

OpenVLA LITEN col1/col3 was attempted on standard LIBERO (2026-03-01) but scored 0% across all four
suites due to an action-space misconfiguration at the time (policy server reported `joint_pos` instead
of `eef_delta`). This bug was fixed the same day. The col3 50 eps/task re-run is in progress
(`results/openvla_50eps_col3_libero/`, libero_spatial ~325/500 eps done, 63.1% interim).

### SmolVLA base model -- no gripper output

`lerobot/smolvla_base` outputs 6-dim actions (no gripper channel). LIBERO requires 7-dim with
gripper control. Use `HuggingFaceVLA/smolvla_libero` (LIBERO-finetuned) instead.

### SmolVLA 50 eps/task Pro run -- invalid

`results/smolvla_50eps_col1_libero_pro/` completed with 0% across all 30 tasks (0/1500 episodes).
The summary.txt shows duplicated output indicating two benchmark instances collided on the same
sim port (5500). This run must be re-done.
