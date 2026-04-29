# Third-Party Attributions

roboeval is an evaluation harness that wraps or interfaces with several third-party simulators,
model checkpoints, and Python packages. The attributions below cover the components that are
downloaded, installed, or invoked at runtime. Each entry lists the upstream project, its license,
the URL where it can be obtained, and how roboeval uses it.

---

## Simulation Environments

### LIBERO
- **License:** MIT
- **URL:** https://github.com/Lifelong-Robot-Learning/LIBERO
- **Citation:** Liu et al. "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning." NeurIPS 2023.
- **Use in roboeval:** Default evaluation benchmark for manipulation tasks (libero_spatial, libero_object, libero_goal, libero_10 suites).

### LIBERO-PRO
- **License:** MIT (inherits LIBERO)
- **URL:** https://github.com/Lifelong-Robot-Learning/LIBERO (extended branch)
- **Use in roboeval:** Extended LIBERO suites with additional perturbation tasks (libero_pro_* suites).

### LIBERO-Infinity
- **License:** MIT
- **URL:** https://github.com/Lifelong-Robot-Learning/LIBERO (libero-infinity extension)
- **Use in roboeval:** Procedurally-generated LIBERO tasks via the Scenic scenario description language (libero_infinity_* suites).

### RoboCasa
- **License:** MIT
- **URL:** https://github.com/robocasa/robocasa
- **Citation:** Nasiriany et al. "RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots." RSS 2024.
- **Use in roboeval:** Kitchen manipulation benchmark (robocasa_kitchen suite).

### RoboTwin
- **License:** MIT
- **URL:** https://github.com/TeleAI-Labs/RoboTwin
- **Citation:** "RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins." arXiv 2024.
- **Use in roboeval:** Dual-arm manipulation benchmark (robotwin_aloha_agilex suite).

---

## VLA Model Checkpoints

Model weights are downloaded from HuggingFace Hub at first run. Each model carries its own
license from the upstream authors. Check the HuggingFace model card before use.

### Pi0.5 (Physical Intelligence)
- **HuggingFace:** `lerobot/pi05_libero_finetuned`
- **License:** Apache 2.0 (see model card)
- **Citation:** Black et al. "π0: A Vision-Language-Action Flow Model for General Robot Control." arXiv 2024.

### SmolVLA (HuggingFace)
- **HuggingFace:** `HuggingFaceVLA/smolvla_libero`
- **License:** Apache 2.0 (see model card)
- **URL:** https://huggingface.co/HuggingFaceVLA

### OpenVLA
- **HuggingFace:** `openvla/openvla-7b-finetuned-libero-*`
- **License:** MIT (see model card)
- **Citation:** Kim et al. "OpenVLA: An Open-Source Vision-Language-Action Model." arXiv 2024.
- **URL:** https://github.com/openvla/openvla

### Cosmos Policy (NVIDIA)
- **HuggingFace:** `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`
- **License:** NVIDIA Research License (see model card)
- **URL:** https://huggingface.co/nvidia

### GR00T N1.6 (NVIDIA)
- **HuggingFace:** `nvidia/GR00T-N1.6-3B`
- **License:** NVIDIA Research License (see model card)
- **URL:** https://huggingface.co/nvidia/GR00T-N1.6-3B

### InternVLA (InternRobotics)
- **HuggingFace:** `InternRobotics/InternVLA-A1-3B-RoboTwin`
- **License:** Apache 2.0 (see model card)
- **URL:** https://huggingface.co/InternRobotics

---

## Python Packages

The following packages are declared as runtime dependencies in `pyproject.toml`. All are
available on PyPI under open-source licenses.

| Package | License | URL |
|---|---|---|
| numpy | BSD-3-Clause | https://numpy.org |
| openai | MIT | https://github.com/openai/openai-python |
| pillow | HPND (PIL-compatible) | https://python-pillow.org |
| opencv-python | MIT | https://github.com/opencv/opencv-python |
| tqdm | MIT + MPL-2.0 | https://github.com/tqdm/tqdm |
| pydantic | MIT | https://docs.pydantic.dev |
| typer | MIT | https://typer.tiangolo.com |
| requests | Apache-2.0 | https://requests.readthedocs.io |
| httpx | BSD-3-Clause | https://www.python-httpx.org |
| fastapi | MIT | https://fastapi.tiangolo.com |
| uvicorn | BSD-3-Clause | https://www.uvicorn.org |
| litellm | MIT | https://github.com/BerriAI/litellm |
| filelock | Unlicense/Public Domain | https://github.com/tox-dev/filelock |
| pyyaml | MIT | https://pyyaml.org |

### LeRobot Framework
- **License:** Apache 2.0
- **URL:** https://github.com/huggingface/lerobot
- **Use in roboeval:** VLA inference backend for Pi0.5 and SmolVLA policy servers. Not a direct Python dependency of roboeval itself; installed in the VLA venv (`.venvs/vla`).

---

*If you believe an attribution is missing or incorrect, please open an issue at
https://github.com/KE7/roboeval/issues.*
