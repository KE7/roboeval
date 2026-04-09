#!/usr/bin/env bash
# Thin wrapper — all logic lives in robo_eval/cli.py.
# Sets libero_pro defaults (LD_LIBRARY_PATH, LIBERO_CONFIG_PATH, MUJOCO_GL) in servers.py.
exec robo-eval servers start sim --sim libero_pro --headless --port "${PORT:-5010}" "$@"
