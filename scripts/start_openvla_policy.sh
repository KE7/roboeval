#!/usr/bin/env bash
# Thin wrapper — all logic lives in robo_eval/cli.py.
# Env vars OPENVLA_MODEL_ID, VLA_PORT, OPENVLA_UNNORM_KEY are forwarded automatically.
exec robo-eval servers start openvla "$@"
