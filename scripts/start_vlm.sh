#!/usr/bin/env bash
# Thin wrapper — all logic lives in robo_eval/cli.py.
# Env vars LITELLM_PORT and VLM_MODEL are forwarded automatically.
exec robo-eval servers start vlm "$@"
