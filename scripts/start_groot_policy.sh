#!/usr/bin/env bash
# Thin wrapper — all logic lives in robo_eval/cli.py.
# Env vars GROOT_MODEL_ID, VLA_PORT are forwarded automatically.
exec robo-eval servers start groot "$@"
