#!/usr/bin/env bash
# Thin wrapper — all logic lives in robo_eval/cli.py.
# Env vars VLA_MODEL and VLA_PORT are forwarded automatically.
exec robo-eval servers start pi05 "$@"
