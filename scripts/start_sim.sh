#!/usr/bin/env bash
# Thin wrapper — all logic lives in robo_eval/cli.py.
# Pass --sim <backend> [--port PORT] [--headless] [--host HOST] as before.
exec robo-eval servers start sim "$@"
