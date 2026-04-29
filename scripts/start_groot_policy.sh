#!/usr/bin/env bash
# Start the GR00T VLA server.
# Env vars GROOT_MODEL_ID, VLA_PORT are forwarded automatically.
exec robo-eval servers start groot "$@"
