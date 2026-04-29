#!/usr/bin/env bash
# Start the OpenVLA server.
# Env vars OPENVLA_MODEL_ID, VLA_PORT, OPENVLA_UNNORM_KEY are forwarded automatically.
exec robo-eval servers start openvla "$@"
