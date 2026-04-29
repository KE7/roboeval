#!/usr/bin/env bash
# Start the pi0.5 server.
# Env vars VLA_MODEL and VLA_PORT are forwarded automatically.
exec robo-eval servers start pi05 "$@"
