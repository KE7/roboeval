#!/usr/bin/env bash
# Start the LIBERO Pro simulator server.
exec robo-eval servers start sim --sim libero_pro --headless --port "${PORT:-5010}" "$@"
