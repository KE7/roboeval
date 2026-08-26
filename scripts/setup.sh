#!/usr/bin/env bash
# =============================================================================
# scripts/setup.sh — developer entry point (shim)
#
# The real script lives inside the package at roboeval/_scripts/setup.sh so
# that it ships in the wheel and `roboeval setup` works from an installed
# package. This shim keeps the documented source-checkout invocation
# (`./scripts/setup.sh <component> ...`) working unchanged.
# =============================================================================

set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_real="$_here/../roboeval/_scripts/setup.sh"

if [[ ! -f "$_real" ]]; then
    echo "error: bundled setup script not found at $_real" >&2
    exit 1
fi

exec bash "$_real" "$@"
