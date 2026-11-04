#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# Non-root container UID intentionally has no /etc/passwd entry. Inductor and
# Triton must not write shared/root caches or query a missing username.
export USER=ps LOGNAME=ps
export TORCHINDUCTOR_CACHE_DIR="$root/.runtime/cache/inductor"
export TRITON_CACHE_DIR="$root/.runtime/cache/triton"
export TORCHINDUCTOR_COMPILE_THREADS=4
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
exec bash "$root/reproduction/supermimic_runtime.sh" "$@"
