#!/usr/bin/env bash
set -euo pipefail
PPO10K_PC2_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "$PPO10K_PC2_ROOT/build/pc2_body_bridge" eth0 192.168.124.25 16840 16841 0
