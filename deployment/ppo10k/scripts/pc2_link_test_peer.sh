#!/usr/bin/env bash
set -eu
cd /home/unitree/ppo10k_20261122
# Test endpoint/domain are separate from the real body bridge.
./build/pc2_body_bridge lo 192.168.124.25 16940 16941 79 > logs/isolated_bridge.log 2>&1 &
PPO_TEST_BRIDGE_PID=$!
trap 'kill "$PPO_TEST_BRIDGE_PID" 2>/dev/null || true; wait "$PPO_TEST_BRIDGE_PID" 2>/dev/null || true' EXIT
./build/body_link_integration dds-peer > logs/isolated_dds_receiver.jsonl 2>&1
