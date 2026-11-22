#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
ssh -o BatchMode=yes unitree@192.168.124.162 \
  'bash /home/unitree/ppo10k_20261122/project/scripts/pc2_link_test_peer.sh' \
  > ../logs/isolated_peer.log 2>&1 &
PPO_TEST_PEER_PID=$!
./build/body_link_integration sender 192.168.124.162 > ../logs/isolated_sender.jsonl 2>&1
PPO_TEST_SENDER_RESULT=$?
wait "$PPO_TEST_PEER_PID"
PPO_TEST_PEER_RESULT=$?
echo "sender=$PPO_TEST_SENDER_RESULT peer=$PPO_TEST_PEER_RESULT"
test "$PPO_TEST_SENDER_RESULT" -eq 0 && test "$PPO_TEST_PEER_RESULT" -eq 0
