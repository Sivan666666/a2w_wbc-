#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p third_party
git clone https://github.com/unitreerobotics/unitree_sdk2.git third_party/unitree_sdk2
git -C third_party/unitree_sdk2 checkout b7fec6020eb93d4c059a6edf19905e12a75a8d3d
git clone https://github.com/unitreerobotics/z1_sdk.git third_party/z1_sdk
git -C third_party/z1_sdk checkout f1af2b42f2a39a5010946e049ad3fe324e2e6f06
