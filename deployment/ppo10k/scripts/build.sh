#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
DEPLOY_PYTHON="${DEPLOY_PYTHON:-python3}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v "$DEPLOY_PYTHON")" \
  -Dpybind11_DIR="$($DEPLOY_PYTHON -m pybind11 --cmakedir)" "$@"
cmake --build build -j4
