#!/usr/bin/env bash
set -euo pipefail
root=/data01/niuwa/sivan/workspace/leggedmanip_wheelv2_20260926
uv=/data01/niuwa/sivan/workspace/locowheeledlegged-repro/runtime/uv
export UV_CACHE_DIR="/data01/niuwa/sivan/.cache/leggedmanip_wheelv2_20260926/uv" TMPDIR="/data01/niuwa/sivan/tmp/leggedmanip_wheelv2_20260926/install"
export UV_LINK_MODE=copy
mkdir -p "$TMPDIR"
py="$root/runtime/env/bin/python"
test -x "$py"
"$uv" pip install --python "$py" --no-deps \
  'rsl-rl-lib==5.0.1' 'tensordict==0.14.2' 'pyvers==0.2.3' 'onnxscript==0.5.4' \
  'onnx_ir==1.0.0' 'orjson==3.12.0' 'onnx==1.22.0' 'ml_dtypes==0.5.4'
# Match the PS1 environment and satisfy the pinned Lab's declared dependencies.
"$uv" pip install --python "$py" --no-deps --no-build-isolation \
  'gymnasium==1.2.1' 'transformers==4.57.6' 'starlette==0.49.1' \
  'flatdict==4.0.0' 'daqp==0.7.2' 'typing-extensions==4.16.0' \
  'importlib-metadata==9.0.1' 'zipp==4.1.0' \
  'tokenizers==0.22.2' 'huggingface-hub==0.36.2' \
  'fastapi==0.141.1' 'annotated-doc==0.0.5'
"$uv" pip install --python "$py" --no-deps --no-build-isolation \
  -e "$root/IsaacLab/source/isaaclab" \
  -e "$root/IsaacLab/source/isaaclab_assets" \
  -e "$root/IsaacLab/source/isaaclab_tasks" \
  -e "$root/IsaacLab/source/isaaclab_rl" \
  -e "$root/LeggedManip_Lab/source/LeggedManip_Lab"
"$py" -c 'import sys, torch, rsl_rl, tensordict; import importlib.metadata as m; print(sys.prefix); print(torch.__version__, rsl_rl.__file__, m.version("rsl-rl-lib"),tensordict.__version__)'
